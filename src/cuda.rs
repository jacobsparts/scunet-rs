//! The CUDA half of the engine: the two fatbins, the kernel-name sets, the launch
//! plumbing, and the graph walker that mirrors `src/cpu.rs` stage for stage.
//!
//! WHICH MODULE A KERNEL LIVES IN IS A PROPERTY OF THE NAME. `lightgpu`'s
//! `cuda/kernels.cu` (the toolkit) and this engine's `cuda/scunet.cu` are disjoint;
//! `module_of` asks the project module first and then the toolkit, so promoting a
//! kernel from one to the other is a one-line change here - and a name in NEITHER
//! fails at construction, where the message can say which list is wrong. Five
//! families have made that trip already: the window gather/scatter, the
//! register-blocked GEMM, the warp LayerNorm and the register-blocked 2x2 pair are
//! toolkit kernels now, and each is still A/B-able against the form it replaced.
//!
//! THE WALKER IS DELIBERATELY THE SAME SHAPE AS THE CPU ONE. Every stage boundary,
//! every shift decision and every buffer sits where `cpu::forward` puts it, because
//! the two are read side by side when a fixture disagrees. The differences are the
//! ones the device forces: weights are uploaded once and addressed by name, and one
//! window attention is three launches (the toolkit's gather, this engine's
//! attention, the toolkit's scatter) around two toolkit `lg_linear`s, where the CPU
//! has a single function.
use std::collections::HashMap;

use lightgpu::vm::{Args, DevBuf, Event, Launch, Module};

use crate::plan::Plan;
use crate::weights::Weights;

/// The toolkit's kernels this graph launches, compiled into the toolkit fatbin's
/// `--entries` by `build.rs`, so a name here the toolkit does not define fails the
/// build rather than the first forward pass.
///
/// `lg_conv3x3s1p1` is `m_head`/`m_tail` and any 3x3 whose `c_in` is 3; every other
/// 3x3 uses `lg_conv3x3_winograd`, the same op at 4/9ths the multiplies when the
/// transform amortises - and it LOSES at `c_in = 3`, which is why the first layer
/// stays on the direct kernel. `lg_gelu_erf` is the MLP's activation, `lg_relu` the
/// conv half's, `lg_add` every residual and skip, `lg_copy` the one copy the walker
/// needs.
///
/// THE FIRST FIVE ARE THE THREE-DIMENSIONAL OPS OF THE GRAPH: the 3x3 convolutions,
/// the 1x1 convolutions, the token linears and the LayerNorms. Each of those has a
/// SECOND, register-blocked form below it in this list, and the switch named on each
/// one A/Bs them - which is how every promoted kernel justified itself. See
/// `build.rs` for what each replacement measured.
pub const TOOLKIT_KERNELS: &[&str] = &[
    "lg_conv3x3s1p1",
    "lg_conv3x3_winograd",
    "lg_conv1x1",
    "lg_linear",
    "lg_layer_norm",
    // The register-blocked forms, all promoted out of this engine's own
    // `cuda/scunet.cu` once they proved general.
    "lg_linear_rb",
    "lg_conv1x1_rb",
    "lg_layer_norm_warp",
    "lg_conv2x2s2",
    "lg_conv_t2x2",
    // The Swin window index map, which `rmbg-rs` also uses now. The attention
    // BETWEEN the two calls is still this engine's own.
    "lg_window_gather",
    "lg_window_scatter",
    // NOT USED BY THE GRAPH. It is a kernel that returns immediately, and it is here
    // so `bench_op("noop")` can measure what ONE LAUNCH costs on this machine - which
    // multiplied by a forward's launch count is the floor no kernel change can go
    // below, and which the event-pair profiler cannot see because its own events cost
    // more than the launches they wrap.
    "lg_noop",
    "lg_gelu_erf",
    "lg_relu",
    "lg_add",
    "lg_copy",
];

/// This engine's own kernels, from `cuda/scunet.cu`, validated by `build.rs`
/// against the source in both directions.
///
/// WHAT IS LEFT IS THE ATTENTION AND ONE COMPARISON ARM. The window attention is
/// the architecture rather than a generic op - a thread owns a whole query-head
/// row, with the axis-wide mask and the learned relative-position bias - so it has
/// no place in a toolkit. The elementwise 2x2 stride-2 convolution and its
/// transposed twin are the `SCUNET_C2X2=elem` arm: the toolkit's register-blocked
/// pair is what the graph runs, and these are what it was measured against.
pub const PROJECT_KERNELS: &[&str] = &[
    "sc_window_attn",
    "sc_window_attn_s",
    "sc_window_attn_r64",
    "sc_conv2x2s2",
    "sc_conv_t2x2",
];

const TOOLKIT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scunet_toolkit.fatbin"));
const PROJECT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scunet_project.fatbin"));

/// Threads per block for the kernels that take one output element per thread.
const BLOCK: usize = 256;

/// The kernel-variant switches and the instruments live in `crate::dev`, which also
/// holds the refusal that keeps them out of a release binary. The gate is there rather
/// than at each call site so the choices below read the same in both kinds of build.
use crate::dev::switch as dev_switch;


/// Grid of `n` one-per-thread work items.
fn grid(n: usize) -> (u32, u32, u32) {
    (((n + BLOCK - 1) / BLOCK) as u32, 1, 1)
}

/// Per-kernel device time, collected only when `SCUNET_PROFILE` is set.
///
/// The event PAIRS are kept, not the elapsed values: reading `cuEventElapsedTime`
/// requires the later event to have completed, and synchronising per kernel would
/// serialise the pipeline being measured. `Cuda::profile_report` does the one
/// synchronise, after which every pair is readable.
struct Profiler {
    spans: Vec<(String, Event, Event)>,
}

impl Profiler {
    fn push(&mut self, name: &str, t0: Event, t1: Event) {
        self.spans.push((name.to_string(), t0, t1));
    }

    /// (name, launches, total ms), sorted by total time descending.
    fn totals(&self) -> Vec<(String, usize, f32)> {
        let mut acc: HashMap<&str, (usize, f32)> = HashMap::new();
        for (name, t0, t1) in &self.spans {
            let ms = t0.elapsed_ms(t1).unwrap_or(0.0);
            let e = acc.entry(name.as_str()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms;
        }
        let mut v: Vec<(String, usize, f32)> =
            acc.into_iter().map(|(k, (n, ms))| (k.to_string(), n, ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }
}

/// WHERE A STAGE SKIP IS WAITING while the up pass is still to come.
///
/// `host` is the default and it is a MEMORY decision: the three skips together are
/// 56 floats per input pixel - 896 MiB at 2048x2048 - and keeping them in host RAM
/// is what makes that size fit on an 8 GB card. The cost is one D2H when the down
/// stage ends and one H2D when the up stage consumes it, and the add back is exact
/// because it is element-wise over DISJOINT ranges: every output element is added
/// exactly once and no sum is reordered, so the two forms agree bit for bit.
///
/// `device` keeps the buffer itself. It costs that 896 MiB, and it exists so the
/// round trip can be MEASURED rather than estimated - no per-kernel instrument in
/// this repository sees a host transfer, `bench_op` included, since it times compute.
enum Skip {
    Host(Vec<f32>),
    Device(DevBuf),
}

/// The CUDA backend: two modules, an upload-once weight cache and a pool of
/// scratch buffers, so a second forward is launches only.
pub struct Cuda<'a> {
    wt: &'a Weights,
    toolkit: Module,
    project: Module,
    /// Every weight the walker has asked for, uploaded once.
    weights: HashMap<String, DevBuf>,
    /// Reusable scratch, keyed by element count.
    pool: HashMap<usize, Vec<DevBuf>>,
    /// Set only when `SCUNET_PROFILE` is in the environment; `run` consults it on
    /// every launch, so the ordinary path pays one `Option` test.
    profiler: Option<std::cell::RefCell<Profiler>>,
    /// Device bytes allocated and not yet released, and the HIGH-WATER of that.
    ///
    /// THE HIGH-WATER IS WHAT DECIDES WHETHER AN IMAGE FITS, and it is not the
    /// number `footprint` reports. `footprint` sums the pool, which counts every
    /// buffer a forward has ever held including ones held at different moments, so
    /// it answers "how much has this backend allocated" rather than "what must the
    /// card have free at once". A pool buffer is still device memory after
    /// `give_back` - it is only released by `release_scratch` - so `live` counts
    /// every byte ever allocated and `peak` is the point at which the card had to
    /// hold the most.
    live: usize,
    peak: usize,
    /// Every allocation, in order, when `SCUNET_MEMTRACE` is set: (label, bytes,
    /// live after). The run up to the peak is what a size is actually made of, and
    /// labels are what separate two buffers of the same byte count.
    trace: Option<std::cell::RefCell<Vec<(String, usize, usize)>>>,
    /// The transfer staging for `add_host`, allocated once and NOT taken from the
    /// pool: a pooled plane can be a whole stage's activation, and holding one of
    /// those for the length of a transfer would raise the peak this exists to lower.
    host_stage: Option<DevBuf>,
}

impl<'a> Cuda<'a> {
    /// Load both fatbins and resolve every kernel the graph can launch, so a name
    /// in neither module fails here rather than at the first forward pass.
    pub fn new(wt: &'a Weights) -> Result<Cuda<'a>, String> {
        let toolkit = Module::load(TOOLKIT_FATBIN)?;
        let project = Module::load(PROJECT_FATBIN)?;
        for k in PROJECT_KERNELS {
            if !project.has(k) {
                return Err(format!(
                    "kernel `{k}` is not in the project fatbin - is it listed in build.rs's \
                     PROJECT_KERNELS and defined in cuda/scunet.cu?"
                ));
            }
        }
        for k in TOOLKIT_KERNELS {
            if !toolkit.has(k) {
                return Err(format!(
                    "kernel `{k}` is not in the toolkit fatbin - is it in lightgpu's \
                     cuda/kernels.cu and src/ops/mod.rs's NAMES?"
                ));
            }
        }
        // `main` refuses the development switches before it gets here, so this is the
        // guard for a caller that drives the backend directly (tests, examples).
        if let Some(name) = crate::dev::refused_switch() {
            return Err(format!(
                "`{name}` is a development switch, and this build is not one - \
                 rebuild with `--features dev` to use it"
            ));
        }
        let profiler = if dev_switch("SCUNET_PROFILE").is_some() {
            Some(std::cell::RefCell::new(Profiler { spans: Vec::new() }))
        } else {
            None
        };
        let trace = if dev_switch("SCUNET_MEMTRACE").is_some() {
            Some(std::cell::RefCell::new(Vec::new()))
        } else {
            None
        };
        Ok(Cuda {
            wt,
            toolkit,
            project,
            weights: HashMap::new(),
            pool: HashMap::new(),
            profiler,
            live: 0,
            peak: 0,
            trace,
            host_stage: None,
        })
    }

    fn module_of(&self, name: &str) -> &Module {
        if self.project.has(name) {
            &self.project
        } else {
            &self.toolkit
        }
    }

    /// Launch a kernel, and - when `SCUNET_PROFILE` is set - time it.
    ///
    /// EVERY LAUNCH GOES THROUGH HERE, which is why the per-kernel breakdown lives
    /// at this one point: this machine denies GPU performance counters to `ncu`
    /// (ERR_NVGPUCTRPERM), so a CUDA event pair around each launch is the only
    /// measurement available.
    ///
    /// IT DOES NOT SYNCHRONISE PER LAUNCH. The two events sit in the default stream
    /// on either side of the kernel, so their elapsed time is that kernel's own span
    /// - and one synchronise before the report makes every pair readable, with the
    /// pipeline intact. A host sync here would drain the queue and measure a
    /// different program.
    fn run(&self, name: &str, launch: Launch, a: &mut Args) -> Result<(), String> {
        let Some(p) = self.profiler.as_ref() else {
            return a.launch(self.module_of(name), name, launch).map_err(|e| format!("{name}: {e}"));
        };
        let (t0, t1) = (Event::new()?, Event::new()?);
        t0.record()?;
        let r = a.launch(self.module_of(name), name, launch).map_err(|e| format!("{name}: {e}"));
        t1.record()?;
        p.borrow_mut().push(name, t0, t1);
        r
    }

    /// Account for `bytes` of device memory now held, and raise the high-water.
    fn note(&mut self, label: &str, bytes: usize) {
        self.live += bytes;
        if self.live > self.peak {
            self.peak = self.live;
        }
        if let Some(t) = self.trace.as_ref() {
            let live = self.live;
            t.borrow_mut().push((label.to_string(), bytes, live));
        }
    }

    fn pool_bytes(&self) -> usize {
        self.pool.values().flat_map(|v| v.iter()).map(|b| b.bytes).sum()
    }

    /// Account for `bytes` of device memory RELEASED, so `live` is what is actually
    /// held and `peak` is the high-water of what the card had to hold at once.
    ///
    /// THIS IS WHAT ANSWERS "WHAT DOES AN IMAGE NEED". A buffer that is given back
    /// stays in the pool and still costs device memory, but a `Scratch` that a new
    /// geometry REPLACES is dropped outright - `Scratch::new` in `scratch_for` - so
    /// the previous stage's buffers really are gone. Adding without subtracting
    /// would make the total grow through a forward and report a peak that no moment
    /// of the forward ever demanded.
    fn free_note(&mut self, label: &str, bytes: usize) {
        self.live = self.live.saturating_sub(bytes);
        if let Some(t) = self.trace.as_ref() {
            let live = self.live;
            t.borrow_mut().push((format!("free {label}"), bytes, live));
        }
    }

    /// A scratch buffer of `n` floats, reused where the size repeats: a forward
    /// cycles through a handful of plane sizes, so the pool settles after the first
    /// call and a steady-state forward allocates nothing.
    ///
    /// A FAILED ALLOCATION SAYS WHAT IT WANTED AND WHAT WAS ALREADY HELD. The
    /// toolkit's error is a bare `cuMemAlloc failed`, which is what made the
    /// 2048x2048 out-of-memory take a separate investigation to attribute; the
    /// message here names the request, the high-water and the pool, so the next
    /// one is diagnosed from its own text.
    fn buf_at(&mut self, label: &str, n: usize) -> DevBuf {
        // BEST FIT: the smallest pooled buffer that is AT LEAST `n` floats, not the
        // one that is exactly `n`.
        //
        // THIS IS WHAT MAKES AN IMAGE FIT. Keyed by exact element count, a stage's
        // plane could only ever be reused by a request of the same size - and a forward
        // walks sizes 2048x2048 -> 1024x1024 -> 512x512 and back up, so every size
        // would keep its OWN set of planes resident for the whole forward. At 2048x2048
        // that pushes the high-water past the card: three 1073 MiB planes from the first
        // stage stay live while the smaller stages allocate three more each, none able
        // to borrow from a bigger neighbour.
        //
        // HANDING OUT A BIGGER BUFFER IS SAFE because every kernel here is told its
        // extents explicitly and touches only the range those name: the tiled 1x1 maps
        // its grid over the pixel count it is given, and the elementwise kernels walk
        // `n` items. Nothing reads past its own arguments, so the tail of a larger
        // buffer is simply never addressed - and it is popped from the pool while in
        // use, so two callers can never hold the same allocation.
        let key = self.pool.keys().filter(|k| **k >= n).min().copied();
        if let Some(k) = key {
            let b = self.pool.get_mut(&k).and_then(|v| v.pop());
            if self.pool.get(&k).map(|v| v.is_empty()).unwrap_or(false) {
                self.pool.remove(&k);
            }
            if let Some(b) = b {
                return b;
            }
        }
        let bytes = n * 4;
        match DevBuf::zeros(bytes) {
            Ok(b) => {
                self.note(label, b.bytes);
                b
            }
            Err(e) => panic!(
                "device allocation of {} MiB for `{}` failed ({}): {:.0} MiB live, {:.0} MiB high-water, {:.0} MiB of that pooled. The high-water is what the card must hold at once, so its distance from the free total is the size the next attempt has to win back.",
                bytes / 1048576,
                label,
                e,
                self.live as f64 / 1048576.0,
                self.peak as f64 / 1048576.0,
                self.pool_bytes() as f64 / 1048576.0
            ),
        }
    }

    fn buf(&mut self, n: usize) -> DevBuf {
        self.buf_at("pool", n)
    }

    /// Like `buf`, but the allocation is labelled in a memory trace. Used by the
    /// walker, where two buffers of the same element count have different roles and
    /// the label is what tells the planebuffers apart.
    fn buf_named(&mut self, label: &str, n: usize) -> DevBuf {
        self.buf_at(label, n)
    }

    /// WHERE THE THREE STAGE SKIPS LIVE while the up pass waits for them.
    ///
    /// `host` is the default and it is a MEMORY decision: the three skips together
    /// are 56 floats per input pixel - 896 MiB at 2048x2048 - and keeping them in
    /// host RAM is what makes that size fit on an 8 GB card. The cost is one D2H and
    /// one H2D each, and the add back is exact because it is element-wise over
    /// DISJOINT ranges (each element is added once; no sum is reordered).
    ///
    /// `device` keeps them in the pool instead. It is what the engine did before the
    /// memory work, it costs 896 MiB at 2048, and it is here because the difference
    /// between the two is the only measurement of what the round trip actually costs
    /// per forward - which no per-kernel instrument in this repository can see,
    /// `bench_op` included, since it times compute and not host transfers.
    fn skip_kind(&self) -> &'static str {
        static K: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        *K.get_or_init(|| match std::env::var("SCUNET_SKIP").as_deref() {
            Ok("device") => "device",
            Ok("host") => "host",
            _ => "auto",
        })
    }

    /// The high-water of device bytes held during the forwards so far, in MiB.
    pub fn peak_mib(&self) -> f64 {
        self.peak as f64 / 1048576.0
    }

    /// The high-water and, when `SCUNET_MEMTRACE` is set, the allocations that
    /// reached it - the ones at least an eighth of the largest single request, so a
    /// size's structure is readable without printing every token buffer.
    pub fn mem_report(&self) {
        println!(
            "  ---- device memory: {:.1} MiB high-water, {:.1} MiB live, {:.1} MiB pooled ----",
            self.peak as f64 / 1048576.0,
            self.live as f64 / 1048576.0,
            self.pool_bytes() as f64 / 1048576.0
        );
        let Some(t) = self.trace.as_ref() else { return };
        let t = t.borrow();
        let top = t.iter().map(|(_, b, _)| *b).max().unwrap_or(0);
        for (label, bytes, live) in t.iter() {
            if *bytes * 8 >= top {
                println!(
                    "  {:>9.1} MiB  live {:>9.1}  {}",
                    *bytes as f64 / 1048576.0,
                    *live as f64 / 1048576.0,
                    label
                );
            }
        }
    }

    /// Return a scratch buffer to the pool, CAPPING each size at what one forward
    /// can hold at once.
    ///
    /// THE CAP IS WHAT KEEPS THE FOOTPRINT HONEST. The pool is keyed by element
    /// count and a forward cycles through a handful of sizes, so the pool settles
    /// after the first call - but a benchmark that walks several IMAGE SIZES in one
    /// process would otherwise accumulate every size's buffers, and `footprint` (which
    /// sums the pool, since a free-VRAM delta is not deterministic here) would then
    /// report every size's need as the largest one's. The cap is the
    /// number of buffers of one size a single forward holds at once, which is a
    /// property of the walker: `Cuda::forward` keeps a few planes plus the per-stage
    /// scratch, and the transformer half reuses one `Scratch` per geometry.
    ///
    /// The surplus is dropped here rather than in a sweep, so a long-running serve
    /// that sees an unusual size once does not keep its buffers forever.
    fn give_back(&mut self, b: DevBuf) {
        // 8 is comfortably above the most buffers of one element count the walker
        // holds simultaneously (a stage's activation, its staging plane and the
        // block dumps) and far below the leak that produced the 4357 MiB reading.
        const MAX_PER_SIZE: usize = 8;
        let v = self.pool.entry(b.bytes / 4).or_default();
        if v.len() < MAX_PER_SIZE {
            v.push(b);
        } else {
            // Over the cap: this buffer is about to be dropped, so it stops costing
            // device memory. Not subtracting it would make `peak` a ceiling on
            // allocations rather than on what the card actually holds.
            self.free_note("pool overflow", b.bytes);
        }
    }

    /// `Scratch` bytes that `scratch_for` is about to free, recorded by the walker
    /// so `live` follows the real lifetime. Held in a field because `scratch_for`
    /// takes the slot and not the backend.
    fn drop_note(&mut self, bytes: usize) {
        self.free_note("scratch replaced", bytes);
    }

    /// The device copy of a weight, uploaded on first use. Every name has been
    /// shape-checked at load, so a miss here is a programming error.
    fn w(&mut self, name: &str) -> u64 {
        if !self.weights.contains_key(name) {
            let b = DevBuf::from_host(self.wt.t(name)).expect("upload weight");
            self.weights.insert(name.to_string(), b);
        }
        self.weights[name].ptr
    }

    /// The device pointer of a bias, or 0 when the checkpoint has none for this op -
    /// the same rule `conv1x1` uses, so a forced-arm measurement matches the graph.
    fn bias_of(&mut self, name: &str) -> u64 {
        if self.wt.has(&format!("{name}.bias")) {
            self.w(&format!("{name}.bias"))
        } else {
            0
        }
    }

    fn upload(&self, v: &[f32]) -> DevBuf {
        DevBuf::from_host(v).expect("upload activation")
    }

    fn download(&self, b: &DevBuf, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        b.download(&mut v).expect("download activation");
        v
    }

    // -- the ops, each the device twin of a function in src/cpu.rs -----------

    /// 3x3, pad 1, no bias: the checkpoint's conv_block and stem/tail weights carry
    /// no bias, so the null pointer is the correct argument, not a placeholder.
    #[allow(clippy::too_many_arguments)]
    fn conv3x3(
        &mut self, x: u64, name: &str, c_in: usize, c_out: usize, h: usize, w: usize, out: u64,
    ) -> Result<(), String> {
        let wp = self.w(&format!("{name}.weight"));
        if c_in >= 32 {
            // The F(4x4, 3x3) form takes the direct kernel's arguments PLUS
            // c_chunk/ocb/act/act_p, and its shared memory is fixed by those: the
            // doc in lightgpu's src/ops/mod.rs is the contract, and a mismatch shows
            // up as CUDA_ERROR_INVALID_VALUE at launch, not as wrong pixels.
            // `c_chunk` is the kernel's own inner loop bound over input channels,
            // and its shared memory is `c_chunk * 32 * 36 * 4` BYTES - the 32 is the
            // tile's rows plus its output channels, not a function of c_in. Passing
            // c_in there asks for hundreds of KB of dynamic shared memory, which the
            // hardware refuses at launch with CUDA_ERROR_INVALID_VALUE. `swin2sr-rs`'s
            // kernel check uses these same two constants.
            const C_CHUNK: usize = 5;
            const OCB: usize = 16;
            let shared = (C_CHUNK * 32 * 36 * 4) as u32;
            let (c_chunk, ocb) = (C_CHUNK, OCB);
            let mut a = Args::new();
            a.ptr(x).ptr(wp).ptr(0).ptr(out)
                .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32)
                .i32(c_chunk as i32).i32(ocb as i32)
                .i32(0).f32(0.0);
            let g = (((w + 15) / 16) as u32, ((h + 15) / 16) as u32, ((c_out + ocb - 1) / ocb) as u32);
            self.run(
                "lg_conv3x3_winograd",
                Launch::new(g, (256, 1, 1)).shared(shared),
                &mut a,
            )
        } else {
            let mut a = Args::new();
            a.ptr(x).ptr(wp).ptr(0).ptr(out)
                .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
            self.run("lg_conv3x3s1p1", Launch::new(grid(c_out * h * w), (BLOCK as u32, 1, 1)), &mut a)
        }
    }

    /// Which 1x1 kernel the graph uses, read once on the first launch so an A/B is a
    /// single environment variable.
    ///
    /// BOTH ARMS ARE TOOLKIT KERNELS NOW, which is the point of the promotion: the
    /// comparison moved to where both kernels live, so the engine only has to record
    /// which one it chose. `rb` (the default) is `lg_conv1x1_rb`, a register-blocked
    /// GEMM over the plane layout; `plain` is `lg_conv1x1`, one output element per
    /// thread, which is the kernel `lg_conv1x1_rb` was measured against.
    ///
    /// The engine's own tiled `sc_conv1x1_t` and its register-blocked
    /// `sc_conv1x1_rb` were the two other arms of that comparison and are gone with
    /// their numbers recorded in `cuda/scunet.cu`: the tiled form lost to the
    /// register-blocked one by 1.9-2.5x, and the engine's own rb was 2.6x slower than
    /// the tiled form at every geometry - the negative result that showed the memory
    /// access pattern, not the load count, was what mattered.
    fn conv1x1_kind(&self) -> &'static str {
        static K: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        *K.get_or_init(|| match dev_switch("SCUNET_1X1").as_deref() {
            Some("plain") => "plain",
            _ => "rb",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1x1(
        &mut self, x: u64, name: &str, c_in: usize, c_out: usize, h: usize, w: usize, out: u64,
    ) -> Result<(), String> {
        let wp = self.w(&format!("{name}.weight"));
        let bp = if self.wt.has(&format!("{name}.bias")) { self.w(&format!("{name}.bias")) } else { 0 };
        let plane = h * w;
        let mut a = Args::new();
        a.ptr(x).ptr(wp).ptr(bp).ptr(out)
            .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
        if self.conv1x1_kind() == "rb" {
            // 4x4 outputs a thread over a 16-wide staged tile, so the block tile is
            // 64x64 and the grid tiles the PIXELS on x and the output channels on y.
            // The plane is the long axis (a 2048x2048 plane is 32768 tiles, past
            // gridDim.y's 65535 cap) and the channel count is at most 512, so this
            // orientation is the safe one at every size the engine runs. THIS IS NOT
            // `rb_grid`: that one is the strided pair's, whose tile is 256 pixels by
            // 8 channels because its threads carry no register tile at all.
            let g = (((plane + 63) / 64) as u32, ((c_out + 63) / 64) as u32, 1);
            return self.run("lg_conv1x1_rb", Launch::new(g, (16, 16, 1)), &mut a);
        }
        self.run("lg_conv1x1", Launch::new(grid(c_out * plane), (BLOCK as u32, 1, 1)), &mut a)
    }

    /// `[rows][c_in]` x `[c_out][c_in]` -> `[rows][c_out]`.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &mut self, x: u64, name: &str, rows: usize, c_in: usize, c_out: usize, out: u64,
    ) -> Result<(), String> {
        let wp = self.w(&format!("{name}.weight"));
        let bp = if self.wt.has(&format!("{name}.bias")) { self.w(&format!("{name}.bias")) } else { 0 };
        let mut a = Args::new();
        a.ptr(x).ptr(wp).ptr(bp).ptr(out).i32(rows as i32).i32(c_in as i32).i32(c_out as i32);
        if self.gemm_kind() == "rb" {
            // 4x4 outputs a thread: the block tile is 64x64, so the grid is the
            // output channels on x and the rows on y. For the token layout the ROWS
            // are the long axis, which is why this orientation is the opposite of the
            // 1x1's - the token count is the pixel count at full resolution.
            let g = (((c_out + 63) / 64) as u32, ((rows + 63) / 64) as u32, 1);
            return self.run("lg_linear_rb", Launch::new(g, (16, 16, 1)), &mut a);
        }
        let g = (((c_out + 15) / 16) as u32, ((rows + 15) / 16) as u32, 1);
        self.run("lg_linear", Launch::new(g, (16, 16, 1)), &mut a)
    }

    /// Which token linear the graph uses, read once so an A/B is one environment
    /// variable. `rb` (the default) is `lg_linear_rb`; `plain` is `lg_linear`, the
    /// 16x16-tile form it was measured against at 2.03x.
    ///
    /// The six tile shapes this engine used to carry as separate kernels - 2x4, 2x8,
    /// 4x4, 4x8, 8x4 and 8x8 - are gone with their register counts recorded in
    /// `cuda/scunet.cu`: on sm_61, 2x4 is 79 registers, 4x4 124, 2x8 127, 4x8 164,
    /// 8x4 190 and 8x8 240, and 256 threads times 240 already exceeds the SM's
    /// 65536-register file. The toolkit keeps the 4x4 shape for that reason.
    fn gemm_kind(&self) -> &'static str {
        static K: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        *K.get_or_init(|| match dev_switch("SCUNET_GEMM").as_deref() {
            Some("plain") => "plain",
            _ => "rb",
        })
    }


    /// Which LayerNorm kernel the graph uses, read once so an A/B is one environment
    /// variable.
    ///
    /// BOTH ARMS ARE TOOLKIT KERNELS: `warp` (the default) is `lg_layer_norm_warp`,
    /// one warp per row with the row held in registers, and `block` is
    /// `lg_layer_norm`, one block per row with shared-memory reduction trees. The
    /// warp form was written in this engine and measured at 2.2x the block form for a
    /// 256-wide row and 8.6x for a 32-wide one - the rows a transformer normalizes
    /// are NARROW, which is exactly where the block form's two trees and two syncs
    /// have nothing to amortise. It is an addition rather than a replacement: the
    /// block form is still the better kernel for the wide rows it was written for.
    fn ln_kind(&self) -> &'static str {
        static K: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        *K.get_or_init(|| match dev_switch("SCUNET_LN").as_deref() {
            Some("block") => "block",
            _ => "warp",
        })
    }

    /// `nn.LayerNorm(c)` over the last axis of `[rows][c]`, the device twin of
    /// `cpu::layer_norm`.
    fn layer_norm(&mut self, x: u64, name: &str, rows: usize, c: usize, out: u64) -> Result<(), String> {
        let which = self.ln_kind();
        self.layer_norm_which(which, x, name, rows, c, out)
    }

    fn layer_norm_which(
        &mut self, which: &str, x: u64, name: &str, rows: usize, c: usize, out: u64,
    ) -> Result<(), String> {
        let wp = self.w(&format!("{name}.weight"));
        let bp = self.w(&format!("{name}.bias"));
        let mut a = Args::new();
        a.ptr(x).ptr(wp).ptr(bp).ptr(out).i32(c as i32).i32(rows as i32).f32(1e-5);
        if which == "block" {
            return self.run("lg_layer_norm", Launch::new((rows as u32, 1, 1), (256, 1, 1)), &mut a);
        }
        // One warp per row, eight warps per block. The row is held in registers, so
        // the width has to fit the kernel's LG_LN_WARP_MAX (128); wider rows take its
        // strided fallback. The register path is the one every released checkpoint
        // takes - the normalized widths here are 32, 64, 128 and 256 tokens per row.
        self.run(
            "lg_layer_norm_warp",
            Launch::new((((rows + 7) / 8) as u32, 1, 1), (256, 1, 1)),
            &mut a,
        )
    }

    fn gelu(&self, x: u64, out: u64, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x).ptr(out).i32(n as i32);
        self.run("lg_gelu_erf", Launch::new(grid(n), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn relu(&self, x: u64, out: u64, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x).ptr(out).i32(n as i32);
        self.run("lg_relu", Launch::new(grid(n), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn add(&self, a_: u64, b: u64, out: u64, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(a_).ptr(b).ptr(out).i32(n as i32);
        self.run("lg_add", Launch::new(grid(n), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn copy(&self, src: u64, dst: u64, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(src).ptr(dst).i64(n as i64);
        self.run("lg_copy", Launch::new(grid(n), (BLOCK as u32, 1, 1)), &mut a)
    }

    /// `out += host`, where `host` lives in the host's RAM rather than on the card.
    ///
    /// THE STAGE SKIPS USE THIS, because a skip-connected network otherwise holds
    /// every down stage's output on the device for the whole up pass - 56 floats per
    /// input pixel, 896 MiB at 2048x2048. Streaming them back CHUNKED is exact: the
    /// kernel adds element-wise over disjoint ranges, so every output element is
    /// added exactly once, no sum is reordered and no chunk boundary is observable.
    /// That is not true of a reduction, and it is the reason this is done by slicing
    /// the OUTPUT rather than by accumulating.
    fn add_host(&mut self, host: &[f32], out: u64, n: usize) -> Result<(), String> {
        const CHUNK: usize = 4 << 20;              // 16 MiB of floats per transfer
        if self.host_stage.is_none() {
            let b = DevBuf::zeros(CHUNK * 4)?;
            self.note("host transfer staging", b.bytes);
            self.host_stage = Some(b);
        }
        let ptr = self.host_stage.as_ref().expect("just allocated").ptr;
        let mut off = 0;
        while off < n {
            let len = CHUNK.min(n - off);
            self.host_stage.as_ref().expect("present").upload(&host[off..off + len])?;
            self.add(ptr, out + (off * 4) as u64, out + (off * 4) as u64, len)?;
            off += len;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gather(
        &self, x: u64, tok: u64, nw: usize, n: usize, nww: usize, win: usize, hp: usize,
        wp: usize, c: usize, shift: usize, w0: usize,
    ) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x).ptr(tok)
            .i32(nw as i32).i32(n as i32).i32(nww as i32).i32(win as i32)
            .i32(hp as i32).i32(wp as i32).i32(c as i32).i32(shift as i32).i32(w0 as i32);
        self.run("lg_window_gather", Launch::new(grid(nw * n * c), (BLOCK as u32, 1, 1)), &mut a)
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter(
        &self, tok: u64, x: u64, nw: usize, n: usize, nww: usize, win: usize, hp: usize,
        wp: usize, c: usize, shift: usize, w0: usize,
    ) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(tok).ptr(x)
            .i32(nw as i32).i32(n as i32).i32(nww as i32).i32(win as i32)
            .i32(hp as i32).i32(wp as i32).i32(c as i32).i32(shift as i32).i32(w0 as i32);
        self.run("lg_window_scatter", Launch::new(grid(nw * n * c), (BLOCK as u32, 1, 1)), &mut a)
    }

    /// The window attention: `qkv` is `[nw*n][3c]` - ONE ROW PER TOKEN, with q, k and
    /// v as its three column blocks, because it comes from a single `lg_linear` with
    /// 3c outputs - `rp` is `[heads][2win-1][2win-1]`, and the output is `[nw*n][c]`
    /// tokens. Reading the qkv as three contiguous planes is the layout bug this
    /// kernel was written with and `tests/cuda_ops.rs`'s degenerate-input tests now
    /// pin down.
    /// Which window-attention kernel the graph uses, read once so a comparison is
    /// one environment variable. `s` (the default) is `sc_window_attn_s`, one block
    /// per (window, head) with that head's k and v staged in shared memory; `row` is
    /// `sc_window_attn`, one block per window with a thread per (query, head) row,
    /// which re-reads k and v from global memory once per query. Both are kept
    /// because the second is the measurement that justified the first.
    fn attn_kind(&self) -> &'static str {
        static K: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        *K.get_or_init(|| match dev_switch("SCUNET_ATTN").as_deref() {
            Some("row") => "row",
            Some("runtime") => "s_runtime",
            _ => "s",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_which(
        &self, which: &str, qkv: u64, rp: u64, out: u64, nw: usize, n: usize, nww: usize,
        win: usize, hp: usize, wp: usize, heads: usize, hd: usize, shift: usize, w0: usize,
    ) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(qkv).ptr(rp).ptr(out)
            .i32(nw as i32).i32(n as i32).i32(nww as i32).i32(win as i32)
            .i32(hp as i32).i32(wp as i32).i32(heads as i32).i32(hd as i32).i32(shift as i32)
            .i32(w0 as i32);
        if which == "row" {
            return self.run(
                "sc_window_attn",
                Launch::new((nw as u32, 1, 1), (n as u32, heads as u32, 1)),
                &mut a,
            );
        }
        // One block per (window, head), k and v staged in shared memory.
        //
        // THE REGISTER FORM IS THE DEFAULT WHERE IT APPLIES. `sc_window_attn_r64` is
        // templated on the released geometry (64 tokens, head width 32), so its 64
        // scores and 32 query values live in REGISTERS; `sc_window_attn_s` takes both
        // as runtime arguments, which puts the same arrays in LOCAL memory - a
        // 512-byte stack frame with zero spills, and about 8.7 GB of local traffic per
        // forward at 256x256. Anything but an 8x8 window with 32-wide heads falls
        // back, so the kernel follows the checkpoint rather than an assumption.
        if which != "s_runtime" && n == 64 && hd == 32 {
            // Two threads per query token, each owning half the keys, so the block is
            // 2*n and the shared memory is k + v + the cross-thread exchange.
            let shared = (3 * n * hd * 4) as u32;
            return self.run(
                "sc_window_attn_r64",
                Launch::new((nw as u32, heads as u32, 1), (2 * n as u32, 1, 1)).shared(shared),
                &mut a,
            );
        }
        let shared = (2 * n * hd * 4) as u32;
        self.run(
            "sc_window_attn_s",
            Launch::new((nw as u32, heads as u32, 1), (n as u32, 1, 1)).shared(shared),
            &mut a,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self, qkv: u64, rp: u64, out: u64, nw: usize, n: usize, nww: usize, win: usize,
        hp: usize, wp: usize, heads: usize, hd: usize, shift: usize, w0: usize,
    ) -> Result<(), String> {
        let k = self.attn_kind();
        self.attention_which(k, qkv, rp, out, nw, n, nww, win, hp, wp, heads, hd, shift, w0)
    }

    /// Which kernel the strided pair uses, read once so an A/B is one environment
    /// variable.
    ///
    /// `rb` (the default) is the toolkit's `lg_conv2x2s2`/`lg_conv_t2x2`: a thread
    /// holds eight accumulators over output channels and reads one activation per
    /// (ci, tap), reusing it eight times. `elem` is this engine's per-output-element
    /// form, kept in `cuda/scunet.cu` because it is the measurement that justified the
    /// register-blocked one (1.72-6.49x for the down form, 2.39-2.60x for the
    /// transposed) and the reference the CPU twin was first matched against.
    fn c2x2_kind(&self) -> &'static str {
        static K: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        *K.get_or_init(|| match dev_switch("SCUNET_C2X2").as_deref() {
            Some("elem") => "elem",
            _ => "rb",
        })
    }

    /// One output channel block per `grid.y`, one output pixel per thread: the grid
    /// the register-blocked strided pair takes. `n_pixels` is the OUTPUT plane's
    /// element count, which differs between the two - the down form halves h and w,
    /// the transposed form quadruples them.
    fn rb_grid(n_pixels: usize, c_out: usize) -> (u32, u32, u32) {
        const OC: usize = 8;
        (grid(n_pixels).0, ((c_out + OC - 1) / OC) as u32, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2x2s2(
        &mut self, x: u64, name: &str, c_in: usize, c_out: usize, h: usize, w: usize, out: u64,
    ) -> Result<(), String> {
        let wp = self.w(&format!("{name}.weight"));
        let mut a = Args::new();
        if self.c2x2_kind() == "rb" {
            // The toolkit's form takes a NULLABLE BIAS before `out`, which the
            // elementwise form below does not have; every 2x2 down convolution in the
            // graph is bias=False, so the null is the correct argument and not a
            // placeholder.
            a.ptr(x).ptr(wp).ptr(0).ptr(out)
                .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
            let g = Self::rb_grid((h / 2) * (w / 2), c_out);
            return self.run("lg_conv2x2s2", Launch::new(g, (BLOCK as u32, 1, 1)), &mut a);
        }
        a.ptr(x).ptr(wp).ptr(out).i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
        self.run(
            "sc_conv2x2s2",
            Launch::new(grid(c_out * (h / 2) * (w / 2)), (BLOCK as u32, 1, 1)),
            &mut a,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_t2x2(
        &mut self, x: u64, name: &str, c_in: usize, c_out: usize, h: usize, w: usize, out: u64,
    ) -> Result<(), String> {
        let wp = self.w(&format!("{name}.weight"));
        let mut a = Args::new();
        if self.c2x2_kind() == "rb" {
            a.ptr(x).ptr(wp).ptr(0).ptr(out)
                .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
            let g = Self::rb_grid(4 * h * w, c_out);
            return self.run("lg_conv_t2x2", Launch::new(g, (BLOCK as u32, 1, 1)), &mut a);
        }
        a.ptr(x).ptr(wp).ptr(out).i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
        self.run(
            "sc_conv_t2x2",
            Launch::new(grid(c_out * 4 * h * w), (BLOCK as u32, 1, 1)),
            &mut a,
        )
    }
}

/// How many tokens the transformer half materialises at once, and therefore how
/// much device memory the token buffers take - INDEPENDENTLY OF THE IMAGE.
///
/// THE TOKEN COUNT EQUALS THE PIXEL COUNT AT FULL RESOLUTION. A block at 1x
/// resolution gathers `(h/win)*(w/win)*win*win = h*w` tokens, so sizing the token
/// buffers by the whole plane makes them grow with the image: at 1024x1024 the
/// m_down1 geometry (trans 64) wanted 4*64 = 256 floats per pixel, which is
/// 1024 MiB for the MLP hidden layer alone and 4352 MiB for the scratch - a
/// 960x960 forward then died with CUDA_ERROR_OUT_OF_MEMORY on an 8 GB card, and
/// 1024x1024 could not launch at all because `lg_linear` ran 1048576 rows and
/// CUDA's grid.y limit is 65535.
///
/// Chunking the WINDOW INDEX fixes both at once and costs nothing arithmetically:
/// attention is per window, every residual is per token, and the gather/scatter
/// index map does not depend on which chunk a window is in. 32768 tokens is four
/// chunks of the 1024x1024 m_down1 stage and keeps the largest token buffer
/// (`hid`, 4*trans floats per token) at 32 MiB for trans = 64; the linear's row
/// count becomes 32768, so its grid.y is 2048.
const TOKEN_BUDGET: usize = 32768;

/// The device buffers one `ConvTransBlock` needs, in floats, sized for one
/// geometry and reused across every block that shares it.
///
/// The token half is sized by `nt_blk`, the chunk of windows actually processed
/// at once, and `nw_blk` records how many windows that is - so the walker can
/// iterate `nw` windows in `nw_blk`-sized steps.
struct Scratch {
    /// The 1x1 expand's output, `[c][h][w]`, and the staging the conv half works in.
    y: DevBuf,
    /// The conv half's two staging buffers, ALLOCATED ONLY WHEN `proj` IS TOO SMALL
    /// to hold them. `staging` explains the reuse and when it applies.
    c1: Option<DevBuf>,
    c2: Option<DevBuf>,
    /// The projection's output and every residual staged in a side buffer. Doubles
    /// as the conv half's staging - see `staging`.
    proj: DevBuf,
    bt: DevBuf,
    /// `[nt_blk][trans]` tokens for the transformer half, and the norm output.
    tok: DevBuf,
    norm: DevBuf,
    /// `[nt_blk][4*trans]`, the MLP's hidden layer.
    hid: DevBuf,
    /// `[nt_blk][3*trans]`, the fused qkv: one row per token, q/k/v as column blocks.
    qkv: DevBuf,
    /// `[nt_blk][trans]`, the attention's output before the projection.
    att: DevBuf,
    c: usize,
    trans: usize,
    n: usize,
    /// Windows processed per chunk.
    nw_blk: usize,
}

impl Scratch {
    fn new(c: usize, trans: usize, h: usize, w: usize, win: usize) -> Result<Scratch, String> {
        let n = h * w;
        let ntok = win * win;
        let nw = (h / win) * (w / win);
        let nw_blk = (TOKEN_BUDGET / ntok).clamp(1, nw.max(1));
        let nt_blk = nw_blk * ntok;
        // The conv half's staging fits inside `proj` when 2*(c - trans) <= c, i.e.
        // when the transformer's half-width is at least the conv half's - true for
        // every released checkpoint (conv_dim == trans == dim/2, so the two sides of
        // the split are equal and the fit is exact). Then the two buffers are not
        // allocated at all and `proj` serves both roles, which it can because the
        // conv half is finished long before the projection writes.
        let staged = (c - trans) * 2 <= c;
        let (c1, c2) = if staged {
            (None, None)
        } else {
            (
                Some(DevBuf::zeros((c - trans) * n * 4)?),
                Some(DevBuf::zeros((c - trans) * n * 4)?),
            )
        };
        Ok(Scratch {
            y: DevBuf::zeros(c * n * 4)?,
            c1,
            c2,
            proj: DevBuf::zeros(c * n * 4)?,
            bt: DevBuf::zeros(nt_blk * trans * 4)?,
            tok: DevBuf::zeros(nt_blk * trans * 4)?,
            norm: DevBuf::zeros(nt_blk * trans * 4)?,
            hid: DevBuf::zeros(nt_blk * 4 * trans * 4)?,
            qkv: DevBuf::zeros(3 * nt_blk * trans * 4)?,
            att: DevBuf::zeros(nt_blk * trans * 4)?,
            c,
            trans,
            n,
            nw_blk,
        })
    }

    fn fits(&self, c: usize, trans: usize, h: usize, w: usize) -> bool {
        self.c == c && self.trans == trans && self.n == h * w
    }

    /// The two staging buffers the conv half needs, as device pointers.
    ///
    /// THEY LIVE INSIDE `proj` WHEN THEY FIT, and that is a memory win rather than a
    /// trick: the conv half runs to completion before the projection writes a byte,
    /// so the two roles never overlap and the staging costs nothing. The condition is
    /// 2*(c - trans) <= c, which holds for every released checkpoint - there
    /// conv_dim and trans are both dim/2, so the fit is exact - and when it does not
    /// hold the two buffers are allocated separately and this is just an accessor.
    fn staging(&self, n: usize) -> (u64, u64) {
        match (self.c1.as_ref(), self.c2.as_ref()) {
            (Some(a), Some(b)) => (a.ptr, b.ptr),
            _ => (
                self.proj.ptr,
                self.proj.ptr + ((self.c - self.trans) * n * 4) as u64,
            ),
        }
    }

    /// The bytes `new` would take, WITHOUT allocating - so the accounting is
    /// charged before the allocation and an out-of-memory message can report it.
    ///
    /// THE SCRATCH IS THE LARGEST SINGLE CONTRIBUTOR at full resolution - at
    /// 2048x2048's down1 geometry (c = 64, trans = 32) it is y + proj at 64 floats a
    /// pixel plus c1 + c2 at 32, i.e. 192 floats a pixel, 3.2 GB - so leaving it out
    /// of the high-water, which is what routing only `Cuda::buf` through the
    /// accounting did, under-reports the requirement by more than half.
    fn bytes_for(c: usize, trans: usize, h: usize, w: usize, win: usize) -> usize {
        let n = h * w;
        let ntok = win * win;
        let nw = (h / win) * (w / win);
        let nw_blk = (TOKEN_BUDGET / ntok).clamp(1, nw.max(1));
        let nt_blk = nw_blk * ntok;
        let plane = |ch: usize| ch * n * 4;
        let tok = |ch: usize| nt_blk * ch * 4;
        // y and proj, plus the two staging buffers ONLY when they do not fit inside
        // proj - see `staging`, which is where the reuse rule lives.
        let staging = if (c - trans) * 2 <= c { 0 } else { plane(c - trans) * 2 };
        plane(c) * 2 + staging + tok(3 * trans) + tok(4 * trans)
    }

    /// Every byte this scratch holds, so its replacement can be accounted for.
    fn bytes(&self) -> usize {
        let mut t = self.y.bytes + self.proj.bytes + self.bt.bytes + self.tok.bytes
            + self.norm.bytes + self.hid.bytes + self.qkv.bytes + self.att.bytes;
        for b in [self.c1.as_ref(), self.c2.as_ref()].into_iter().flatten() {
            t += b.bytes;
        }
        t
    }
}

/// The scratch for one geometry, reused across every block that shares it.
///
/// ALLOCATING PER BLOCK WOULD BE 28 x 10 DEVICE BUFFERS PER FORWARD. A stage's
/// blocks all have the same geometry - the stage's width and plane - and there are
/// only seven distinct (c, trans, h, w) tuples in the whole graph, so one scratch
/// per tuple turns the steady-state forward into launches with no allocation at
/// all. `fits` is what distinguishes a stage change from a repeat.
fn scratch_for(
    slot: &mut Option<Scratch>, c: usize, trans: usize, h: usize, w: usize, win: usize,
) -> Result<&mut Scratch, String> {
    if !slot.as_ref().map(|s| s.fits(c, trans, h, w)).unwrap_or(false) {
        *slot = Some(Scratch::new(c, trans, h, w, win)?);
    }
    Ok(slot.as_mut().expect("just filled"))
}

impl<'a> Cuda<'a> {
    /// One `ConvTransBlock` on NCHW, the device twin of `cpu::conv_trans_block`.
    ///
    /// THE TRANSFORMER HALF NEVER TRANSPOSES. The CPU twin builds `[h][w][c]` by
    /// hand at this boundary because its attention works in that layout; the gather
    /// reads `[nw][n][c]` tokens straight out of the NCHW plane with the shift
    /// folded in, which IS that layout - so this function has no transpose in it,
    /// and the boundary the CPU twin documents is the gather's index map.
    #[allow(clippy::too_many_arguments)]
    fn conv_trans_block(
        &mut self, x: u64, prefix: &str, dims: (usize, usize), h: usize, w: usize, shift: usize,
        out: u64, s: &mut Scratch,
    ) -> Result<(), String> {
        let (conv_dim, trans_dim) = dims;
        let (c, n) = (conv_dim + trans_dim, h * w);
        let win = self.wt.window;
        let heads = trans_dim / self.wt.head_dim;
        let hd = self.wt.head_dim;
        let nww = w / win;
        let nw = (h / win) * nww;
        let ntok = win * win;

        // 1x1 expand: c -> c.
        self.conv1x1(x, &format!("{prefix}.conv1_1"), c, c, h, w, s.y.ptr)?;

        // Conv half, in place in the first conv_dim planes of y: 3x3 -> ReLU -> 3x3,
        // then the residual. conv_block has no bias in the checkpoint.
        //
        // `staging` lends the two buffers, WHICH ARE OFTEN NOT THERE: for every
        // released checkpoint the conv half's two planes fit inside `proj`, which is
        // dead at this point, so the pair costs nothing. See `Scratch::staging`.
        let yc = s.y.ptr;
        let (s1, s2) = s.staging(n);
        self.conv3x3(yc, &format!("{prefix}.conv_block.0"), conv_dim, conv_dim, h, w, s1)?;
        self.relu(s1, s1, conv_dim * n)?;
        self.conv3x3(s1, &format!("{prefix}.conv_block.2"), conv_dim, conv_dim, h, w, s2)?;
        self.add(s2, yc, yc, conv_dim * n)?;

        // Transformer half: the plane starts at channel conv_dim of y.
        //
        // IN CHUNKS OF WINDOWS, so the token buffers do not grow with the image.
        // Every step below is per window or per token and the index map is global,
        // so the loop is arithmetically the same forward - it is only the SIZE of
        // the buffers that changes. See TOKEN_BUDGET for the two failures this
        // removes (the 8 GB card running out of memory at 960x960, and the grid.y
        // overflow at 1024x1024).
        let yt = s.y.ptr + (conv_dim * n * 4) as u64;
        let four = 4 * trans_dim;
        let rp = self.w(&format!("{prefix}.trans_block.msa.relative_position_params"));
        let mut w0 = 0;
        while w0 < nw {
            let wb = s.nw_blk.min(nw - w0);          // windows in this chunk
            let nb = wb * ntok;                      // ... and its token count
            self.gather(yt, s.tok.ptr, wb, ntok, nww, win, h, w, trans_dim, shift, w0)?;
            // x = x + msa(ln1(x)): normalise the tokens, then lift to qkv.
            self.layer_norm(s.tok.ptr, &format!("{prefix}.trans_block.ln1"), nb, trans_dim, s.norm.ptr)?;
            self.linear(
                s.norm.ptr, &format!("{prefix}.trans_block.msa.embedding_layer"), nb, trans_dim,
                3 * trans_dim, s.qkv.ptr,
            )?;
            self.attention(
                s.qkv.ptr, rp, s.att.ptr, wb, ntok, nww, win, h, w, heads, hd, shift, w0,
            )?;
            self.linear(
                s.att.ptr, &format!("{prefix}.trans_block.msa.linear"), nb, trans_dim, trans_dim,
                s.bt.ptr,
            )?;
            self.add(s.bt.ptr, s.tok.ptr, s.tok.ptr, nb * trans_dim)?;
            // x = x + mlp(ln2(x))
            self.layer_norm(s.tok.ptr, &format!("{prefix}.trans_block.ln2"), nb, trans_dim, s.norm.ptr)?;
            self.linear(s.norm.ptr, &format!("{prefix}.trans_block.mlp.0"), nb, trans_dim, four, s.hid.ptr)?;
            self.gelu(s.hid.ptr, s.hid.ptr, nb * four)?;
            self.linear(s.hid.ptr, &format!("{prefix}.trans_block.mlp.2"), nb, four, trans_dim, s.bt.ptr)?;
            self.add(s.bt.ptr, s.tok.ptr, s.tok.ptr, nb * trans_dim)?;
            // Back into the second half of y, at the same index the gather read.
            self.scatter(s.tok.ptr, yt, wb, ntok, nww, win, h, w, trans_dim, shift, w0)?;
            w0 += wb;
        }

        // 1x1 project on the concatenation, then the residual over the whole block.
        //
        // IN PLACE WHEN THE CALLER ASKS FOR IT, which is what lets the walker drop a
        // whole full-resolution plane: `x` is untouched until this add (every stage
        // above writes `y` or `proj`), so `out == x` is well defined and the copy
        // becomes unnecessary. The walker passes `out == x`; `block_dump` passes a
        // separate buffer so the input survives for comparison.
        self.conv1x1(s.y.ptr, &format!("{prefix}.conv1_2"), c, c, h, w, s.proj.ptr)?;
        if out != x {
            self.copy(x, out, c * n)?;
        }
        self.add(s.proj.ptr, out, out, c * n)?;
        Ok(())
    }

    /// ONE block on the device for one plane, so a divergence can be localised to a
    /// block instead of guessed at from the output image. `cpu::conv_trans_block` is
    /// the reference and takes the same arguments in the same order; a test compares
    /// the two directly, which is how a wrong argument order or a wrong plane offset
    /// gets named rather than inferred.
    pub fn block_dump(
        &mut self, prefix: &str, h: usize, w: usize, shift: usize, x: &[f32],
    ) -> Result<Vec<f32>, String> {
        let dims = self.wt.block_dims(prefix);
        let c = dims.0 + dims.1;
        let n = h * w;
        let xd = self.upload(x);
        let out = self.buf(c * n);
        let mut s = Scratch::new(c, dims.1, h, w, self.wt.window)?;
        self.conv_trans_block(xd.ptr, prefix, dims, h, w, shift, out.ptr, &mut s)?;
        let y = self.download(&out, c * n);
        self.give_back(out);
        self.give_back(xd);
        Ok(y)
    }

    /// ONE op on the device, for the op-level comparison against `cpu.rs`. `which`
    /// names the kernel and the shapes are explicit, so a test can hold both sides to
    /// the same arithmetic; the block test narrows a failure to a `ConvTransBlock`,
    /// and this narrows it to a launch inside one.
    pub fn op_probe(
        &mut self, which: &str, name: &str, x: &[f32], c_in: usize, c_out: usize, h: usize,
        w: usize,
    ) -> Result<Vec<f32>, String> {
        let n = h * w;
        let xd = self.upload(x);
        let out = self.buf(c_out * n);
        match which {
            "conv3x3" => self.conv3x3(xd.ptr, name, c_in, c_out, h, w, out.ptr)?,
            "conv1x1" => self.conv1x1(xd.ptr, name, c_in, c_out, h, w, out.ptr)?,
            // The register-blocked variant, forced on regardless of the env var, so
            // the two 1x1 kernels can be timed against each other.
            "conv1x1_rb" => {
                let wp = self.w(&format!("{name}.weight"));
                let bp = self.bias_of(name);
                let mut a = Args::new();
                a.ptr(xd.ptr).ptr(wp).ptr(bp).ptr(out.ptr)
                    .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
                let g = (((h * w + 63) / 64) as u32, ((c_out + 63) / 64) as u32, 1);
                self.run("lg_conv1x1_rb", Launch::new(g, (16, 16, 1)), &mut a)?
            }
            "conv2x2s2" => self.conv2x2s2(xd.ptr, name, c_in, c_out, h, w, out.ptr)?,
            "conv_t2x2" => self.conv_t2x2(xd.ptr, name, c_in, c_out, h, w, out.ptr)?,
            other => return Err(format!("op_probe: no such op `{other}`")),
        }
        let y = self.download(&out, c_out * n);
        self.give_back(out);
        self.give_back(xd);
        Ok(y)
    }


    /// The scratch for one geometry, accounting for a REPLACED one.
    ///
    /// `scratch_for` drops the old `Scratch` when the geometry changes, which frees
    /// its device memory outright - it never enters the pool. Without subtracting it
    /// here, `live` would only ever grow and the reported high-water would be the sum
    /// of every stage's scratch rather than the most the card ever had to hold.
    fn scratch_slot<'s>(
        &mut self, slot: &'s mut Option<Scratch>, c: usize, trans: usize, h: usize, w: usize,
        win: usize,
    ) -> Result<&'s mut Scratch, String> {
        let rebuilt = !slot.as_ref().map(|s| s.fits(c, trans, h, w)).unwrap_or(false);
        if rebuilt {
            // DROP THE REPLACED SCRATCH FIRST, not when the new one is assigned. Taking
            // it here frees a stage's buffers before the next stage's are allocated;
            // leaving it in place overlaps the two sets for the length of one
            // allocation, which at 2048x2048 is 2.1 GB of scratch alive twice over.
            if let Some(old) = slot.take() {
                let b = old.bytes();
                self.drop_note(b);
            }
            // Charge the accounting BEFORE the allocation, so a failure inside
            // `scratch_for` reports a `live` that already includes what it wanted.
            let want = Scratch::bytes_for(c, trans, h, w, win);
            self.note("ConvTransBlock scratch", want);
        }
        scratch_for(slot, c, trans, h, w, win)
    }

    /// Time ONE op with its buffers RESIDENT, so the number is the kernel's own and
    /// not the PCIe round trip.
    ///
    /// `op_probe` uploads its input and downloads its output on every call, which is
    /// right for a correctness check and wrong for a measurement: at down1's geometry
    /// that is 16.8 MB up and 16.8 MB down around each launch, which both inflates
    /// the time and hides what the kernel itself does. This keeps the buffers on the
    /// device across every iteration and synchronises once, at the end.
    ///
    /// Rows/c_in/c_out are explicit so the caller can time the same volume in either
    /// layout: `nchw = true` is the `[c][h][w]` plane a 1x1 convolution walks,
    /// `nchw = false` is the `[rows][c_in]` token layout `lg_linear` takes.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_op(
        &mut self, which: &str, name: &str, c_in: usize, c_out: usize, h: usize, w: usize,
        iters: usize,
    ) -> Result<f64, String> {
        let n = h * w;
        let xd = self.buf(c_in * n);
        // THE OUTPUT BUFFER IS SIZED BY THE ARM, NOT BY `c_out * n`. Three families
        // write a different volume: the transposed 2x2 conv writes FOUR output pixels
        // per input pixel, the row norms write `c_in` elements per row, and the two
        // window index maps move `c_in * n` floats. With the generic `c_out * n` those
        // arms wrote past the end of a buffer they did not own - three of the `budget`
        // example's rows have been measured that way - and the failure mode is not a
        // wrong number but CUDA_ERROR_ILLEGAL_ADDRESS reported against the NEXT launch,
        // on a buffer far from the culprit. (`bench_attn` has always sized its own, for
        // exactly this reason: qkv is `[n][3c]`, which `c_in * n` does not provide.)
        let out_elems = match which {
            "conv_t2x2" | "conv_t2x2_elem" | "conv_t2x2_rb" => 4 * c_out * n,
            "layer_norm" | "layer_norm_warp" | "layer_norm_block" | "gather" | "scatter" => {
                c_in * n
            }
            _ => c_out * n,
        };
        let out = self.buf(out_elems);
        let launch = |me: &mut Self| -> Result<(), String> {
            match which {
                // `conv1x1` dispatches by env var, which makes it a useless
                // measurement arm - the "toolkit" column of `k1x1` silently measured
                // the tiled kernel twice. Each variant has its own name here.
                // `conv1x1` and `linear` dispatch by env var, which makes them
                // useless as measurement arms - the "toolkit" column of `k1x1` once
                // silently measured the tiled kernel twice. Each kernel has its own
                // name here.
                "conv1x1" => me.conv1x1(xd.ptr, name, c_in, c_out, h, w, out.ptr),
                "conv1x1_rb" => {
                    let wp = me.w(&format!("{name}.weight"));
                    let bp = me.bias_of(name);
                    let mut a = Args::new();
                    a.ptr(xd.ptr).ptr(wp).ptr(bp).ptr(out.ptr)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
                    let g = (((n + 63) / 64) as u32, ((c_out + 63) / 64) as u32, 1);
                    me.run("lg_conv1x1_rb", Launch::new(g, (16, 16, 1)), &mut a)
                }
                "conv1x1_toolkit" => {
                    let wp = me.w(&format!("{name}.weight"));
                    let bp = me.bias_of(name);
                    let mut a = Args::new();
                    a.ptr(xd.ptr).ptr(wp).ptr(bp).ptr(out.ptr)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
                    // c_out * pixels work items, NOT `n`: `lg_conv1x1` gives one
                    // thread one output ELEMENT, and the elements are c_out planes of
                    // h*w. Passing the plane size launches 1/c_out of the work and
                    // reports a fantasy rate (it read 0.078 ms, i.e. 6.9 TFLOP/s, on
                    // this card's ~7 TFLOP/s at 1600 MHz).
                    me.run("lg_conv1x1", Launch::new(grid(c_out * n), (BLOCK as u32, 1, 1)), &mut a)
                }
                "linear" => me.linear(xd.ptr, name, n, c_in, c_out, out.ptr),
                // THE TOOLKIT'S PLAIN GEMM, so the register-blocked replacement can be
                // A/B'd inside one window instead of across two processes.
                "linear_toolkit" => {
                    let wp = me.w(&format!("{name}.weight"));
                    let bp = if me.wt.has(&format!("{name}.bias")) {
                        me.w(&format!("{name}.bias"))
                    } else {
                        0
                    };
                    let mut a = Args::new();
                    a.ptr(xd.ptr).ptr(wp).ptr(bp).ptr(out.ptr)
                        .i32(n as i32).i32(c_in as i32).i32(c_out as i32);
                    let g = (((c_out + 15) / 16) as u32, ((n + 15) / 16) as u32, 1);
                    me.run("lg_linear", Launch::new(g, (16, 16, 1)), &mut a)
                }
                "conv3x3" => me.conv3x3(xd.ptr, name, c_in, c_out, h, w, out.ptr),
                // The window attention needs ITS OWN buffer sizes - qkv is `[n][3c]`
                // and the output `[n][c]`, which the `c_in * n` / `c_out * n` pair
                // this function allocates does NOT provide. Measuring it here ran the
                // kernel off the end of both buffers; `bench_attn` is the arm that
                // sizes them correctly.
                "attn" | "attn_row" => {
                    let win = me.wt.window;
                    let nww = w / win;
                    let nw = (h / win) * nww;
                    let qkvd = xd.ptr;
                    let rpd = me.w(&format!("{name}.trans_block.msa.relative_position_params"));
                    let heads = c_out;
                    let hd = c_in / heads;
                    let which = if which == "attn_row" { "row" } else { me.attn_kind() };
                    me.attention_which(
                        which, qkvd, rpd, out.ptr, nw, n * win * win, nww, win, h, w, heads, hd, 0, 0,
                    )
                }
                // The elementwise ops and the window index maps. They are each a
                // fraction of a percent of a forward, and WITHOUT them the budget
                // example cannot be summed against a measured forward at all - an
                // unexplained few percent is exactly where a wrong ranking hides.
                // The two strided convolutions, at `c_in * n` in and `c_out * n` out
                // where `n = h*w` is the INPUT plane - which is the shape the walker
                // hands them (`conv2x2s2` halves the plane, `conv_t2x2` doubles it, so
                // the output buffer here is larger than `c_out * n` and the caller
                // must size `c_out` for the doubled geometry).
                "conv2x2s2" => me.conv2x2s2(xd.ptr, name, c_in, c_out, h, w, out.ptr),
                "conv_t2x2" => me.conv_t2x2(xd.ptr, name, c_in, c_out, h, w, out.ptr),
                // The per-element forms, forced on regardless of the env var, so the
                // two can be timed against each other in one window.
                "conv2x2s2_elem" => {
                    let wp = me.w(&format!("{name}.weight"));
                    let mut a = Args::new();
                    a.ptr(xd.ptr).ptr(wp).ptr(out.ptr)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
                    me.run("sc_conv2x2s2", Launch::new(grid(c_out * (h / 2) * (w / 2)), (BLOCK as u32, 1, 1)), &mut a)
                }
                "conv_t2x2_elem" => {
                    let wp = me.w(&format!("{name}.weight"));
                    let mut a = Args::new();
                    a.ptr(xd.ptr).ptr(wp).ptr(out.ptr)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
                    me.run("sc_conv_t2x2", Launch::new(grid(c_out * 4 * h * w), (BLOCK as u32, 1, 1)), &mut a)
                }
                "conv2x2s2_rb" => {
                    let wp = me.w(&format!("{name}.weight"));
                    let mut a = Args::new();
                    a.ptr(xd.ptr).ptr(wp).ptr(0).ptr(out.ptr)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
                    let g = Self::rb_grid((h / 2) * (w / 2), c_out);
                    me.run("lg_conv2x2s2", Launch::new(g, (BLOCK as u32, 1, 1)), &mut a)
                }
                "conv_t2x2_rb" => {
                    let wp = me.w(&format!("{name}.weight"));
                    let mut a = Args::new();
                    a.ptr(xd.ptr).ptr(wp).ptr(0).ptr(out.ptr)
                        .i32(c_in as i32).i32(c_out as i32).i32(h as i32).i32(w as i32);
                    let g = Self::rb_grid(4 * h * w, c_out);
                    me.run("lg_conv_t2x2", Launch::new(g, (BLOCK as u32, 1, 1)), &mut a)
                }
                "layer_norm" => me.layer_norm(xd.ptr, name, n, c_in, out.ptr),
                "layer_norm_warp" => me.layer_norm_which("warp", xd.ptr, name, n, c_in, out.ptr),
                "layer_norm_block" => me.layer_norm_which("block", xd.ptr, name, n, c_in, out.ptr),
                "gelu" => me.gelu(xd.ptr, out.ptr, n),
                "relu" => me.relu(xd.ptr, out.ptr, n),
                "add" => me.add(xd.ptr, xd.ptr, out.ptr, n),
                "copy" => me.copy(xd.ptr, out.ptr, n),
                // The gather/scatter move `nw * win * win * c` floats, which is the
                // `c_in * n` the caller allocated exactly when the plane is a whole
                // number of windows wide - which `bench_op`'s callers guarantee.
                "gather" | "scatter" => {
                    let win = me.wt.window;
                    let nww = w / win;
                    let nw = (h / win) * nww;
                    let shift = if which == "scatter" { 1 } else { 0 };
                    if which == "gather" {
                        me.gather(xd.ptr, out.ptr, nw, win * win, nww, win, h, w, c_in, shift, 0)
                    } else {
                        me.scatter(xd.ptr, out.ptr, nw, win * win, nww, win, h, w, c_in, shift, 0)
                    }
                }
                // THE LAUNCH ITSELF, with no work behind it. A forward makes a few
                // thousand launches, so the per-launch cost is multiplied by
                // thousands - and this repository has no other way to see it. The
                // event-pair profiler cannot: its own events cost more than the
                // launches it measures, which is why it inflates short kernels.
                "noop" => {
                    // `lg_noop` takes no arguments at all, so it needs its own empty
                    // argument list rather than the one the arms above share.
                    let mut na = Args::new();
                    me.run("lg_noop", Launch::new((1, 1, 1), (32, 1, 1)), &mut na)
                }
                other => return Err(format!("bench_op: no such op `{other}`")),
            }
        };
        for _ in 0..3 {
            launch(self)?;
        }
        lightgpu::vm::sync()?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            launch(self)?;
        }
        lightgpu::vm::sync()?;
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        self.give_back(out);
        self.give_back(xd);
        Ok(ms)
    }

    /// Time the window attention alone at one geometry, with the buffers sized as the
    /// graph sizes them: `qkv` is `[nw*n][3c]` and the output `[nw*n][c]`.
    ///
    /// This exists because the profile made the attention the largest single cost
    /// (31% of device time at 256x256) and the shared-memory rewrite of it showed no
    /// end-to-end change - so the kernel had to be isolated before either conclusion
    /// was drawn. `which` is `row` (one block per window, a thread per query-head row)
    /// or `s` (one block per window-head, k and v staged in shared memory).
    #[allow(clippy::too_many_arguments)]
    pub fn bench_attn(
        &mut self, which: &str, prefix: &str, c: usize, heads: usize, hd: usize, h: usize,
        w: usize, iters: usize,
    ) -> Result<f64, String> {
        let win = self.wt.window;
        let nww = w / win;
        let nw = (h / win) * nww;
        let ntok = win * win;
        let nt = nw * ntok;
        let qkvd = self.buf(3 * nt * c);
        let out = self.buf(nt * c);
        let rpd = self.w(&format!("{prefix}.trans_block.msa.relative_position_params"));
        let launch = |me: &mut Self| -> Result<(), String> {
            me.attention_which(
                which, qkvd.ptr, rpd, out.ptr, nw, ntok, nww, win, h, w, heads, hd, 0, 0,
            )
        };
        for _ in 0..3 {
            launch(self)?;
        }
        lightgpu::vm::sync()?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            launch(self)?;
        }
        lightgpu::vm::sync()?;
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        self.give_back(out);
        self.give_back(qkvd);
        Ok(ms)
    }

    /// `lg_linear` on the device, one op, for the same purpose as `op_probe`.
    pub fn linear_probe(
        &mut self, name: &str, x: &[f32], rows: usize, c_in: usize, c_out: usize,
    ) -> Result<Vec<f32>, String> {
        let xd = self.upload(x);
        let out = self.buf(rows * c_out);
        self.linear(xd.ptr, name, rows, c_in, c_out, out.ptr)?;
        let y = self.download(&out, rows * c_out);
        self.give_back(out);
        self.give_back(xd);
        Ok(y)
    }

    /// `lg_layer_norm` on the device, one op, for the same purpose.
    pub fn layer_norm_probe(
        &mut self, name: &str, x: &[f32], rows: usize, c: usize,
    ) -> Result<Vec<f32>, String> {
        let xd = self.upload(x);
        let out = self.buf(rows * c);
        self.layer_norm(xd.ptr, name, rows, c, out.ptr)?;
        let y = self.download(&out, rows * c);
        self.give_back(out);
        self.give_back(xd);
        Ok(y)
    }

    /// The attention KERNEL alone, on a caller-supplied `[tokens][3c]` qkv and
    /// `[heads][span][span]` bias. No gather, no layer norm, no linear: this is what
    /// tells a kernel bug apart from a wiring bug when the pipeline around it has
    /// already been proven step by step.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kernel_probe(
        &mut self, h: usize, w: usize, c: usize, heads: usize, hd: usize, shift: usize,
        qkv: &[f32], rp: &[f32],
    ) -> Result<Vec<f32>, String> {
        let win = self.wt.window;
        let nww = w / win;
        let nw = (h / win) * nww;
        let ntok = win * win;
        let nt = nw * ntok;
        let qd = self.upload(qkv);
        let rd = self.upload(rp);
        let att = self.buf(nt * c);
        self.attention(qd.ptr, rd.ptr, att.ptr, nw, ntok, nww, win, h, w, heads, hd, shift, 0)?;
        let y = self.download(&att, nt * c);
        self.give_back(att);
        self.give_back(qd);
        self.give_back(rd);
        Ok(y)
    }

    /// The gather alone, so the device's index map can be compared against the
    /// reference's own `pix` formula rather than only against the scatter.
    pub fn gather_probe(
        &mut self, h: usize, w: usize, c: usize, shift: usize, x: &[f32],
    ) -> Result<Vec<f32>, String> {
        let win = self.wt.window;
        let nww = w / win;
        let nw = (h / win) * nww;
        let ntok = win * win;
        let xd = self.upload(x);
        let tok = self.buf(nw * ntok * c);
        self.gather(xd.ptr, tok.ptr, nw, ntok, nww, win, h, w, c, shift, 0)?;
        let y = self.download(&tok, nw * ntok * c);
        self.give_back(tok);
        self.give_back(xd);
        Ok(y)
    }

    /// GATHER then SCATTER on the same plane: the round trip must be the identity,
    /// because both halves use `sc_pos` with the same shift. A mismatch here is an
    /// index-map bug and it is separable from the attention, which sits between the
    /// two in the graph.
    pub fn window_round_trip(
        &mut self, h: usize, w: usize, c: usize, shift: usize, x: &[f32],
    ) -> Result<Vec<f32>, String> {
        let win = self.wt.window;
        let nww = w / win;
        let nw = (h / win) * nww;
        let ntok = win * win;
        let xd = self.upload(x);
        let tok = self.buf(nw * ntok * c);
        let out = self.buf(c * h * w);
        self.gather(xd.ptr, tok.ptr, nw, ntok, nww, win, h, w, c, shift, 0)?;
        self.scatter(tok.ptr, out.ptr, nw, ntok, nww, win, h, w, c, shift, 0)?;
        let y = self.download(&out, c * h * w);
        self.give_back(tok);
        self.give_back(out);
        self.give_back(xd);
        Ok(y)
    }

    /// The transformer half of one block on the device, from a `[h][w][c]` plane -
    /// the layout `cpu::attention` takes - so the pair can be compared directly.
    ///
    /// THE OUTPUT IS THE GATHER'S `[tokens][c]` ORDER, and `cpu::attention` returns a
    /// scattered `[h*w][c]` buffer. Comparing the two directly is comparing two
    /// indexings; `tests/cuda_ops.rs` scatters this result with the reference's own
    /// `pix` map before the comparison.
    pub fn attention_probe(
        &mut self, prefix: &str, h: usize, w: usize, c: usize, shift: usize, x: &[f32],
    ) -> Result<Vec<f32>, String> {
        let win = self.wt.window;
        let heads = c / self.wt.head_dim;
        let hd = self.wt.head_dim;
        let nww = w / win;
        let nw = (h / win) * nww;
        let ntok = win * win;
        let nt = nw * ntok;
        let xd = self.upload(x);
        let tok = self.buf(nt * c);
        let norm = self.buf(nt * c);
        let qkv = self.buf(3 * nt * c);
        let att = self.buf(nt * c);
        let out = self.buf(nt * c);
        self.gather(xd.ptr, tok.ptr, nw, ntok, nww, win, h, w, c, shift, 0)?;
        self.layer_norm(tok.ptr, &format!("{prefix}.trans_block.ln1"), nt, c, norm.ptr)?;
        self.linear(norm.ptr, &format!("{prefix}.trans_block.msa.embedding_layer"), nt, c, 3 * c, qkv.ptr)?;
        let rp = self.w(&format!("{prefix}.trans_block.msa.relative_position_params"));
        self.attention(qkv.ptr, rp, att.ptr, nw, ntok, nww, win, h, w, heads, hd, shift, 0)?;
        self.linear(att.ptr, &format!("{prefix}.trans_block.msa.linear"), nt, c, c, out.ptr)?;
        let y = self.download(&out, nt * c);
        for b in [tok, norm, qkv, att, out] {
            self.give_back(b);
        }
        self.give_back(xd);
        Ok(y)
    }

    /// The whole network on the padded plane: the device twin of
    /// `cpu::forward_dump`, same stages, same shift decisions, same skip order.
    ///
    /// EVERY BUFFER IS OWNED AND SWAPPED, exactly as the CPU twin swaps its `Vec`s:
    /// a block reads `cur` and writes `tmp`, then the two swap, so no raw pointer
    /// outlives the statement that reads it and the borrow checker is satisfied
    /// without a single `unsafe`.
    pub fn forward(&mut self, x: &[f32], hp: usize, wp: usize) -> Result<Vec<f32>, String> {
        let (in_nc, dim, win) = (self.wt.in_nc, self.wt.dim, self.wt.window);
        let config = self.wt.config;
        let block_dims = |wt: &Weights, p: &str| wt.block_dims(p);
        let n = hp * wp;

        let xd = self.upload(x);

        // m_head: 3x3, in_nc -> dim. `head` is x1, which the m_tail skip needs at
        // the very end, so it is kept and the down stages run on a copy.
        // m_head goes STRAIGHT INTO `cur`, and the skip it produces for `m_tail` is
        // RECOMPUTED at the end rather than held for the whole forward.
        //
        // A FULL-RESOLUTION PLANE IS THE EXPENSIVE THING HERE, and this one is only
        // needed at the very end. Keeping it costs 64 floats a pixel - 1073 MiB at
        // 2048x2048, against the ~4.8 GB another tenant leaves free - and the
        // recomputation is one 3x3 convolution from the 3-channel input, 0.3% of the
        // network's FLOPs. It is also bit-identical: same kernel, same input, same
        // order.
        let mut cur = self.buf_named("current activation", dim * n);
        self.conv3x3(xd.ptr, "m_head.0", in_nc, dim, hp, wp, cur.ptr)?;

        // Three down stages. The stage's OUTPUT - after the stride-2 conv - is the
        // skip, and it is that buffer the up stage adds before its transposed
        // convolution, at the same resolution and channel count.
        // The three stage skips, HOST-resident by default - see `skip_kind` and the
        // note where they are written.
        // WHICH FORMS THE SKIPS TAKE, decided before the down pass begins.
        //
        // `auto` sizes the three of them - 2 * width * (hh/2) * (ww/2) floats each,
        // which is what the stage's stride-2 conv produces - and takes the DEVICE
        // form only if they all fit in HALF the memory free right now. The half is
        // not a fudge: the up pass re-allocates the whole activation ladder at
        // increasing size, so a policy that consumed all the free memory to hold
        // skips would trade a transfer for an allocation failure.
        //
        // The host form is what makes 2048x2048 fit, so it stays the fallback and
        // `SCUNET_SKIP=host`/`device` force either one.
        // A PER-STAGE DECISION AGAINST A MEASURED BUDGET, not an up-front estimate.
        //
        // The budget is WHATEVER IS FREE BEYOND WHAT THE FORWARD ITSELF NEEDS, and
        // each skip is charged against it as it is produced. Both halves of that
        // matter. A skip is sized `out_w * oh * ow` with `out_w = 2 * width`, twice
        // what a channels-times-pixels reading suggests, and an all-or-nothing answer
        // cannot degrade: a size one byte over the line loses the whole win, and a size
        // one byte under it is an allocation failure at the very end of the forward -
        // on the one requirement that must keep working.
        //
        // Subtracting the forward's own requirement is what makes the policy degrade
        // instead of gamble. The requirement is `memguard::device_need`, the measured
        // host-skip high-water; on a quiet card at 2048x2048 that leaves room for the
        // three skips, and with another tenant holding memory it leaves nothing and
        // every skip goes to the host. A blind fraction of free memory would not: a
        // quarter of a mostly-occupied 8 GB card authorises skips on a forward that
        // then needs more than the card has.
        let kind = self.skip_kind();
        // The same function the guard uses, so budget and guard cannot disagree about
        // what a size costs. It keeps a margin on the measurement, because
        // under-estimating it is the direction that authorises skips the forward
        // cannot afford.
        let need = crate::memguard::device_need(n);
        let mut budget: Option<usize> = match kind {
            "device" => None,                          // no limit, for A/B
            "host" => Some(0),                         // nothing on the device
            _ => Some(lightgpu::vm::free_vram().unwrap_or(0).saturating_sub(need)),
        };
        let mut spent = 0usize;
        let announce = self.trace.is_some() || std::env::var("SCUNET_SKIP").as_deref() == Ok("auto");
        if announce {
            println!(
                "  skip budget: {:.1} MiB free at entry, {:.1} MiB live, need {:.1} MiB -> headroom {:.1} MiB",
                lightgpu::vm::free_vram().unwrap_or(0) as f64 / 1048576.0,
                self.live as f64 / 1048576.0,
                need as f64 / 1048576.0,
                budget.unwrap_or(0) as f64 / 1048576.0
            );
        }
        let mut skips: Vec<Option<Skip>> = (0..3).map(|_| None).collect();
        // One scratch per geometry, rebuilt only when a stage changes the width or
        // the plane - `fits` is what tells a stage change from a repeat.
        let mut slot: Option<Scratch> = None;
        let (mut hh, mut ww) = (hp, wp);
        for k in 0..3 {
            let name = ["down1", "down2", "down3"][k];
            let count = config[k];
            let (ch, _) = block_dims(self.wt, &format!("m_{name}.0"));
            let width = 2 * ch;
            // IN PLACE: a block's only read of its input is at the very end, after
            // `y` and `proj` have been written, so `out == x` is well defined and the
            // walker needs no second full-resolution plane. That plane is 32 floats a
            // pixel at the largest stage - 537 MiB at 2048x2048 - and it was the
            // buffer the 2048 allocation died on.
            for i in 0..count {
                let prefix = format!("m_{name}.{i}");
                let dims = block_dims(self.wt, &prefix);
                let b = self.scratch_slot(&mut slot, width, dims.1, hh, ww, win)?;
                let shift = if i % 2 == 1 { win / 2 } else { 0 };
                self.conv_trans_block(cur.ptr, &prefix, dims, hh, ww, shift, cur.ptr, b)?;
            }
            let (oh, ow) = (hh / 2, ww / 2);
            let out_w = 2 * width;
            let down = self.buf_named("stage skip", out_w * oh * ow);
            self.conv2x2s2(cur.ptr, &format!("m_{name}.{count}"), width, out_w, hh, ww, down.ptr)?;
            // The stage's output is BOTH the skip and the next stage's input, and the
            // next stage writes in place - so the two roles need separate buffers and
            // the copy is what separates them. Order matters: `cur` is given back
            // only AFTER its replacement is taken, or the pool can hand the same
            // buffer back and the copy becomes a self-copy.
            let next = self.buf(out_w * oh * ow);
            self.copy(down.ptr, next.ptr, out_w * oh * ow)?;
            self.give_back(cur);
            // SKIPS LIVE IN HOST RAM, and that is the single largest memory saving
            // available. A skip-connected network holds the down pass's outputs for
            // the whole up pass, and SCUNet's three skips are 56 floats per input
            // pixel - 896 MiB at 2048x2048, against the ~4.8 GB another tenant leaves
            // on this card. They are written once and read once, so the cost of
            // moving them is one D2H and one H2D each - and the ADD back is
            // element-wise over DISJOINT ranges, which cannot change the arithmetic
            // (each element is added exactly once, no accumulation is reordered, and
            // no split point is observable).
            // The charge is the WALKER's own element count for this skip, so the
            // policy and the allocation cannot disagree about how big it is.
            let skip_bytes = out_w * oh * ow * 4;
            let device_skip = match budget.as_mut() {
                None => true,
                Some(left) => {
                    if skip_bytes <= *left {
                        *left -= skip_bytes;
                        spent += skip_bytes;
                        true
                    } else {
                        false
                    }
                }
            };
            if announce {
                println!(
                    "  skip {}: {:.1} MiB -> {} (device skips so far {:.1} MiB)",
                    k + 1,
                    skip_bytes as f64 / 1048576.0,
                    if device_skip { "device" } else { "host" },
                    spent as f64 / 1048576.0
                );
            }
            skips[k] = Some(if device_skip {
                // The device form keeps the buffer itself and never downloads: that is
                // exactly what the measurement compares against. The buffer already
                // came from `buf_named`, which does not pool it, so it cannot be
                // handed to anything else while the skip owns it.
                Skip::Device(down)
            } else {
                let v = self.download(&down, out_w * oh * ow);
                self.give_back(down);
                Skip::Host(v)
            });
            cur = next;
            hh = oh;
            ww = ow;
        }

        // The body, at the same resolution, no stride-2 conv.
        {
            let count = config[3];
            let (ch, _) = block_dims(self.wt, "m_body.0");
            let width = 2 * ch;
            for i in 0..count {
                let prefix = format!("m_body.{i}");
                let dims = block_dims(self.wt, &prefix);
                let b = self.scratch_slot(&mut slot, width, dims.1, hh, ww, win)?;
                let shift = if i % 2 == 1 { win / 2 } else { 0 };
                self.conv_trans_block(cur.ptr, &prefix, dims, hh, ww, shift, cur.ptr, b)?;
            }
        }

        // Three up stages: the transposed convolution on `cur + skip`, then the
        // blocks. The skip is added BEFORE the transposed convolution.
        for k in 0..3 {
            let name = ["up3", "up2", "up1"][k];
            let count = config[4 + k];
            let (ci, co) = self.wt.up_conv_channels(name);
            let joined = cur;
            // TAKE THE SKIP, do not borrow it: an up stage consumes its skip once and
            // never needs it again, so releasing it here means the three skip buffers
            // are not all live across the whole graph. They are held only as long as
            // the down pass that produced them, which is the minimum any
            // skip-connected network can ask for.
            let skip = skips[2 - k].take().expect("the skip for this up stage");
            match skip {
                Skip::Host(v) => self.add_host(&v, joined.ptr, ci * hh * ww)?,
                Skip::Device(b) => {
                    // One whole-plane launch, the same element-wise add over the same
                    // disjoint range - so the device and host forms agree bit for bit.
                    self.add(b.ptr, joined.ptr, joined.ptr, ci * hh * ww)?;
                    self.give_back(b);
                }
            }
            let (nh, nw_) = (hh * 2, ww * 2);
            let up = self.buf(co * nh * nw_);
            self.conv_t2x2(joined.ptr, &format!("m_{name}.0"), ci, co, hh, ww, up.ptr)?;
            self.give_back(joined);
            let src = up;
            for i in 0..count {
                let prefix = format!("m_{name}.{}", i + 1);
                let dims = block_dims(self.wt, &prefix);
                let b = self.scratch_slot(&mut slot, co, dims.1, nh, nw_, win)?;
                // The blocks of an up stage run W, SW, W, SW starting UNSHIFTED: the
                // reference's comprehension index never sees the prepended
                // ConvTranspose2d, so list index 0 does not invert the pattern. Getting
                // this backwards produces a plausible image that is wrong everywhere,
                // which is why tests/parity.rs compares against the upstream torch
                // module and not just against the CPU backend.
                let shift = if i % 2 == 1 { win / 2 } else { 0 };
                self.conv_trans_block(src.ptr, &prefix, dims, nh, nw_, shift, src.ptr, b)?;
            }
            cur = src;
            hh = nh;
            ww = nw_;
        }

        // m_head recomputed, then m_tail over y + x1, then the crop (in backend.rs).
        let head = self.buf_named("m_head recomputed for m_tail", dim * n);
        self.conv3x3(xd.ptr, "m_head.0", in_nc, dim, hp, wp, head.ptr)?;
        self.add(head.ptr, cur.ptr, cur.ptr, dim * n)?;
        self.give_back(head);
        let out = self.buf(in_nc * n);
        self.conv3x3(cur.ptr, "m_tail.0", dim, in_nc, hp, wp, out.ptr)?;
        let y = self.download(&out, in_nc * n);
        self.give_back(out);
        self.give_back(cur);
        self.give_back(xd);
        // THE SCRATCH IS A LOCAL, so Rust frees its device memory the moment this
        // function returns - and `live` has to follow, or it keeps counting bytes the
        // card has already been given back. The walker rebuilds the scratch on the
        // next forward and notes it again, so the omission compounded once per
        // forward: at 1024x1024 `live` climbed 268 MiB a forward - exactly one
        // 64-float-per-pixel plane - and reached 3952 MiB after twelve, while
        // `nvidia-smi` held the whole process at 1642 MiB. `scratch_slot` already
        // accounts for the REPLACEMENT case; this is the same subtraction at the end.
        if let Some(last) = slot.take() {
            self.drop_note(last.bytes());
        }
        Ok(y)
    }
}

impl crate::backend::Backend for Cuda<'_> {
    fn name(&self) -> &'static str {
        "cuda"
    }

    /// The device bytes this backend holds: weights uploaded so far plus every
    /// scratch buffer in its pool. DETERMINISTIC, unlike a free-VRAM delta - the
    /// driver's own free-memory figure moves with other processes' allocations and
    /// with the driver's own caching, and reading it around a forward produced
    /// numbers that disagreed by a factor of seven between two sizes of the same
    /// run.
    ///
    /// This is the number that answers "what does an N x N image cost on the
    /// device": after one forward the pool holds exactly the buffers that forward
    /// needed, and the weights are the checkpoint.
    fn footprint(&self) -> (usize, usize) {
        let w: usize = self.weights.values().map(|b| b.bytes).sum();
        let p: usize = self.pool.values().flat_map(|v| v.iter()).map(|b| b.bytes).sum();
        (w, p)
    }

    /// Where the device time went, most expensive kernel first. Needs the profiler,
    /// i.e. `SCUNET_PROFILE` set before `Cuda::new` - otherwise the report is empty
    /// and this prints nothing.
    fn profile_report(&self) {
        let Some(p) = self.profiler.as_ref() else { return };
        let _ = lightgpu::vm::sync();
        let totals = p.borrow().totals();
        let all: f32 = totals.iter().map(|(_, _, ms)| ms).sum();
        println!("  ---- device time per kernel (total {all:.1} ms) ----");
        for (name, n, ms) in totals {
            println!("  {ms:9.1} ms  {n:6} launches  {name}");
        }
    }

    fn forward(&mut self, x: &[f32], plan: &Plan) -> Result<Vec<f32>, String> {
        self.forward(x, plan.hp, plan.wp)
    }

    fn mem_report(&self) {
        Cuda::mem_report(self);
    }

    /// Free every pooled scratch buffer. The weights stay: they are the checkpoint,
    /// they are the same bytes for every image, and re-uploading them per size would
    /// put the checkpoint's cost into each case's timing.
    fn release_scratch(&mut self) {
        let b = self.pool_bytes();
        self.pool.clear();
        self.free_note("release_scratch", b);
        // RESET THE HIGH-WATER TO WHAT IS STILL HELD. `peak` only ever rises, so a
        // benchmark that walks several sizes in one process would report every later
        // size's high-water as at least the largest earlier size's - the reading that
        // makes a small size look like it needs a big one. After this the next size's
        // peak is its OWN requirement, from the weights plus a steady state.
        self.peak = self.live;
    }
}
