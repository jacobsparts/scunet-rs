//! WILL THIS FORWARD FIT? A refusal BEFORE the allocation, on both backends.
//!
//! WHY. The two backends fail differently without it, and neither failure is one a
//! caller can act on. The CPU path has no fallible allocation anywhere - no
//! `try_reserve`, every buffer is `vec![0.0; n]` - so a pass larger than the machine
//! does not return an error, it ABORTS: under `ulimit -v 2097152` a 1024x1024
//! forward dies with exit code 134 (SIGABRT) and the bare line `memory allocation of
//! 268435456 bytes failed`, naming no image size, no stage and no remedy. The CUDA
//! path reports a failed allocation well (`Cuda::buf_at` names the request, the
//! high-water and the pool) but only once it has already failed - and on a card this
//! engine SHARES, that is after another tenant has taken the memory.
//!
//! So the question is asked first, and answered against numbers this engine has
//! MEASURED rather than against a sum over the graph's buffer names. Both figures
//! below were fitted to the peak of a running forward (see the Memory section of
//! README.md) and each is checked from both sides: the model predicts what the
//! instrument then reports.
//!
//! THE FIGURES ARE MODELS, NOT MEASUREMENTS, and `check_*` says so when it
//! refuses. A model can be wrong in the refusal direction - turning away a pass
//! that would have fitted - and that is the direction it is tuned to be wrong in,
//! because the alternative on a machine with a swap file is not an error at all:
//! it is a pass that takes an hour.
//!
//! WHERE IT IS CALLED. `backend::run`, which is the single entry point every
//! caller shares - the CLI, `tests/`, and the fixture checker - so the guard
//! cannot be forgotten by one of them.
use crate::plan::Plan;
use crate::weights::Weights;

/// Multiplier over the modelled activations for allocator slack: the model counts
/// the buffers the walker holds, and the allocator's own overhead, its fragmentation
/// and the page tables are not in that count.
pub const SLACK: f64 = 1.25;

/// Bytes of workspace a forward needs that do not scale with the image - the two
/// chunked staging buffers this engine keeps at a fixed size by construction
/// (`MLP_CHUNK_BYTES` and one `TOKEN_BUDGET` chunk of attention tokens).
pub const WORKSPACE: usize = 32 << 20;

/// Bytes of device memory a forward needs PER PADDED PIXEL, at the minimum, with
/// the three stage skips in host RAM.
///
/// THIS IS THE SAME NUMBER THE SKIP POLICY USES, deliberately: `Cuda::forward`
/// sizes each skip against `device_need(plane)`, so the policy and the guard cannot
/// disagree about what a size costs. Fitted to the measured HOST-SKIP high-water,
/// which is the FLOOR: the skip policy only ever moves the skips further onto the
/// device, never off it, so no image needs less than the host-skip figure.
///
/// ```text
///   size   measured   this model   ratio
///    256     92.0        107.6      1.17
///   1024   1052.0       1242.0      1.18
///   1536   2588.0       2759.5      1.07
///   2048   4828.0       4880.0      1.01
/// ```
///
/// It is over at every point, by between 1% and 18%, and it has to be: the
/// measurement moves with the card's other tenants, and UNDER-counting is the
/// direction that lets the first allocation fail. The line through the two end
/// points would be 1147 bytes a pixel; the margin above that is deliberate.
///
/// (`auto`, the default policy, measures HIGHER than these - 1132/2788/5196 - because
/// it takes the skips onto the device while the card has room. A guard built on that
/// would refuse every image between the two figures, so it is built on the floor.)
pub const DEVICE_BYTES_PER_PX: usize = 1210;
/// The non-pixel part of the device requirement - the checkpoint and the fixed pool.
pub const DEVICE_BASE: usize = 32 << 20;

/// The smallest device footprint a `plane`-pixel forward can have, in bytes.
#[inline]
pub fn device_need(plane: usize) -> usize {
    DEVICE_BYTES_PER_PX * plane + DEVICE_BASE
}

/// Floats of activation a CPU forward holds per PADDED pixel, at its worst moment.
///
/// THE WORST MOMENT IS `m_tail`, and the terms are the walker's own buffers, which
/// is why this is a count of NAMED things rather than a tuned constant:
///
/// ```text
///   cur   dim     the up pass's output, about to be read by m_head's add
///   head  dim     recomputed for m_tail (`forward`), not held from the top
///   out   in_nc   m_tail's output
///   y + proj + tok + bt, in BlockScratch: 3*dim
/// ```
///
/// `BlockScratch` is `3 * dim` and not the `c + 4 * dim` a glance at the struct
/// suggests: the widest block in the ladder is `c = dim` (the transformer half is a
/// constant `dim` at every stage and the conv half is `dim/2` at the outermost
/// stage), the conv half's two staging planes live INSIDE `proj`, and `c1`/`c2` are
/// empty on this path. `tb` is `2 * trans = dim`.
///
/// So 6*dim + in_nc = 387 floats = 1548 bytes a padded pixel at the published
/// checkpoint, against a measured 1548 (97.5 / 390.0 / 1560.0 MiB at 256 / 512 /
/// 1024, i.e. 1.52 KB/px, the same figure at every size).
#[inline]
pub fn cpu_floats_per_px(wt: &Weights) -> usize {
    6 * wt.dim + wt.in_nc
}

/// The host-memory model of one CPU forward.
#[derive(Debug, Clone, Copy)]
pub struct CpuPlan {
    /// Activations at the worst moment, in bytes.
    pub acts: usize,
    /// Fixed-size workspace, in bytes. Does not scale with the image.
    pub workspace: usize,
    /// The checkpoint, as mapped.
    pub weights: usize,
    /// What to compare against `MemAvailable`: `acts + workspace`, with `SLACK`
    /// applied, plus the weights at face value.
    pub peak: usize,
    /// The padded plane the model is for.
    pub plane: usize,
}

impl CpuPlan {
    pub fn of(wt: &Weights, h: usize, w: usize) -> CpuPlan {
        let p = Plan::new(h, w, wt.window);
        let plane = p.plane();
        let acts = cpu_floats_per_px(wt).saturating_mul(plane).saturating_mul(4);
        let workspace = WORKSPACE;
        let weights = wt.bytes as usize;
        // NOT `(acts + workspace + weights) * SLACK`: the slack is for the buffers
        // the model cannot see, and the checkpoint is not one of them - it is one
        // number this engine knows exactly.
        let peak = ((acts + workspace) as f64 * SLACK) as usize + weights;
        CpuPlan { acts, workspace, weights, peak, plane }
    }
}

/// Free HOST memory in bytes, from `/proc/meminfo`.
///
/// `MemAvailable`, NEVER `MemFree`. `MemFree` excludes reclaimable page cache, so
/// on a machine that has read a file recently it reports a tiny number and this
/// guard would refuse passes that fit comfortably - which is the failure mode of a
/// guard that is too aggressive: it is an outage, and it does not look like one.
pub fn host_available() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        let rest = line.strip_prefix("MemAvailable:")?;
        let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
        return Some(kb.saturating_mul(1024));
    }
    None
}

/// This process's address-space limit and current use, from `/proc/self/limits` and
/// `/proc/self/statm`, as `(remaining, limit)` in bytes.
///
/// WHY THIS IS HERE AT ALL. `MemAvailable` does not know about `RLIMIT_AS`, so a
/// process run under `ulimit -v` sees a machine with gigabytes free and then dies
/// inside its first large allocation - which is exactly the failure this file was
/// written for. It is also the case a container's memory limit presents in some
/// runtimes, and it is two lines of `/proc` to ask about. Returns `None` when the
/// limit is `unlimited` or unreadable, which is the usual case.
fn rlimit_as() -> Option<(usize, usize)> {
    let lim = std::fs::read_to_string("/proc/self/limits").ok()?;
    let line = lim.lines().find(|l| l.starts_with("Max address space"))?;
    // Columns are fixed-width and the SOFT limit is the first one after the name.
    let mut it = line.split_whitespace();
    it.next(); it.next(); it.next();                  // "Max", "address", "space"
    let soft = it.next()?;
    if soft == "unlimited" {
        return None;
    }
    let limit: usize = soft.parse().ok()?;
    // `statm`'s first field is the total program size in PAGES, i.e. VmSize.
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: usize = statm.split_whitespace().next()?.parse().ok()?;
    let used = pages.saturating_mul(4096);
    Some((limit.saturating_sub(used), limit))
}

/// Total host memory, for the message.
fn host_total() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Bytes as a human figure. MiB below a GiB, GiB above it, because the two numbers
/// in this message are the only thing the reader is going to act on.
pub fn fmt_bytes(n: usize) -> String {
    const MIB: f64 = 1048576.0;
    let mib = n as f64 / MIB;
    if mib < 1024.0 {
        format!("{mib:.0} MiB")
    } else {
        format!("{:.2} GiB", mib / 1024.0)
    }
}

/// Refuse a CPU pass that will not fit in host memory, BEFORE it allocates.
///
/// Fails OPEN: if `/proc/meminfo` cannot be read (not Linux, or a sandbox) the
/// caller proceeds, because this guard exists to turn a crash into a message and
/// not to become a new way for the engine to fail.
pub fn check_cpu(wt: &Weights, h: usize, w: usize) -> Result<CpuPlan, String> {
    let plan = CpuPlan::of(wt, h, w);
    let avail = match (host_available(), rlimit_as()) {
        // THE SMALLER OF THE TWO IS THE TRUTH: a process under `ulimit -v` cannot use
        // the machine's free memory, however much of it there is.
        (Some(free), Some((left, _))) => free.min(left),
        (Some(free), None) => free,
        (None, Some((left, _))) => left,
        (None, None) => return Ok(plan),
    };
    if plan.peak <= avail {
        return Ok(plan);
    }
    Err(refuse_cpu(&plan, w, h, avail, host_total()))
}

/// The refusal, as a function of the plan and the free figure, so it can be tested
/// without arranging for the machine to be out of memory.
fn refuse_cpu(plan: &CpuPlan, w: usize, h: usize, avail: usize, total: Option<usize>) -> String {
    let total = total.map(|t| format!(" of {}", fmt_bytes(t))).unwrap_or_default();
    format!(
        "not enough HOST memory for a {w}x{h} pass on the CPU\n\
         scunet: the padded plane is {p}x{p} pixels, so it needs about {peak} \
         ({acts} of activations held at the worst moment + {ws} of workspace, \
         plus {slack}% for allocator slack, and the {wts} checkpoint)\n\
         scunet: {} is available{total}, and the machine's swap is not counted - \
         a pass that goes there does not fail, it stops being usable\n\
         scunet: that figure is a MODEL of this walker's buffers fitted to measured \
         peak RSS, not a live measurement\n\
         scunet: a smaller image, the CUDA engine if this build has one, or more \
         memory is what fits",
        fmt_bytes(avail),
        wts = fmt_bytes(plan.weights),
        acts = fmt_bytes(plan.acts),
        ws = fmt_bytes(plan.workspace),
        peak = fmt_bytes(plan.peak),
        p = (plan.plane as f64).sqrt().round() as usize,
        slack = ((SLACK - 1.0) * 100.0).round() as usize,
    )
}

/// Refuse a CUDA pass that will not fit in the DEVICE's free memory, before the
/// first allocation of the forward.
///
/// Gated on the feature so a CPU-only build never touches the driver - `lightgpu`
/// is not optional, so the call would link, but asking a machine with no driver
/// about its free VRAM is not a question worth asking.
///
/// The requirement is `device_need`, the engine's own measured minimum - the same
/// number `Cuda::forward`'s skip policy sizes each stage's skip against. Free VRAM
/// is read at this instant, so it already accounts for this process's checkpoint
/// and for whatever the card's other tenants hold, which on this machine is a real
/// term rather than a hypothetical one.
///
/// Fails open in the same sense as `check_cpu`: if the driver cannot be queried,
/// the caller proceeds and the walker's own reporting takes over.
#[cfg(feature = "cuda")]
pub fn check_cuda(plane: usize) -> Result<usize, String> {
    let need = device_need(plane);
    let Ok((free, total)) = lightgpu::vm::vram() else {
        return Ok(need);
    };
    if need <= free {
        return Ok(need);
    }
    Err(format!(
        "not enough DEVICE memory for a {plane}-pixel plane ({p}x{p})\n\
         scunet: the engine's own high-water for this size is {need}, against \
         {free} free of {total} on this device\n\
         scunet: that high-water is measured, and it is the SMALLEST this forward \
         needs - the skip policy moves the three stage skips onto the device only \
         while they fit on top of it\n\
         scunet: the card is shared, so free memory here includes other processes' \
         allocations and moves between one launch and the next\n\
         scunet: a smaller image, or the CPU engine, is what fits",
        p = (plane as f64).sqrt().round() as usize,
        need = fmt_bytes(need),
        free = fmt_bytes(free),
        total = fmt_bytes(total),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `../models/scunet-color-real-psnr.safetensors`, or None when the checkpoint
    /// has not been converted on this machine.
    ///
    /// The three tests below are about the numbers a PUBLISHED CHECKPOINT rounds to -
    /// its `dim` and `in_nc` decide what a pixel costs - so there is nothing to stand
    /// in for it, and they SKIP (not fail) when it is absent, exactly as
    /// `tests/parity.rs` does. Without that, a clone of this repository passes the
    /// build and then panics in `cargo test` on a file only the author's machine has.
    fn checkpoint() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../models/scunet-color-real-psnr.safetensors");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    /// The floor of the model is the ONE size everyone knows: at 1024x1024 the
    /// engine measured 1560.0 MiB of peak live, and the model must not be far below
    /// that - a guard that under-counts is a guard that lets the abort through.
    #[test]
    fn the_model_is_at_least_the_measurement() {
        let Some(ckpt) = checkpoint() else {
            eprintln!("skipping: ../models/scunet-color-real-psnr.safetensors not converted");
            return;
        };
        let wt = Weights::load(&ckpt).unwrap();
        let p = CpuPlan::of(&wt, 1024, 1024);
        // The DERIVATION, restated as an assertion rather than as a quoted number:
        // `(6*dim + in_nc)` floats a pixel, which is 387 floats at this checkpoint.
        let per_px = p.acts / (1024 * 1024);
        assert_eq!(per_px, (6 * wt.dim + wt.in_nc) * 4);
        assert_eq!(per_px, 1548, "the published checkpoint's figure moved");
        assert!(p.peak > 1560 * 1048576, "peak {} must exceed the measured 1560 MiB", p.peak);
        // And it must be the same figure in BYTES A PIXEL at every size, which is
        // what the measurement showed.
        for s in [256usize, 384, 512, 768, 1024] {
            let q = CpuPlan::of(&wt, s, s);
            let pp = (q.acts / q.plane) as f64;
            assert!((pp - 1548.0).abs() < 1.0, "{s}: {pp} bytes a pixel");
        }
    }

    /// The device model is the HOST-skip requirement, which is the minimum a
    /// forward has - the skip policy only ever moves the skips further onto the
    /// device. Measured with `SCUNET_SKIP=host`: 92.0 MiB at 256, 1052.0 at 1024,
    /// 2588.0 at 1536, 4828.0 at 2048. The model must be at or above every one of
    /// them: UNDER-counting is the direction that lets the first allocation fail.
    ///
    /// (The default `auto` measures higher - 1132/2788/5196 - because it takes the
    /// skips onto the device while there is room. That is not the floor, and a guard
    /// built on it would refuse every image between the two figures.)
    #[test]
    fn the_device_model_covers_the_host_skip_high_water() {
        assert!(device_need(1 << 20) < device_need(2 << 20));
        for (s, mib) in [(256usize, 92.0), (1024, 1052.0), (1536, 2588.0), (2048, 4828.0)] {
            let got = device_need(s * s) as f64 / 1048576.0;
            assert!(got >= mib, "{s}: model {got:.1} MiB is under the measured {mib} MiB");
            // ...and not absurdly over it, or the guard refuses images that fit. The
            // fit's worst overshoot is 18% at 256 and 1024 (see the constant).
            assert!(got <= mib * 1.30, "{s}: model {got:.1} MiB is far over {mib} MiB");
        }
    }

    /// The refusal must FIRE, and it must say what the caller can do about it. Both
    /// halves matter: a guard that refuses silently is a mystery, and one that
    /// refuses without a remedy is a dead end.
    #[test]
    fn the_refusal_fires_and_names_the_numbers() {
        let Some(ckpt) = checkpoint() else {
            eprintln!("skipping: ../models/scunet-color-real-psnr.safetensors not converted");
            return;
        };
        let wt = Weights::load(&ckpt).unwrap();
        let plan = CpuPlan::of(&wt, 1024, 1024);
        // The shape of the message is what a caller reads, so assert on the parts
        // that are decisions rather than on its formatting.
        let msg = refuse_cpu(&plan, 1024, 1024, 512 << 20, Some(16 << 30));
        for want in ["not enough HOST memory", "1024x1024", "MiB", "smaller image", "MODEL"] {
            assert!(msg.contains(want), "the refusal does not mention `{want}`:\n{msg}");
        }
        // And the figure it quotes is the model, which is the measurement plus slack.
        assert!(msg.contains(&fmt_bytes(plan.acts)), "no activations figure:\n{msg}");
    }

    /// The guard's direction: it must let a pass that fits through. This is the
    /// failure mode that would make it an outage rather than a diagnostic - a guard
    /// tuned to be safe by refusing everything.
    ///
    /// The bound checked is that the model's claim stays under 4 GiB up to 1024x1024,
    /// which is the largest size this box has run on its 32 GiB of host memory. That
    /// is a real property and not a tautology: it is the same inequality a caller
    /// with a normal machine needs to hold.
    #[test]
    fn a_pass_that_fits_is_not_refused() {
        let Some(ckpt) = checkpoint() else {
            eprintln!("skipping: ../models/scunet-color-real-psnr.safetensors not converted");
            return;
        };
        let wt = Weights::load(&ckpt).unwrap();
        for s in [64usize, 256, 384, 512, 768, 1024] {
            let p = CpuPlan::of(&wt, s, s);
            assert!(p.peak < 4 << 30, "{s}: {} claimed, which no machine here needs", fmt_bytes(p.peak));
        }
        // ...and the model is monotone in the size, or it is not a model of this.
        let mut prev = 0;
        for s in [64usize, 256, 384, 512, 768, 1024] {
            let q = CpuPlan::of(&wt, s, s).peak;
            assert!(q > prev, "{s}: {q} did not grow");
            prev = q;
        }
    }

    /// The device refusal on a plane no card of this class can hold. Safe to run
    /// against a real device: it is a pure function of the plane and the free
    /// figure, and allocates nothing.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_device_refusal_fires_on_a_plane_that_cannot_fit() {
        // 8192x8192 is 67 Mpixels: 1210 bytes each is 75 GiB, against an 8 GB card.
        let plane = 8192 * 8192;
        let msg = check_cuda(plane).expect_err("8192x8192 cannot fit on this device");
        for want in ["not enough DEVICE memory", "8192x8192", "smaller image", "shared"] {
            assert!(msg.contains(want), "the refusal does not mention `{want}`:\n{msg}");
        }
        // ...and a size that fits is not refused.
        assert!(check_cuda(256 * 256).is_ok());
    }

    #[test]
    fn fmt_bytes_switches_at_a_gib() {
        assert_eq!(fmt_bytes(1024 * 1048576), "1.00 GiB");
        assert_eq!(fmt_bytes(512 * 1048576), "512 MiB");
    }
}
