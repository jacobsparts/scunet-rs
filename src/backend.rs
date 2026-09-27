//! The backend trait, so the CLI and the fixture checker are written once.
//!
//! SCUNet HAS NO TILE LOOP, and this file is where that is recorded rather than
//! silently omitted. The sibling Swin2SR engine must tile: its input padding is a
//! REFLECTION of the whole plane, so a tile's border is not derivable from the tile,
//! and `swin2sr-rs/src/backend.rs` carries the overlap arithmetic that repairs it.
//! SCUNet pads by REPLICATION instead - `nn.ReplicationPad2d`, i.e. edge clamping -
//! and a convolution over a clamped edge reads exactly the values it would read from
//! the interior of a larger padded tensor. So one padded pass over the whole image is
//! exact for any size, and the trait below needs no tile geometry at all.
//!
//! That is a claim about the arithmetic, not a hope: `tests/parity.rs` runs an 80x64
//! fixture that pads on BOTH axes, where a tiling or padding mistake cannot hide.
use crate::plan::Plan;
use crate::weights::Weights;

/// Where the arithmetic happens. `Cpu` and `Gpu` differ in nothing else: the pad, the
/// crop and the [0,1] float contract live in `run` below, once.
pub trait Backend {
    /// For the run header: "cpu" or "cuda".
    fn name(&self) -> &'static str;

    /// `[c][hp][wp]` at the padded geometry -> the same shape, the network's output.
    /// The buffer is borrowed, not owned: each backend keeps its own scratch.
    fn forward(&mut self, x: &[f32], plan: &Plan) -> Result<Vec<f32>, String>;

    /// Device bytes this backend is holding, as (weights, scratch), after the
    /// forwards it has already run. Deterministic and free of driver noise, unlike
    /// a free-memory reading around a launch - see `Cuda::footprint`.
    fn footprint(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Print the per-kernel device-time breakdown, when this backend collects one
    /// (`SCUNET_PROFILE` for CUDA) - see `Cuda::profile_report`. A no-op here so a
    /// caller can ask unconditionally.
    fn profile_report(&self) {}

    /// Print the device-memory high-water, and the allocations that reached it
    /// when `SCUNET_MEMTRACE` is set. A no-op here so a caller can ask
    /// unconditionally.
    ///
    /// The HIGH-WATER is the number that decides whether an image fits, and it is
    /// not `footprint`: that sums the pool, which counts every buffer a forward has
    /// ever held including ones held at different moments.
    fn mem_report(&self) {}

    /// Drop the pooled scratch, so the next forward allocates afresh.
    ///
    /// TWO THINGS NEED THIS. A benchmark that walks several input sizes in one
    /// process otherwise reports the SUM of every size's buffers as one size's
    /// footprint - the small sizes' plane buffers stay pooled and are counted again.
    /// And a server that sees one unusual size keeps its buffers until the process
    /// ends, which is a leak with a bound but still a leak.
    fn release_scratch(&mut self) {}
}

/// Replicate-pad, run, crop - the whole I/O contract of the model, once.
///
/// The caller hands in `[c][h][w]` in [0,1] (the CLI divides the uint8 image by 255
/// and does not subtract a mean or divide by a standard deviation, because SCUNet
/// was not trained with one).
pub fn run<B: Backend + ?Sized>(
    be: &mut B, x: &[f32], c: usize, h: usize, w: usize, wt: &Weights,
) -> Result<Vec<f32>, String> {
    let plan = Plan::new(h, w, wt.window);
    crate::plan::check_min_size(h, w, wt.window)?;
    // WILL IT FIT? Asked HERE rather than at each call site, so the CLI, `tests/`
    // and the fixture checker are all covered by one copy of it - and asked BEFORE
    // the padding below, which is itself a full-resolution allocation.
    //
    // The two backends fail differently without this and both are worth avoiding.
    // The CPU path has no fallible allocation anywhere, so a pass larger than the
    // machine ABORTS (observed: exit 134, `memory allocation of 268435456 bytes
    // failed`, no image size and no stage); the CUDA path does report a failed
    // allocation, but only after the card has already refused, which on a shared
    // device is after another tenant took the bytes. See `src/memguard.rs`.
    match be.name() {
        // `Cuda` is the only backend that is not host memory. A build without the
        // cuda feature cannot reach this arm, so the name test is the whole check.
        #[cfg(feature = "cuda")]
        "cuda" => {
            crate::memguard::check_cuda(plan.plane())?;
        }
        _ => {
            crate::memguard::check_cpu(wt, h, w)?;
        }
    }
    let padded = if plan.hp == h && plan.wp == w {
        x.to_vec()
    } else {
        crate::plan::pad_replicate(x, c, h, w, plan.hp, plan.wp)
    };
    let y = be.forward(&padded, &plan)?;
    if y.len() != c * plan.plane() {
        return Err(format!(
            "{}: the backend returned {} floats for a {}x{} plane of {c} channels",
            be.name(),
            y.len(),
            plan.hp,
            plan.wp
        ));
    }
    Ok(crate::plan::crop(&y, c, h, w, plan.hp, plan.wp))
}
