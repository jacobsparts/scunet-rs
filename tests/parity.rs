//! The Rust backends against the validated reference, on the golden fixtures.
//!
//! The fixtures were produced by `tools/reference.py`, which agrees with an
//! independent functional transcription of `network_scunet.py` to 3.1e-6 on the full
//! model. The tolerance here is 2e-3: an exact transcription lands at ~1e-6, a single
//! wrong tap, a transposed layout, an inverted shift pattern, a missing skip or the
//! conv/trans halves swapped in the concatenation is >=1e-2. Three orders of
//! magnitude of headroom in each direction is what makes this test worth having;
//! tightening it to 1e-4 would start failing on accumulation-order differences
//! between the reference's numpy einsums and this Rust, which are not bugs.
//!
//! The test SKIPS (not fails) when the converted checkpoint is absent, so the
//! repository builds and tests cleanly without the 72 MB weights.
use std::path::{Path, PathBuf};

use scunet::cpu::Cpu;

#[path = "device_lock.rs"]
mod device_lock;
use scunet::fixture::Fixture;
use scunet::{plan, Weights};

const TOL: f32 = 2e-3;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// `../models/scunet-color-real-psnr.safetensors`, or None if the checkpoint has not
/// been converted on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = repo().join("../models/scunet-color-real-psnr.safetensors");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// The same check on the device. SKIPS (not fails) when the build has no CUDA
/// feature or the machine has no usable device, so a CPU-only checkout still runs
/// the suite; the fixtures are identical, and so is the tolerance, because a device
/// backend that is off by more than accumulation order is wrong for the same
/// reasons.
#[cfg(feature = "cuda")]
fn check_cuda(fixture: &str) {
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping cuda {fixture}: checkpoint not converted");
        return;
    };
    let wt = Weights::load(&ckpt).expect("checkpoint loads");
    let f = Fixture::load(repo().join("tests/data").join(fixture)).expect("fixture loads");
    // NO SKIP HERE ONCE THE CHECKPOINT EXISTS: the CUDA feature is on, so a device
    // failure is a failure. The earlier `match` returned silently on any error and
    // printed to stderr, which read as `ok` in the test summary - a skip that reports
    // a pass is worse than no test, and it hid a real divergence for a while.
    let _guard = device_lock::device_lock();
    let mut be = scunet::cuda::Cuda::new(&wt).expect("a usable device with --features cuda");
    let got = scunet::backend::run(&mut be, &f.input, f.c, f.h, f.w, &wt).expect("cuda forward");
    let (worst, at, mean) = f.compare(&got);
    let (y, x, c) = f.locate(at);
    assert!(
        worst <= TOL,
        "cuda {fixture}: worst |diff| {worst:.3e} at (y {y}, x {x}, channel {c}), mean {mean:.3e} \
         - tolerance {TOL:.0e}. When this fails while the CPU check passes, suspect the window \
         attention first (__expf against the CPU's .exp(), and the -FLT_MAX/4 mask sentinel) \
         and the winograd kernel's summation order second (it replaces lg_conv3x3s1p1 wherever \
         c_in >= 32)."
    );
    eprintln!("cuda {fixture}: worst {worst:.3e} mean {mean:.3e} (y {y}, x {x}, c {c})");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_color_real_psnr_80x64() {
    check_cuda("color_real_psnr.bin");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_color_real_psnr_64x64() {
    check_cuda("color_real_psnr_64.bin");
}

fn check(fixture: &str) {
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping {fixture}: ../models/scunet-color-real-psnr.safetensors not converted");
        return;
    };
    let wt = Weights::load(&ckpt).expect("checkpoint loads");
    let f = Fixture::load(repo().join("tests/data").join(fixture)).expect("fixture loads");
    assert_eq!(f.variant, wt.arch.name(), "fixture was made with a different checkpoint");
    assert_eq!(f.c, wt.in_nc, "fixture channels disagree with the checkpoint");
    assert_eq!(f.win, wt.window);

    let mut be = Cpu::new(&wt).expect("cpu backend");
    let got = scunet::backend::run(&mut be, &f.input, f.c, f.h, f.w, &wt).expect("forward");
    let (worst, at, mean) = f.compare(&got);
    let (y, x, c) = f.locate(at);
    assert!(
        worst <= TOL,
        "{fixture}: worst |diff| {worst:.3e} at (y {y}, x {x}, channel {c}), mean {mean:.3e} - \
         tolerance {TOL:.0e}. A diff of 1e-2 or more means a structural mismatch (a tap, a \
         layout, the shift pattern, a skip), not numerics."
    );
    eprintln!("{fixture}: worst {worst:.3e} mean {mean:.3e} (y {y}, x {x}, c {c})");
}

#[test]
fn color_real_psnr_80x64() {
    // Pads to 128x64: both axes exercise replicate padding, and a shifted window
    // straddles the padded border.
    check("color_real_psnr.bin");
}

#[test]
fn color_real_psnr_64x64() {
    // No padding at all: the shifted-window mask straddles interior borders only.
    check("color_real_psnr_64.bin");
}

/// The geometry the engine claims: replicate padding to a multiple of 64, exact for
/// any size, crop back. This does not need the checkpoint.
#[test]
fn plan_geometry() {
    let p = plan::Plan::new(80, 64, 8);
    assert_eq!((p.hp, p.wp), (128, 64));
    assert_eq!(plan::Plan::new(256, 256, 8).hp, 256);
    assert_eq!(plan::Plan::new(257, 1, 8).hp, 320);
    // For the published window (8), 64 = 8 * 8, so EVERY input is fine once padded -
    // which is why the engine needs no minimum size at all. The guard exists for a
    // window large enough that the 1/8-resolution body would not fill one window,
    // and that is the case this asserts on.
    assert!(plan::check_min_size(1, 1, 8).is_ok());
    assert!(plan::check_min_size(1, 1, 16).is_err());
}
