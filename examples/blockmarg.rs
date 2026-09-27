//! THE MARGIN BETWEEN THE DEVICE AND THE CPU TWIN ON A `ConvTransBlock`, in numbers.
//!
//! WHY. `tests/cuda_ops.rs::block_chunks_beyond_the_token_budget` compares the device
//! against `cpu::conv_trans_block` at four geometries, on the worst absolute difference
//! scaled by the data's own magnitude, at a tolerance of 3e-5. It failed ONCE in about
//! thirty runs and has not reproduced, and the assertion's own message is the only
//! diagnostic it produces. A test that fails once and then passes is either a
//! knife-edge tolerance or a real intermittent bug, and the difference between those
//! two answers is the MARGIN - which nothing printed.
//!
//!     cargo run --release --features cuda --example blockmarg -- [reps]
use scunet::cuda::Cuda;
use scunet::Weights;

fn seq(n: usize, a: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * a).sin() * 0.7 + 0.3 * a).sin()).collect()
}

fn main() -> Result<(), String> {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let mut be = Cuda::new(&wt)?;
    for (prefix, h, w, shift) in [
        ("m_body.0", 576usize, 64usize, 0usize),
        ("m_body.0", 576, 64, 4),
        ("m_down1.0", 512, 72, 0),
        ("m_down1.0", 512, 72, 4),
    ] {
        let dims = wt.block_dims(prefix);
        let c = dims.0 + dims.1;
        let x = seq(c * h * w, 0.71);
        let mut want = vec![0.0f32; c * h * w];
        let mut sc = scunet::cpu::BlockScratch::new();
        scunet::cpu::conv_trans_block(&x, dims, h, w, wt.window, shift, &wt, prefix, &mut want, &mut sc);
        // The scale the test divides by, so the printed margin is directly comparable
        // with the tolerance it asserts against.
        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let mut worst = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut at = 0usize;
        for r in 0..reps.max(1) {
            let got = be.block_dump(prefix, h, w, shift, &x).expect("block on device");
            let (mut wa, mut wr, mut ia) = (0.0f32, 0.0f32, 0usize);
            for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                let d = (a - b).abs();
                if d > wa {
                    wa = d;
                    ia = i;
                }
                let eps = 1e-3 * scale;
                wr = wr.max(d / a.abs().max(b.abs()).max(eps));
            }
            if wa > worst || r == 0 {
                worst = wa;
                worst_rel = wr;
                at = ia;
            }
            if r == 0 || wa >= worst {
                println!(
                    "  {prefix:<12} {h}x{w} shift{shift}  worst |diff| {wa:.3e}  scale {scale:.3e}  \
                     -> {:.4e} of scale (tolerance 3.0e-5)  rel {wr:.3e}  at {ia}",
                    wa / scale
                );
            }
        }
        let _ = (worst, worst_rel, at);
    }
    Ok(())
}
