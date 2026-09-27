//! WHICH SIDE IS NONDETERMINISTIC, and WHERE the divergence lives.
//!
//! `examples/blockmarg.rs` caught `m_body.0` at 576x64 shift0 differing from
//! `cpu::conv_trans_block` by up to 1.7e3 (relative 1.999) at a DIFFERENT INDEX on
//! each run, once every ~40 repetitions. Comparing DEVICE AGAINST DEVICE shows the
//! device is the nondeterministic side: the CPU reference is bit-stable.
//!
//! WHAT THIS ADDS is the SHAPE of the divergence, which is what localizes it. A
//! `ConvTransBlock` is a conv half (channels `0..conv`) and a transformer half
//! (`conv..c`), joined by a 1x1 projection that mixes all `c` channels per pixel. So:
//!
//!   * a large error in the conv half only, on some pixels, is a conv-half kernel;
//!   * a large error spread across ALL channels of some pixels is a transformer-half
//!     kernel, because the projection smears one bad token across every channel;
//!   * a large error in the transformer half only, on some pixels, is the gather,
//!     the attention or the scatter;
//!   * differences of one ulp everywhere are a reassociation, not a race.
//!
//!     cargo run --release --features cuda --example blockdet -- [reps]
use scunet::cuda::Cuda;
use scunet::Weights;

fn seq(n: usize, a: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * a).sin() * 0.7 + 0.3 * a).sin()).collect()
}

fn main() -> Result<(), String> {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let mut be = Cuda::new(&wt)?;
    for (prefix, h, w, shift) in [
        ("m_body.0", 576usize, 64usize, 0usize),
        ("m_body.0", 576, 64, 4),
        ("m_down1.0", 512, 72, 0),
    ] {
        let dims = wt.block_dims(prefix);
        let (conv, c) = (dims.0, dims.0 + dims.1);
        let n = h * w;
        let x = seq(c * n, 0.71);
        let first = be.block_dump(prefix, h, w, shift, &x)?;
        let scale = first.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let tol = 1e-3 * scale;
        let mut bad_runs = 0usize;
        for r in 1..reps {
            let got = be.block_dump(prefix, h, w, shift, &x)?;
            let mut any = 0usize;
            let mut big = 0usize;
            let mut nan = 0usize;
            // Where the LARGE differences are: per channel, and per half.
            let mut big_conv = 0usize;
            let mut big_trans = 0usize;
            let mut pix_conv: std::collections::BTreeSet<usize> = Default::default();
            let mut pix_trans: std::collections::BTreeSet<usize> = Default::default();
            for i in 0..first.len() {
                let a = first[i];
                let b = got[i];
                if a.is_nan() || b.is_nan() {
                    nan += 1;
                }
                let d = (a - b).abs();
                if d != 0.0 {
                    any += 1;
                }
                if !(d <= tol) {
                    big += 1;
                    // NCHW: element `i` is channel `i / n` at pixel `i % n`. The
                    // earlier version of this file divided by `c` instead of `n`,
                    // which is the TOKEN-major convention and reported channel and
                    // pixel ranges that named nothing.
                    let (ch, pix) = (i / n, i % n);
                    if ch < conv {
                        big_conv += 1;
                        pix_conv.insert(pix);
                    } else {
                        big_trans += 1;
                        pix_trans.insert(pix);
                    }
                }
            }
            if big > 0 || nan > 0 {
                bad_runs += 1;
                // The SHAPE of the bad region: the channel and pixel ranges, and the
                // first few indices as (channel, y, x). A rectangular region names a
                // tiled kernel's block; a scattered one is a race.
                let (mut ch_lo, mut ch_hi) = (usize::MAX, 0usize);
                let (mut px_lo, mut px_hi) = (usize::MAX, 0usize);
                let mut sample: Vec<String> = Vec::new();
                for i in 0..first.len() {
                    if !((first[i] - got[i]).abs() <= tol) {
                        let (ch, pix) = (i / n, i % n);
                        ch_lo = ch_lo.min(ch);
                        ch_hi = ch_hi.max(ch);
                        px_lo = px_lo.min(pix);
                        px_hi = px_hi.max(pix);
                        if sample.len() < 6 {
                            sample.push(format!(
                                "(ch {ch}, y {}, x {}, {:.4} -> {:.4})",
                                pix / w, pix % w, first[i], got[i]
                            ));
                        }
                    }
                }
                println!(
                    "    run {r}: {any} differ at all, {big} beyond tolerance, {nan} NaN; \
                     conv half {big_conv} over {} pixels, trans half {big_trans} over {} pixels (of {n}); \
                     channels {ch_lo}..{ch_hi} of {c}, pixels {px_lo}..{px_hi} of {n}",
                    pix_conv.len(), pix_trans.len()
                );
                println!("      {}", sample.join(" "));
                // The PIXEL pattern: the first few differing pixels as (y, x) with how
                // many channels each has, and the sorted unique stride between them.
                let mut pix: Vec<usize> = pix_conv.iter().copied().collect();
                pix.sort_unstable();
                let show: Vec<String> = pix
                    .iter()
                    .take(8)
                    .map(|&p| {
                        let nch = (0..c).filter(|&ch| {
                            !((first[ch * n + p] - got[ch * n + p]).abs() <= tol)
                        }).count();
                        let lo = (0..c)
                            .filter(|&ch| !((first[ch * n + p] - got[ch * n + p]).abs() <= tol))
                            .min();
                        format!("(y {}, x {}, {nch} ch from {:?})", p / w, p % w, lo)
                    })
                    .collect();
                let strides: Vec<usize> = pix.windows(2).map(|s| s[1] - s[0]).collect();
                let mut uniq = strides.clone();
                uniq.sort_unstable();
                uniq.dedup();
                println!("      pixels: {}", show.join(" "));
                println!("      {} pixels, strides {:?}", pix.len(), &uniq[..uniq.len().min(8)]);
            }
        }
        println!("  {prefix} {h}x{w} shift{shift}: {bad_runs} of {} runs diverge beyond tolerance", reps - 1);
    }
    Ok(())
}
