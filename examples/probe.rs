//! Development: one ConvTransBlock, dumped internally, to compare against the numpy
//! reference's internals for the same block.
//!
//!     cargo run --release --example probe -- m_down1.0 0 [in.bin] [outprefix]
//!     cargo run --release --example probe -- m_down1.1 4 /tmp/probe_in.bin /tmp/probe2
//!
//! The input file is raw f32, `[c][h][w]`; the channel count is read off the weights.
use std::fs;

use scunet::cpu;
use scunet::Weights;

fn load(path: &str) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn save(path: &str, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(path, b).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let prefix = a.first().cloned().unwrap_or_else(|| "m_down1.0".into());
    let shift: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(0);
    let infile = a.get(2).cloned().unwrap_or_else(|| "/tmp/probe_in.bin".into());
    let outpre = a.get(3).cloned().unwrap_or_else(|| "/tmp/probe".into());
    // The plane size: 64x64 by default (the m_down1 blocks), 32x32 for m_down2.
    let whw: Vec<usize> = a
        .get(4)
        .map(|s| s.split('x').map(|t| t.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![64, 64]);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors").unwrap();
    let x = load(&infile);
    let (h, w) = (whw[0], whw[1]);
    let dims = wt.block_dims(&prefix);
    let c = dims.0 + dims.1;
    if x.len() != c * h * w {
        eprintln!("input is {} floats, expected {}", x.len(), c * h * w);
        std::process::exit(1);
    }
    println!("prefix {prefix} dims {dims:?} shift {shift}");
    let mut sc = cpu::BlockScratch::new();
    let mut stage = cpu::StageDump::default();
    let mut out = vec![0.0f32; c * h * w];
    cpu::conv_trans_block_dump(&x, dims, h, w, wt.window, shift, &wt, &prefix, &mut out, &mut sc, &mut stage);
    save(&format!("{outpre}_out_rust.bin"), &out);
    for (name, v) in [
        ("conv1_1", stage.expand.as_ref()),
        ("convhalf", stage.conv_half.as_ref()),
        ("transhalf", stage.trans_half.as_ref()),
        ("res", stage.res.as_ref()),
        ("msa", stage.msa.as_ref()),
    ] {
        if let Some(v) = v {
            save(&format!("{outpre}_{name}_rust.bin"), v);
        }
    }
    println!("wrote {outpre}_*_rust.bin");
}
