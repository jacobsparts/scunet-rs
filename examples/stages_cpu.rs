//! WHERE THE ENGINE'S OWN CPU FORWARD SPENDS ITS TIME, stage by stage.
//!
//! WHY. `budget_cpu.rs` times each op class in isolation and its honest sum came to
//! ~725 ms at 256x256 where the engine measures ~1359 - a 1.9x gap. An isolated arm
//! cannot see what the walker does BETWEEN the ops it names: the per-stage
//! `vec![0.0; ...]` allocations, their zero-fill, the whole-plane `clone()`s that
//! seed each stage, and the residual copies inside each block.
//!
//! HOW. `cpu::forward_dump` takes a hook called at every stage boundary with the
//! stage's name, and `cpu::forward` is that same function with a no-op hook - so this
//! times the production path, not a copy of it. Timestamps between consecutive hook
//! calls are the stage's own cost, allocations included.
//!
//!     cargo run --release --features cuda --example stages_cpu -- 256 [iters]
use std::time::Instant;

use scunet::plan::Plan;
use scunet::weights::Weights;

fn main() -> Result<(), String> {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(size, size, wt.window);
    let (hp, wp) = (p.hp, p.wp);
    let x = vec![0.5f32; wt.in_nc * hp * wp];
    let n = hp * wp;

    let mut per: Vec<(String, f64)> = Vec::new();
    let mut total = 0.0f64;
    for it in 0..iters + 1 {
        let mut last = Instant::now();
        let mut acc: Vec<(String, f64)> = Vec::new();
        {
            let mut hook = |name: &str, _plane: &[f32]| {
                let now = Instant::now();
                acc.push((name.to_string(), now.duration_since(last).as_secs_f64() * 1e3));
                last = now;
            };
            let _ = scunet::cpu::forward_dump(&wt, &x, hp, wp, &mut hook)?;
        }
        if it == 0 {
            continue;                       // warmup: the first pass pays for page faults
        }
        total += acc.iter().map(|(_, ms)| ms).sum::<f64>();
        // The stages are named identically every iteration, so accumulate in order.
        if per.is_empty() {
            per = acc;
        } else {
            for (dst, src) in per.iter_mut().zip(acc.iter()) {
                dst.1 += src.1;
            }
        }
    }
    let mean = |v: f64| v / iters as f64;
    println!("CPU ENGINE STAGES at {hp}x{wp}, {} threads, mean of {iters} timed passes (the walker's own allocations and copies are inside their stage)", rayon::current_num_threads());
    println!("  {:<22} {:>9} {:>8} {:>7}", "stage", "ms", "% total", "MB");
    let sum: f64 = per.iter().map(|(_, v)| mean(*v)).sum();
    // The channel count of each stage's largest plane, for the MB column: read from
    // the block dims of the stage itself so it cannot drift from the model.
    for (name, v) in &per {
        let mb = stage_mb(name, &wt, n);
        println!("  {name:<22} {:>9.1} {:>7.1}% {:>7.1}", mean(*v), 100.0 * mean(*v) / sum, mb);
    }
    println!("  {:<22} {:>9.1}   (the hook's own spans sum; the engine measured {:.1} ms)", "SUM of stages", sum, mean(total));
    Ok(())
}

/// The bytes the stage's biggest plane holds, for context in the table.
fn stage_mb(name: &str, wt: &Weights, n: usize) -> f64 {
    // Only the seven stage keys have `ConvTransBlock`s; `m_head` and `m_tail` are
    // single 3x3 convolutions at `dim`, and asking for their block dims panics.
    let key = ["down1", "down2", "down3", "body", "up3", "up2", "up1"]
        .iter()
        .find(|k| name.starts_with(&format!("m_{k}")))
        .copied();
    let dim = match key {
        // An up stage's blocks are indexed from 1: `m_upN.0` is its transposed
        // convolution. `m_up3_blocks`/`m_body`/`m_downN_blocks` all start at 0.
        Some(k) if k.starts_with("up") => 2 * wt.block_dims(&format!("m_{k}.1")).0,
        Some(k) => 2 * wt.block_dims(&format!("m_{k}.0")).0,
        None => wt.dim,
    };
    (dim * n * 4) as f64 / 1048576.0
}
