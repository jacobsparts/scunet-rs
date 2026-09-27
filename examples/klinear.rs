//! The transformer half's four linear shapes, timed per launch at their real
//! geometry, so the next GEMM change is attributable.
//!
//! WHY. After the attention and 1x1 fixes, the profile puts `lg_linear` first at 52
//! ms per forward on a 256x256 image - but that is ~4.3 GFLOP of work, i.e. 82
//! GFLOP/s, where the same kernel measured 774 GFLOP/s on the down1 volume. The
//! shapes here are the reason: `c_in` is 32 (two k-steps per 16x16 tile) and `c_out`
//! is as small as 32, so a tile's two `__syncthreads` and its shared loads are not
//! amortised by the 16 multiply-adds each thread does per k-step.
//!
//!     cargo run --release --features cuda --example klinear -- 256 30
use scunet::plan::Plan;
use scunet::weights::Weights;

fn main() -> Result<(), String> {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(size, size, wt.window);
    let mut be = scunet::cuda::Cuda::new(&wt)?;
    let win = wt.window;

    let h = p.hp;
    let stages: [(&str, usize, usize, usize); 7] = [
        // stage, plane resolution, block index, blocks in the stage
        ("down1", h, 0, 4),
        ("down2", h / 2, 0, 4),
        ("down3", h / 4, 0, 4),
        ("body", h / 8, 0, 4),
        ("up3", h / 8, 1, 4),
        ("up2", h / 4, 1, 4),
        ("up1", h / 2, 1, 4),
    ];
    println!("transformer linears at {size}x{size}, {iters} iterations, buffers resident");
    println!(
        "  {:<7} {:>7} {:>6} {:>16} {:>7} {:>7} {:>9} {:>9} {:>9} {:>7}",
        "stage", "rows", "trans", "op", "c_in", "c_out", "rb ms", "toolkit", "rb GF/s", "x"
    );
    let mut total = 0.0f64;
    let mut tot_tk = 0.0f64;
    for (name, res, idx, blocks) in stages {
        let prefix = format!("m_{name}.{idx}");
        let (_, trans) = wt.block_dims(&prefix);
        let nw = (res / win) * (res / win);
        let rows = nw * win * win;
        // The walker chunks the token dimension at TOKEN_BUDGET, so the row count a
        // launch actually sees is min(rows, budget) - measured, not assumed.
        let rows = rows.min(32768);
        let four = 4 * trans;
        let ops: [(&str, &str, usize, usize); 4] = [
            ("qkv", "trans_block.msa.embedding_layer", trans, 3 * trans),
            ("msa.linear", "trans_block.msa.linear", trans, trans),
            ("mlp.0", "trans_block.mlp.0", trans, four),
            ("mlp.2", "trans_block.mlp.2", four, trans),
        ];
        for (op, suffix, c_in, c_out) in ops {
            let full = format!("{prefix}.{suffix}");
            let ms = be.bench_op("linear", &full, c_in, c_out, rows, 1, iters)?;
            let tk = be.bench_op("linear_toolkit", &full, c_in, c_out, rows, 1, iters)?;
            let flops = 2.0 * (c_in as f64) * (c_out as f64) * (rows as f64);
            println!(
                "  {name:<7} {rows:>7} {trans:>6} {op:>16} {c_in:>7} {c_out:>7} {ms:>9.3} {tk:>9.3} {:>9.1} {:>7.2}",
                flops / (ms / 1e3) / 1e9,
                tk / ms
            );
            total += ms * blocks as f64;
            tot_tk += tk * blocks as f64;
        }
    }
    println!("  ---- per forward, all stages and blocks: register-blocked {total:.1} ms, toolkit {tot_tk:.1} ms ({:.2}x) ----", tot_tk / total);
    Ok(())
}
