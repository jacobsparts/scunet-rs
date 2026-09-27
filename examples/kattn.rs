//! The window attention alone, at each stage geometry, with correct buffer sizes.
//!
//! WHY IT EXISTS. The per-kernel profile put `sc_window_attn` at 31% of device
//! time, and the obvious fix - stage each head's k and v in shared memory instead
//! of letting all 64 query threads re-read them from global - showed NO end-to-end
//! change. That is the kind of result that needs the kernel isolated before the
//! conclusion is drawn either way, so this times the two kernels against each
//! other with the buffers sized as the graph sizes them.
//!
//!     cargo run --release --features cuda --example kattn -- 256 50
use scunet::plan::Plan;
use scunet::weights::Weights;

fn main() -> Result<(), String> {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(size, size, wt.window);
    let mut be = scunet::cuda::Cuda::new(&wt)?;

    let h = p.hp;
    let stages: [(&str, usize, usize); 7] = [
        ("down1", h, 0),
        ("down2", h / 2, 0),
        ("down3", h / 4, 0),
        ("body", h / 8, 0),
        ("up3", h / 8, 1),
        ("up2", h / 4, 1),
        ("up1", h / 2, 1),
    ];
    println!("window attention at {size}x{size}, {iters} iterations, buffers resident");
    println!(
        "  {:<7} {:>10} {:>6} {:>6} {:>6}  {:>10} {:>10}   {}",
        "stage", "plane", "c", "heads", "hd", "row ms", "shared ms", "speedup"
    );
    let (mut tr, mut ts) = (0.0f64, 0.0f64);
    for (name, res, idx) in stages {
        let prefix = format!("m_{name}.{idx}");
        let (_, trans) = wt.block_dims(&prefix);
        let heads = trans / wt.head_dim;
        let hd = wt.head_dim;
        // The attention's own geometry: `c` is the token width (trans), `heads` the
        // head count, and the kernels need qkv `[tokens][3c]` and out `[tokens][c]`.
        let row = be.bench_attn("row", &prefix, trans, heads, hd, res, res, iters)?;
        let sh = be.bench_attn("s", &prefix, trans, heads, hd, res, res, iters)?;
        tr += row;
        ts += sh;
        println!(
            "  {name:<7} {res:>4}x{res:<4} {trans:>6} {heads:>6} {hd:>6}  {row:>10.3} {sh:>10.3}   {:.2}x",
            row / sh
        );
    }
    println!(
        "  {:<36}  {tr:>10.3} {ts:>10.3}   {:.2}x total per forward",
        "TOTAL", tr / ts
    );
    Ok(())
}
