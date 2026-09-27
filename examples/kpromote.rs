//! Development: the two kernel comparisons that put a pair of kernels into the
//! lightgpu toolkit, each timed against its counterpart in ONE window.
//!
//!     cargo run --release --features cuda --example kpromote -- 20
//!
//! The A/B has to be inside one process: this card's clock swings between 139 and
//! 1887 MHz across windows, so two runs are not two measurements.
use scunet::plan::Plan;
use scunet::weights::Weights;

fn main() -> Result<(), String> {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(256, 256, wt.window);
    let mut be = scunet::cuda::Cuda::new(&wt)?;

    println!("A. LAYER NORM: warp (one warp per row, registers) vs block (one block per row)");
    // The width has to be the WIDTH OF THAT TENSOR: `bench_op` feeds `c_in` to the
    // kernel as `ne0`, so a guessed number runs off the end of the uploaded weight
    // vector (CUDA_ERROR_ILLEGAL_ADDRESS, learned the hard way). Every normalized
    // width in this checkpoint is 32, 64, 128 or 256; `ln1` and `ln2` are the same
    // width, so this is the whole range the engine runs - and 256 is already past
    // LG_LN_WARP_MAX = 128, i.e. it is the kernel's strided FALLBACK path.
    let pairs: [(&str, usize); 6] = [
        ("m_down1.0.trans_block.ln1", 32),
        ("m_down2.0.trans_block.ln1", 64),
        ("m_down3.0.trans_block.ln1", 128),
        ("m_body.0.trans_block.ln1", 256),
        ("m_body.0.trans_block.ln2", 256),
        ("m_down3.0.trans_block.ln2", 128),
    ];
    println!("  {:>7} {:>6} {:>10} {:>10} {:>8}", "rows", "ne0", "warp ms", "block", "x");
    for rows in [8192usize, 32768] {
        for (name, ne0) in pairs {
            let a = be.bench_op("layer_norm_warp", name, ne0, 0, rows, 1, iters)?;
            let tk = be.bench_op("layer_norm_block", name, ne0, 0, rows, 1, iters)?;
            let b = be.bench_op("layer_norm_warp", name, ne0, 0, rows, 1, iters)?;
            let sc = a.min(b);
            println!("  {rows:>7} {ne0:>6} {sc:>10.3} {tk:>10.3} {:>8.2}", tk / sc);
        }
    }

    println!("B. THE 2x2 STRIDE-2 PAIR: elem (one thread per output) vs rb (register-blocked, the toolkit's)");
    println!("  {:<6} {:>8} {:>6} {:>6} {:>10} {:>10} {:>8}", "stage", "plane", "c_in", "c_out", "elem ms", "rb ms", "x");
    let mut hh = p.hp;
    let mut ww = p.wp;
    for stage in 0..7usize {
        let name = ["down1", "down2", "down3", "body", "up3", "up2", "up1"][stage];
        let count = wt.config[stage];
        if stage < 3 {
            let (ch, _) = wt.block_dims(&format!("m_{name}.0"));
            let width = 2 * ch;
            let out_w = 2 * width;
            let nm = format!("m_{name}.{count}");
            let a = be.bench_op("conv2x2s2_elem", &nm, width, out_w, hh, ww, iters)?;
            let b = be.bench_op("conv2x2s2_rb", &nm, width, out_w, hh, ww, iters)?;
            println!("  {name:<6} {hh:>4}x{ww:<3} {width:>6} {out_w:>6} {a:>10.3} {b:>10.3} {:>8.2}", a / b);
            hh /= 2;
            ww /= 2;
        } else if stage > 3 {
            // The transposed conv of an up stage is `m_<stage>.0` - the BLOCKS are
            // `m_<stage>.1..`, which is what `base` indexes for the token linears.
            let (ci, co) = wt.up_conv_channels(name);
            let nm = format!("m_{name}.0");
            let a = be.bench_op("conv_t2x2_elem", &nm, ci, co, hh, ww, iters)?;
            let b = be.bench_op("conv_t2x2_rb", &nm, ci, co, hh, ww, iters)?;
            println!("  {name:<6} {hh:>4}x{ww:<3} {ci:>6} {co:>6} {a:>10.3} {b:>10.3} {:>8.2}", a / b);
            hh *= 2;
            ww *= 2;
        }
    }
    Ok(())
}
