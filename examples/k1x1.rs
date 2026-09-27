//! One kernel's time at each geometry the graph actually runs it, so a variant
//! can be compared in seconds instead of through a 1-second forward pass.
//!
//! WHY IT EXISTS. The per-kernel profile named `lg_conv1x1` as 41% of device time
//! on a 256x256 image. Comparing variants through a full forward mixes in every
//! other kernel; this isolates one launch at one geometry.
//!
//! THE BUFFERS STAY RESIDENT. `Cuda::bench_op` holds the input and output on the
//! device across every iteration and synchronises once at the end. Using
//! `op_probe` (which uploads on every call) measured the PCIe round trip as much
//! as the kernel: at down1's geometry that is 16.8 MB up and 16.8 MB down around a
//! launch that is itself a few milliseconds, which is what made the first
//! comparison of these kernels read 23 ms instead of 4.6 ms.
//!
//! TWO ARMS, AND THIS IS WHERE THE PROMOTION WAS DECIDED. A 1x1 convolution is a
//! matrix multiply over channels at every pixel, so one register-blocked GEMM over
//! the plane layout serves it exactly as the token layout serves the linears.
//! `rb` is the toolkit's `lg_conv1x1_rb` - the kernel the graph dispatches to -
//! and `plain` is `lg_conv1x1`, one output element per thread, which it beats by
//! 10.25x summed over the seven geometries below.
//!
//! The engine's own two 1x1 kernels were the other arms of this comparison and
//! this is the instrument that retired them: its 16x16 tiled `sc_conv1x1_t` lost to
//! the register-blocked GEMM by 1.9-2.5x at every geometry (3.075 ms against 1.345
//! summed per forward), and its register-blocked `sc_conv1x1_rb` was 2.6x slower
//! than the tiled one (55.723 ms) - the negative result that showed the memory
//! access pattern, not the number of loads, was what the kernel had to fix.
//!
//!     cargo run --release --features cuda --example k1x1 -- 256 20
use scunet::plan::Plan;
use scunet::weights::Weights;

fn main() -> Result<(), String> {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(size, size, wt.window);
    let mut be = scunet::cuda::Cuda::new(&wt)?;

    // Each stage with the resolution its BLOCKS' conv1x1 runs at, and the block
    // index conv1_1 lives at: 0 for the down stages and the body, 1 for the up
    // stages, because an up stage's list index 0 is its transposed convolution.
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
    let kernels = ["conv1x1_rb", "conv1x1_toolkit"];

    println!("1x1 convolution at {size}x{size} (padded {}x{}), {iters} iterations, buffers resident", p.hp, p.wp);
    println!(
        "  {:<7} {:>10} {:>6} {:>10} {:>10} {:>9} {:>12}",
        "stage", "plane", "c", "rb ms", "plain", "plain/rb", "GFLOP/s rb"
    );
    let mut tot = [0.0f64; 2];
    let mut gflops_tot = 0.0f64;
    for (name, res, idx) in stages {
        let prefix = format!("m_{name}.{idx}");
        let (ch, _) = wt.block_dims(&prefix);
        let c = 2 * ch;
        let flops = 2.0 * (c as f64) * (c as f64) * ((res * res) as f64);
        let mut ms = [0.0f64; 2];
        for (i, kernel) in kernels.iter().enumerate() {
            ms[i] = be.bench_op(kernel, &format!("{prefix}.conv1_1"), c, c, res, res, iters)?;
            tot[i] += ms[i];
        }
        gflops_tot += flops;
        println!(
            "  {name:<7} {res:>4}x{res:<4} {c:>6} {:>10.3} {:>10.3} {:>9.2} {:>12.1}",
            ms[0],
            ms[1],
            ms[1] / ms[0],
            flops / ms[0] / 1.0e6
        );
    }
    println!("  {:<24} {:>10.3} {:>10.3}", "TOTAL per forward", tot[0], tot[1]);
    println!(
        "  the graph runs the register-blocked GEMM: plain is {:.2}x it ({:.1} GFLOP/s against {:.1})",
        tot[1] / tot[0],
        gflops_tot / tot[0] / 1.0e6,
        gflops_tot / tot[1] / 1.0e6
    );
    Ok(())
}
