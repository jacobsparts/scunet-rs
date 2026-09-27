
// Print the graph's geometry table: for each stage, the block width, plane and
// the bytes a 1x1 convolution moves at that stage. This is the arithmetic behind
// the memory-bound analysis of `lg_conv1x1`, which the kernel profile named as
// the largest single cost (41% of device time).
use scunet::{plan::Plan, Weights};

fn main() -> Result<(), String> {
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let size = std::env::args().nth(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(256);
    let p = Plan::new(size, size, wt.window);
    let (mut hh, mut ww) = (p.hp, p.wp);
    let names = ["down1", "down2", "down3", "body", "up3", "up2", "up1"];
    println!("{size}x{size}: padded {}x{}", p.hp, p.wp);
    let mut total_bytes = 0usize;
    let mut total_flops = 0.0f64;
    for (k, name) in names.iter().enumerate() {
        let (nh, nw) = if k < 3 {
            (hh / 2, ww / 2)
        } else if k == 3 {
            (hh, ww)
        } else {
            (hh * 2, ww * 2)
        };
        let count = wt.config[k];
        // the blocks of this stage run at the PREVIOUS resolution for down stages
        let (bh, bw) = if k < 3 { (hh, ww) } else if k == 3 { (hh, ww) } else { (hh, ww) };
        let (ch, _) = wt.block_dims(&format!("m_{name}.0"));
        let c = 2 * ch;
        let n = bh * bw;
        // conv1_1 and conv1_2 each: read c*n, write c*n, read c*c weights
        let bytes = count * 2 * (2 * c * n + c * c) * 4;
        let flops = count as f64 * 2.0 * (2.0 * (c as f64) * (c as f64) * (n as f64));
        total_bytes += bytes;
        total_flops += flops;
        println!(
            "  {name:5} blocks {:2} at {bh}x{bw} c={c:4}  n={n:6}  1x1 traffic {:7.1} MiB  {:6.1} GFLOP",
            count, bytes as f64 / 1048576.0, flops / 1e9
        );
        hh = nh;
        ww = nw;
    }
    println!(
        "  TOTAL 1x1: {:.1} MiB moved, {:.1} GFLOP; at 320 GB/s the floor is {:.0} ms",
        total_bytes as f64 / 1048576.0,
        total_flops / 1e9,
        total_bytes as f64 / 320e9 * 1e3
    );
    Ok(())
}
