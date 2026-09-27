//! Inference timing and footprint, per input size and per device.
//!
//! The measurement the README quotes and the one kernel work should be judged by:
//! `backend::run` end to end, which is pad + forward + crop - the whole path a user's
//! image takes, including the replicate padding that a large image pays for.
//!
//!     cargo run --release --features cuda --example bench -- \
//!         -m ../models/scunet-color-real-psnr.safetensors \
//!         --device gpu --sizes 256,512,1024,2048 --iters 5
//!
//! WHAT IS REPORTED, and why each number is there:
//!
//! * `min` and `median` of the per-iteration wall clock. `min` is the least
//!   perturbed sample and is the honest one for comparing kernels; `median` is what
//!   a user mostly sees. The first iteration after a warm-up is still slow on a
//!   device whose clocks have not ramped, so at least one warm-up run is required
//!   and three is the default.
//! * `MP/s` as the throughput the size actually buys, and `GFLOP/s` against the
//!   model's own `flops_conv`/`flops_attention`, so a result can be compared with
//!   the hardware's peak rather than only with another run of itself.
//! * On CUDA, an EXACT device footprint rather than a free-VRAM delta: the bytes of
//!   every weight uploaded plus every buffer in the scratch pool (`Backend::footprint`).
//!   The driver's free-memory figure moves with other processes' allocations and with
//!   the driver's own caching, and reading it around a forward produced numbers that
//!   disagreed by a factor of seven between two sizes of the same run. `peakRSS` is the
//!   host side, from `VmHWM`.
//!
//!   THE FOOTPRINT IS PER SIZE EVEN IN ONE PROCESS, and that took a fix: the pool is
//!   keyed by element count and `give_back` only ever pushed, so a run that walked
//!   64, 128, ... 1024 summed the buffers of every size it had seen and reported
//!   4357 MiB at 1024x1024 where the truth is 1569. `Cuda::give_back` now caps each
//!   size at the buffers one forward actually holds and frees the surplus.
//!
//! The input is the deterministic plane `tools/bench_torch.py`'s `raw_input`
//! generates, so a PyTorch run on the same (h, w) sees identical numbers
//! value for value.
use std::time::Instant;

use scunet::plan;
use scunet::weights::Weights;

/// The golden-ratio fractional sequence, value for value as
/// `tools/bench_torch.py`'s `raw_input`.
fn seq(n: usize) -> Vec<f32> {
    let t = 0.618_033_988_749_894_9f32;
    let mut x = 0.0f32;
    (0..n)
        .map(|_| {
            let v = x;
            x += t;
            if x >= 1.0 {
                x -= 1.0;
            }
            v
        })
        .collect()
}

/// The process's peak resident set, in MiB, from /proc/self/status.
fn peak_rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

struct Args {
    weights: String,
    device: String,
    sizes: Vec<usize>,
    iters: usize,
    warmup: usize,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        weights: "../models/scunet-color-real-psnr.safetensors".into(),
        device: "cpu".into(),
        sizes: vec![256, 512, 1024, 2048],
        iters: 5,
        warmup: 3,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "-m" | "--model" | "--weights" => a.weights = val()?,
            // `gpu` is what the engine's own CLI calls this device; `cuda` is what
            // this benchmark has always called it. Both are accepted.
            "--device" => {
                a.device = match val()?.as_str() {
                    "gpu" => "cuda".to_string(),
                    other => other.to_string(),
                }
            }
            "--iters" => a.iters = val()?.parse().map_err(|e| format!("--iters: {e}"))?,
            "--warmup" => a.warmup = val()?.parse().map_err(|e| format!("--warmup: {e}"))?,
            "--sizes" => {
                a.sizes = val()?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().map_err(|e| format!("--sizes: {e}")))
                    .collect::<Result<Vec<_>, _>>()?
            }
            "--help" | "-h" => {
                println!("bench [-m PATH] [--device cpu|gpu] [--sizes h,w,...] [--iters N] [--warmup N]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(a)
}

fn main() -> Result<(), String> {
    let args = parse()?;
    let wt = Weights::load(&args.weights)?;
    let c = wt.in_nc;
    println!(
        "{}: {} in_nc {} dim {} window {} config {:?}, {} threads",
        args.weights,
        wt.arch.name(),
        c,
        wt.dim,
        wt.window,
        wt.config,
        rayon::current_num_threads()
    );
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10}",
        "size", "min ms", "med ms", "mean ms", "MP/s", "GFLOP/s", "VRAM MiB", "peakRSS"
    );

    // One boxed backend: the loop below is identical for either device, and the
    // `?Sized` bound on `backend::run` is what makes a trait object usable here.
    let mut be: Box<dyn scunet::Backend> = match args.device.as_str() {
        "cpu" => Box::new(scunet::cpu::Cpu::new(&wt)?),
        #[cfg(feature = "cuda")]
        "cuda" => Box::new(scunet::cuda::Cuda::new(&wt)?),
        other => {
            return Err(format!(
                "--device {other} is not available in this build ({}); use --features cuda for a device",
                if cfg!(feature = "cuda") { "cpu|cuda" } else { "cpu" }
            ))
        }
    };

    for &h in &args.sizes {
        let w = h;
        let plan = plan::Plan::new(h, w, wt.window);
        let x = seq(c * h * w);
        let flops = (wt.flops_conv(plan.hp, plan.wp) + wt.flops_attention(plan.hp, plan.wp)) as f64;

        let run = |be: &mut dyn scunet::Backend| -> Result<Vec<f64>, String> {
            for _ in 0..args.warmup {
                scunet::backend::run(be, &x, c, h, w, &wt)?;
            }
            let mut ts = Vec::with_capacity(args.iters);
            for _ in 0..args.iters {
                let t = Instant::now();
                scunet::backend::run(be, &x, c, h, w, &wt)?;
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            if std::env::var("BENCH_SAMPLES").is_ok() {
                let v: Vec<String> = ts.iter().map(|x| format!("{x:.1}")).collect();
                println!("    samples {h}x{w}: [{}]", v.join(", "));
            }
            Ok(ts)
        };

        // VRAM is measured as a HIGH-WATER: the free-before value minus the lowest
        // free value seen after any single forward. Sizing this as "free before minus
        // free after" reports 0 for a size whose buffers the pool had already grown
        // for - it measures the pool's final state, not its peak, and a repeat size
        // therefore looked like it used no device memory at all.
        let ts = run(be.as_mut())?;
        // EXACT, not a free-VRAM delta: the weights the backend has uploaded plus
        // every buffer in its scratch pool. See `Cuda::footprint` for why the
        // driver's own free-memory figure is not usable for this.
        let (wb, sb) = be.footprint();
        let vram = (wb + sb) as f64 / (1024.0 * 1024.0);

        let mut sorted = ts.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = sorted[0];
        let med = sorted[sorted.len() / 2];
        let mean = ts.iter().sum::<f64>() / ts.len() as f64;
        let mp = (h * w) as f64 / (med / 1e3) / 1e6;
        let gflops = flops / (med / 1e3) / 1e9;
        if std::env::var("BENCH_PROFILE").is_ok() {
            be.profile_report();
        }
        if std::env::var("BENCH_MEM").is_ok() {
            be.mem_report();
        }
        println!(
            "{h:>6} {:>10.1} {:>10.1} {:>10.1} {:>9.2} {:>9.1} {:>10.1} {:>9.0}M",
            min,
            med,
            mean,
            mp,
            gflops,
            vram,
            peak_rss_mib().unwrap_or(0.0)
        );
        // Hand the scratch back BETWEEN sizes, so each line of a multi-size run is
        // that size's own footprint. Without it the smaller sizes' plane buffers stay
        // pooled and are counted again in the next line's total (512 read 465 MiB
        // behind a 256 that had been measured first, against 443 on its own).
        be.release_scratch();
    }
    Ok(())
}
