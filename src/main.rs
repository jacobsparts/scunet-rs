//! SCUNet real-world image denoising, standalone.
//!
//! Usage:
//!   scunet -m scunet-color-real-psnr.safetensors -i noisy.png -o clean.png
//!   scunet -m ... -i noisy.png -o clean.png --device cpu
//!
//! The model file is the output of `tools/convert.py`, not the original .pth.
use std::path::PathBuf;
use std::time::Instant;

use scunet::cpu::Cpu;
use scunet::Weights;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> ! {
    eprintln!(
        "scunet {VERSION} - SCUNet real-world image denoising on lightgpu

USAGE:
    scunet -m <weights.safetensors> -i <in.png> -o <out.png> [options]

OPTIONS:
    -m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
    -i, --input <path>    input PNG, 8-bit, RGB or grayscale
    -o, --output <path>   where to write the denoised PNG
        --device <dev>    gpu or cpu (default: gpu when this build has CUDA and a
                          driver, cpu otherwise; `cuda` is accepted for gpu)
        --cpu             same as --device cpu
        --gpu             same as --device gpu, and refuses to fall back
    -q, --quiet           no progress output
    -h, --help            this text
    -V, --version         print the version

The model takes the image at its native resolution - there is no tile size to
choose and no `--tile` flag."
    );
    std::process::exit(2)
}

struct Args {
    weights: PathBuf,
    input: PathBuf,
    output: PathBuf,
    device: String,
    /// Set only when the caller NAMED the GPU. Without it, a GPU that cannot be
    /// brought up is not fatal: the engine falls back to the CPU backend, which is
    /// what lets one binary run on a machine with no NVIDIA driver at all. A
    /// CPU-only build has no GPU to fail to bring up, so it never reads this.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    force_gpu: bool,
    quiet: bool,
}

fn parse() -> Args {
    let mut i = std::env::args().skip(1);
    let mut weights = None;
    let mut input = None;
    let mut output = None;
    let mut device = if cfg!(feature = "cuda") { "gpu" } else { "cpu" }.to_string();
    let mut force_gpu = false;
    let mut quiet = false;
    while let Some(a) = i.next() {
        let mut take = |what: &str| -> String {
            i.next().unwrap_or_else(|| {
                eprintln!("scunet: {what} needs a value");
                usage()
            })
        };
        match a.as_str() {
            "-m" | "--model" | "--weights" => weights = Some(PathBuf::from(take("--model"))),
            "-i" | "--input" | "--in" => input = Some(PathBuf::from(take("--input"))),
            "-o" | "--output" | "--out" => output = Some(PathBuf::from(take("--output"))),
            "--device" => {
                device = take("--device");
                force_gpu = device == "gpu" || device == "cuda";
            }
            "--cpu" => device = "cpu".to_string(),
            "--gpu" => {
                device = "gpu".to_string();
                force_gpu = true;
            }
            "-q" | "--quiet" => quiet = true,
            "-h" | "--help" => usage(),
            "-V" | "--version" => {
                println!("scunet {VERSION}");
                std::process::exit(0);
            }
            // A DRIVER SCRIPT SHARED ACROSS THE FAMILY MAY PASS THIS, so answer with
            // the reason instead of ignoring it: this engine runs a whole image at
            // once, and a silently accepted `--tile` would let a caller believe it
            // had bounded the memory the process will use.
            "--tile" | "--tile-pad" => {
                eprintln!("scunet: `{a}` is not supported: this engine runs a whole image at once");
                eprintln!("scunet: (SCUNet pads by replication to a multiple of 64 and has no tile geometry)");
                std::process::exit(2);
            }
            other => {
                eprintln!("scunet: unknown argument `{other}`");
                usage();
            }
        }
    }
    let weights = weights.unwrap_or_else(|| {
        eprintln!("scunet: --model is required (see tools/convert.py)");
        usage()
    });
    let (Some(input), Some(output)) = (input, output) else {
        eprintln!("scunet: --input and --output are required");
        usage()
    };
    Args { weights, input, output, device, force_gpu, quiet }
}

fn main() {
    // ASKED BEFORE THE DEVICE IS CHOSEN, because the refusal must not be maskable by
    // anything that follows it: a refused switch surfaced through the CUDA backend's
    // constructor would look like "the GPU is unavailable" and send the run to the CPU
    // at a fifth of the speed, which is a worse outcome than an error.
    if let Some(name) = scunet::dev::refused_switch() {
        eprintln!(
            "scunet: `{name}` is a development switch, and this build is not one - \
             rebuild with `--features dev` to use it"
        );
        std::process::exit(2);
    }
    let args = parse();
    let wt = match Weights::load(&args.weights) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("scunet: {e}");
            std::process::exit(1);
        }
    };
    if !args.quiet {
        eprintln!(
            "scunet {VERSION}: {} ({} in_nc {}, dim {}, window {})",
            wt.arch.name(), args.weights.display(), wt.in_nc, wt.dim, wt.window
        );
    }
    let img = match scunet::image::read(&args.input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("scunet: {e}");
            std::process::exit(1);
        }
    };
    if img.c != wt.in_nc {
        eprintln!(
            "scunet: {}: {} channels, but this checkpoint takes {} (a {} model)",
            args.input.display(),
            img.c,
            wt.in_nc,
            wt.arch.name()
        );
        std::process::exit(2);
    }

    let t0 = Instant::now();
    let out = match args.device.as_str() {
        #[cfg(feature = "cuda")]
        "gpu" | "cuda" => match scunet::cuda::Cuda::new(&wt) {
            Ok(mut be) => {
                if !args.quiet {
                    eprintln!("scunet: device cuda");
                }
                match scunet::backend::run(&mut be, &img.data, img.c, img.h, img.w, &wt) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("scunet: {e}");
                        std::process::exit(1);
                    }
                }
            }
            // NAMING THE GPU IS A REQUEST; NOT NAMING IT IS NOT. gpu is the default
            // in a CUDA build, so a machine with no driver must still work: the
            // driver failing to come up is not an error unless the caller asked for
            // the GPU by name.
            Err(e) if args.force_gpu => {
                eprintln!("scunet: {e}");
                std::process::exit(1);
            }
            Err(e) => {
                // NOT PROGRESS OUTPUT, SO `--quiet` DOES NOT SILENCE IT: which
                // backend ran is a property of the result.
                eprintln!("scunet: cuda: {e}");
                eprintln!("scunet: falling back to the CPU backend (--gpu forces the GPU)");
                run_cpu(&wt, &img)
            }
        },
        #[cfg(not(feature = "cuda"))]
        "gpu" | "cuda" => {
            eprintln!("scunet: this build has no cuda feature; use --device cpu");
            std::process::exit(2);
        }
        "cpu" => run_cpu(&wt, &img),
        other => {
            eprintln!("scunet: unknown device `{other}` (gpu or cpu)");
            std::process::exit(2);
        }
    };
    if !args.quiet {
        eprintln!("scunet: {}x{} -> {}x{} in {:.2}s", img.w, img.h, img.w, img.h, t0.elapsed().as_secs_f64());
    }
    if let Err(e) = scunet::image::write(&args.output, &out, img.c, img.h, img.w) {
        eprintln!("scunet: {e}");
        std::process::exit(1);
    }
}

fn run_cpu(wt: &Weights, img: &scunet::image::Image) -> Vec<f32> {
    let mut be = match Cpu::new(wt) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("scunet: {e}");
            std::process::exit(1);
        }
    };
    match scunet::backend::run(&mut be, &img.data, img.c, img.h, img.w, wt) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("scunet: {e}");
            std::process::exit(1);
        }
    }
}
