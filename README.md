# scunet

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [nafnet-rs](https://github.com/jacobsparts/nafnet-rs),
[rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[maxim-rs](https://github.com/jacobsparts/maxim-rs),
[swin2sr-rs](https://github.com/jacobsparts/swin2sr-rs),
[ifan-rs](https://github.com/jacobsparts/ifan-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs),
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs) and
[nightenh-rs](https://github.com/jacobsparts/nightenh-rs), all built on the
[lightgpu toolkit](https://github.com/jacobsparts/lightgpu);
[adaptive-enhance](https://github.com/jacobsparts/adaptive-enhance) is a CPU-only
toolset that does not use it, and
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives them all.

[SCUNet](https://github.com/cszn/SCUNet) real-world image denoising - the
Swin-Conv-UNet that pairs a Swin transformer block with a convolutional block in
every stage - as a single self-contained binary. Feed it a noisy photograph, get
back the same photograph with the noise taken out. No Python, PyTorch, ONNX
Runtime or CUDA toolkit needed at runtime.

```
scunet -m scunet-color-real-psnr.safetensors -i noisy.png -o clean.png
```

![A 512x512 crop of a clean frame with synthetic noise beside the same crop
denoised by this engine and by PyTorch - all three at their measured
PSNR](docs/before-after.png)

* **Both backends in one executable**: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and the
  CPU path otherwise, so one binary covers a machine with no NVIDIA driver at
  all; `--device cpu|gpu` overrides that choice, and `--gpu` refuses to fall
  back.
* **No tile size to choose, and none to get wrong.** SCUNet pads by replication
  to a multiple of 64, and a convolution over a clamped edge reads exactly what
  the interior of a larger padded tensor would - so one pass over the whole
  image is exact at any size, and there is no `--tile` flag. A 2048x2048
  denoise takes about 8.5 s here, under 5.2 GB of device memory.
* **Smaller than PyTorch on the same card**: about 1.1-1.25x slower to run,
  holding about a third of the device memory PyTorch does. PyTorch cannot run a
  2048x2048 denoise on this 8 GB card at all.
* **Verified per op, not just per image.** Every CUDA kernel is compared against
  its CPU twin, the whole forward against a PyTorch transcription of the upstream
  network at 1.2e-06, and `tools/compare.py` closes the loop by running the
  upstream module itself under torch and diffing it stage by stage against what
  the engine produced - 16 named planes, worst 4.0e-05 on the 64x64 fixture. The
  comparison is against upstream, so a mistake made identically in both backends
  cannot hide.

## Download

Prebuilt binaries and the eight converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/scunet-rs/releases) - the checkpoints
sit next to the binaries as files, with no archive to unpack. Both binaries run
on the CPU; they differ only in whether CUDA support is compiled in.

| asset | contents | notes |
|---|---|---|
| `scunet-linux-x86_64` | CPU + CUDA, auto-selected | x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present. GPU path needs a compute capability 6.1+ GPU |
| `scunet-linux-x86_64-cpu-only` | CPU only | same, with nothing NVIDIA-related included - `--device gpu` is refused |
| 8 `scunet-*.safetensors` checkpoints | every published SCUNet model | 72 MB each; the name to pass to `-m` is in the table below |

```sh
chmod +x scunet-linux-x86_64
./scunet-linux-x86_64 -m scunet-color-real-psnr.safetensors -i noisy.png -o clean.png
```

The `chmod` is not decoration: a download does not carry the executable
bit through, and a binary that has lost it fails with `Permission denied`
before it can print anything.

## Build

```sh
cargo build --release
# CPU only, no CUDA toolkit or driver needed at build time either:
cargo build --release --no-default-features
```

The default build needs `nvcc` (set `NVCC=` if it is not on `PATH`) and produces
one binary with both backends. The `--no-default-features` build contains only
the CPU path, which reports `this build has no cuda feature; use --device cpu`
if asked for the GPU rather than failing obscurely. The kernels cover `sm_61`,
`sm_75`, `sm_80` and compute capability 8.0 PTX, so the GPU path runs on Pascal
(GTX 10-series) through Ampere, and on anything newer via the PTX.

`lightgpu` is a normal Cargo dependency on
[its repository](https://github.com/jacobsparts/lightgpu), so a clone of this
project builds on its own.

## Choosing a checkpoint

SCUNet publishes **eight** checkpoints, and this engine runs all of them:

| checkpoint | input | denoises |
|---|---|---|
| `scunet_color_real_psnr` | colour | real photographs, PSNR-oriented |
| `scunet_color_real_gan` | colour | real photographs, GAN-oriented (sharper, lower PSNR) |
| `scunet_color_15` / `_25` / `_50` | colour | Gaussian noise at sigma 15 / 25 / 50 |
| `scunet_gray_15` / `_25` / `_50` | grayscale | Gaussian noise at sigma 15 / 25 / 50 |

The releases carry all eight, converted, with the underscores in the checkpoint's
name replaced by dashes: `scunet_color_15` becomes
`scunet-color-15.safetensors`. So the table above is also the list of files to
hand to `-m`, and `scunet-color-real-psnr.safetensors` is the one to reach for on
a photograph from a camera.
The two `_real_` checkpoints are the ones the authors train blind: instead of a
fixed amount of Gaussian noise, a randomly shuffled degradation sequence - blur,
noise, resampling, compression - is applied to clean photographs, so the network
learns to undo whatever it is given rather than a known sigma. The sigma-numbered
models expect the noise they were named for, and a mismatch shows up as either
leftover grain or smeared detail. The grayscale models take a single-channel
input; this engine checks the checkpoint's `in_nc` against the image and refuses
the pair if they disagree, rather than guessing.

They are the SCUNet authors' work, released under
[KAIR](https://github.com/cszn/KAIR) - the official downloader fetches all
eight:

```sh
python main_download_pretrained_models.py --models "SCUNet" --model_dir model_zoo
```

Convert one to the `.safetensors` this engine loads - no torch needed, and the
architecture is read out of the weights rather than guessed from the file name:

```sh
python3 tools/convert.py scunet_color_real_psnr.pth scunet-color-real-psnr.safetensors
```

`tools/compare.py` is the other half of that: it runs the upstream module under
torch on a fixture, diffs it against `tools/reference.py`, and - with
`--dump <dir>` - against every named stage `examples/dump.rs` wrote, so a
divergence is localised rather than merely detected. It needs torch; the Rust
tests do not, which is why the numpy transcription above exists.

`tools/convert.py --help` lists the three flags it takes: `--variant` and
`--in-nc`, needed only for a file whose name is not one of the eight (the
architecture is otherwise read out of the weights), and `--dim`, which is 64 for
all eight.

The checkpoint is used **exactly as published**: no mean subtraction, no
standard deviation, no quantisation, no re-training. A pixel in [0, 1] in, a
pixel in [0, 1] out.

## Usage

```
scunet -m scunet-color-real-psnr.safetensors -i noisy.png -o clean.png
scunet -m model.safetensors -i in.png -o out.png --device cpu
scunet -m model.safetensors -i in.png -o out.png --gpu
```

```
-m, --model <path>    converted .safetensors checkpoint (see Choosing a checkpoint)
-i, --input <path>    input PNG, 8-bit, RGB or grayscale
-o, --output <path>   where to write the denoised PNG
    --device <dev>    gpu or cpu (default: gpu when this build has CUDA and a
                      driver, cpu otherwise; `cuda` is accepted for gpu)
    --cpu             same as --device cpu
    --gpu             same as --device gpu, and refuses to fall back
-q, --quiet           no progress output
-h, --help            this text
-V, --version         print the version
```

The output is a PNG the same size as the input. RGB is denoised as RGB; a
grayscale input goes through a grayscale checkpoint; an input with an alpha
channel has the alpha dropped, because the model has no fourth channel.

An image that will not fit is refused **before** it is allocated, with the
numbers: the size, what it needs and what is available. On the CPU path that
refusal is also what stands between a too-large image and an `abort()` - the CPU
path has no fallible allocation, so without the check the process would die with
a bare allocator message naming no image size and no remedy.

## Performance

Measured on the machine this was built on - an 8 GB GTX 1080, which is **shared
with other tenants and thermally throttled**, so the two engines are always
measured by alternating them inside one window and what is comparable is the
ratio. Milliseconds from different windows are not.

| size | PyTorch CUDA | scunet CUDA | PyTorch CPU (24 threads) | scunet CPU |
| ---- | ------------ | ----------- | ------------------------ | ---------- |
| 128  |              |             | 83-89 ms                 | 381-391 ms |
| 256  | 103-113 ms   | 129-130 ms  | 387-1834 ms              | 1.25-1.29 s |
| 512  | 411-441 ms   | 484-488 ms  | 6.0-19.9 s               | 5.06-5.16 s |
| 1024 | 1.68-1.69 s  | 1.89-1.90 s | 19.9-20.7 s              | 21.99 s    |
| 2048 | out of memory | **8.2-9.0 s** | | |

Device memory for a 1024x1024 denoise: **1132 MiB here against 3088 MiB for
PyTorch** - 2.7x less, and the reason PyTorch OOMs at 2048 where this engine
does not. Host memory for the whole process at 1024x1024, peak RSS: **1688 MiB
against PyTorch's 3713**.

On the GPU the engine runs at about 550-600 GFLOP/s, flat at about 2.0 us per
pixel from 64x64 up to 2048x2048; on the CPU at 52-58 GFLOP/s, and the CPU ratio
against PyTorch narrows with size (about 4x behind at 128x128, 1.07-1.13x at
1024x1024) because PyTorch is cache-resident at small sizes and memory-bound at
large ones - which is also why the CUDA ratio is the one to read if you have a
GPU.

[Engineering notes](docs/TECH_NOTES.md) has the full measurements, the budget of
where the device time goes, what each kernel change was worth, and how a
GPU concurrency bug was found and fixed.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE).

This is an independent implementation of the SCUNet architecture, which is by
[Kai Zhang](https://github.com/cszn) and Apache-2.0 licensed - the vendored
upstream it was written against is at
[cszn/SCUNet](https://github.com/cszn/SCUNet). `tools/network_scunet.py` is that
project's network module, kept for validation, and `tools/reference.py` is a
transcription of it; both are derived works and are not covered by this
repository's copyright.

The **checkpoints** are the SCUNet authors' work as well, released on the
[KAIR releases page](https://github.com/cszn/KAIR/releases/tag/v1.0) under the
same Apache-2.0 terms, and that is the licence the eight `.safetensors` files
attached to the releases are redistributed under. Each is a format conversion,
by `tools/convert.py`, of the official `.pth` of the same name in the table
above - the tensors are not modified, requantised or re-trained, and the header
that names the architecture is written from them rather than from the file name.
The original `.pth` files are not redistributed here.
