# scunet

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

SCUNet real-world image denoising in one self-contained binary: feed it a noisy
photograph, get back the same photograph with the noise taken out. No Python,
PyTorch, ONNX Runtime or CUDA toolkit needed.

```
scunet -m scunet-color-real-psnr.safetensors -i noisy.png -o clean.png
```

![A 512x512 crop of a clean frame with synthetic noise beside the same crop
denoised by this engine and by PyTorch - all three at their measured
PSNR](docs/before-after.png)

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when a CUDA driver is available and the
  CPU path otherwise, so one binary covers a machine with no NVIDIA driver at
  all; `--device cpu|gpu` overrides that choice, and `--gpu` refuses to fall
  back.
* No tile size to choose: SCUNet pads by replication to a multiple of 64, and a
  convolution over a clamped edge reads exactly what the interior of a larger
  padded tensor would - so one pass over the whole image is exact at any size.

Both backends reproduce the upstream PyTorch implementation's output to within
float32 rounding precision (worst stage difference 4.0e-05 on the 64x64 fixture).

## Download

Prebuilt binary and the eight converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/scunet-rs/releases).

| asset | what it is |
|---|---|
| `scunet-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); falls back to the CPU path when no NVIDIA driver is present, the GPU path needs a compute capability 6.1+ GPU |
| 8 `scunet-*.safetensors` checkpoints | every published SCUNet model, 72 MB each; see Models |

```sh
chmod +x scunet-linux-x86_64
./scunet-linux-x86_64 -m scunet-color-real-psnr.safetensors -i noisy.png -o clean.png
```

## Models

SCUNet publishes eight checkpoints, and this engine runs all of them. `scunet-color-real-psnr.safetensors` is the one to reach for on a photograph from a camera.

| checkpoint | input | denoises |
|---|---|---|
| `scunet_color_real_psnr` | colour | real photographs, PSNR-oriented |
| `scunet_color_real_gan` | colour | real photographs, GAN-oriented (sharper, lower PSNR) |
| `scunet_color_15` / `_25` / `_50` | colour | Gaussian noise at sigma 15 / 25 / 50 |
| `scunet_gray_15` / `_25` / `_50` | grayscale | Gaussian noise at sigma 15 / 25 / 50 |

The releases carry all eight, with the underscores in the checkpoint's name
replaced by dashes: `scunet_color_15` becomes `scunet-color-15.safetensors`. The
two `_real_` checkpoints are the blind ones, trained on a shuffled degradation
sequence rather than a fixed sigma; the sigma-numbered models expect the noise
they were named for, and a mismatch shows up as leftover grain or smeared detail.
The grayscale models take a single-channel input; the engine checks the
checkpoint's `in_nc` against the image and refuses the pair if they disagree.

Convert any SCUNet `.pth` to the `.safetensors` this engine loads - no torch
needed, and the architecture is read out of the weights rather than guessed from
the file name:

```sh
python3 tools/convert.py scunet_color_real_psnr.pth scunet-color-real-psnr.safetensors
```

## Usage

```
scunet -m scunet-color-real-psnr.safetensors -i noisy.png -o clean.png
scunet -m model.safetensors -i in.png -o out.png --device cpu
scunet -m model.safetensors -i in.png -o out.png --gpu
```

```
-m, --model <path>    converted .safetensors checkpoint
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
channel has the alpha dropped, because the model has no fourth channel. An image
that will not fit is refused before it is allocated, with the numbers.

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE). This is an independent implementation of the SCUNet
architecture, which is by [Kai Zhang](https://github.com/cszn) and Apache-2.0
licensed - the vendored upstream is at
[cszn/SCUNet](https://github.com/cszn/SCUNet). The **checkpoints** are the SCUNet
authors' work, released on the
[KAIR releases page](https://github.com/cszn/KAIR/releases/tag/v1.0) under the
same Apache-2.0 terms, and that is the licence the eight `.safetensors` files
attached to the releases are redistributed under.
