#!/usr/bin/env python3
"""Compare the TORCH upstream module against tools/reference.py and against the engine.

Why this exists
---------------
`tools/reference.py` is a torch-free numpy transcription of the upstream network,
and it is what the Rust tests answer to. That makes the numpy file the authority -
so the one thing that must not stay implicit is why the numpy file is right. It is
right because it agrees with the upstream module itself, and this script is that
measurement: it runs `network_scunet.py` under torch, on the same input, and diffs
the two.

    python3 tools/compare.py                       # the 64x64 fixture, torch vs reference.py
    python3 tools/compare.py --input ../tests/data/color-real-psnr/input.npy
    python3 tools/compare.py --dump /tmp/rustdump  # ... and against the Rust engine

With `--dump` it also reads what `examples/dump.rs` wrote and diffs every named
stage of the engine against the same stage of the torch module:

    cargo run --release --example dump -- color_real_psnr_64.bin /tmp/rustdump
    python3 tools/compare.py --dump /tmp/rustdump

The stage names are the dump's own, and the mapping to torch is mechanical: a hook
on the upstream `m_<stage>` Sequential captures its output, and on the down stages a
hook on the trailing stride-2 convolution gives `m_<stage>_blocks`; on the up stages
a hook on the leading ConvTranspose2d gives `m_<stage>_up`; the input to `m_tail` is
`m_tail_in`. A plane the dump does not contain is simply not compared.

LAYOUT. The torch module is NCHW with a batch of 1, the engine (and the dump) is
`[c][h][w]`, batchless. Both are row-major over the same ordering, so the
comparison is element for element.

THE NUMBERS. An exact transcription lands at ~1e-6 - that is what
`tests/parity.rs` and `src/cpu.rs` quote - and a single wrong tap, a transposed
layout, an inverted shift pattern or the conv/trans halves swapped in the
concatenation shows up as >=1e-2. This script prints both ends of that range, and
exits non-zero if anything exceeds `--tol`.

Requires torch (and `thop`, which the vendored module imports at top level) in the
interpreter that runs it, plus both checkpoints - the converted `.safetensors` for
the architecture metadata and the upstream `.pth` for torch to load. Neither is in
the repository. `tools/reference.py` needs only numpy, which is why the day-to-day
parity loop does not go through here.
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reference  # noqa: E402  (the numpy transcription, in this directory)

# `network_scunet` is imported inside `build_torch` rather than here ON PURPOSE: it
# does `from thop import profile` at top level, so a module-level import makes even
# `--help` fail in an interpreter without torch's neighbourhood, with a traceback
# instead of the sentence that says what to install.

# The stages the dump names, in the order the walker reaches them.
PLANES = [
    "m_head",
    "m_down1_blocks", "m_down1",
    "m_down2_blocks", "m_down2",
    "m_down3_blocks", "m_down3",
    "m_body",
    "m_up3_up", "m_up3",
    "m_up2_up", "m_up2",
    "m_up1_up", "m_up1",
    "m_tail_in", "m_tail",
]
STAGES = ["m_head", "m_down1", "m_down2", "m_down3", "m_body",
          "m_up3", "m_up2", "m_up1", "m_tail"]


def pad_to(h, w, multiple=64):
    """The engine's and the upstream module's shared padding rule."""
    return -(-h // multiple) * multiple, -(-w // multiple) * multiple


def build_torch(pth, in_nc, dim, config, input_resolution):
    """The upstream module, weights loaded strictly, in eval mode.

    `input_resolution` IS NOT THE PADDED SIZE, and passing the padded size is a
    silent way to change the network. Both `ConvTransBlock` and `Block` do

        if self.input_resolution <= self.window_size:
            self.type = 'W'

    so the SHIFT PARITY of a stage flips when the stage's resolution collapses to
    one window: at 64x64 the body stage sits at 8x8, and `input_resolution // 8 == 8
    <= window_size 8` forces every body block to the unshifted type. The engine, and
    `tools/reference.py` behind it, always run the `W, SW, W, SW` pattern; upstream's
    own test scripts leave `input_resolution` at its 256 default for exactly this
    reason. Passing the padded size here produced a 4.4 disagreement at m_body that
    looked like an engine bug and is not one.

    The constructor prints one line per block ("Block Initial Type: ..."), which is
    upstream's own logging and 28 lines of it here, so it is captured and dropped.
    """
    try:
        import torch
        from network_scunet import SCUNet
    except ImportError as e:
        raise SystemExit(
            f"this script needs torch and thop (the vendored network_scunet.py imports "
            f"thop at top level): {e}\n"
            f"interpreter: {sys.executable}\n"
            f"tools/reference.py, which the Rust tests use, needs only numpy."
        )

    with contextlib.redirect_stdout(io.StringIO()):
        model = SCUNet(in_nc=in_nc, config=config, dim=dim, drop_path_rate=0.0,
                       input_resolution=input_resolution)
    sd = torch.load(pth, map_location="cpu")
    sd = sd.get("params", sd)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def capture(model):
    """Every named plane, by hooks on the module the walker names it after."""
    got = {}

    def pre(name):
        def hook(mod, args):
            got[name] = args[0].detach()
        return hook

    def post(name):
        def hook(mod, inp, out):
            got[name] = out.detach()
        return hook

    for s in STAGES:
        mod = getattr(model, s)
        mod.register_forward_pre_hook(pre("m_tail_in" if s == "m_tail" else s + "_in"))
        mod.register_forward_hook(post(s))
        if s.startswith("m_down"):
            # The trailing stride-2 convolution: its input is the stage's blocks.
            mod[-1].register_forward_pre_hook(pre(s + "_blocks"))
        elif s.startswith("m_up"):
            # The leading transposed convolution, which doubles the resolution.
            mod[0].register_forward_hook(post(s + "_up"))
    return got


def diff(a, b):
    """(max |a-b|, mean |a-b|, index of the worst) over a flattened pair."""
    d = np.abs(np.asarray(a, dtype=np.float64).ravel()
               - np.asarray(b, dtype=np.float64).ravel())
    if d.size == 0:
        return 0.0, 0.0, 0
    return float(d.max()), float(d.mean()), int(d.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default=str(HERE.parent / "../models/scunet_color_real_psnr.pth"),
                    help="the upstream checkpoint (not the converted one)")
    ap.add_argument("--weights",
                    default=str(HERE.parent / "../models/scunet-color-real-psnr.safetensors"),
                    help="the converted checkpoint, for the architecture metadata")
    ap.add_argument("--input", default=str(HERE.parent / "tests/data/color-real-psnr/input-64.npy"),
                    help="the input .npy, the same one the dump was made from")
    ap.add_argument("--dump", default=None, help="a directory written by examples/dump.rs")
    ap.add_argument("--input-resolution", type=int, default=256,
                    help="the upstream module's input_resolution, whose 256 default "
                         "its own test scripts rely on; see build_torch")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="fail above this (default 2e-3, the Rust tests' tolerance)")
    args = ap.parse_args()

    # Both checkpoints are needed and neither is in the repository: the converted
    # `.safetensors` carries the architecture metadata and the `.pth` is what torch
    # loads. A fresh clone has neither, so name the missing one instead of letting
    # the reader raise a bare FileNotFoundError.
    for label, path in (("converted checkpoint", args.weights), ("upstream checkpoint", args.pth)):
        if not Path(path).exists():
            raise SystemExit(
                f"missing {label}: {path}\n"
                f"the converted file comes from tools/convert.py (and the released "
                f"assets), the upstream .pth from the SCUNet downloader; see the "
                f"README's 'Choosing a checkpoint' section."
            )

    w, meta = reference.read_safetensors(args.weights)
    in_nc = int(meta["in_nc"])
    dim = int(meta["dim"])
    config = [int(v) for v in meta["config"].split(",")]

    x = np.load(args.input).astype(np.float32)
    h, w_px = x.shape[-2:]
    ph, pw = pad_to(h, w_px)

    print(f"input {args.input} {x.shape}, padded to {ph}x{pw}")
    print(f"checkpoint {args.pth}")
    print(f"architecture in_nc {in_nc}, dim {dim}, config {config} (from the container's metadata)")
    print(f"upstream input_resolution {args.input_resolution} "
          f"(its own default; the padded size {ph} would flip a stage's shift parity)")
    print()

    # The torch module, on the same input. Its own forward pads, so the input goes
    # in unpadded and the padded planes come out of the hooks.
    model = build_torch(args.pth, in_nc, dim, config, args.input_resolution)
    got = capture(model)
    import torch
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).numpy()
    print(f"torch {torch.__version__} ran the upstream module: {y_torch.shape}")

    # reference.py, the numpy transcription.
    ref = reference.SCUNet(w, meta)
    y_ref = ref.forward(x)
    mx, mean, at = diff(y_torch, y_ref)
    print(f"  upstream torch vs tools/reference.py, full model: "
          f"max |diff| {mx:.3g}, mean {mean:.3g} (worst at flat index {at})")
    print()

    if args.dump:
        d = Path(args.dump)
        files = {p.stem: p for p in sorted(d.glob("*.bin"))}
        print(f"{'plane':<16} {'floats':>10}  {'max |diff|':>11} {'mean':>10}  verdict")
        worst = (0.0, "")
        for name in PLANES:
            if name not in got or name not in files:
                continue
            plane = np.fromfile(files[name], dtype="<f4")
            t = got[name].numpy().ravel()
            if plane.size != t.size:
                print(f"{name:<16} {plane.size:>10}  shape mismatch: the dump is "
                      f"{plane.size} floats, torch {t.size} - was it made at {ph}x{pw}?")
                continue
            mx, mean, _ = diff(plane, t)
            verdict = "ok" if mx <= args.tol else "OVER TOLERANCE"
            if mx > worst[0]:
                worst = (mx, name)
            print(f"{name:<16} {plane.size:>10}  {mx:>11.3g} {mean:>10.3g}  {verdict}")
        print()
        print(f"worst stage: {worst[1]} at {worst[0]:.3g} (tolerance {args.tol:g})")
        if worst[0] > args.tol:
            print("FAIL: the engine and the upstream module disagree by more than the "
                  "tolerance on a named stage")
            return 1
        print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
