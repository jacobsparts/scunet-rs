#!/usr/bin/env python3
"""Time the PyTorch reference for SCUNet, on CPU and CUDA, at the sizes the engine
is measured at. The counterpart of `examples/bench.rs`, and the source of the
"vs PyTorch" column in the README.

Why this file exists: `tools/reference.py` is a torch-FREE transcription used for
parity, so it can say whether the engine is right but not how the upstream model
performs. This runs the upstream `network_scunet.py` itself.

    /home/jacob/torchenv311/bin/python tools/bench_torch.py \
        --model ../models/scunet_color_real_psnr.pth --sizes 256,512,1024 \
        --iters 5 --warmup 3

What is measured, and why it is the same thing the Rust bench measures:

* `SCUNet.forward` end to end, which already includes the ReplicationPad2d to a
  multiple of 64 and the crop - the identical I/O contract as `backend::run`.
* The input is the golden-ratio plane `raw_input` below generates, which is the
  same sequence `examples/bench.rs`'s `seq` builds - so both sides run on
  numerically identical data, value for value.
* `input_resolution` is part of the model's construction (the WMSA shift and its
  relative-position table are built from it), so the model is rebuilt per size -
  exactly as the upstream test scripts do for each image.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_scunet import SCUNet  # the vendored copy in this directory  # noqa: E402


def raw_input(h, w, c):
    """The golden-ratio fractional sequence, [c][h][w] - the same plane
    `examples/bench.rs` uses, so a comparison is on identical numbers."""
    n = c * h * w
    i = np.arange(n, dtype=np.float64)
    t = i * 0.6180339887498949
    return (t - np.floor(t)).astype(np.float32).reshape(c, h, w)


def build(ckpt, in_nc, dim, config, resolution):
    model = SCUNet(in_nc=in_nc, config=config, dim=dim, drop_path_rate=0.0,
                   input_resolution=resolution)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("params", sd)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def time_run(model, x, device, iters, warmup):
    """Median/min seconds per forward, and the peak memory the run held."""
    x = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            y = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            y = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    peak = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else float("nan")
    return float(ts.min()), float(np.median(ts)), peak, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/scunet_color_real_psnr.pth")
    ap.add_argument("--sizes", default="256,512,1024")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--in-nc", type=int, default=3)
    ap.add_argument("--config", default="4,4,4,4,4,4,4")
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    config = [int(v) for v in args.config.split(",")]
    sizes = [int(v) for v in args.sizes.split(",")]
    devices = args.devices.split(",")
    print(f"torch {torch.__version__} cuda {torch.version.cuda}, threads {args.threads}")
    print(f"{'size':>6} {'device':>6} {'min ms':>10} {'med ms':>10} {'MP/s':>8} {'peak MiB':>9}")

    for h in sizes:
        for dev in devices:
            if dev == "cuda" and not torch.cuda.is_available():
                print(f"{h:>6} {'cuda':>6}   (no device)")
                continue
            model = build(args.model, args.in_nc, args.dim, config, h)
            model = model.to(dev)
            x = torch.from_numpy(raw_input(h, h, args.in_nc)).unsqueeze(0)
            lo, med, peak, _ = time_run(model, x, dev, args.iters, args.warmup)
            mp = (h * h) / (med * 1e3) / 1e3
            print(f"{h:>6} {dev:>6} {lo * 1e3:>10.1f} {med * 1e3:>10.1f} {mp:>8.2f} {peak:>9.1f}")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
