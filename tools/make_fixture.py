#!/usr/bin/env python3
"""Pack a reference (.npy in, .npy out) pair into the SCUF fixture container.

The reference is `tools/reference.py`, which needs only numpy - so the fixtures for
this engine are reproducible on any machine, without torch. The container is read by
`src/fixture.rs`; the layout is

    magic   "SCUF"                  4 bytes
    version u32                     1
    h, w, c, win                    u32 each
    vlen    u32                     length of the variant string
    variant utf-8                   vlen bytes
    input   f32[h*w*c]              NCHW, the values the network sees (already in [0,1])
    output  f32[h*w*c]              what the reference produced, before any cropping

Usage:
    python3 tools/make_fixture.py --dir tests/data/color-real-psnr \
        --variant color-real-psnr --out tests/data/color_real_psnr.bin
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding input*.npy/output*.npy")
    ap.add_argument("--variant", required=True, help="the checkpoint variant the reference used")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default="", help="input STEM.npy / output STEM.npy (default: no suffix)")
    args = ap.parse_args()

    d = Path(args.dir)
    ix = d / f"input{args.stem}.npy"
    ox = d / f"output{args.stem}.npy"
    x = np.load(ix).astype(np.float32)
    y = np.load(ox).astype(np.float32)
    if x.ndim != 4 or x.shape[0] != 1:
        raise SystemExit(f"{ix}: expected [1][c][h][w], got {x.shape}")
    if x.shape != y.shape:
        raise SystemExit(f"{ix} is {x.shape}, {ox} is {y.shape}")
    _, c, h, w = x.shape
    v = args.variant.encode("utf-8")
    blob = bytearray()
    blob += b"SCUF"
    blob += struct.pack("<IIIIII", 1, h, w, c, args.window, len(v))
    blob += v
    blob += np.ascontiguousarray(x).tobytes()
    blob += np.ascontiguousarray(y).tobytes()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    print(f"{out}: {h}x{w}x{c} {args.variant}, {len(blob)} bytes ({len(blob)/1024/1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
