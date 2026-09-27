#!/usr/bin/env python3
"""Convert an SCUNet .pth checkpoint into the .safetensors this engine loads.

SCUNet is Kai Zhang's Swin-Conv-UNet (https://github.com/cszn/SCUNet, Apache-2.0);
the released weights live on the KAIR releases page
(https://github.com/cszn/KAIR/releases/tag/v1.0) and are the SCUNet authors'
work, covered by that project's licence, not by scunet-rs's.

    python3 tools/convert.py scunet_color_real_psnr.pth scunet-color-real-psnr.safetensors

No torch. The `.pth` is a zip whose `data.pkl` is a pickled `state_dict`; the
tensors themselves are separate members under `archive/data/<storage key>`. Both
halves are read directly here, for the same reason the other engines in the
family convert without a second toolchain: the conversion is a byte copy plus a
header, and torch would be a large dependency to do arithmetic on shapes.

Two things the file does NOT let us copy blindly:

* **Tensors are not always contiguous.** `WMSA.__init__` rebuilds
  `relative_position_params` with `view(...).transpose(1,2).transpose(0,1)`, so
  the saved tensors have a non-contiguous stride: `m_body.0`'s is shape
  `[8, 15, 15]` over a storage laid out `[p][q][head]`. Copying the storage would
  silently transpose the table. The reader below walks the stride.

* **The stride is also where the shape comes from.** Every tensor records the
  C-order shape it should be written as; the converter checks that the shape is
  exactly a permutation of the storage and writes the permuted copy.

`--arch` writes the architecture constants into `__metadata__` the way
nafnet-rs's converter does, because the Rust side reads them back rather than
guessing from the file name: there are eight published SCUNet checkpoints and a
converted file that does not say which one it is cannot be shape-checked.
"""
import argparse
import collections
import json
import os
import pickle
import re
import struct
import sys
import zipfile

import numpy as np

# The eight published checkpoints, from main_download_pretrained_models.py. All
# use config [4,4,4,4,4,4,4] and dim 64; `in_nc` is 1 for the grayscale pair
# (the -15/-25/-50 noise levels exist in both) and 3 for the colour ones.
KNOWN = {
    "scunet_gray_15": (1, "gray-15"),
    "scunet_gray_25": (1, "gray-25"),
    "scunet_gray_50": (1, "gray-50"),
    "scunet_color_15": (3, "color-15"),
    "scunet_color_25": (3, "color-25"),
    "scunet_color_50": (3, "color-50"),
    "scunet_color_real_psnr": (3, "color-real-psnr"),
    "scunet_color_real_gan": (3, "color-real-gan"),
}

DTYPES = {"FloatStorage": ("F32", "<f4", 4), "HalfStorage": ("F16", "<f2", 2),
          "DoubleStorage": ("F64", "<f8", 8), "LongStorage": ("I64", "<i8", 8),
          "IntStorage": ("I32", "<i4", 4), "ByteStorage": ("U8", "u1", 1),
          "BoolStorage": ("BOOL", "?", 1)}


class _Unpickler(pickle.Unpickler):
    """A pickler for a torch state_dict that never imports torch.

    `_rebuild_tensor_v2(storage, offset, size, stride, ...)` is reduced to the
    four things a converter needs, and the storage itself - fetched through
    `persistent_load` - to its (key, dtype, numel).
    """

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return self._rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return lambda *a, **k: ("storage_cls", name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unexpected global {module}.{name}")

    @staticmethod
    def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
        return {"storage": storage, "offset": int(storage_offset),
                "size": tuple(int(s) for s in size),
                "stride": tuple(int(s) for s in stride)}

    def persistent_load(self, pid):
        # ('storage', StorageClass, key, location, numel)
        if not (isinstance(pid, tuple) and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        _, cls, key, _location, numel = pid
        # `cls` is whatever find_class returned for the storage class: an
        # instance of the `("storage_cls", "<Name>Storage")` lambda's result when
        # the pickler resolves it through the class path, or the callable itself
        # when it does not. Normalise rather than trust one of the two.
        if isinstance(cls, tuple):
            _, cls = cls
        if callable(cls):
            name = getattr(cls, "__name__", None) or repr(cls)
            if name == "<lambda>":
                name = getattr(cls, "_lg_name", "FloatStorage")
            cls = name
        return ("storage_ref", str(cls), str(key), int(numel))


def load_state_dict(path):
    with zipfile.ZipFile(path) as z:
        members = z.namelist()
        pkl = next((m for m in members if m.endswith("data.pkl")), None)
        if pkl is None:
            raise SystemExit(f"{path}: no data.pkl member - not a saved state_dict?")
        prefix = os.path.dirname(pkl)
        with z.open(pkl) as f:
            state = _Unpickler(f).load()
        storages = {}
        for m in members:
            if m.startswith(prefix + "/data/"):
                storages[os.path.basename(m)] = z.read(m)
    if not isinstance(state, dict):
        raise SystemExit(f"{path}: data.pkl did not hold a dict")
    return state, storages


def materialise(entry, storages):
    """Return (array, canonical_name) for one state_dict entry.

    The saved tensor may be a view with an arbitrary stride; numpy's
    `as_strided` + `copy` walks it, and the C-order copy is what gets written.
    """
    key = entry["storage"][2]
    dtype = entry["storage"][1]
    raw = storages.get(key)
    if raw is None:
        raise SystemExit(f"storage {key} is missing from the archive")
    if dtype not in DTYPES:
        raise SystemExit(f"unsupported storage dtype {dtype}")
    st_name, st_code, st_size = DTYPES[dtype]
    n = len(raw) // st_size
    st = np.frombuffer(raw, dtype=st_code, count=n)
    offset, shape, stride = entry["offset"], entry["size"], entry["stride"]
    if stride == () and shape == ():
        return np.array(st[offset], dtype=st_code), st_name
    # numpy strides are in bytes; the saved strides are in elements.
    view = np.lib.stride_tricks.as_strided(
        st[offset:], shape=shape,
        strides=tuple(s * st_size for s in stride), writeable=False)
    return np.ascontiguousarray(view), st_name


def shape_of(state, key):
    return tuple(state[key]["size"])


def arch_from_weights(state):
    """Infer (in_nc, config, dim) from the weights themselves.

    `config[i]` is the number of ConvTransBlocks before the stage's strided
    conv, and each block is `m_<stage>.<i>`, so the largest index + 1 is the
    count. This is the same inference the other engines' converters do, and it
    makes a converted file's metadata a statement about the tensors actually in
    it rather than about its file name.
    """
    def blocks(stage, first_block):
        # The up stages put their ConvTranspose2d at index 0, so their first
        # ConvTransBlock is at 1; counting max(index) + 1 would call that 5 blocks.
        idx = [int(k.split(".")[1]) for k in state
               if k.startswith(f"m_{stage}.") and k.count(".") >= 2
               and k.split(".")[1].isdigit()
               and "trans_block" in k]
        return (max(idx) + 1 - first_block) if idx else 0

    cfg = [blocks(s, 1 if s.startswith("up") else 0)
           for s in ("down1", "down2", "down3", "body", "up3", "up2", "up1")]
    in_nc = shape_of(state, "m_head.0.weight")[0] and shape_of(state, "m_head.0.weight")[1]
    dim = shape_of(state, "m_head.0.weight")[0]
    return in_nc, cfg, dim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--variant", default=None,
                    help="gray-15, gray-25, gray-50, color-15, color-25, color-50, "
                         "color-real-psnr, color-real-gan (default: from the file name)")
    ap.add_argument("--in-nc", type=int, default=None, help="1 or 3 (default: from the weights)")
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()

    state, storages = load_state_dict(args.src)
    state = {k: v for k, v in state.items()
             if isinstance(v, dict) and "storage" in v}
    if "m_head.0.weight" not in state:
        raise SystemExit(f"{args.src}: no m_head.0.weight - not an SCUNet checkpoint?")

    in_nc_w, cfg, dim_w = arch_from_weights(state)
    in_nc = args.in_nc or in_nc_w
    dim = args.dim or dim_w
    if in_nc not in (1, 3):
        raise SystemExit(f"{args.src}: m_head has {in_nc} input channels; SCUNet is 1 or 3")

    variant = args.variant
    if variant is None:
        base = re.sub(r"\.pth$", "", os.path.basename(args.src))
        for known, (kc, kv) in KNOWN.items():
            if base == known:
                variant = kv
                if kc != in_nc:
                    print(f"warning: {base} is published with in_nc={kc}, "
                          f"but the weights have {in_nc}", file=sys.stderr)
                break
        else:
            variant = f"in{in_nc}-unknown"
    else:
        # A named variant is a claim about the weights; check it.
        for _known, (kc, kv) in KNOWN.items():
            if kv == variant and kc != in_nc:
                print(f"warning: variant {variant} is published with in_nc={kc}, "
                      f"but the weights have {in_nc}", file=sys.stderr)

    if cfg != [4, 4, 4, 4, 4, 4, 4]:
        print(f"warning: the weights are config={cfg}; SCUNet publishes [4,4,4,4,4,4,4]",
              file=sys.stderr)

    metadata = {
        "format": "pt",
        "arch": "scunet",
        "variant": variant,
        "in_nc": str(in_nc),
        "dim": str(dim),
        "config": ",".join(str(v) for v in cfg),
        "head_dim": "32",
        "window_size": "8",
    }

    tensors = []
    for name, entry in state.items():
        arr, st_name = materialise(entry, storages)
        tensors.append((name, st_name, arr))
    tensors.sort(key=lambda kv: kv[0])

    offset = 0
    header = {}
    for name, st_name, arr in tensors:
        nbytes = arr.size * arr.dtype.itemsize
        header[name] = {"dtype": st_name, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += (nbytes + 7) & ~7  # 8-byte alignment for the f32/bf16 paths
    header["__metadata__"] = metadata
    hjson = json.dumps(header, separators=(",", ":")).encode()

    header_bytes = 8 + len(hjson)
    pad = (-header_bytes) & 7
    with open(args.dst, "wb") as f:
        f.write(struct.pack("<Q", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        written = 0
        for name, _st, arr in tensors:
            want = header[name]["data_offsets"][0]
            if written < want:
                f.write(b"\0" * (want - written))
                written = want
            raw = arr.tobytes()
            f.write(raw)
            written += len(raw)

    total = sum(arr.size for _, _, arr in tensors)
    nblocks = sum(cfg)
    print(f"{args.dst}: {len(tensors)} tensors, {total} values "
          f"({total / 1e6:.2f} M), variant {variant}, in_nc {in_nc}, dim {dim}, "
          f"config {cfg} ({nblocks} ConvTransBlocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
