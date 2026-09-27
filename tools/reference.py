#!/usr/bin/env python3
"""A torch-free numpy reference for SCUNet, and the source of the test fixtures.

Why this file exists
--------------------

Every other engine in the family checks its CUDA and CPU backends against a
torch transcription of the upstream model. SCUNet is no different in principle -
`network_scunet.py` from https://github.com/cszn/SCUNet is the definition - but
this file is a *second*, independent reading of that definition that runs with
numpy alone, so the day-to-day parity loop does not need a torch install:

    python3 tools/reference.py --check                     # self-test vs the .pth
    python3 tools/reference.py --fixture ../tests/data/tiny

`tools/compare.py` remains the authority on torch agreement (it runs the upstream
module itself); this file is what the Rust tests can regenerate anywhere.

The arithmetic below is written to mirror `network_scunet.py` statement by
statement, because the whole point is to disagree with the Rust when the Rust is
wrong. Where an equivalent-looking formulation exists (im2col convolutions, a
batched softmax over a different axis, an einsum that reassociates the sum), the
formulation chosen is the one the reference uses, even when a faster one exists.

Conventions
-----------

* NCHW throughout, as every consumer engine in the family uses.
* A "stage" is a list of ConvTransBlocks, optionally followed by the strided
  convolution that halves the resolution (`m_down*`) or prefixed by the
  transposed convolution that doubles it (`m_up*`).
* The window shift is applied on the h/w axes of the WHOLE plane, before
  windowing, exactly as `torch.roll` does in the reference - not per window.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# safetensors (read side, for the fixtures and the self-test)
# ---------------------------------------------------------------------------


def read_safetensors(path):
    """{name: array} plus the `__metadata__` map."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    meta = header.pop("__metadata__", {})
    dtypes = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}
    out = {}
    for name, info in header.items():
        dt = dtypes.get(info["dtype"])
        if dt is None:
            raise SystemExit(f"{name}: unsupported dtype {info['dtype']}")
        a, b = info["data_offsets"]
        arr = np.frombuffer(blob[a:b], dtype=dt).reshape(info["shape"])
        out[name] = np.ascontiguousarray(arr, dtype=np.float32)
    return out, meta


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def conv2d(x, w, b=None, stride=1, pad=0):
    """NCHW convolution, the direct form.

    `w` is [c_out][c_in][kh][kw]. Accumulation order is ky, kx, ci - the order
    `lg_conv3x3s1p1` documents, so a mismatch is a numerics difference and not
    a reordering.
    """
    n, c_in, h, wd = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w, (x.shape, w.shape)
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    acc = np.zeros((n, c_out, oh, ow), dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            # [n][c_in][oh*ow] slice of the input at this tap
            patch = xp[:, :, ky:ky + oh * stride:stride, kx:kx + ow * stride:stride]
            # (c_in, c_out) x (n, c_in, oh*ow) -> (n, c_out, oh*ow)
            acc += np.einsum("oi,nip->nop", w[:, :, ky, kx],
                             patch.reshape(n, c_in, -1), optimize=True
                             ).reshape(n, c_out, oh, ow)
    if b is not None:
        acc += b.reshape(1, -1, 1, 1)
    return acc


def conv_transpose2x2(x, w):
    """ConvTranspose2d(2, stride=2), the up-sampling the reference uses.

    `w` is [c_in][c_out][2][2] - torch's ConvTranspose layout, which is the
    *reverse* of the Conv layout. Equivalent to scattering each input pixel's
    four taps, which is what `Op::ConvT2x2` does on the Rust side.
    """
    n, c_in, h, wd = x.shape
    c_in_w, c_out, kh, kw = w.shape
    assert c_in == c_in_w and (kh, kw) == (2, 2), (x.shape, w.shape)
    out = np.zeros((n, c_out, 2 * h, 2 * wd), dtype=np.float32)
    for dy in range(2):
        for dx in range(2):
            # (c_in, c_out) x (n, c_in, h*wd) -> (n, c_out, h*wd)
            out[:, :, dy::2, dx::2] = np.einsum(
                "io,nip->nop", w[:, :, dy, dx], x.reshape(n, c_in, -1),
                optimize=True).reshape(n, c_out, h, wd)
    return out


def layer_norm(x, w, b, eps=1e-5):
    """nn.LayerNorm over the LAST axis (the channel axis of [b][h][w][c])."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    """timm/torch's default GELU, the erf form (`nn.GELU()`)."""
    from math import sqrt
    return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))


def _erf(x):
    # math.erf on the array, skipping scipy. Slow but exact, and this file is a
    # reference rather than a fast path.
    v = np.vectorize(__import__("math").erf, otypes=[np.float32])
    return v(x.astype(np.float64)).astype(np.float32)


def rel_pos_table(params, window_size):
    """The [ws*ws][ws*ws] relative-position bias for one attention head.

    `params` is [2*ws-1][2*ws-1] for one head. The reference builds

        cord = [[i, j] for i in range(ws) for j in range(ws)]      # [ws*ws][2]
        relation = cord[:, None, :] - cord[None, :, :] + ws - 1    # [ws*ws][ws*ws][2]
        raw_rp[:, relation[..., 0], relation[..., 1]]              # [ws*ws][ws*ws]

    so entry (p, q) is `params[dy + ws - 1][dx + ws - 1]` with dy, dx the
    row/column difference between query-window-pixel p and key-window-pixel q.
    The window is ws x ws laid out row-major, so p = (py, px) and q = (qy, qx).
    """
    ws = window_size
    cord = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing="ij"), axis=-1)
    cord = cord.reshape(-1, 2)              # p -> (py, px), row-major
    rel = cord[:, None, :] - cord[None, :, :] + ws - 1
    return params[rel[..., 0], rel[..., 1]]  # [ws*ws][ws*ws]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class SCUNet:
    def __init__(self, weights, meta):
        self.w = weights
        self.in_nc = int(meta["in_nc"])
        self.dim = int(meta["dim"])
        self.config = [int(v) for v in meta["config"].split(",")]
        self.head_dim = int(meta["head_dim"])
        self.window_size = int(meta["window_size"])

    # -- one Swin window-attention block -----------------------------------
    def wmsa(self, x, prefix, shifted):
        """x: [b][h][w][c] (the reference's layout inside Block)."""
        ws = self.window_size
        b, h, w, c = x.shape
        if shifted:
            x = np.roll(np.roll(x, -ws // 2, axis=1), -ws // 2, axis=2)
        nw_h, nw_w = h // ws, w // ws
        # 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c'
        xt = x.reshape(b, nw_h, ws, nw_w, ws, c).transpose(0, 1, 3, 2, 4, 5)
        # -> 'b (w1 w2) (p1 p2) c'
        xt = xt.reshape(b, nw_h * nw_w, ws * ws, c)
        qkv = xt @ self.w[prefix + ".embedding_layer.weight"].T + \
            self.w[prefix + ".embedding_layer.bias"]
        n_heads = c // self.head_dim
        qkv = qkv.reshape(b, nw_h * nw_w, ws * ws, 3, n_heads, self.head_dim)
        # 'b nw np (threeh c) -> threeh b nw np h c', i.e. the three chunks are
        # split, then the head axis. The reference folds (threeh c) -> the reshape
        # above already made h and c separate, so only the three-way chunk and the
        # axis move are left.
        # 'b nw np (three h) c -> three h b nw np c'. The reference writes the
        # chunk and the head split as one rearrange ('threeh c'); doing the head
        # axis separately here keeps each step checkable against it.
        qkv = qkv.transpose(3, 0, 1, 2, 4, 5)      # (three, b, nw, np, h, hd)
        q, k, v = (qkv[i].transpose(3, 0, 1, 2, 4) for i in range(3))  # (h, b, nw, np, hd)
        scale = self.head_dim ** -0.5
        # q, k are (h, b, nw, np, hd). An explicit matmul over the last axis
        # rather than a labelled einsum: with h == 1 the labels of a labelled
        # einsum are ambiguous against the remaining axes and silently produce a
        # tensor whose nominal axis order is not the reference's.
        sim = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(scale)
        rp = self.w[prefix + ".relative_position_params"]  # [n_heads][2ws-1][2ws-1]
        # 'h p q -> h 1 1 p q'
        # [n_heads][ws*ws][ws*ws], broadcast as [h][1][1][p][q] against
        # sim's [h][b][nw][np][np].
        table = np.stack([rel_pos_table(rp[i], ws) for i in range(n_heads)])
        sim = sim + table[:, None, None, :, :]
        if shifted:
            sim = sim + self._mask(nw_h, nw_w, ws)
        # softmax over the last axis, in the same order the reference's
        # nn.functional.softmax(dim=-1) uses.
        m = sim.max(axis=-1, keepdims=True)
        e = np.exp(sim - m)
        probs = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        out = np.matmul(probs, v)      # (h, b, nw, np, hd)
        # 'h b w p c -> b w p (h c)'
        out = out.transpose(1, 2, 3, 0, 4).reshape(b, nw_h * nw_w, ws * ws, c)
        out = out @ self.w[prefix + ".linear.weight"].T + self.w[prefix + ".linear.bias"]
        # 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c'
        out = out.reshape(b, nw_h, nw_w, ws, ws, c).transpose(0, 1, 3, 2, 4, 5)
        out = out.reshape(b, nw_h * ws, nw_w * ws, c)
        if shifted:
            out = np.roll(np.roll(out, ws // 2, axis=1), ws // 2, axis=2)
        return out

    def _mask(self, nw_h, nw_w, ws):
        """The attention mask for a shifted window, as [1][1][nw][np][np].

        True becomes -inf in the reference (`sim.masked_fill_(attn_mask, -inf)`),
        which is why the -inf case is spelled out here rather than folded into the
        softmax: a fully masked row must not appear, and if one ever does the two
        implementations must produce the same NaN.
        """
        p = ws
        s = p - p // 2
        mask = np.zeros((nw_h, nw_w, p, p, p, p), dtype=bool)
        mask[-1, :, :s, :, s:, :] = True
        mask[-1, :, s:, :, :s, :] = True
        mask[:, -1, :, :s, :, s:] = True
        mask[:, -1, :, s:, :, :s] = True
        # 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)': a pure merge of
        # adjacent axes. Permuting p2 with p3 here instead would leave whole rows
        # masked and produce NaN through the softmax, which is exactly how this
        # port got it wrong once.
        m = mask.reshape(1, 1, nw_h * nw_w, p * p, p * p)
        return np.where(m, np.float32("-inf"), np.float32(0.0))

    # -- one transformer block (the "trans" half of a ConvTransBlock) -------
    def trans_block(self, x, prefix, shifted):
        # x arrives as [b][c][h][w]; the reference rearranges to [b][h][w][c].
        xt = x.transpose(0, 2, 3, 1)
        xt = xt + self.wmsa(layer_norm(xt, self.w[prefix + ".ln1.weight"],
                                       self.w[prefix + ".ln1.bias"]), prefix + ".msa", shifted)
        h = layer_norm(xt, self.w[prefix + ".ln2.weight"], self.w[prefix + ".ln2.bias"])
        h = h @ self.w[prefix + ".mlp.0.weight"].T + self.w[prefix + ".mlp.0.bias"]
        h = gelu(h)
        h = h @ self.w[prefix + ".mlp.2.weight"].T + self.w[prefix + ".mlp.2.bias"]
        xt = xt + h
        return xt.transpose(0, 3, 1, 2)

    # -- one ConvTransBlock -------------------------------------------------
    def conv_trans_block(self, x, prefix, shifted):
        conv_dim, trans_dim = self._dims(prefix)
        # torch.split(conv1_1(x), (conv_dim, trans_dim), dim=1): the split must be
        # materialised as contiguous arrays. A non-contiguous column slice that is
        # later reshaped (which the transformer does) silently reorders data, and
        # the reorder does not show up when the block is probed with a small
        # hand-made input.
        y = conv2d(x, self.w[prefix + ".conv1_1.weight"], self.w[prefix + ".conv1_1.bias"])
        conv_x = np.ascontiguousarray(y[:, :conv_dim])
        trans_x = np.ascontiguousarray(y[:, conv_dim:])
        # `conv_x = self.conv_block(conv_x) + conv_x` in network_scunet.py: the RESIDUAL
        # is the split's value, not the block's intermediate. Writing this as a chain of
        # rebindings (`conv_x = relu(...)` then `... + conv_x`) silently adds the ReLU's
        # output to itself instead - a 0.72 error on the block output, and the bug that
        # this reference carried until the Rust disagreed with the fixture and the
        # upstream source settled it.
        cx = conv_x
        h = conv2d(cx, self.w[prefix + ".conv_block.0.weight"], pad=1)
        h = np.maximum(h, 0.0)
        conv_x = conv2d(h, self.w[prefix + ".conv_block.2.weight"], pad=1) + cx
        trans_x = self.trans_block(trans_x, prefix + ".trans_block", shifted)
        res = conv2d(np.concatenate((conv_x, trans_x), axis=1),
                     self.w[prefix + ".conv1_2.weight"], self.w[prefix + ".conv1_2.bias"])
        return x + res

    def _dims(self, prefix):
        w = self.w[prefix + ".conv1_1.weight"]
        total, conv_dim = w.shape[0], w.shape[0] - self.w[prefix + ".trans_block.msa.linear.weight"].shape[0]
        # conv1_1 is (conv_dim+trans_dim) -> (conv_dim+trans_dim); trans_dim is
        # the input width of the transformer's final projection.
        trans_dim = self.w[prefix + ".trans_block.msa.linear.weight"].shape[0]
        return total - trans_dim, trans_dim

    # -- a whole stage: blocks, then optionally the strided conv ------------
    def stage(self, x, name, count, shifted_start, tail=None):
        # The reference builds each stage's `type` as `'W' if not i % 2 else 'SW'`
        # over the stage list, so block 0 of m_down1/m_down2/m_down3/m_body is an
        # UNSHIFTED window-attention block. `shifted_start` exists so the up
        # stages, whose list begins with the transposed convolution, can shift
        # the same pattern by one position.
        for i in range(count):
            shifted = bool(shifted_start ^ (i % 2))
            x = self.conv_trans_block(x, f"m_{name}.{i}", shifted)
        if tail is not None:
            x = conv2d(x, self.w[f"m_{name}.{count}.weight"], stride=2)
        return x

    def forward(self, x):
        """x: [n][c][h][w], float32 in [0, 1] before the padding."""
        n, c, h, w = x.shape
        pad_b = int(np.ceil(h / 64) * 64 - h)
        pad_r = int(np.ceil(w / 64) * 64 - w)
        # nn.ReplicationPad2d((0, right, 0, bottom))
        if pad_b or pad_r:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_b), (0, pad_r)), mode="edge")
        cfg = self.config

        x1 = conv2d(x, self.w["m_head.0.weight"], pad=1)
        x2 = self.stage(x1, "down1", cfg[0], False, tail=True)
        x3 = self.stage(x2, "down2", cfg[1], False, tail=True)
        x4 = self.stage(x3, "down3", cfg[2], False, tail=True)
        body = self.stage(x4, "body", cfg[3], False)
        # m_up3.0 is the ConvTranspose2d; the blocks follow at 1..count.
        y = conv_transpose2x2(body + x4, self.w["m_up3.0.weight"])
        y = self._up_blocks(y, "up3", cfg[4])
        y = conv_transpose2x2(y + x3, self.w["m_up2.0.weight"])
        y = self._up_blocks(y, "up2", cfg[5])
        y = conv_transpose2x2(y + x2, self.w["m_up1.0.weight"])
        y = self._up_blocks(y, "up1", cfg[6])
        y = conv2d(y + x1, self.w["m_tail.0.weight"], pad=1)
        return y[:, :, :h, :w]

    def _up_blocks(self, x, name, count):
        for i in range(count):
            # The reference builds each stage as [ConvTranspose2d] + [ConvTransBlock(...)
            # for i in range(count)] and computes the block's type from the
            # COMPREHENSION's index - the ConvTranspose2d is a separately prepended list
            # element and consumes no i. So the up stages' blocks run W, SW, W, SW
            # starting UNSHIFTED, exactly like the down stages; the earlier reading
            # ("the transpose occupies index 0, so the blocks invert the pattern") is
            # wrong and inverts the parity of every up-stage block. Settled against the
            # torch module: with the inverted pattern m_up3 was off by 9.77, with this
            # one the two agree to ~1e-6.
            x = self.conv_trans_block(x, f"m_{name}.{i + 1}", bool(i % 2))
        return x


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def load(weights_path):
    w, meta = read_safetensors(weights_path)
    return SCUNet(w, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../models/scunet-color-real-psnr.safetensors")
    ap.add_argument("--fixture", default=None, help="write a fixture directory here")
    ap.add_argument("--check", action="store_true", help="run the shape self-test")
    args = ap.parse_args()

    model = load(args.weights)
    print(f"model: in_nc {model.in_nc}, dim {model.dim}, config {model.config}, "
          f"window {model.window_size}, head_dim {model.head_dim}")
    if args.check:
        rng = np.random.default_rng(0)
        x = rng.random((1, model.in_nc, 32, 48), dtype=np.float32)
        y = model.forward(x)
        print(f"self-test: {x.shape} -> {y.shape}, "
              f"mean {float(y.mean()):.6f}, min {float(y.min()):.6f}, max {float(y.max()):.6f}")
    if args.fixture:
        os.makedirs(args.fixture, exist_ok=True)
        rng = np.random.default_rng(1234)
        # 80x64: not a multiple of 64 in either axis (h+48, w+0) and big enough
        # for a shifted window to straddle a border - the case the window mask
        # and the replication padding both exist for.
        x = rng.random((1, model.in_nc, 80, 64), dtype=np.float32).astype(np.float32)
        np.save(os.path.join(args.fixture, "input.npy"), x)
        y = model.forward(x)
        np.save(os.path.join(args.fixture, "output.npy"), y)
        # And a tiny one that exercises the shifted-window mask hard: 8x8 of
        # windows means the two shifted rows/columns are interior.
        x2 = rng.random((1, model.in_nc, 64, 64), dtype=np.float32).astype(np.float32)
        np.save(os.path.join(args.fixture, "input-64.npy"), x2)
        np.save(os.path.join(args.fixture, "output-64.npy"), model.forward(x2))
        print(f"fixture: {args.fixture}/input.npy, output.npy, input-64.npy, output-64.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
