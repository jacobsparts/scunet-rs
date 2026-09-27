//! The CPU backend: one function per op, plain Rust, no device.
//!
//! This is the path for a machine with no usable GPU, and it is held to the same
//! standard as the device one: correct against the validated reference, and as fast
//! as this machine can be made to run it. It is NOT the reference - the reference is
//! `tools/reference.py`. That reference is itself checked against the upstream
//! `network_scunet.py` under torch - end to end to 1.2e-06 on the 64x64 fixture and
//! 1.6e-06 on the 80x64 one - because a check that only compared the two backends to
//! each other would pass just as happily if both were wrong the same way.
//!
//! NCHW IS THE CANONICAL LAYOUT. An activation is `[c][h][w]`. The one exception is
//! inside window attention, which reads and writes `[window][token][c]`; the
//! gather/scatter pair below is the only place that mapping appears.
//!
//! MEMORY. The forward's peak is `6*dim + in_nc` floats per padded pixel -
//! `src/memguard.rs` models it, and `examples/cpu_mem.rs` measures it. Every buffer
//! that would grow with the IMAGE rather than with a stage is chunked: the attention
//! window index at `TOKEN_BUDGET` windows and the MLP's hidden layer at
//! `MLP_CHUNK_BYTES` bytes.
//!
//! THE SHAPE OF THE NETWORK, once, because every function below is a line of it:
//!
//!   x1 = m_head(replicate_pad(x))            3x3, in_nc -> dim
//!   x2 = down1(x1)                          4 ConvTransBlocks + 2x2 stride-2 conv
//!   x3 = down2(x2)                          ... at dim, 2*dim, 4*dim, 8*dim
//!   x4 = down3(x3)
//!   b  = body(x4)                           4 ConvTransBlocks, no downsample
//!   y  = up3(b + x4)                        ConvTranspose2d + 4 ConvTransBlocks
//!   y  = up2(y + x3)                        ... the skip is added BEFORE the up-conv
//!   y  = up1(y + x2)
//!   out = m_tail(y + x1)  [..., :h, :w]     3x3, dim -> in_nc, then crop
//!
//! A ConvTransBlock is: 1x1 expand, split, [3x3 - ReLU - 3x3] + residual on the conv
//! half, a Swin `Block` on the transformer half, 1x1 project, and a residual add
//! around the whole thing. The two halves are concatenated in that order, `conv_x`
//! first - swapping them is a wrong image that still looks like an image.
//!
//! PARALLELISM. Every op is split across the machine's cores with rayon, and the
//! split is ALWAYS OVER INDEPENDENT OUTPUT ELEMENTS: each output is accumulated by
//! one thread in the same order the sequential version used, so the parallel result
//! is bit-identical to the sequential one and the fixture tolerance is unaffected.
//! Splitting a reduction across threads would change the summation order, which is a
//! different answer this engine cannot afford - the tolerance in `tests/parity.rs`
//! is already only three orders of magnitude wide.
use rayon::prelude::*;

use crate::plan::Plan;
use crate::weights::Weights;

const LN_EPS: f32 = 1e-5;

/// Elements below which an op stays on one thread: a `par_chunks` over three
/// elements costs more in wakeups than it saves, and the graph is full of small ops
/// (the norms, the 1x1s at small sizes).
const PAR_MIN: usize = 1 << 14;

#[inline]
fn par_on(n: usize) -> bool {
    n >= PAR_MIN && rayon::current_num_threads() > 1
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

/// `[c_in][h][w]` -> `[c_out][h][w]`, 3x3, pad 1, bias-free.
///
/// The pad is zero, as `nn.Conv2d(..., 3, 1, 1)` implies - the module passes no
/// `padding_mode`, so it is zeros and NOT the replication the input padding uses.
/// These are different paddings in the same network and conflating them is one of
/// the two mistakes that produce a plausible-looking wrong image.
pub fn conv3x3(
    x: &[f32], c_in: usize, c_out: usize, h: usize, w: usize, wt: &[f32], out: &mut [f32],
) {
    debug_assert_eq!(wt.len(), c_out * c_in * 9);
    debug_assert_eq!(x.len(), c_in * h * w);
    debug_assert_eq!(out.len(), c_out * h * w);
    // WHY THIS IS SHAPED THE WAY IT IS. The obvious formulation walks the output
    // pixels and, for each one, loops over the input channels with nine taps inside.
    // The channel loop strides by `h * w` so nothing vectorises, and the output pixel
    // is the innermost thing touched, so the whole output plane is read and written
    // once per input channel - about 2 GB of traffic for this one op at 256x256, or
    // ~105 ms at this machine's memory bandwidth. The budget measured it at 49
    // GFLOP/s across 24 cores, and it is the largest single op in the CPU forward.
    //
    // Here the channel loop is inside an output ROW, and each channel's nine taps
    // are applied as nine contiguous row operations. The innermost statement is a
    // slice zip, which is a vector instruction, and the accumulator row is a few KB
    // that stays in L1 while all `c_in` planes stream past it.
    //
    // THE ACCUMULATION ORDER IS UNCHANGED, which is what makes this safe to do at
    // all: for one output pixel the terms are still added input-channel ascending and
    // within a channel in ky-then-kx order, because there is one accumulator per
    // output pixel either way and only the iteration order over INDEPENDENT pixels
    // differs. The zero-padding taps at a plane's edge need no condition: the tap is
    // simply skipped, and a skipped tap contributes exactly what `at` returned, 0.0.
    // (The one value where "skipped" and "added zero" differ is a signed zero, which
    // needs every tap of a pixel to be zero; see the note in the module docs.)
    let run = |co: usize, row: &mut [f32]| {
        for yy in 0..h {
            let dst = &mut row[yy * w..(yy + 1) * w];
            dst.fill(0.0);
            for ci in 0..c_in {
                let p = &x[ci * h * w..(ci + 1) * h * w];
                let k = &wt[(co * c_in + ci) * 9..(co * c_in + ci) * 9 + 9];
                for ky in 0..3usize {
                    let sy = yy as isize + ky as isize - 1;
                    if sy < 0 || sy >= h as isize {
                        continue;                       // the padded row: no contribution
                    }
                    let s = &p[sy as usize * w..(sy as usize + 1) * w];
                    for kx in 0..3usize {
                        let kk = k[ky * 3 + kx];
                        // The output columns whose source cell is inside the plane.
                        // For kx = 1 that is every column; for 0 and 2 it is all but
                        // one, which is the padding.
                        let x0 = (1isize - kx as isize).max(0) as usize;
                        let x1 = ((w as isize) + 1 - kx as isize).min(w as isize) as usize;
                        let src = &s[x0 + kx - 1..x1 + kx - 1];
                        for (a, b) in dst[x0..x1].iter_mut().zip(src.iter()) {
                            *a += kk * b;
                        }
                    }
                }
            }
        }
    };
    if par_on(c_out * h * w) {
        out.par_chunks_mut(h * w)
            .enumerate()
            .for_each(|(co, row)| run(co, row));
    } else {
        for (co, row) in out.chunks_mut(h * w).enumerate() {
            run(co, row);
        }
    }
}

/// `[c_in][h][w]` -> `[c_out][h][w]` with a 2x2 kernel, stride 2, no padding:
/// the downsampler that follows each of the three down stages.
pub fn conv2x2s2(
    x: &[f32], c_in: usize, c_out: usize, h: usize, w: usize, wt: &[f32], out: &mut [f32],
) {
    let (oh, ow) = (h / 2, w / 2);
    debug_assert_eq!(wt.len(), c_out * c_in * 4);
    debug_assert_eq!(out.len(), c_out * oh * ow);
    let run = |co: usize, row: &mut [f32]| {
        for y in 0..oh {
            for xx in 0..ow {
                let mut acc = 0.0f32;
                for ci in 0..c_in {
                    let p = &x[ci * h * w..(ci + 1) * h * w];
                    let k = &wt[(co * c_in + ci) * 4..(co * c_in + ci) * 4 + 4];
                    acc += k[0] * p[(2 * y) * w + 2 * xx];
                    acc += k[1] * p[(2 * y) * w + 2 * xx + 1];
                    acc += k[2] * p[(2 * y + 1) * w + 2 * xx];
                    acc += k[3] * p[(2 * y + 1) * w + 2 * xx + 1];
                }
                row[y * ow + xx] = acc;
            }
        }
    };
    if par_on(c_out * oh * ow) {
        out.par_chunks_mut(oh * ow).enumerate().for_each(|(co, row)| run(co, row));
    } else {
        for (co, row) in out.chunks_mut(oh * ow).enumerate() {
            run(co, row);
        }
    }
}

/// 1x1 convolution, which is a matmul over channels at every pixel.
///
/// `wt` is `[c_out][c_in]`, `bias` may be empty.
pub fn conv1x1(
    x: &[f32], c_in: usize, c_out: usize, n: usize, wt: &[f32], bias: &[f32], out: &mut [f32],
) {
    debug_assert_eq!(wt.len(), c_out * c_in);
    debug_assert_eq!(out.len(), c_out * n);
    // THE INNER LOOP IS OVER COLUMNS, NOT CHANNELS, and that is the whole point.
    // Written the obvious way round the channel loop strides by `n`, so the compiler
    // cannot vectorise it and every one of the `c_in` terms is a separate scalar load:
    // the CPU budget measured this op at 38 GFLOP/s over 24 cores, where the machine's
    // AVX2 peak is several hundred. Turning the loop inside out makes the innermost
    // body a contiguous zip of two slices, which is a vector instruction.
    //
    // IT IS BIT-IDENTICAL, not merely close. For one output pixel the terms are
    // still added in exactly the order the serial loop adds them - the bias first, then
    // `k[0] * x[0][i]`, `k[1] * x[1][i]`, ... by ascending input channel - because an
    // accumulator is per output pixel either way and only the ITERATION ORDER over
    // independent accumulators has changed. Blocked over columns so the working set
    // stays in L1 rather than a whole 256 KB plane.
    let run = |co: usize, row: &mut [f32]| {
        let k = &wt[co * c_in..(co + 1) * c_in];
        let b = if bias.is_empty() { 0.0 } else { bias[co] };
        let mut jb = 0usize;
        while jb < n {
            let je = (jb + CONV1X1_BLK).min(n);
            let dst = &mut row[jb..je];
            dst.fill(b);
            for (ci, &kk) in k.iter().enumerate() {
                let src = &x[ci * n + jb..ci * n + je];
                for (a, s) in dst.iter_mut().zip(src.iter()) {
                    *a += kk * s;
                }
            }
            jb = je;
        }
    };
    if par_on(c_out * n) {
        out.par_chunks_mut(n).enumerate().for_each(|(co, row)| run(co, row));
    } else {
        for (co, row) in out.chunks_mut(n).enumerate() {
            run(co, row);
        }
    }
}

/// `ConvTranspose2d(2, stride=2)`: the up-sampler before each up stage's blocks.
///
/// `wt` is `[c_in][c_out][2][2]` - torch's transposed-convolution layout, the REVERSE
/// of the conv layout, and the reason this is its own function rather than a flag on
/// `conv3x3`. The operation is a scatter with NO TAP FLIP:
///
/// ```text
/// out[2i + dy][2j + dx] += sum_ci wt[ci][co][dy][dx] * x[ci][i][j]
/// ```
///
/// which is what `torch.nn.functional.conv_transpose2d` computes and what
/// `tools/reference.py`'s `conv_transpose2x2` computes. Flipping the tap indices
/// gives a mirrored up-sample: an image, but not this one.
pub fn conv_t2x2(
    x: &[f32], c_in: usize, c_out: usize, h: usize, w: usize, wt: &[f32], out: &mut [f32],
) {
    let (oh, ow) = (2 * h, 2 * w);
    debug_assert_eq!(wt.len(), c_in * c_out * 4);
    debug_assert_eq!(out.len(), c_out * oh * ow);
    out.fill(0.0);
    // Over output channels: each accumulates over its own c_in slice, so the split is
    // still over independent outputs and the order of the sum is unchanged.
    let run = |co: usize, row: &mut [f32]| {
        for ci in 0..c_in {
            let p = &x[ci * h * w..(ci + 1) * h * w];
            let k = &wt[(ci * c_out + co) * 4..(ci * c_out + co) * 4 + 4];
            for y in 0..h {
                for xx in 0..w {
                    let v = p[y * w + xx];
                    row[(2 * y) * ow + 2 * xx] += k[0] * v;
                    row[(2 * y) * ow + 2 * xx + 1] += k[1] * v;
                    row[(2 * y + 1) * ow + 2 * xx] += k[2] * v;
                    row[(2 * y + 1) * ow + 2 * xx + 1] += k[3] * v;
                }
            }
        }
    };
    if par_on(c_out * oh * ow) {
        out.par_chunks_mut(oh * ow).enumerate().for_each(|(co, row)| run(co, row));
    } else {
        for (co, row) in out.chunks_mut(oh * ow).enumerate() {
            run(co, row);
        }
    }
}

// ---------------------------------------------------------------------------
// Elementwise and normalisation
// ---------------------------------------------------------------------------

/// `x += y`, elementwise.
#[inline]
pub fn add_into(x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    if par_on(y.len()) {
        y.par_iter_mut().zip(x.par_iter()).for_each(|(a, b)| *a += *b);
    } else {
        for (a, b) in y.iter_mut().zip(x.iter()) {
            *a += *b;
        }
    }
}

/// In-place ReLU.
#[inline]
pub fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

/// `nn.GELU()` (the exact erf form) in place. `lightgpu::ops::cpu` has the same op
/// and the device kernel is checked against it; this is a separate spelling only
/// because the CPU path must not depend on the device crate's feature set.
pub fn gelu_erf_inplace(x: &mut [f32]) {
    // ELEMENT-WISE, so chunking it cannot change a result - each output depends on
    // its own input alone. It is the MLP's activation over `4 * c` channels a token,
    // and it needs the same parallel split as `relu`: single-threaded it is a 150x
    // outlier against that twin, and the third-largest op in the whole graph. The
    // `erf` itself is the same f64 series and the same cast the reference uses,
    // because that is what the parity test agrees on.
    let run = |chunk: &mut [f32]| {
        for v in chunk.iter_mut() {
            let t = *v as f64;
            *v = (0.5 * t * (1.0 + erf(t / std::f64::consts::SQRT_2))) as f32;
        }
    };
    if par_on(x.len()) {
        x.par_chunks_mut(GELU_CHUNK).for_each(run);
    } else {
        run(x);
    }
}

/// erf to double precision, as `libm`/glibc's, from Abramowitz & Stegun 7.1.26's
/// rational approximation refined by one Newton step - which is what torch's CPU
/// kernel effectively does at f32 and what the reference's `math.erf` does at f64.
/// `lightgpu::ops::cpu::gelu_erf` is the toolkit's twin of the CUDA kernel; the
/// parity test compares the two so they cannot drift.
pub fn erf(x: f64) -> f64 {
    // A&S 7.1.26, |error| < 1.5e-7, then one Newton iteration on the ODE erf' = 2/sqrt(pi) e^-x^2
    // is not valid for the inverse, so instead use the better 7.1.28 series.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// LayerNorm over the LAST axis of `[n][c]`: the op `nn.LayerNorm(dim)` applies inside
/// a Swin block, where the tensor is in `[h][w][c]` order and the normalised axis is
/// the channel one. The window attention below therefore works on a `[tokens][c]`
/// copy - the same tensor, transposed.
pub fn layer_norm(
    x: &[f32], c: usize, wgt: &[f32], bias: &[f32], out: &mut [f32],
) {
    debug_assert_eq!(x.len() % c, 0);
    let run = |(dst, src): (&mut [f32], &[f32])| {
        let mu = src.iter().sum::<f32>() / c as f32;
        let var = src.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for i in 0..c {
            dst[i] = (src[i] - mu) * inv * wgt[i] + bias[i];
        }
    };
    if par_on(x.len()) {
        out.par_chunks_mut(c).zip(x.par_chunks(c)).for_each(|p| run(p));
    } else {
        for p in out.chunks_mut(c).zip(x.chunks(c)) {
            run(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Window attention
// ---------------------------------------------------------------------------

/// One attention window's edge length, `win`, and `win*win` tokens.
///
/// THE MASK IS ADDED, NOT EXPONENTIATED, and it is `-FLT_MAX/4` rather than `-inf`:
/// the toolkit's convention (cuda/CONVENTIONS.md) is that an infinite logit breaks an
/// online-softmax rescale, and the reference's own `masked_fill_(-inf)` would produce
/// a NaN the moment a row were fully masked - which it never is in this mask, but a
/// finite sentinel is what makes the two implementations comparable at all.
const MASK_SENTINEL: f32 = -f32::MAX / 4.0;

/// Elements a GELU chunk hands to one thread. Large enough that the cost of the
/// `erf` dominates the split, small enough to fill 24 cores evenly.
const GELU_CHUNK: usize = 1 << 14;

/// Bytes of MLP hidden buffer one token chunk may occupy. The MLP is applied to one
/// token at a time, so the chunk boundary can fall anywhere; this only decides how
/// large the `4c`-wide buffer is - see `trans_block_dump`.
const MLP_CHUNK_BYTES: usize = 8 << 20;

/// Windows one `attention` chunk stages at a time, in TOKENS - the same budget and
/// the same reasoning as `cuda.rs`'s constant of that name, so the two backends
/// chunk the image at the same point.
///
/// WHY IT EXISTS. `attention` computes a window's tokens into a staging buffer and
/// scatters them afterwards, and that buffer is `nw * n * c` floats for EVERY window
/// in the plane. It is one of only two buffers in this file that grow with the IMAGE
/// rather than with a stage - 128 MiB at 1024x1024 for the first stage alone, against
/// 4 MiB chunked - and the CUDA walker has chunked the same index since the 960x960
/// OOM. 32768 tokens is 512 windows at SCUNet's 8x8, and the chunk loop scatters each
/// chunk before starting the next, so the buffer's size is a constant of the model.
const TOKEN_BUDGET: usize = 32768;

/// Destination rows one thread transposes into `[h][w][c]`. The work per row is one
/// move an element, so this wants to be small: 64 rows is 64 * c moves, and at the
/// smallest stage's `c` that is still far above the cost of the split.
const TRANSPOSE_ROWS: usize = 64;

/// `[c][n]` -> `[n][c]`, where `n = h * w`: the layout change the transformer half
/// needs, because it normalises and attends over the CHANNEL axis.
///
/// PUBLIC SO THAT IT IS MEASURED WHERE IT RUNS. This is the loop the engine calls,
/// so `examples/budget_cpu.rs` times this and not a copy of it - an instrument that
/// reimplements the thing it measures is measuring the wrong program.
///
/// PARALLEL OVER DESTINATION ROW BLOCKS, so every thread's writes are contiguous
/// while its reads stride across the `c` planes. The other way round - splitting over
/// channels, as the original serial loop's index order suggests - puts every thread's
/// destination addresses `c` apart, which is the classic slow transpose. Each
/// destination cell is a single move, so this is element-wise and cannot change a
/// result.
pub fn to_hwc(src: &[f32], c: usize, n: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), c * n);
    debug_assert_eq!(dst.len(), n * c);
    let run_t = |blk: &mut [f32], base: usize| {
        for (li, row) in blk.chunks_mut(c).enumerate() {
            let i = base + li;
            for ci in 0..c {
                row[ci] = src[ci * n + i];
            }
        }
    };
    if par_on(n * c) {
        let cw = TRANSPOSE_ROWS.max(1) * c;
        dst.par_chunks_mut(cw)
            .enumerate()
            .for_each(|(bi, blk)| run_t(blk, bi * TRANSPOSE_ROWS.max(1)));
    } else {
        run_t(dst, 0);
    }
}

/// `[n][c]` -> `[c][n]`, the inverse of `to_hwc`.
///
/// PARALLEL OVER OUTPUT CHANNELS, which is the mirror of the forward split: the
/// destination is `[c][n]` here, so one thread owning one channel writes `n`
/// contiguous elements. Element-wise, so identical to the serial loop it replaces.
pub fn from_hwc(src: &[f32], c: usize, n: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), n * c);
    debug_assert_eq!(dst.len(), c * n);
    let run_b = |ci: usize, row: &mut [f32]| {
        for i in 0..n {
            row[i] = src[i * c + ci];
        }
    };
    if par_on(n * c) {
        dst.par_chunks_mut(n).enumerate().for_each(|(ci, row)| run_b(ci, row));
    } else {
        for (ci, row) in dst.chunks_mut(n).enumerate() {
            run_b(ci, row);
        }
    }
}

/// Columns one `conv1x1` block accumulates at a time. The accumulator IS the output
/// row, so the block size is what keeps that row - read and written once per input
/// channel - in the L1: 1024 floats is 4 KB, and it has to be read and written `c_in`
/// times because the channel index cannot be reordered.
const CONV1X1_BLK: usize = 1024;

/// The inverse of `gather`: `[nw][n][c]` tokens back into `[c][h][w]`, applying the
/// same shift modulo, so the two are exact inverses at the same `shift`.
///
/// THE PARALLEL SPLIT IS OVER CHANNELS, and this is the one op where that is the
/// natural axis: every output element is written by exactly one (window, token)
/// pair, so no two threads touch the same address and the written value does not
/// depend on the order they run in.
pub fn scatter(
    tok: &[f32], plane: usize, c: usize, hp: usize, wp: usize, win: usize, shift: usize, out: &mut [f32],
) {
    let n = win * win;
    let nww = wp / win;
    let nw = (hp / win) * nww;
    debug_assert_eq!(tok.len(), nw * n * c);
    debug_assert_eq!(out.len(), c * plane);
    let run = |ch: usize, dst_plane: &mut [f32]| {
        for wi in 0..nw {
            let (wh, ww) = (wi / nww, wi % nww);
            let toks = &tok[wi * n * c..(wi + 1) * n * c];
            for t in 0..n {
                let (i, j) = (t / win, t % win);
                let y = (wh * win + i + shift) % hp;
                let xx = (ww * win + j + shift) % wp;
                dst_plane[y * wp + xx] = toks[t * c + ch];
            }
        }
    };
    if par_on(c * plane) {
        out.par_chunks_mut(plane).enumerate().for_each(|(ch, d)| run(ch, d));
    } else {
        for (ch, d) in out.chunks_mut(plane).enumerate() {
            run(ch, d);
        }
    }
}


// ---------------------------------------------------------------------------
// Window attention, in the reference's [h][w][c] layout
// ---------------------------------------------------------------------------
//
// The transformer half of a block never leaves [h][w][c]: that is the reference's
// layout inside `Block`, and staying in it removes every permute the reference does
// not perform. Windowing is an index computation - token (i, j) of window (wh, ww) is
// plane pixel (wh*win + i, ww*win + j) - not a gather into a token buffer.
//
// The block is PRE-NORM, as `Block.forward` is: `x = x + msa(ln1(x))`, then
// `x = x + mlp(ln2(x))`. The sibling Swin2SR engine's blocks are POST-NORM; getting
// the direction backwards yields a plausible image that is wrong everywhere, which is
// what `tests/parity.rs` exists to catch.

/// `sum(a[i] * b[i])`, the accumulation order the reference's matmuls use.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// The relative-position bias table for one head, `[n][n]` with `n = win*win`.
///
/// ```text
/// entry (p, q) = params[dy + win - 1][dx + win - 1]
/// ```
///
/// with `dy, dx` the row/column difference between query pixel p and key pixel q in
/// the window's row-major layout. Building it from a FLATTENED cord vector instead of
/// the 2-D `(i, j)` pairs is the mistake that looks harmless - the table still has the
/// right shape - and shifts every bias by one row.
pub fn rel_pos_table(params: &[f32], win: usize, out: &mut [f32]) {
    let n = win * win;
    let span = 2 * win - 1;
    debug_assert_eq!(params.len(), span * span);
    debug_assert_eq!(out.len(), n * n);
    for p in 0..n {
        let (py, px) = (p / win, p % win);
        for q in 0..n {
            let (qy, qx) = (q / win, q % win);
            let dy = py + win - 1 - qy;
            let dx = px + win - 1 - qx;
            out[p * n + q] = params[dy * span + dx];
        }
    }
}

/// Whether the shifted-window mask forbids query token `p` from attending to key
/// token `t`, for the window at grid position `(wh, ww)`.
///
/// The reference's `generate_mask` written as four conditions instead of four boolean
/// assignments on a 6-D array. EACH EDGE COMPARES THE SAME AXIS ON BOTH SIDES: on the
/// last window ROW, a query row in the first `s` rows may not see a key row in the last
/// `s` and vice versa; on the last window COLUMN, the same for columns, with
/// `s = win - win/2`. The tempting alternative - one shared axis pair for both edges -
/// has the right density and is wrong in 29696 of 31744 cells of a 64x64 plane.
#[inline]
fn mask_hit(p: usize, t: usize, wh: usize, ww: usize, nwh: usize, nww: usize, win: usize) -> bool {
    let s = win - win / 2;
    // The reference's generate_mask is four assignments on a 6-D tensor
    // [w1][w2][p1][p2][p3][p4], with (p1, p2) the QUERY token as (row, column) and
    // (p3, p4) the KEY token as (row, column):
    //
    //     attn_mask[-1, :, :s, :, s:, :] = True      (A)
    //     attn_mask[-1, :, s:, :, :s, :] = True      (B)
    //     attn_mask[:, -1, :, :s, :, s:] = True      (C)
    //     attn_mask[:, -1, :, s:, :, :s] = True      (D)
    //
    // (A)|(B) slice p1 and p3: the last window ROW compares the query's row with the
    // key's ROW. (C)|(D) slice p1 and p4: the last window COLUMN compares the query's
    // column with the key's COLUMN. Checked cell by cell against the reference's own
    // tensor over a 3x3 grid of windows (36864 triples): the ROW edge matches
    // "query row vs key ROW" exactly and the COLUMN edge matches "query COLUMN vs key
    // COLUMN" exactly - each edge compares the SAME axis on both sides. The plausible
    // reading, "the two edges share one axis pair", is off in 29696 of 31744 cells on
    // a 64x64 plane: the right density in the wrong places.
    let pr = p / win;
    let pc = p % win;
    let tr = t / win;
    let tc = t % win;
    let row_edge = wh == nwh - 1;
    let col_edge = ww == nww - 1;
    let row = (pr < s && tr >= s) || (pr >= s && tr < s);
    let col = (pc < s && tc >= s) || (pc >= s && tc < s);
    (row_edge && row) || (col_edge && col)
}

/// One block's window self-attention over `[h][w][c]`, writing into `out`.
///
/// `shift` is `win/2` for a shifted block and `0` for an unshifted one; both the roll
/// in and the roll out are index computations with a modulo, exactly
/// `torch.roll(x, (-s, -s))` and `torch.roll(o, (s, s))`.
///
/// `emb_w` is `[3c][c]` - the q rows first, then k, then v, as `qkv.chunk(3, dim=0)`
/// implies - `lin_w` is `[c][c]`, and `rp` is `[heads][2w-1][2w-1]`.
///
/// THE SPLIT IS OVER WINDOWS, and each window owns its scratch, so the parallel and
/// serial paths agree bit for bit. The qkv embedding and the projection are the
/// expensive parts of this op and both are inside the per-window work, which is why
/// the split is here rather than over the channels.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    x: &[f32], h: usize, w: usize, c: usize, win: usize, shift: usize,
    emb_w: &[f32], emb_b: &[f32], lin_w: &[f32], lin_b: &[f32], rp: &[f32], out: &mut [f32],
) {
    let n = win * win;
    let nwh = h / win;
    let nww = w / win;
    let span = 2 * win - 1;
    let heads = rp.len() / (span * span);
    let hd = c / heads;
    let scale = (hd as f32).powf(-0.5);
    let shifted = shift != 0;
    debug_assert_eq!(x.len(), h * w * c);
    debug_assert_eq!(out.len(), h * w * c);
    debug_assert_eq!(emb_w.len(), 3 * c * c);
    debug_assert_eq!(lin_w.len(), c * c);
    debug_assert_eq!(rp.len(), heads * span * span);
    debug_assert_eq!(heads * hd, c);

    /// Per-window scratch, owned by one worker at a time.
    ///
    /// `pix` IS IN HERE rather than a fresh `Vec` per window. It is the window's `n`
    /// source pixel indices, and at 1024x1024 a block has 16384 windows and a forward
    /// has 28 blocks: one `Vec` per window is 458k allocations of 512 bytes each, which
    /// is most of this path's allocation count and all of its churn for nothing - the
    /// buffer is written fresh each window, so there was never anything to preserve.
    struct Win {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        sim: Vec<f32>,
        rel: Vec<f32>,
        o: Vec<f32>,
        pix: Vec<usize>,
    }
    let nw = nwh * nww;
    // THE WINDOW INDEX IS CHUNKED, which is what the CUDA walker does with the same
    // number. `stage` holds every window's tokens at once (`nw * n * c` floats), so at
    // 1024x1024 the first stage's is 128 MiB, and it is one of the two buffers in this
    // file that grow with the IMAGE rather than with a stage. A chunk of `TOKEN_BUDGET`
    // windows makes it 4 MiB and constant. The split costs nothing arithmetically:
    // attention is per window, and every window scatters to pixels of its own.
    let nw_blk = (TOKEN_BUDGET / n).clamp(1, nw);
    let mut stage = vec![0.0f32; nw_blk * n * c];
    let run = |wi: usize, win_buf: &mut Win, dst: &mut [f32]| {
        let (wh, ww) = (wi / nww, wi % nww);
        // The plane pixel index token t of this window reads, after the roll. Written
        // into the worker's own buffer - see `Win::pix` above.
        for (t, p) in win_buf.pix.iter_mut().enumerate() {
            let (i, j) = (t / win, t % win);
            *p = ((wh * win + i + shift) % h) * w + (ww * win + j + shift) % w;
        }
        let pix = &win_buf.pix;
        for head in 0..heads {
            let base = head * hd;
            for t in 0..n {
                let xv = &x[pix[t] * c..pix[t] * c + c];
                for r in 0..hd {
                    win_buf.q[t * hd + r] =
                        dot(&emb_w[(base + r) * c..(base + r + 1) * c], xv) + emb_b[base + r];
                    win_buf.k[t * hd + r] = dot(&emb_w[(c + base + r) * c..(c + base + r + 1) * c], xv)
                        + emb_b[c + base + r];
                    win_buf.v[t * hd + r] =
                        dot(&emb_w[(2 * c + base + r) * c..(2 * c + base + r + 1) * c], xv)
                            + emb_b[2 * c + base + r];
                }
            }
            rel_pos_table(
                &rp[head * span * span..(head + 1) * span * span],
                win,
                &mut win_buf.rel,
            );
            for p in 0..n {
                for t in 0..n {
                    let mut s = dot(
                        &win_buf.q[p * hd..(p + 1) * hd],
                        &win_buf.k[t * hd..(t + 1) * hd],
                    ) * scale;
                    s += win_buf.rel[p * n + t];
                    if shifted && mask_hit(p, t, wh, ww, nwh, nww, win) {
                        s = MASK_SENTINEL;
                    }
                    win_buf.sim[p * n + t] = s;
                }
                // Softmax over the key axis: subtract the row max, exponentiate,
                // divide by the row sum - the reference's `softmax(dim=-1)` order.
                let row = &mut win_buf.sim[p * n..(p + 1) * n];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                for v in row.iter_mut() {
                    *v *= inv;
                }
                for r in 0..hd {
                    let mut acc = 0.0f32;
                    for t in 0..n {
                        acc += win_buf.sim[p * n + t] * win_buf.v[t * hd + r];
                    }
                    win_buf.o[p * c + base + r] = acc;
                }
            }
        }
        // The output projection for this window's tokens.
        for p in 0..n {
            let tok = &win_buf.o[p * c..(p + 1) * c];
            let row = &mut dst[p * c..(p + 1) * c];
            for r in 0..c {
                row[r] = dot(&lin_w[r * c..(r + 1) * c], tok) + lin_b[r];
            }
        }
    };
    let fresh = || Win {
        q: vec![0.0; n * hd],
        k: vec![0.0; n * hd],
        v: vec![0.0; n * hd],
        sim: vec![0.0; n * n],
        rel: vec![0.0; n * n],
        o: vec![0.0; n * c],
        pix: vec![0; n],
    };
    // ONE CHUNK OF THE WINDOW INDEX AT A TIME, scattered as soon as the chunk that
    // produced it is done. A window's destination pixels are its own and no two windows
    // share one, so the chunk boundary can fall anywhere and the result is the same
    // arithmetic in the same order.
    for base in (0..nw).step_by(nw_blk) {
        let nch = nw_blk.min(nw - base);
        let view = &mut stage[..nch * n * c];
        if par_on(nch * n * c) {
            // Each window owns its slice of `stage` and its own scratch, so the result
            // is a pure function of the window index and does not depend on the thread
            // count. `base + j` rather than `j`: the chunk is an offset, not a restart.
            view.par_chunks_mut(n * c)
                .enumerate()
                .for_each_init(fresh, |b, (j, chunk)| run(base + j, b, chunk));
        } else {
            let mut b = fresh();
            for j in 0..nch {
                run(base + j, &mut b, &mut view[j * n * c..(j + 1) * n * c]);
            }
        }
        // Scatter this chunk's tokens to the rolled plane position.
        for j in 0..nch {
            let wi = base + j;
            let (wh, ww) = (wi / nww, wi % nww);
            for t in 0..n {
                let (i, jj) = (t / win, t % win);
                let dst = ((wh * win + i + shift) % h) * w + (ww * win + jj + shift) % w;
                out[dst * c..dst * c + c]
                    .copy_from_slice(&view[j * n * c + t * c..j * n * c + (t + 1) * c]);
            }
        }
    }
}

/// The transformer half of a `ConvTransBlock`: pre-norm window attention then a
/// pre-norm 2-layer MLP with GELU, both with residual adds.
///
/// The reference keeps `[h][w][c]` throughout `Block`, and so does this - the caller
/// transposes at the `ConvTransBlock` boundary, which is the only place the two
/// layouts meet.
#[allow(clippy::too_many_arguments)]
pub fn trans_block(
    x: &[f32], h: usize, w: usize, c: usize, win: usize, shift: usize,
    ln1_w: &[f32], ln1_b: &[f32], emb_w: &[f32], emb_b: &[f32], lin_w: &[f32], lin_b: &[f32],
    rp: &[f32], ln2_w: &[f32], ln2_b: &[f32], mlp0_w: &[f32], mlp0_b: &[f32], mlp2_w: &[f32],
    mlp2_b: &[f32], out: &mut [f32], scratch: &mut Vec<f32>,
) {
    trans_block_dump(
        x, h, w, c, win, shift, ln1_w, ln1_b, emb_w, emb_b, lin_w, lin_b, rp, ln2_w, ln2_b,
        mlp0_w, mlp0_b, mlp2_w, mlp2_b, out, scratch, None,
    )
}

/// `trans_block` with an optional report of the attention output (`[h][w][c]`, the
/// reference's layout), for the block-interior differ.
#[allow(clippy::too_many_arguments)]
pub fn trans_block_dump(
    x: &[f32], h: usize, w: usize, c: usize, win: usize, shift: usize,
    ln1_w: &[f32], ln1_b: &[f32], emb_w: &[f32], emb_b: &[f32], lin_w: &[f32], lin_b: &[f32],
    rp: &[f32], ln2_w: &[f32], ln2_b: &[f32], mlp0_w: &[f32], mlp0_b: &[f32], mlp2_w: &[f32],
    mlp2_b: &[f32], out: &mut [f32], scratch: &mut Vec<f32>, mut msa: Option<&mut Vec<f32>>,
) {
    let n = h * w;
    scratch.resize(c * n * 2, 0.0);
    let (norm, att) = scratch.split_at_mut(c * n);
    // x = x + msa(ln1(x))
    layer_norm(x, c, ln1_w, ln1_b, norm);
    attention(norm, h, w, c, win, shift, emb_w, emb_b, lin_w, lin_b, rp, att);
    if let Some(m) = msa.as_deref_mut() {
        m.clear();
        m.extend_from_slice(att);
    }
    out.copy_from_slice(x);
    add_into(att, out);
    // x = x + mlp(ln2(x))
    layer_norm(out, c, ln2_w, ln2_b, norm);
    {
        // mlp0: [4c][c] over every token, GELU, mlp2: [c][4c].
        //
        // THE TOKEN AXIS IS CHUNKED, because `hidden` is the largest single buffer this
        // path allocates: `n * 4c` floats, which at 1024x1024 is 512 MiB for the first
        // stage - four times the plane it came from, since `4c` is four floats per
        // token per channel. It is the analogue of the CUDA walker's `TOKEN_BUDGET`
        // chunk, and it is exact for the same reason attention's is: the MLP is
        // applied to one token at a time - `mlp0` is `[4c][c]` over that token's `c`
        // channels and `mlp2` is `[c][4c]` back - so a token's output depends on that
        // token alone and the chunk boundary can fall anywhere. The chunk also keeps
        // the working set near L2, which is a second reason it is not a loss.
        let ntok = n;
        let four = mlp0_w.len() / c;
        // Sized in BYTES rather than in tokens, because `hidden` is four floats per
        // token per channel and the same token count is therefore a very different
        // buffer at each stage's width. 8 MiB is the cap: large enough that a chunk
        // carries real work for the parallel split, small enough to be a constant of
        // the model rather than of the image.
        let tblk = (MLP_CHUNK_BYTES / (four.max(1) * 4)).max(1).min(ntok.max(1));
        let mut hidden = vec![0.0f32; tblk * four];
        for base in (0..ntok).step_by(tblk) {
            let nch = tblk.min(ntok - base);
            // `hb`, not `h`: `h` is the plane HEIGHT at this point in the signature,
            // and shadowing it with the hidden slice reads as if the height had changed.
            let hb = &mut hidden[..nch * four];
            let run = |t: usize, dst: &mut [f32]| {
                let src = &norm[t * c..(t + 1) * c];
                for r in 0..four {
                    dst[r] = dot(&mlp0_w[r * c..(r + 1) * c], src) + mlp0_b[r];
                }
            };
            if par_on(nch * four) {
                hb.par_chunks_mut(four)
                    .enumerate()
                    .for_each(|(j, d)| run(base + j, d));
            } else {
                for (j, d) in hb.chunks_mut(four).enumerate() {
                    run(base + j, d);
                }
            }
            gelu_erf_inplace(hb);
            let run2 = |t: usize, dst: &mut [f32]| {
                let src = &hb[t * four..(t + 1) * four];
                for r in 0..c {
                    dst[r] = dot(&mlp2_w[r * four..(r + 1) * four], src) + mlp2_b[r];
                }
            };
            let dst = &mut att[base * c..(base + nch) * c];
            if par_on(nch * c) {
                dst.par_chunks_mut(c).enumerate().for_each(|(j, d)| run2(j, d));
            } else {
                for (j, d) in dst.chunks_mut(c).enumerate() {
                    run2(j, d);
                }
            }
        }
    }
    add_into(att, out);
}

/// A `ConvTransBlock` up to but not including its residual add: 1x1 expand, split, a
/// 3x3 residual pair on the conv half, the transformer block on the other half, and
/// the 1x1 project - the result left in `scratch.proj`.
///
/// The two halves are concatenated `conv_x` FIRST, as `torch.cat((conv_x, trans_x))`
/// does; swapping them is a wrong image that still looks like an image.
fn block_project(
    x: &[f32], dims: (usize, usize), h: usize, w: usize, win: usize, shift: usize,
    wt: &Weights, prefix: &str, scratch: &mut BlockScratch,
) {
    let (conv_dim, trans_dim) = dims;
    let c = conv_dim + trans_dim;
    let n = h * w;
    scratch.resize(c, h, w, trans_dim);

    // 1x1 expand: c -> c.
    conv1x1(
        x, c, c, n,
        wt.t(&format!("{prefix}.conv1_1.weight")),
        wt.t(&format!("{prefix}.conv1_1.bias")),
        &mut scratch.y,
    );

    // Conv half: 3x3 - ReLU - 3x3, then the residual, WRITTEN BACK into the first half
    // of `scratch.y`. The projection below reads `y`, and it must see the conv half's
    // OUTPUT: leaving the result in a side buffer (`c2`) and projecting `y` would feed
    // the 1x1 expand's raw output to `conv1_2`, which is a wrong image that still looks
    // like an image.
    {
        // THE STAGING PLANES LIVE INSIDE `proj`. They are dead weight of their own:
        // the conv half finishes long before `conv1_2` writes the projection, and
        // `2 * conv_dim <= c` holds for every released checkpoint - an EQUALITY there,
        // since every block has `conv_dim == trans_dim == dim / 2` - so the pair fits
        // `proj` exactly and costs nothing. At 1024x1024 that saves 256 MiB of scratch.
        debug_assert!(2 * conv_dim <= c, "the conv half's staging needs 2*conv_dim <= c");
        let (c1, c2) = scratch.proj.split_at_mut(conv_dim * n);
        let (yc, _) = scratch.y.split_at(conv_dim * n);
        conv3x3(yc, conv_dim, conv_dim, h, w, wt.t(&format!("{prefix}.conv_block.0.weight")), c1);
        relu(c1);
        conv3x3(c1, conv_dim, conv_dim, h, w, wt.t(&format!("{prefix}.conv_block.2.weight")), c2);
        add_into(yc, c2);
        scratch.y[..conv_dim * n].copy_from_slice(c2);
    }

    // Transformer half: NCHW -> [h][w][c], the block, and back.
    {
        // Resize only when the length actually CHANGES: `resize(..., 0.0)` would
        // zero-fill the whole buffer before every element of it is overwritten - 16.8
        // MB of stores a block at 256x256, for nothing. The transpose below writes
        // every element either way.
        if scratch.tok.len() != n * trans_dim {
            scratch.tok.resize(n * trans_dim, 0.0);
        }
        let (_, yt) = scratch.y.split_at(conv_dim * n);
        to_hwc(yt, trans_dim, n, &mut scratch.tok);
        let tb = std::mem::take(&mut scratch.tb);
        let mut tb = tb;
        trans_block(
            &scratch.tok, h, w, trans_dim, win, shift,
            wt.t(&format!("{prefix}.trans_block.ln1.weight")),
            wt.t(&format!("{prefix}.trans_block.ln1.bias")),
            wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.weight")),
            wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.bias")),
            wt.t(&format!("{prefix}.trans_block.msa.linear.weight")),
            wt.t(&format!("{prefix}.trans_block.msa.linear.bias")),
            wt.t(&format!("{prefix}.trans_block.msa.relative_position_params")),
            wt.t(&format!("{prefix}.trans_block.ln2.weight")),
            wt.t(&format!("{prefix}.trans_block.ln2.bias")),
            wt.t(&format!("{prefix}.trans_block.mlp.0.weight")),
            wt.t(&format!("{prefix}.trans_block.mlp.0.bias")),
            wt.t(&format!("{prefix}.trans_block.mlp.2.weight")),
            wt.t(&format!("{prefix}.trans_block.mlp.2.bias")),
            &mut scratch.bt,
            &mut tb,
        );
        scratch.tb = tb;
        // Back to NCHW, into the second half of `scratch.y`. The two halves are
        // disjoint, so this is a reborrow of the same buffer the forward transpose
        // read from - not an aliasing write.
        let (_, yt) = scratch.y.split_at_mut(conv_dim * n);
        from_hwc(&scratch.bt, trans_dim, n, yt);
    }

    // 1x1 project on the concatenation, then the residual over the whole block.
    conv1x1(
        &scratch.y, c, c, n,
        wt.t(&format!("{prefix}.conv1_2.weight")),
        wt.t(&format!("{prefix}.conv1_2.bias")),
        &mut scratch.proj,
    );
}

/// One `ConvTransBlock`, IN PLACE: `buf` holds the block's input on entry and
/// `buf + proj` on exit.
///
/// WHY THIS EXISTS. The block's only read of its input is the residual add at the end
/// - everything in between happens inside `BlockScratch` - so a block can be applied
/// to its own buffer. That is worth two full-resolution planes at every stage: the
/// stage's working copy and the `tmp` a ping-pong pair would need - 512 MiB of them
/// at 1024x1024, for no arithmetic reason.
pub fn conv_trans_block_inplace(
    buf: &mut [f32], dims: (usize, usize), h: usize, w: usize, win: usize, shift: usize,
    wt: &Weights, prefix: &str, scratch: &mut BlockScratch,
) {
    block_project(buf, dims, h, w, win, shift, wt, prefix, scratch);
    // `buf` was only READ above, so this is the residual add and not a self-copy.
    add_into(&scratch.proj, buf);
}

/// `conv_trans_block_inplace` through a separate output buffer, for the callers that
/// need the input preserved - `tests/cuda_ops.rs`'s per-op checks and
/// `examples/probe.rs`'s block-interior differ.
#[allow(clippy::too_many_arguments)]
pub fn conv_trans_block(
    x: &[f32], dims: (usize, usize), h: usize, w: usize, win: usize, shift: usize,
    wt: &Weights, prefix: &str, out: &mut [f32], scratch: &mut BlockScratch,
) {
    block_project(x, dims, h, w, win, shift, wt, prefix, scratch);
    out.copy_from_slice(x);
    add_into(&scratch.proj, out);
}

/// The internals of one `ConvTransBlock`, for the block-interior differ in
/// `examples/probe.rs`. `None` for every field unless that differ asked for it.
#[derive(Default)]
pub struct StageDump {
    pub expand: Option<Vec<f32>>,
    pub conv_half: Option<Vec<f32>>,
    pub trans_half: Option<Vec<f32>>,
    pub res: Option<Vec<f32>>,
    /// The attention output of the block's Swin half, in `[h][w][c]`.
    pub msa: Option<Vec<f32>>,
}

/// `conv_trans_block` with the internals recorded. Only `examples/probe.rs` calls this;
/// the production path is `conv_trans_block`, which is the same code with `dump` None.
#[allow(clippy::too_many_arguments)]
pub fn conv_trans_block_dump(
    x: &[f32], dims: (usize, usize), h: usize, w: usize, win: usize, shift: usize,
    wt: &Weights, prefix: &str, out: &mut [f32], scratch: &mut BlockScratch, dump: &mut StageDump,
) {
    let (conv_dim, trans_dim) = dims;
    let c = conv_dim + trans_dim;
    let n = h * w;
    scratch.resize(c, h, w, trans_dim);
    conv1x1(
        x, c, c, n,
        wt.t(&format!("{prefix}.conv1_1.weight")),
        wt.t(&format!("{prefix}.conv1_1.bias")),
        &mut scratch.y,
    );
    dump.expand = Some(scratch.y.clone());
    {
        // This path needs the conv half's planes to outlive the call, so it sizes them
        // itself - `BlockScratch::resize` does not, because the production path stages
        // them inside `proj` (see `block_project`).
        if scratch.c1.len() != conv_dim * n {
            scratch.c1 = vec![0.0; conv_dim * n];
            scratch.c2 = vec![0.0; conv_dim * n];
        }
        let (yc, _) = scratch.y.split_at(conv_dim * n);
        conv3x3(yc, conv_dim, conv_dim, h, w, wt.t(&format!("{prefix}.conv_block.0.weight")), &mut scratch.c1);
        relu(&mut scratch.c1);
        conv3x3(&scratch.c1, conv_dim, conv_dim, h, w, wt.t(&format!("{prefix}.conv_block.2.weight")), &mut scratch.c2);
        add_into(yc, &mut scratch.c2);
        dump.conv_half = Some(scratch.c2.clone());
        scratch.y[..conv_dim * n].copy_from_slice(&scratch.c2);
    }
    {
        let (_, yt) = scratch.y.split_at(conv_dim * n);
        let tx = &mut scratch.tok;
        tx.clear();
        tx.resize(n * trans_dim, 0.0);
        for ci in 0..trans_dim {
            for i in 0..n {
                tx[i * trans_dim + ci] = yt[ci * n + i];
            }
        }
        let mut tb = std::mem::take(&mut scratch.tb);
        let mut msa = Vec::new();
        trans_block_dump(
            &scratch.tok, h, w, trans_dim, win, shift,
            wt.t(&format!("{prefix}.trans_block.ln1.weight")),
            wt.t(&format!("{prefix}.trans_block.ln1.bias")),
            wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.weight")),
            wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.bias")),
            wt.t(&format!("{prefix}.trans_block.msa.linear.weight")),
            wt.t(&format!("{prefix}.trans_block.msa.linear.bias")),
            wt.t(&format!("{prefix}.trans_block.msa.relative_position_params")),
            wt.t(&format!("{prefix}.trans_block.ln2.weight")),
            wt.t(&format!("{prefix}.trans_block.ln2.bias")),
            wt.t(&format!("{prefix}.trans_block.mlp.0.weight")),
            wt.t(&format!("{prefix}.trans_block.mlp.0.bias")),
            wt.t(&format!("{prefix}.trans_block.mlp.2.weight")),
            wt.t(&format!("{prefix}.trans_block.mlp.2.bias")),
            &mut scratch.bt,
            &mut tb,
            Some(&mut msa),
        );
        dump.msa = Some(msa);
        scratch.tb = tb;
        let (_, yt) = scratch.y.split_at_mut(conv_dim * n);
        for ci in 0..trans_dim {
            for i in 0..n {
                yt[ci * n + i] = scratch.bt[i * trans_dim + ci];
            }
        }
        dump.trans_half = Some(yt.to_vec());
    }
    conv1x1(
        &scratch.y, c, c, n,
        wt.t(&format!("{prefix}.conv1_2.weight")),
        wt.t(&format!("{prefix}.conv1_2.bias")),
        &mut scratch.proj,
    );
    dump.res = Some(scratch.proj.clone());
    out.copy_from_slice(x);
    add_into(&scratch.proj, out);
}

/// The buffers one `ConvTransBlock` needs, reused across every block in a forward.
pub struct BlockScratch {
    /// The 1x1 expand's output, `[c][h][w]`.
    y: Vec<f32>,
    /// The conv half's two 3x3 outputs. EMPTY on the production path, which stages
    /// them inside `proj` - see `block_project`. They exist for
    /// `conv_trans_block_dump`, which needs them to outlive the call.
    c1: Vec<f32>,
    c2: Vec<f32>,
    /// `[h][w][c]` staging for the transformer half and its result.
    tok: Vec<f32>,
    bt: Vec<f32>,
    /// The projection's output.
    proj: Vec<f32>,
    /// `trans_block`'s own scratch.
    tb: Vec<f32>,
    /// The current channel count, so `resize` can tell a stage change from a repeat.
    c: usize,
    h: usize,
    w: usize,
    trans_dim: usize,
}

impl BlockScratch {
    pub fn new() -> BlockScratch {
        BlockScratch {
            y: Vec::new(),
            c1: Vec::new(),
            c2: Vec::new(),
            tok: Vec::new(),
            bt: Vec::new(),
            proj: Vec::new(),
            tb: Vec::new(),
            c: 0,
            h: 0,
            w: 0,
            trans_dim: 0,
        }
    }

    fn resize(&mut self, c: usize, h: usize, w: usize, trans_dim: usize) {
        if (c, h, w, trans_dim) == (self.c, self.h, self.w, self.trans_dim) && !self.y.is_empty() {
            return;
        }
        self.c = c;
        self.h = h;
        self.w = w;
        self.trans_dim = trans_dim;
        let n = h * w;
        self.y = vec![0.0; c * n];
        // Not allocated: the production path stages the conv half inside `proj`, so
        // these two planes would be dead weight carried through the whole forward.
        // `conv_trans_block_dump` sizes them itself when it is called.
        self.c1 = Vec::new();
        self.c2 = Vec::new();
        self.tok = vec![0.0; trans_dim * n];
        self.bt = vec![0.0; trans_dim * n];
        self.proj = vec![0.0; c * n];
        self.tb = Vec::new();
    }
}


// ---------------------------------------------------------------------------
// The network
// ---------------------------------------------------------------------------

/// The whole network on the padded plane: `[c][hp][wp]` in, `[c][hp][wp]` out.
///
/// This is `SCUNet.forward` with the padding and the crop removed (they are in
/// `backend.rs`), statement for statement:
///
/// ```text
/// x1 = m_head(x0)
/// x2 = m_down1(x1); x3 = m_down2(x2); x4 = m_down3(x3)
/// x  = m_body(x4)
/// x  = m_up3(x + x4); x = m_up2(x + x3); x = m_up1(x + x2)
/// x  = m_tail(x + x1)
/// ```
///
/// THE SKIP IS ADDED BEFORE THE UP-CONVOLUTION, at the SAME resolution: `m_up1`'s
/// transposed convolution is `(128 -> 64)` and it is applied to `x + x2` where both
/// are 128 channels at 1/2 resolution, so adding the skip AFTER the up-conv does not
/// even type-check.
pub fn forward(wt: &Weights, x: &[f32], hp: usize, wp: usize) -> Result<Vec<f32>, String> {
    forward_dump(wt, x, hp, wp, &mut |_, _| {})
}

/// The walker with a stage hook: `f(name, plane)` after every stage boundary. `forward`
/// calls this with a no-op hook, so the diagnostic path IS the production path - a
/// divergence found here is a divergence in the engine, not in a re-implementation of
/// it written for the occasion.
pub fn forward_dump(
    wt: &Weights, x: &[f32], hp: usize, wp: usize, f: &mut dyn FnMut(&str, &[f32]),
) -> Result<Vec<f32>, String> {
    let dim = wt.dim;
    if x.len() != wt.in_nc * hp * wp {
        return Err(format!(
            "expected {} floats ({} channels of {hp}x{wp}), got {}",
            wt.in_nc * hp * wp,
            wt.in_nc,
            x.len()
        ));
    }
    let n = hp * wp;

    // Scratch: one buffer per resolution step, all NCHW.
    let mut b = BlockScratch::new();
    // THE HEAD IS NOT KEPT. `m_head`'s output is read twice - as the first down
    // stage's input, and at the very end as the skip `m_tail` consumes - so holding it
    // means a `dim`-channel full-resolution plane alive for the whole forward: 256 MiB
    // at 1024x1024, and the largest single thing this function would own. Recomputing
    // it at the end costs 0.3% of the network's FLOPs and nothing else, because the
    // convolution is a pure function of the INPUT - which is what makes the trade
    // available at all.
    let mut cur = vec![0.0f32; dim * n];
    conv3x3(x, wt.in_nc, dim, hp, wp, wt.t("m_head.0.weight"), &mut cur);
    f("m_head", &cur);

    // Three down stages. Each is `config[k]` blocks at its own width, then a stride-2
    // convolution that halves the plane and doubles the channels; the stage's OUTPUT
    // (after the stride-2 conv) is what the skip adds.
    let mut skips: Vec<Vec<f32>> = Vec::new();
    let mut hh = hp;
    let mut ww = wp;
    for k in 0..3 {
        let name = ["down1", "down2", "down3"][k];
        let count = wt.config[k];
        let (ch, _) = wt.block_dims(&format!("m_{name}.0"));
        let width = 2 * ch;
        // The stage's blocks, IN PLACE IN `cur`. No `tmp` and no copy: the block does
        // not read its input except in the final residual add, so a ping-pong pair
        // would keep two full-resolution planes alive at every stage for nothing.
        for i in 0..count {
            let prefix = format!("m_{name}.{i}");
            let dims = wt.block_dims(&prefix);
            conv_trans_block_inplace(
                &mut cur, dims, hh, ww, wt.window,
                if i % 2 == 1 { wt.window / 2 } else { 0 },
                wt, &prefix, &mut b,
            );
        }
        // The stride-2 tail convolution halves the plane and doubles the channels.
        let (oh, ow) = (hh / 2, ww / 2);
        let out_w = 2 * width;
        let mut down = vec![0.0f32; out_w * oh * ow];
        conv2x2s2(
            &cur, width, out_w, hh, ww, wt.t(&format!("m_{name}.{count}.weight")), &mut down,
        );
        f(&format!("m_{name}_blocks"), &cur);
        f(&format!("m_{name}"), &down);
        // The skip is the stage's OUTPUT, i.e. x2 = m_down1(x1) - NOT the stage's
        // input. It is added to the up stage's input BEFORE the transposed
        // convolution, at the same resolution and the same channel count.
        //
        // THE SKIP IS A COPY, AND THAT COPY IS INHERENT: the next stage's blocks
        // rewrite `cur` in place, so the skip must be its own buffer. Each skip is
        // released the moment the up pass has consumed it (the up loop `pop`s, since
        // the stages take them in reverse), rather than all three staying resident to
        // the end of the forward.
        skips.push(down.clone());
        cur = down;
        hh = oh;
        ww = ow;
    }

    // The body, at the same resolution and in place: no stride-2 convolution and no
    // second buffer.
    {
        let count = wt.config[3];
        for i in 0..count {
            let prefix = format!("m_body.{i}");
            let dims = wt.block_dims(&prefix);
            conv_trans_block_inplace(
                &mut cur, dims, hh, ww, wt.window,
                if i % 2 == 1 { wt.window / 2 } else { 0 },
                wt, &prefix, &mut b,
            );
        }
        f("m_body", &cur);
    }

    // Three up stages: ConvTranspose2d on `x + skip`, then the stage's blocks.
    for k in 0..3 {
        let name = ["up3", "up2", "up1"][k];
        let count = wt.config[4 + k];
        let (ci, co) = wt.up_conv_channels(name);
        // x + skip, IN PLACE IN `cur` AT THE LOWER RESOLUTION. The join is element-wise
        // over disjoint ranges and `cur` is dead the moment the transposed convolution
        // has read it, so a separate joined buffer - a full lower-resolution plane per
        // up stage - has nothing to preserve. The skip itself is POPPED rather than
        // indexed: the stages consume them in reverse, and this is where each one is
        // released for good.
        let skip = skips.pop().expect("one skip per down stage");
        add_into(&skip, &mut cur);
        drop(skip);
        let (nh, nw) = (hh * 2, ww * 2);
        let mut up = vec![0.0f32; co * nh * nw];
        conv_t2x2(&cur, ci, co, hh, ww, wt.t(&format!("m_{name}.0.weight")), &mut up);
        f(&format!("m_{name}_up"), &up);
        // The up stages' blocks are IN PLACE too. The assignment also releases the
        // lower-resolution plane above: the up pass builds a bigger ladder than it
        // tears down, and holding both rungs at once is 256 MiB at 1024x1024 that the
        // forward does not need.
        cur = up;
        for i in 0..count {
            let prefix = format!("m_{name}.{}", i + 1);
            let dims = wt.block_dims(&prefix);
            // The blocks of an up stage run W, SW, W, SW starting UNSHIFTED. The
            // reference builds the stage as [ConvTranspose2d] + [ConvTransBlock for i in
            // range(count)] and takes the type from the comprehension's i, which never
            // sees the prepended transpose - so the transpose at list index 0 does NOT
            // invert the pattern. Getting this backwards produces a plausible image that
            // is wrong everywhere; `tests/parity.rs` catches it because it compares
            // against the upstream module rather than against this walker.
            let shifted = i % 2 == 1;
            conv_trans_block_inplace(
                &mut cur, dims, nh, nw, wt.window,
                if shifted { wt.window / 2 } else { 0 },
                wt, &prefix, &mut b,
            );
        }
        f(&format!("m_{name}"), &cur);
        hh = nh;
        ww = nw;
    }

    // m_tail over y + x1. `head` is computed HERE rather than held from the top of the
    // function - same convolution, same input, so the same bits (see the note at
    // `cur`'s declaration).
    let mut head = vec![0.0f32; dim * n];
    conv3x3(x, wt.in_nc, dim, hp, wp, wt.t("m_head.0.weight"), &mut head);
    add_into(&head, &mut cur);
    f("m_tail_in", &cur);
    let mut out = vec![0.0f32; wt.in_nc * hp * wp];
    conv3x3(&cur, dim, wt.in_nc, hp, wp, wt.t("m_tail.0.weight"), &mut out);
    f("m_tail", &out);
    Ok(out)
}

/// The CPU backend: just the weights. `forward` allocates its scratch and drops it at
/// the end of the call, which is what makes the memory guard's model - one forward's
/// peak - a true statement about this process's live set.
pub struct Cpu<'a> {
    wt: &'a Weights,
}

impl<'a> Cpu<'a> {
    pub fn new(wt: &'a Weights) -> Result<Cpu<'a>, String> {
        Ok(Cpu { wt })
    }
}

impl crate::backend::Backend for Cpu<'_> {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn forward(&mut self, x: &[f32], plan: &Plan) -> Result<Vec<f32>, String> {
        forward(self.wt, x, plan.hp, plan.wp)
    }
}
