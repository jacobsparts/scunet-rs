// SCUNet's own kernels: the Swin window ATTENTION of one ConvTransBlock, and the
// elementwise 2x2 stride-2 convolution and transposed convolution that halve and
// double each stage - the latter pair kept as the comparison arm for the toolkit's
// register-blocked `lg_conv2x2s2` / `lg_conv_t2x2`, which is what the graph runs.
//
// Everything else the graph does is a toolkit kernel (see build.rs). The window
// gather/scatter, the register-blocked GEMM, the warp LayerNorm and the
// register-blocked 2x2 pair were all written here first and are toolkit ops now;
// the notes below each one record what the move kept and what it measured.
//
// NUMERICS: the same flags the toolkit is built with - ftz off, prec-div and
// prec-sqrt on, fmad on, no fast math. The attention mask is applied as
// -FLT_MAX/4 rather than -inf, so a row that were fully masked would stay finite
// through the softmax rescale; this mask never produces one, but the CPU twin and
// this kernel must agree if it ever did.
//
// NCHW THROUGHOUT. An activation is `[c][h][w]`; the token layout `[tokens][c]`
// appears only inside a block, and `[3][tokens][c]` only for the fused qkv.
#include <cuda_runtime.h>

// The largest attention window the per-thread logit array can hold (win = 8 in
// every released checkpoint, so n = 64). Checked against the plan at launch
// rather than silently overflowing.
#define SC_MAX_N 64
// The largest head width the query row and the output accumulators hold. The
// default checkpoint has hd = 32 at the widest stage.
#define SC_MAX_HD 64
// The largest head count the block can be laid out for: blockDim is
// (n <= 64, heads <= 64).
#define SC_MAX_HEADS 64

// ---------------------------------------------------------------------------
// The 2x2 stride-2 down convolution and its transposed twin.
//
// Both take NCHW planes and the weight layouts torch defines, and those are NOT
// the same: `nn.Conv2d` is [c_out][c_in][ky][kx], `nn.ConvTranspose2d` is
// [c_in][c_out][ky][kx]. The toolkit has the 3x3 forms only, so this pair is the
// engine's.
// ---------------------------------------------------------------------------

// out[co][y][x] = sum_{ky,kx,ci} wt[co][ci][ky][kx] * in[ci][2y+ky][2x+kx].
// Accumulation order ky, kx, ci - the order src/cpu.rs::conv2x2s2 uses, so a
// difference here is a bug and not a reordering.
extern "C" __global__ void sc_conv2x2s2(
    const float *__restrict__ in, const float *__restrict__ wt, float *__restrict__ out,
    int c_in, int c_out, int h, int w)
{
    const int ow = w / 2, oh = h / 2;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)c_out * oh * ow) return;
    const int x = (int)(idx % ow);
    const long t = idx / ow;
    const int y = (int)(t % oh);
    const int oc = (int)(t / oh);
    const size_t ip = (size_t)h * w;
    float acc = 0.0f;
    for (int ky = 0; ky < 2; ++ky) {
        for (int kx = 0; kx < 2; ++kx) {
            const size_t off = (size_t)(2 * y + ky) * w + (2 * x + kx);
            const float *wp = wt + ((size_t)oc * c_in) * 4 + ky * 2 + kx;
            for (int ci = 0; ci < c_in; ++ci) {
                acc += wp[(size_t)ci * 4] * in[(size_t)ci * ip + off];
            }
        }
    }
    out[idx] = acc;
}

// out[co][2y+ky][2x+kx] += sum_ci wt[ci][co][ky][kx] * in[ci][y][x], with NO tap
// flip - torch's conv_transpose2d and the reference's `conv_transpose2x2`. One
// thread per OUTPUT element, accumulating over ci in the CPU twin's order.
extern "C" __global__ void sc_conv_t2x2(
    const float *__restrict__ in, const float *__restrict__ wt, float *__restrict__ out,
    int c_in, int c_out, int h, int w)
{
    const int ow = 2 * w, oh = 2 * h;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)c_out * oh * ow) return;
    const int ox = (int)(idx % ow);
    const long t = idx / ow;
    const int oy = (int)(t % oh);
    const int oc = (int)(t / oh);
    const int iy = oy >> 1, ky = oy & 1;
    const int ix = ox >> 1, kx = ox & 1;
    const size_t ip = (size_t)h * w;
    float acc = 0.0f;
    for (int ci = 0; ci < c_in; ++ci) {
        const float wv = wt[((size_t)ci * c_out + oc) * 4 + ky * 2 + kx];
        acc += wv * in[(size_t)ci * ip + (size_t)iy * w + ix];
    }
    out[idx] = acc;
}

// ---------------------------------------------------------------------------
// The window gather and scatter are TOOLKIT ops now.
//
// They were `sc_window_gather` / `sc_window_scatter` plus the shared `sc_pos`
// index helper. The toolkit's `lg_window_gather` / `lg_window_scatter` are the
// same map with the two generalisations this graph itself needed - a cyclic
// `shift` and a chunk base `w0` - so the engine now calls those and passes both.
//
// WHAT WAS WORTH KEEPING FROM THE ORIGINAL NOTE, because it is about the CALLER:
//
// THE INDEX MAP IS THE DANGEROUS PART, and the toolkit shares one map between the
// two directions, so a mistake in it moves BOTH sides and cannot hide behind the
// pair agreeing with itself. What catches it is the golden fixture, not the pair.
// `tests/cuda_ops.rs` also holds the pair to a round trip at a shifted and an
// unshifted call.
//
// THE SHIFT'S SIGN IS LOAD-BEARING. The reference rolls the input by -shift
// before windowing and the output by +shift after; reading the UNROLLED plane at
// (wh*win + i + shift) for both is the same thing, and the opposite sign is a
// 99.7%-wrong attention that reads exactly like a mask bug. The modulo is a wrap,
// as `torch.roll` is.
//
// The ATTENTION stays here: it is the architecture, not a generic op.
// ---------------------------------------------------------------------------

// The shifted-window mask of the reference's `WMSA.generate_mask`.
//
// The reference writes four assignments on a 6-D tensor
// [w1][w2][p1][p2][p3][p4], with (p1, p2) the QUERY token as (row, column) and
// (p3, p4) the KEY token:
//
//     attn_mask[-1, :, :s, :, s:, :] = True
//     attn_mask[-1, :, s:, :, :s, :] = True
//     attn_mask[:, -1, :, :s, :, s:] = True
//     attn_mask[:, -1, :, s:, :, :s] = True
//
// i.e. on the last window ROW the query's row is compared with the key's ROW,
// and on the last window COLUMN the query's column with the key's COLUMN - EACH
// EDGE COMPARES THE SAME AXIS ON BOTH SIDES. The tempting alternative ("the two
// edges share one axis pair") has the right density and is wrong in 29696 of
// 31744 cells of a 64x64 plane; it was verified cell for cell against the
// reference's own tensor over a 3x3 grid of windows, 36864 triples, 0
// mismatches.
__device__ __forceinline__ bool sc_masked(int p, int t, int wh, int ww, int nwh, int nww, int win)
{
    const int s = win - win / 2;
    const int pr = p / win, pc = p % win;
    const int tr = t / win, tc = t % win;
    const bool row_edge = (wh == nwh - 1);
    const bool col_edge = (ww == nww - 1);
    const bool row_hit = (pr < s && tr >= s) || (pr >= s && tr < s);
    const bool col_hit = (pc < s && tc >= s) || (pc >= s && tc < s);
    return (row_edge && row_hit) || (col_edge && col_hit);
}

// Window attention: one block per window, blockDim (n, heads) = (64, up to 64).
//
// ONE THREAD OWNS A WHOLE (query, head) ROW. A block-wide softmax would need the
// logits in shared memory, tree reductions and three syncs, and it would make the
// summation order depend on the block size - so the device result would differ
// from the CPU twin's by more than rounding and the two could not be compared at
// any tolerance worth having. A row here is `n * (2*hd + 4)` flops, which a
// thread does in the time a sync costs. The 64 logits live in a local array
// (indexed by a runtime k, so in L1-cached local memory).
//
// `qkv` is `[nw*n][3c]`: ONE ROW PER TOKEN, with q in the first c columns of the
// row, then k, then v. That is what a single `lg_linear` with 3c outputs produces
// from the gathered tokens, and it is the layout the reference gets from
// `embedding_layer` followed by `chunk(3)` - the three chunks are COLUMN blocks of
// one row, not three separate planes. Reading it as `[3][nw*n][c]` - three
// contiguous planes, each indexed with a row stride of c - looks equivalent and is
// not: a `[tok][3c]` row is 3c wide, so the k and v rows come from the middle of a
// neighbouring token's q block, which is a wrong answer at every geometry rather
// than a rounding difference. The OUTPUT is `[nw*n][c]`, one row per token, because
// `msa.linear` consumes it as `c_in = c`.
//
// `rp` is `[heads][2*win-1][2*win-1]`, packed per head exactly as the checkpoint
// stores it. Unlike the toolkit's attention kernels there is no 1/sqrt(hd)
// missing: the reference scales by `self.scale = head_dim ** -0.5`.
extern "C" __global__ void sc_window_attn(
    const float *__restrict__ qkv, const float *__restrict__ rp, float *__restrict__ out,
    int nw, int n, int nww, int win, int hp, int wp, int heads, int hd, int shift, int w0)
{
    const int wl = blockIdx.x;             // window index WITHIN this chunk
    const int wi = w0 + wl;                // ... and its global index, for the mask
    const int c = heads * hd;
    // One row per token, 3c wide: q, k and v are the three column blocks of the row.
    const int stride = 3 * c;
    const int masked = shift > 0;
    const float scale = rsqrtf((float)hd);
    const float neg = -3.402823466e+38f / 4.0f;   // -FLT_MAX/4, never -inf
    const int nwh = hp / win;

    const int q = threadIdx.x;              // query token in the window
    const int h = threadIdx.y;              // head
    const int off = h * hd;
    const int span = 2 * win - 1;

    const float *qrow = qkv + ((size_t)wl * n + q) * stride + off;
    float qr[SC_MAX_HD];
    for (int d = 0; d < hd; ++d) qr[d] = qrow[d];

    float logits[SC_MAX_N];
    float mx = -3.402823466e+38f;
    for (int k = 0; k < n; ++k) {
        const float *krow = qkv + ((size_t)wl * n + k) * stride + c + off;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int d = 0;
        for (; d + 4 <= hd; d += 4) {
            a0 += qr[d] * krow[d];
            a1 += qr[d + 1] * krow[d + 1];
            a2 += qr[d + 2] * krow[d + 2];
            a3 += qr[d + 3] * krow[d + 3];
        }
        for (; d < hd; ++d) a0 += qr[d] * krow[d];
        float s = ((a0 + a1) + (a2 + a3)) * scale;
        // The learned relative-position bias: entry (p, t) is
        // rp[head][pr + win - 1 - tr][pc + win - 1 - tc] with (pr, pc) the query
        // token as (row, column) and (tr, tc) the key token, both row-major in
        // the window.
        const int dy = q / win + win - 1 - k / win;
        const int dx = q % win + win - 1 - k % win;
        s += rp[((size_t)h * span + dy) * span + dx];
        if (masked && sc_masked(q, k, wi / nww, wi % nww, nwh, nww, win)) s = neg;
        logits[k] = s;
        mx = fmaxf(mx, s);
    }
    float sum = 0.f;
    for (int k = 0; k < n; ++k) {
        const float e = __expf(logits[k] - mx);
        logits[k] = e;
        sum += e;
    }
    const float inv = 1.0f / sum;
    for (int d = 0; d < hd; ++d) {
        const int r = d;
        float acc = 0.f;
        for (int k = 0; k < n; ++k) {
            acc += logits[k] * qkv[((size_t)wl * n + k) * stride + 2 * c + off + r];
        }
        out[((size_t)wl * n + q) * c + off + r] = acc * inv;
    }
}


// Window attention, SHARED-MEMORY k and v: one block per (window, head).
//
// WHY THIS EXISTS (examples/k1x1.rs, buffers resident). `sc_window_attn` below, kept
// as the reference when the two are compared, is one block per window with blockDim
// (n, heads) and ONE THREAD PER (query, head) row. Each of the n query threads of a head walks ALL n key rows and ALL n
// value rows, so the same 2*n*hd floats are read from GLOBAL memory n times over:
// at n = 64, hd = 32 that is 16 KB read 64 times per head, per window, per launch.
// Measured, that made the attention 31% of this engine's device time on a 256x256
// image (276 ms of 895 ms over three forwards, 2.56 ms per launch).
//
// THE FIX IS ONE BLOCK PER (window, head) and a shared tile of that head's k and v.
// blockIdx.x is the window, blockIdx.y the head, and 64 threads cover the window's
// 64 query tokens. The block loads k[.][hd] and v[.][hd] once - 2*n*hd floats, 16 KB
// at the default checkpoint's hd = 32, which fits the 48 KB a block gets without
// opting in - and every query thread then reads them from shared memory. The global
// traffic per (window, head) falls from 2*n*hd*n floats to 2*n*hd, a factor of n.
//
// THE ARITHMETIC AND THE ORDER ARE UNCHANGED, which is why this is not a numerical
// risk: every output still accumulates over k ascending, the logits still live in a
// per-thread array, and the softmax still runs in one thread over the whole row -
// so the kernel and `cpu::attention` remain comparable at the same tolerance, and
// `tests/cuda_ops.rs` holds them to it. Staging in shared memory changes where a
// value is read from, not the order it is added in.
//
// The q row stays in registers: q is read once per thread and used by every k.
extern "C" __global__ void sc_window_attn_s(
    const float *__restrict__ qkv, const float *__restrict__ rp, float *__restrict__ out,
    int nw, int n, int nww, int win, int hp, int wp, int heads, int hd, int shift, int w0)
{
    // One head of one window: k and v rows, each n x hd.
    extern __shared__ float kv[];             // [2][n][hd]
    float *ks = kv;
    float *vs = kv + (size_t)n * hd;

    const int wl = blockIdx.x;                // window index within this chunk
    const int wi = w0 + wl;                   // ... and its global index, for the mask
    const int h = blockIdx.y;                 // head
    const int c = heads * hd;
    const int stride = 3 * c;                 // one row per token, q/k/v as column blocks
    const int masked = shift > 0;
    const float scale = rsqrtf((float)hd);
    const float neg = -3.402823466e+38f / 4.0f;
    const int nwh = hp / win;
    const int off = h * hd;
    const int span = 2 * win - 1;

    // Stage this head's k and v. A thread copies one element of each row per step, so
    // the loads are coalesced along hd and the row stride 3c is the only gap.
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const float *krow = qkv + ((size_t)wl * n + t) * stride + c + off;
        const float *vrow = qkv + ((size_t)wl * n + t) * stride + 2 * c + off;
        for (int d = 0; d < hd; ++d) {
            ks[(size_t)t * hd + d] = krow[d];
            vs[(size_t)t * hd + d] = vrow[d];
        }
    }
    __syncthreads();

    const int q = threadIdx.x;
    const float *qrow = qkv + ((size_t)wl * n + q) * stride + off;
    float qr[SC_MAX_HD];
    for (int d = 0; d < hd; ++d) qr[d] = qrow[d];

    float logits[SC_MAX_N];
    float mx = -3.402823466e+38f;
    for (int k = 0; k < n; ++k) {
        const float *krow = ks + (size_t)k * hd;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int d = 0;
        for (; d + 4 <= hd; d += 4) {
            a0 += qr[d] * krow[d];
            a1 += qr[d + 1] * krow[d + 1];
            a2 += qr[d + 2] * krow[d + 2];
            a3 += qr[d + 3] * krow[d + 3];
        }
        for (; d < hd; ++d) a0 += qr[d] * krow[d];
        float s = ((a0 + a1) + (a2 + a3)) * scale;
        const int dy = q / win + win - 1 - k / win;
        const int dx = q % win + win - 1 - k % win;
        s += rp[((size_t)h * span + dy) * span + dx];
        if (masked && sc_masked(q, k, wi / nww, wi % nww, nwh, nww, win)) s = neg;
        logits[k] = s;
        mx = fmaxf(mx, s);
    }
    float sum = 0.f;
    for (int k = 0; k < n; ++k) {
        const float e = __expf(logits[k] - mx);
        logits[k] = e;
        sum += e;
    }
    const float inv = 1.0f / sum;
    for (int d = 0; d < hd; ++d) {
        float acc = 0.f;
        for (int k = 0; k < n; ++k) acc += logits[k] * vs[(size_t)k * hd + d];
        out[((size_t)wl * n + q) * c + off + d] = acc * inv;
    }
}


// ---------------------------------------------------------------------------
// THIS FILE NOW HOLDS TWO KERNEL FAMILIES THAT WERE BUILT HERE FIRST AND MOVED
// TO THE TOOLKIT, plus the attention. Both of the moves are recorded here
// because the REASONS are what the docs quote, and the numbers are the evidence
// for the toolkit's defaults.
//
// LAYER NORM (`sc_layer_norm`, now `lg_layer_norm_warp`). One warp per row, the
// row held in registers, the toolkit's own shuffle reduction. It was 10.2% of a
// 256x256 forward at 105 ms because `lg_layer_norm` gives one BLOCK per row with
// two shared-memory reduction trees and two syncs, for an op whose arithmetic is
// two reductions over 32-128 floats. Measured against each other: 2.2x at a
// 256-wide row and 8.6x at 32 wide. It is an ADDITION rather than a replacement
// because `lg_layer_norm` is still the better kernel for the wide rows it was
// written for, which is why the toolkit carries both.
//
// THE STRIDED PAIR (`sc_conv2x2s2_rb` / `sc_conv_t2x2_rb`, now `lg_conv2x2s2` /
// `lg_conv_t2x2`). Measured with `examples/budget.rs` at 256x256, the elementwise
// forms above came to 21.7 ms and 10.8 ms per forward - 13% of the whole forward
// for under 1% of its FLOPs, at 148.7 and 299.1 GFLOP/s.
//
// THE REUSE IS OVER CHANNELS, NOT SPACE, and that is the whole design. Both are
// 2x2 kernels at stride 2 (the down form) or a 2x2 scatter (the transposed form),
// so an input element belongs to exactly ONE output's tap set - there is no
// spatial halo to share, and a shared-memory tile of the plane would reduce
// nothing. What the input element IS shared across is the `c_out` channels: at
// down1 that is 128 outputs reading the same activation. So a thread holds eight
// accumulators over output channels and reads one activation into a register,
// reusing it eight times.
//
// THE WEIGHT LOADS ARE UNIFORM ACROSS THE WARP - every thread in a block reads
// the same `oc` set at the same `(ci, ky, kx)` - so the hardware broadcasts them
// and L1 serves each in one transaction. That is the load that would otherwise
// dominate: per (ci, ky, kx) a thread issues eight weight loads and eight FMAs.
//
// THE ACCUMULATION ORDER IS UNCHANGED, which is why the move is a pure speedup:
// the down form still runs ky, kx, ci outermost to innermost exactly as the
// kernel above (and as `src/cpu.rs::conv2x2s2`), the transposed form still runs
// ci alone, and the eight accumulators are independent sums. The toolkit's
// versions add a nullable bias as the accumulator's INITIAL value, matching
// `lg_conv3x3s1p1`, which passing null reproduces bit for bit.
//
// WHY THE ELEMENTWISE FORMS ABOVE STAY. They are the `SCUNET_C2X2=elem`
// comparison arm - the measurement that justified the register-blocked form - and
// they are the reference this engine's CPU twin was first matched against. A
// counterexample that cannot be re-run is not a counterexample.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The register-blocked GEMM is a TOOLKIT op now: `lg_linear_rb` (token layout)
// and `lg_conv1x1_rb` (plane layout), one `lg_rb_body` with a layout flag.
//
// ONE BODY IS THE POINT. The 1x1 convolution and the token linear are the same
// matrix multiply in the two layouts this engine already had, and the attempts to
// write them as separate kernels - the tiled 1x1 above, the six GEMM tile shapes
// here - are what `examples/k1x1.rs` and `examples/klinear.rs` measured against
// each other until one winner remained.
//
// WHAT THE TOOLKIT KEPT. A 4x4 output tile per thread with a 16-wide staged tile
// and 256 threads a block, and a tile shape above 4x4 rejected on REGISTERS
// rather than on load ratio: `nvcc -Xptxas -v` on sm_61 gives 2x4 79 registers,
// 4x4 124 (two blocks an SM), 2x8 127, 4x8 164, 8x4 190 and 8x8 240 (one block
// an SM), and 256 threads times 240 already exceeds the SM's 65536-register file.
//
// THE ACCUMULATION ORDER IS UNCHANGED - k ascends through the whole of c_in in
// tiles of BK, ascending within a tile, so every output's sum runs over the input
// channels in the order `lg_linear`, `src/cpu.rs` and the kernels above use - and
// the BIAS ORDER of each replaced kernel was reproduced separately: `lg_conv1x1`
// folds the bias into the accumulator BEFORE the first multiply, `lg_linear` adds
// it after the last. Both instantiations do it, which is what makes a caller's
// swap bit-exact rather than rounding-level.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// THE WINDOW ATTENTION, WITH THE SCORES IN REGISTERS.
//
// WHY THIS EXISTS. `examples/budget.rs` at 256x256 makes the window attention the
// largest single cost in the graph - 24.6 ms of a 102 ms forward, 24% - and it is
// nowhere near a hardware limit. The attention is 7.8 GFLOP per forward at that size,
// so 24.6 ms is 316 GFLOP/s, under 4% of this card's fp32 peak. Neither global nor
// shared bandwidth explains it: the whole attention moves about 486 MB (20 GB/s), and
// every shared read it makes is a BROADCAST, because the address depends on the key
// and the channel but not on the query, so 32 lanes share one fetch.
//
// THE PROBLEM IS WHERE THE SCORES LIVE. `sc_window_attn_s` keeps `qr[hd]` and
// `logits[n]` in arrays indexed by RUNTIME loop bounds, and `nvcc -Xptxas -v` says so
// outright: 48 registers, ZERO spill bytes, and a 512-BYTE STACK FRAME. A stack frame
// with no spills means the arrays went to LOCAL MEMORY, which on this architecture is
// global memory addressed through L1. So each of the 64 uses of `qr[d]` and each pass
// over `logits` is a memory access: about 9.2 KB per thread, 14848 blocks of 64
// threads, roughly 8.7 GB per forward. That is the 24.6 ms.
//
// TEMPLATING ALONE DOES NOT FIT. Making the bounds compile-time (`n` = 64 tokens,
// `hd` = 32) does turn the arrays into registers, but 64 scores plus 32 query values
// plus the unrolled float4 loads come to 255 registers with 476 bytes of spills - the
// whole register file for one thread, and a block of 64 of them then runs one block
// to an SM. So the score vector is SPLIT ACROSS TWO THREADS instead: a block of 128
// covers one (window, head) with two threads per query token, each owning half the
// keys, and the per-thread arrays are 32 + 32.
//
// THE THREAD INDEXING IS CHOSEN FOR THE BROADCAST. `q = tid & 63` and
// `half = tid >> 6` puts queries 0..63 in the first two warps and the same queries
// again in the last two, so every lane of a warp reads the SAME k and v element -
// which is what keeps a shared fetch to one transaction. The obvious alternative
// (`q = tid >> 1`, `half = tid & 1`) interleaves the halves within a warp and doubles
// every shared transaction.
//
// WHAT IS AND IS NOT BIT-IDENTICAL. The dot products are: the same four-way-unrolled
// accumulation with the same `((a0+a1)+(a2+a3))` association, the same
// relative-position bias, the same mask, the same `__expf`. The maximum is:
// `fmaxf` is exact and associative, so max over the first half and the second half
// and then over those two is the same number as one pass over all 64. TWO SUMS ARE
// REASSOCIATED, and they are the only arithmetic difference: the softmax denominator
// becomes (sum of the first 32) + (sum of the last 32) where the row kernel adds all
// 64 in sequence, and the weighted sum is combined from two half-accumulators the
// same way. Both are sums of 64 positive terms, so the difference is a rounding of
// order 1e-7 relative - which is why `tests/cuda_ops.rs` holds the attention to 2e-4
// and not to the bit.
//
// k and v are still staged in shared memory, and the float4 loads below change only
// the number of load instructions, never the order of additions.
// ---------------------------------------------------------------------------

template <int N, int HD>
__device__ void sc_attn_s_body(
    const float *__restrict__ qkv, const float *__restrict__ rp, float *__restrict__ out,
    int nw, int n, int nww, int win, int hp, int wp, int heads, int hd, int shift, int w0,
    float *ks, float *vs, float *xsh)
{
    constexpr int NH = N / 2;                 // keys per thread
    const int wl = blockIdx.x;                // window index within this chunk
    const int wi = w0 + wl;                   // ... and its global index, for the mask
    const int h = blockIdx.y;                 // head
    const int c = heads * hd;
    const int stride = 3 * c;                 // one row per token, q/k/v as column blocks
    const int masked = shift > 0;
    const float scale = rsqrtf((float)hd);
    const float neg = -3.402823466e+38f / 4.0f;
    const int nwh = hp / win;
    const int off = h * hd;
    const int span = 2 * win - 1;

    const int tid = threadIdx.x;
    const int q = tid & (N - 1);
    const int half = tid >> 6;                // 0 or 1: which half of the keys
    const int k0 = half * NH;

    // Stage this head's k and v: one row per thread, 128 threads for 128 rows.
    {
        const int t = tid;
        if (t < N) {
            const float *krow = qkv + ((size_t)wl * n + t) * stride + c + off;
            for (int d = 0; d < HD; d += 4) {
                *(float4 *)(ks + (size_t)t * HD + d) = *(const float4 *)(krow + d);
            }
        } else {
            const int t2 = t - N;
            const float *vrow = qkv + ((size_t)wl * n + t2) * stride + 2 * c + off;
            for (int d = 0; d < HD; d += 4) {
                *(float4 *)(vs + (size_t)t2 * HD + d) = *(const float4 *)(vrow + d);
            }
        }
    }
    __syncthreads();

    const float *qrow = qkv + ((size_t)wl * n + q) * stride + off;
    float qr[HD];
#pragma unroll
    for (int d = 0; d < HD; d += 4) {
        const float4 qq = *(const float4 *)(qrow + d);
        qr[d] = qq.x;
        qr[d + 1] = qq.y;
        qr[d + 2] = qq.z;
        qr[d + 3] = qq.w;
    }

    float logits[NH];
    float mx = -3.402823466e+38f;
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        const int k = k0 + j;
        const float *krow = ks + (size_t)k * HD;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int d = 0; d < HD; d += 4) {
            const float4 kk = *(const float4 *)(krow + d);
            a0 += qr[d] * kk.x;
            a1 += qr[d + 1] * kk.y;
            a2 += qr[d + 2] * kk.z;
            a3 += qr[d + 3] * kk.w;
        }
        float s = ((a0 + a1) + (a2 + a3)) * scale;
        const int dy = q / win + win - 1 - k / win;
        const int dx = q % win + win - 1 - k % win;
        s += rp[((size_t)h * span + dy) * span + dx];
        if (masked && sc_masked(q, k, wi / nww, wi % nww, nwh, nww, win)) s = neg;
        logits[j] = s;
        mx = fmaxf(mx, s);
    }

    // The row maximum, exchanged through shared memory: exact, so this is the same
    // number the single-thread version computed.
    xsh[half * N + q] = mx;
    __syncthreads();
    const float mxall = fmaxf(xsh[q], xsh[N + q]);

    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        const float e = __expf(logits[j] - mxall);
        logits[j] = e;
        sum += e;
    }
    // `xsh` IS REUSED THREE TIMES AND EVERY REUSE NEEDS A BARRIER BEFORE THE WRITE.
    // It holds the row maxima, then the softmax sums, then the two halves'
    // accumulators - and a write to `xsh[half*N+q]` lands on the very slots the
    // PREVIOUS use's readers are still reading, because every thread reads BOTH
    // halves (`xsh[q]` and `xsh[N+q]`) while every thread writes only its own. A
    // barrier after the read is what orders the two; the barrier below the write only
    // orders the write against the NEXT read.
    //
    // WITHOUT THIS ONE, a fast thread's sum overwrote a slow thread's `mxall` slot,
    // giving that thread the wrong denominator - an INTERMITTENT error of up to 2.0
    // in relative terms on one window row, which the 1x1 projection then smeared
    // across all 512 channels of those 8 pixels. Measured before the fix: 9 divergent
    // runs in 399 on `m_body.0` at 576x64, and 0 in 399 once both barriers were in.
    __syncthreads();
    xsh[half * N + q] = sum;
    __syncthreads();
    const float inv = 1.0f / (xsh[q] + xsh[N + q]);

    // The weighted sum over this half's keys, then the two halves combined. `xsh` is
    // reused as the exchange: the low half writes its accumulator, the high half adds
    // its own and stores.
    float acc[HD];
#pragma unroll
    for (int d = 0; d < HD; ++d) acc[d] = 0.f;
#pragma unroll
    for (int j = 0; j < NH; ++j) {
        const float p = logits[j];
        const float *vrow = vs + (size_t)(k0 + j) * HD;
#pragma unroll
        for (int d = 0; d < HD; d += 4) {
            const float4 vv = *(const float4 *)(vrow + d);
            acc[d] += p * vv.x;
            acc[d + 1] += p * vv.y;
            acc[d + 2] += p * vv.z;
            acc[d + 3] += p * vv.w;
        }
    }
    // The second reuse: this write covers `xsh[0..N*HD)` and so lands on the slots
    // `inv` above was just read from. Every thread has read them by now only if a
    // barrier says so, and the barrier inside the branch below is TOO LATE - it
    // orders this write against the OTHER half's read, not against every thread's
    // read of `inv`. Warps are entirely `half == 0` or `half == 1` (q is the low six
    // bits), so the branch is warp-uniform and the barrier in it is well defined.
    __syncthreads();
    if (half == 0) {
#pragma unroll
        for (int d = 0; d < HD; d += 4) {
            *(float4 *)(xsh + (size_t)q * HD + d) =
                make_float4(acc[d], acc[d + 1], acc[d + 2], acc[d + 3]);
        }
        __syncthreads();
    } else {
        __syncthreads();
        float *orow = out + ((size_t)wl * n + q) * c + off;
#pragma unroll
        for (int d = 0; d < HD; d += 4) {
            const float4 lo = *(const float4 *)(xsh + (size_t)q * HD + d);
            const float4 r = make_float4(lo.x + acc[d], lo.y + acc[d + 1],
                                         lo.z + acc[d + 2], lo.w + acc[d + 3]);
            *(float4 *)(orow + d) = make_float4(r.x * inv, r.y * inv, r.z * inv, r.w * inv);
        }
    }
}

// The instantiation the released checkpoints use: an 8x8 window (64 tokens) and a
// head width of 32, with 128 threads (two per query token). A different window or
// head width falls back to `sc_window_attn_s`, which takes both as runtime arguments.
extern "C" __global__ void sc_window_attn_r64(
    const float *__restrict__ qkv, const float *__restrict__ rp, float *__restrict__ out,
    int nw, int n, int nww, int win, int hp, int wp, int heads, int hd, int shift, int w0)
{
    extern __shared__ float kv[];
    sc_attn_s_body<64, 32>(qkv, rp, out, nw, n, nww, win, hp, wp, heads, hd, shift, w0,
                           kv, kv + (size_t)n * hd, kv + 2 * (size_t)n * hd);
}
