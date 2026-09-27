//! WHERE A FORWARD'S DEVICE TIME GOES, by shape rather than by wall clock.
//!
//! WHY THIS EXISTS. The profiler in `Cuda::run` wraps every launch in a CUDA event
//! pair, which is the only per-kernel measurement this machine allows (`ncu` is
//! denied its counters, ERR_NVGPUCTRPERM) - but a short kernel's span is dominated by
//! the events themselves, so the totals are not absolute. Measured with resident
//! buffers, the same 28 `lg_linear` launches cost 33.8 ms per forward where the
//! profile charged 52, and the RANKING it produced may be wrong for the same reason.
//! The next optimization has to be chosen against a number that can be reproduced.
//!
//! THE METHOD. Time one op of each class at the geometry the graph runs it at, with
//! buffers resident and the pipeline drained once at the end, then multiply by the
//! number of times the walker launches it. The sum is a SAMPLED FORWARD: each arm is
//! measured with only its own buffers in the pool, so a memory-bound op is measured
//! with a colder cache than the real forward gives it. It is therefore a slightly
//! pessimistic upper bound, which is the useful direction for choosing a target.
//!
//!     cargo run --release --features cuda --example budget -- 256 [iters]
//!
//! The graph walked here is `Cuda::forward`'s, stage for stage, including the up
//! stages' block indexing (`i + 1`) - the same iteration `Weights::flops_conv` uses,
//! so the two FLOP totals can be compared and a missing op is visible as a gap.
use std::collections::BTreeMap;

use scunet::plan::Plan;
use scunet::weights::Weights;

#[derive(Default, Clone)]
struct Acc {
    ms: f64,
    calls: usize,
    flops: f64,
    bytes: f64,
}

/// The graph's constants, so the block helper does not need half of `Cuda` passed in.
struct Ctx {
    win: usize,
    ntok: usize,
    /// Windows the walker processes per token chunk.
    nw_blk: usize,
}

/// Accumulate one op class.
#[allow(clippy::too_many_arguments)]
fn tag(a: &mut BTreeMap<&'static str, Acc>, k: &'static str, ms: f64, calls: usize, flops: f64, bytes: f64) {
    let e = a.entry(k).or_default();
    e.ms += ms * calls as f64;
    e.calls += calls;
    e.flops += flops * calls as f64;
    e.bytes += bytes * calls as f64;
}

/// One `ConvTransBlock` at `(h, w)`, every op it launches timed at the size the
/// launch actually sees.
#[allow(clippy::too_many_arguments)]
fn bench_block(
    be: &mut scunet::cuda::Cuda, wt: &Weights, ctx: &Ctx, iters: usize,
    a: &mut BTreeMap<&'static str, Acc>, prefix: &str, h: usize, w: usize,
) -> Result<(), String> {
    let (conv, trans) = wt.block_dims(prefix);
    let c = conv + trans;
    let n = h * w;
    let nwin = (h / ctx.win) * (w / ctx.win);
    let chunks = (nwin + ctx.nw_blk - 1) / ctx.nw_blk;
    let nb = ctx.nw_blk.min(nwin.max(1)) * ctx.ntok;   // tokens one chunk holds
    let four = 4 * trans;
    let fma = |k: usize, o: usize| 2.0 * (k * o * n) as f64;
    // A plane op's traffic: read the input, write the output, plus the weights.
    let plane_bytes = |k: usize, o: usize, kern: usize| {
        ((k * n + o * n) * 4) as f64 + (k * o * kern * 4) as f64
    };

    let t = be.bench_op("conv1x1", &format!("{prefix}.conv1_1"), c, c, h, w, iters)?;
    tag(a, "conv1_1 (1x1 c->c)", t, 1, fma(c, c), plane_bytes(c, c, 1));
    // conv_block: two 3x3 on the conv half, with a ReLU and a residual add between.
    let t = be.bench_op("conv3x3", &format!("{prefix}.conv_block.0"), conv, conv, h, w, iters)?;
    tag(a, "conv_block 3x3 x2", t, 2, 2.0 * 9.0 * (conv * conv * n) as f64, plane_bytes(conv, conv, 9));
    let t = be.bench_op("relu", "", 1, 1, conv * n, 1, iters)?;
    tag(a, "relu", t, 1, 0.0, (2 * conv * n * 4) as f64);
    let t = be.bench_op("add", "", 1, 1, conv * n, 1, iters)?;
    tag(a, "add (conv resid)", t, 1, 0.0, (3 * conv * n * 4) as f64);

    // Transformer half, per chunk.
    // WHOLE-PLANE ARMS, calls = 1. `bench_attn` (and this gather/scatter) runs over
    // EVERY window at once, so multiplying by the chunk count would count the same
    // work `chunks` times - which is exactly what an early version of this file did,
    // reporting the attention at 162 ms where it is 81. Chunking does not change the
    // total work (the sum of the chunks' windows is the plane's), only the number of
    // launches, so one whole-plane measurement IS the graph's cost.
    let t = be.bench_op("gather", "", trans, 0, h, w, iters)?;
    tag(a, "gather", t, 1, 0.0, (2 * trans * n * 4) as f64);
    let t = be.bench_op("layer_norm", &format!("{prefix}.trans_block.ln1"), trans, 0, nb, 1, iters)?;
    tag(a, "layer_norm x2", t, 2 * chunks, 0.0, 2.0 * (2 * nb * trans * 4) as f64);
    let t = be.bench_op("linear", &format!("{prefix}.trans_block.msa.embedding_layer"), trans, 3 * trans, nb, 1, iters)?;
    tag(a, "linear qkv", t, chunks, 2.0 * (trans * 3 * trans * nb) as f64, 0.0);
    let hd = wt.head_dim;
    let heads = trans / hd;
    let t = be.bench_attn("s", prefix, trans, heads, hd, h, w, iters)?;
    tag(a, "attention (window)", t, 1, 0.0, 0.0);
    let t = be.bench_op("linear", &format!("{prefix}.trans_block.msa.linear"), trans, trans, nb, 1, iters)?;
    tag(a, "linear msa.linear", t, chunks, 2.0 * (trans * trans * nb) as f64, 0.0);
    let t = be.bench_op("add", "", 1, 1, nb * trans, 1, iters)?;
    tag(a, "add (token resid x2)", t, 2 * chunks, 0.0, 3.0 * (nb * trans * 4) as f64);
    let t = be.bench_op("linear", &format!("{prefix}.trans_block.mlp.0"), trans, four, nb, 1, iters)?;
    tag(a, "linear mlp.0", t, chunks, 2.0 * (trans * four * nb) as f64, 0.0);
    let t = be.bench_op("gelu", "", 1, 1, nb * four, 1, iters)?;
    tag(a, "gelu", t, chunks, 0.0, (2 * nb * four * 4) as f64);
    let t = be.bench_op("linear", &format!("{prefix}.trans_block.mlp.2"), four, trans, nb, 1, iters)?;
    tag(a, "linear mlp.2", t, chunks, 2.0 * (four * trans * nb) as f64, 0.0);
    let t = be.bench_op("scatter", "", trans, 0, h, w, iters)?;
    tag(a, "scatter", t, 1, 0.0, (2 * trans * n * 4) as f64);

    let t = be.bench_op("conv1x1", &format!("{prefix}.conv1_2"), c, c, h, w, iters)?;
    tag(a, "conv1_2 (1x1 c->c)", t, 1, fma(c, c), plane_bytes(c, c, 1));
    let t = be.bench_op("add", "", 1, 1, c * n, 1, iters)?;
    tag(a, "add (block resid)", t, 1, 0.0, (3 * c * n * 4) as f64);
    Ok(())
}

fn main() -> Result<(), String> {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(size, size, wt.window);
    let mut be = scunet::cuda::Cuda::new(&wt)?;
    let (hp, wp) = (p.hp, p.wp);
    let ntok = wt.window * wt.window;
    let nw_all = (hp / wt.window) * (wp / wt.window);
    let ctx = Ctx {
        win: wt.window,
        ntok,
        nw_blk: (32768 / ntok).clamp(1, nw_all.max(1)),
    };
    let mut a: BTreeMap<&'static str, Acc> = BTreeMap::new();

    let names = ["down1", "down2", "down3", "body", "up3", "up2", "up1"];
    let mut hh = hp;
    let mut ww = wp;
    for stage in 0..7usize {
        let name = names[stage];
        let count = wt.config[stage];
        let base = if stage >= 4 { 1 } else { 0 };
        for b in 0..count {
            let prefix = format!("m_{name}.{}", b + base);
            bench_block(&mut be, &wt, &ctx, iters, &mut a, &prefix, hh, ww)?;
        }
        if stage < 3 {
            let (ch, _) = wt.block_dims(&format!("m_{name}.0"));
            let width = 2 * ch;
            let (oh, ow) = (hh / 2, ww / 2);
            let out_w = 2 * width;
            let t = be.bench_op("conv2x2s2", &format!("m_{name}.{count}"), width, out_w, hh, ww, iters)?;
            tag(&mut a, "conv2x2s2 (stride 2)", t, 1, 2.0 * 4.0 * (width * out_w * oh * ow) as f64,
                (width * hh * ww * 4) as f64);
            hh = oh;
            ww = ow;
        } else if stage > 3 {
            let (ci, co) = wt.up_conv_channels(name);
            let (nh, nw_) = (hh * 2, ww * 2);
            let t = be.bench_op("conv_t2x2", &format!("m_{name}.0"), ci, co, hh, ww, iters)?;
            tag(&mut a, "conv_t2x2 (transposed)", t, 1, 2.0 * 4.0 * (ci * co * hh * ww) as f64,
                (co * nh * nw_ * 4) as f64);
            hh = nh;
            ww = nw_;
        }
    }
    // m_head twice (the second for m_tail's skip) and m_tail once.
    let t = be.bench_op("conv3x3", "m_head.0", wt.in_nc, wt.dim, hp, wp, iters)?;
    tag(&mut a, "m_head 3x3 x2", t, 2, 2.0 * 9.0 * (wt.in_nc * wt.dim * hp * wp) as f64, 0.0);
    let t = be.bench_op("conv3x3", "m_tail.0", wt.dim, wt.in_nc, hp, wp, iters)?;
    tag(&mut a, "m_tail 3x3", t, 1, 2.0 * 9.0 * (wt.dim * wt.in_nc * hp * wp) as f64, 0.0);
    // The host skips: one D2H when the down stage ends and one H2D when the
    // matching up stage consumes it, 4 bytes a float. Their size is the stage's
    // width (`2 * the block's conv half`) on the HALVED plane the stride-2 conv
    // produced, which is what the up stage adds back.
    let mut sk = hp / 2;
    let mut sk_floats = 0.0f64;
    for stage in 0..3 {
        let nm = ["down1", "down2", "down3"][stage];
        let (ch, _) = wt.block_dims(&format!("m_{nm}.0"));
        sk_floats += (2 * ch) as f64 * (sk * sk) as f64;
        sk /= 2;
    }
    let skip_floats = sk_floats;
    tag(&mut a, "skip D2H+H2D (host)", 0.0, 0, 0.0, 2.0 * skip_floats * 4.0);

    let total: f64 = a.values().map(|v| v.ms).sum();
    let flops: f64 = a.values().map(|v| v.flops).sum();
    let model = (wt.flops_conv(hp, wp) + wt.flops_attention(hp, wp)) as f64;
    println!("BUDGET at {hp}x{wp}, {iters} iterations per arm (each op with RESIDENT buffers)");
    println!(
        "  {:<24} {:>8} {:>7} {:>9} {:>9} {:>7}",
        "op", "ms", "launch", "GFLOP/s", "GB/s", "% sum"
    );
    // THE LAUNCH FLOOR. A forward is a few thousand launches, and the sum above
    // measures only the work behind them. Timing `lg_noop` - a kernel that returns
    // immediately - gives the per-launch cost on this machine, which multiplied by
    // the launch count is the floor no kernel change can go below.
    let per_launch = be.bench_op("noop", "", 1, 0, 1, 1, 200)?;
    let launches: usize = a.values().map(|v| v.calls).sum();
    println!(
        "  launch overhead: {:.1} us a launch (lg_noop) x {} launches = {:.1} ms",
        per_launch * 1e3,
        launches,
        per_launch * launches as f64
    );

    let mut rows: Vec<(&str, Acc)> = a.into_iter().collect();
    rows.sort_by(|x, y| y.1.ms.total_cmp(&x.1.ms));
    for (k, v) in &rows {
        if v.calls == 0 {
            continue;
        }
        let gf = if v.flops > 0.0 { v.flops / (v.ms / 1e3) / 1e9 } else { f64::NAN };
        let gb = if v.bytes > 0.0 { v.bytes / (v.ms / 1e3) / 1e9 } else { f64::NAN };
        println!(
            "  {k:<24} {:>8.1} {:>7} {:>9.1} {:>9.1} {:>6.1}%",
            v.ms, v.calls, gf, gb, 100.0 * v.ms / total
        );
    }
    println!("  {:<24} {:>8.1}", "SUM of sampled ops", total);
    println!(
        "  sampled FLOPs {:.1} GFLOP; the model's own count is {:.1} GFLOP",
        flops / 1e9,
        model / 1e9
    );
    println!(
        "  over the sampled sum: {:.1} GFLOP/s (the model's FLOPs over the sampled time)",
        model / (total / 1e3) / 1e9
    );
    Ok(())
}
