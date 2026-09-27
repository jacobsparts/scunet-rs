//! WHERE A FORWARD'S CPU TIME GOES, by shape rather than by wall clock.
//!
//! THE CPU TWIN OF `budget.rs`, and it exists for the same reason: the CUDA path
//! was optimized against a per-op measurement at real geometry with resident
//! buffers, and the CPU path - which is ~10x PyTorch's at 128x128, the one size
//! where torch's own timing is reproducible - has never been measured at all.
//! Choosing a target from the total (3695 ms at 256x256) means guessing.
//!
//! WHAT IT MEASURES. Each op class at the channel counts, plane sizes and token
//! counts the graph actually gives it, then multiplies by the number of times the
//! walker calls it. `attention` and `trans_block` are timed whole as well, so the
//! transformer half can be split into its window attention and the rest.
//!
//!     cargo run --release --features cuda --example budget_cpu -- 256 [iters]
//!
//! THE TRANSPOSES ARE REAL WORK ON THIS PATH AND ARE COUNTED. The GPU walker keeps
//! `[rows][c]` tokens throughout; the CPU twin has no such layout, it transposes
//! NCHW -> `[h][w][c]` before each transformer half and back after - 2 * n * c
//! element moves each way, per block, single-threaded.
use std::collections::BTreeMap;
use std::time::Instant;

use scunet::plan::Plan;
use scunet::weights::Weights;

#[derive(Default, Clone)]
struct Acc {
    ms: f64,
    calls: usize,
    flops: f64,
    bytes: f64,
}

#[allow(clippy::too_many_arguments)]
fn tag(a: &mut BTreeMap<&'static str, Acc>, k: &'static str, ms: f64, calls: usize, flops: f64, bytes: f64) {
    let e = a.entry(k).or_default();
    e.ms += ms * calls as f64;
    e.calls += calls;
    e.flops += flops * calls as f64;
    e.bytes += bytes * calls as f64;
}

/// One op, `iters` times, with everything resident. Returns ms per call.
fn time<T>(iters: usize, mut f: impl FnMut() -> T) -> f64 {
    f();                                    // warm the caches and the allocator
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() -> Result<(), String> {
    let size: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let wt = Weights::load("../models/scunet-color-real-psnr.safetensors")?;
    let p = Plan::new(size, size, wt.window);
    let (hp, wp) = (p.hp, p.wp);
    let win = wt.window;
    let mut a: BTreeMap<&'static str, Acc> = BTreeMap::new();
    let fma = |k: usize, o: usize, n: usize| 2.0 * (k * o * n) as f64;
    let plane = |k: usize, o: usize, n: usize| ((k * n + o * n) * 4) as f64;

    let block = |a: &mut BTreeMap<&'static str, Acc>, prefix: &str, h: usize, w: usize| {
        let (conv, trans) = wt.block_dims(prefix);
        let c = conv + trans;
        let n = h * w;
        let four = 4 * trans;

        // The two layouts the transformer half needs, allocated ONCE so their
        // allocation is not charged to whichever op happens to be timed first.
        let mut x = vec![0.0f32; c * n];
        for (i, v) in x.iter_mut().enumerate() { *v = (i % 97) as f32 * 0.01 - 0.5; }
        let mut y = vec![0.0f32; c * n];
        let mut c1 = vec![0.0f32; conv * n];
        let mut c2 = vec![0.0f32; conv * n];
        let mut tok = vec![0.0f32; n * trans];
        let mut bt = vec![0.0f32; n * trans];

        let t = time(iters, || {
            scunet::cpu::conv1x1(&x, c, c, n, wt.t(&format!("{prefix}.conv1_1.weight")),
                wt.t(&format!("{prefix}.conv1_1.bias")), &mut y)
        });
        tag(a, "conv1_1 (1x1 c->c)", t, 1, fma(c, c, n), plane(c, c, n) + (c * c * 4) as f64);

        let t = time(iters, || {
            scunet::cpu::conv3x3(&x, conv, conv, h, w, wt.t(&format!("{prefix}.conv_block.0.weight")), &mut c1)
        });
        tag(a, "conv_block 3x3 x2", t, 2, 2.0 * 9.0 * (conv * conv * n) as f64, plane(conv, conv, n));
        let t = time(iters, || scunet::cpu::relu(&mut c1));
        tag(a, "relu", t, 1, 0.0, (2 * conv * n * 4) as f64);
        let t = time(iters, || scunet::cpu::add_into(&c1, &mut c2));
        tag(a, "add (conv resid)", t, 1, 0.0, (3 * conv * n * 4) as f64);

        // THE TRANSPOSES, calling the ENGINE'S OWN FUNCTIONS. An earlier version of
        // this file inlined a copy of the old serial loops, so it kept reporting 46 ms
        // for a transpose the walker had already parallelised - an instrument that
        // reimplements the thing it measures measures the wrong program.
        let (_, yt) = x.split_at(conv * n);
        let t = time(iters, || scunet::cpu::to_hwc(yt, trans, n, &mut tok));
        tag(a, "transpose to [h][w][c]", t, 1, 0.0, (2 * n * trans * 4) as f64);
        let t = time(iters, || scunet::cpu::from_hwc(&tok, trans, n, &mut bt));
        tag(a, "transpose back to NCHW", t, 1, 0.0, (2 * n * trans * 4) as f64);

        let emb = format!("{prefix}.trans_block.msa.embedding_layer");
        let lin = format!("{prefix}.trans_block.msa.linear");
        let m0 = format!("{prefix}.trans_block.mlp.0");
        let m2 = format!("{prefix}.trans_block.mlp.2");
        let ln1 = format!("{prefix}.trans_block.ln1");
        let rp = format!("{prefix}.trans_block.msa.relative_position_params");
        let mut out = vec![0.0f32; n * trans];

        // THE WINDOW ATTENTION ALONE, which is the CUDA budget's second-largest op.
        let t = time(iters, || {
            scunet::cpu::attention(&tok, h, w, trans, win, 0,
                wt.t(&format!("{emb}.weight")), wt.t(&format!("{emb}.bias")),
                wt.t(&format!("{lin}.weight")), wt.t(&format!("{lin}.bias")),
                wt.t(&rp), &mut out)
        });
        tag(a, "attention (gather+attn+scatter)", t, 1, 0.0, 0.0);
        // THE SHIFTED FORM IS A DIFFERENT COST and the graph alternates: half the
        // blocks shift by win/2, and only those evaluate the mask - `mask_hit` per
        // (p, t) pair, with four integer divisions in it. Timed separately because an
        // unshifted-only arm once made this budget understate the engine by 1.8x.
        let t = time(iters, || {
            scunet::cpu::attention(&tok, h, w, trans, win, win / 2,
                wt.t(&format!("{emb}.weight")), wt.t(&format!("{emb}.bias")),
                wt.t(&format!("{lin}.weight")), wt.t(&format!("{lin}.bias")),
                wt.t(&rp), &mut out)
        });
        // ACCUMULATED like every other arm. A plain insert put the LAST BLOCK's time
        // in the map, which is the full-resolution one, and comparing that against the
        // 28-block accumulated mean made the shifted form look twice as cheap.
        tag(a, "attention SHIFTED (win/2)", t, 1, 0.0, 0.0);

        // The linears inside the transformer half, timed alone so their share is
        // visible. Each is a CONV1X1 OVER TOKENS, which is the same matmul the CUDA
        // budget times as `linear`: qkv is trans -> 3*trans, mlp.0 is trans -> 4*trans.
        // NOTHING HERE IS ALSO COUNTED ELSEWHERE: the attention arm above includes the
        // qkv embedding and the output projection, but not these, so the arms add up.
        // (`trans_block` as a whole was an arm here once and it double counted every
        // one of them, which is how a budget stops being a decomposition.)
        let t = time(iters, || {
            scunet::cpu::conv1x1(&tok, trans, 3 * trans, n,
                wt.t(&format!("{emb}.weight")), wt.t(&format!("{emb}.bias")), &mut out)
        });
        tag(a, "linear qkv", t, 1, fma(trans, 3 * trans, n), 0.0);
        let mut o4 = vec![0.0f32; n * four];
        let t = time(iters, || {
            scunet::cpu::conv1x1(&tok, trans, four, n,
                wt.t(&format!("{m0}.weight")), wt.t(&format!("{m0}.bias")), &mut o4)
        });
        tag(a, "linear mlp.0", t, 1, fma(trans, four, n), 0.0);
        let t = time(iters, || scunet::cpu::gelu_erf_inplace(&mut o4));
        tag(a, "gelu", t, 1, 0.0, (2 * n * four * 4) as f64);
        let t = time(iters, || {
            scunet::cpu::conv1x1(&o4, four, trans, n,
                wt.t(&format!("{m2}.weight")), wt.t(&format!("{m2}.bias")), &mut out)
        });
        tag(a, "linear mlp.2", t, 1, fma(four, trans, n), 0.0);
        let t = time(iters, || {
            scunet::cpu::layer_norm(&tok, trans, wt.t(&format!("{ln1}.weight")),
                wt.t(&format!("{ln1}.bias")), &mut out)
        });
        tag(a, "layer_norm x2", t, 2, 0.0, (2 * 2 * n * trans * 4) as f64);
        let t = time(iters, || scunet::cpu::add_into(&tok, &mut bt));
        tag(a, "add (token resid x2)", t, 2, 0.0, (3 * n * trans * 4) as f64);

        let t = time(iters, || {
            scunet::cpu::conv1x1(&x, c, c, n, wt.t(&format!("{prefix}.conv1_2.weight")),
                wt.t(&format!("{prefix}.conv1_2.bias")), &mut y)
        });
        tag(a, "conv1_2 (1x1 c->c)", t, 1, fma(c, c, n), plane(c, c, n) + (c * c * 4) as f64);
        let t = time(iters, || scunet::cpu::add_into(&y, &mut x));
        tag(a, "add (block resid)", t, 1, 0.0, (3 * c * n * 4) as f64);

        // A WHOLE BLOCK, as a CHECK on the arms above rather than a term of them: it
        // contains every one of them, so it is excluded from the sum by name. When the
        // arms were double counted this was the only arm that could not lie, and with
        // an honest decomposition it is how the remaining gap gets localized.
        let mut bs = scunet::cpu::BlockScratch::new();
        let mut bo = vec![0.0f32; c * n];
        let t = time(iters, || {
            scunet::cpu::conv_trans_block(&x, (conv, trans), h, w, win, 0, &wt, prefix, &mut bo, &mut bs)
        });
        tag(a, "BLOCK TOTAL (double counts)", t, 1, 0.0, 0.0);
    };

    let names = ["down1", "down2", "down3", "body", "up3", "up2", "up1"];
    let mut hh = hp;
    let mut ww = wp;
    for stage in 0..7usize {
        let name = names[stage];
        let count = wt.config[stage];
        let base = if stage >= 4 { 1 } else { 0 };
        for b in 0..count {
            let prefix = format!("m_{name}.{}", b + base);
            block(&mut a, &prefix, hh, ww);
        }
        if stage < 3 {
            let (ch, _) = wt.block_dims(&format!("m_{name}.0"));
            let width = 2 * ch;
            let (oh, ow) = (hh / 2, ww / 2);
            let out_w = 2 * width;
            let x = vec![0.0f32; width * hh * ww];
            let mut o = vec![0.0f32; out_w * oh * ow];
            let t = time(iters, || {
                scunet::cpu::conv2x2s2(&x, width, out_w, hh, ww, wt.t(&format!("m_{name}.{count}.weight")), &mut o)
            });
            tag(&mut a, "conv2x2s2 (stride 2)", t, 1, 2.0 * 4.0 * (width * out_w * oh * ow) as f64,
                (width * hh * ww * 4) as f64);
            hh = oh;
            ww = ow;
        } else if stage > 3 {
            let (ci, co) = wt.up_conv_channels(name);
            let (nh, nw_) = (hh * 2, ww * 2);
            let x = vec![0.0f32; ci * hh * ww];
            let mut o = vec![0.0f32; co * nh * nw_];
            let t = time(iters, || {
                scunet::cpu::conv_t2x2(&x, ci, co, hh, ww, wt.t(&format!("m_{name}.0.weight")), &mut o)
            });
            tag(&mut a, "conv_t2x2 (transposed)", t, 1, 2.0 * 4.0 * (ci * co * hh * ww) as f64,
                (co * nh * nw_ * 4) as f64);
            hh = nh;
            ww = nw_;
        }
    }
    let x = vec![0.0f32; wt.in_nc * hp * wp];
    let mut o = vec![0.0f32; wt.dim * hp * wp];
    let t = time(iters, || {
        scunet::cpu::conv3x3(&x, wt.in_nc, wt.dim, hp, wp, wt.t("m_head.0.weight"), &mut o)
    });
    tag(&mut a, "m_head 3x3 x2", t, 2, 2.0 * 9.0 * (wt.in_nc * wt.dim * hp * wp) as f64, 0.0);
    let t = time(iters, || {
        scunet::cpu::conv3x3(&o, wt.dim, wt.in_nc, hp, wp, wt.t("m_tail.0.weight"), &mut x.clone())
    });
    tag(&mut a, "m_tail 3x3", t, 1, 2.0 * 9.0 * (wt.dim * wt.in_nc * hp * wp) as f64, 0.0);

    // `attention SHIFTED` is a COMPARISON, not a term: the graph runs the attention
    // about half shifted and half unshifted, and the unshifted arm above is the one
    // charged in the decomposition. Excluded by name from the sum.
    let is_note = |k: &&str| *k == "attention SHIFTED (win/2)" || *k == "BLOCK TOTAL (double counts)";
    let total: f64 = a.iter().filter(|(k, _)| !is_note(k)).map(|(_, v)| v.ms).sum();
    let flops: f64 = a.iter().filter(|(k, _)| !is_note(k)).map(|(_, v)| v.flops).sum();
    let model = (wt.flops_conv(hp, wp) + wt.flops_attention(hp, wp)) as f64;
    println!("CPU BUDGET at {hp}x{wp}, {iters} iterations per arm, {} threads", rayon::current_num_threads());
    println!("  {:<30} {:>9} {:>7} {:>9} {:>8} {:>7}", "op", "ms", "calls", "GFLOP/s", "GB/s", "% sum");
    let mut rows: Vec<(&str, Acc)> = a.into_iter().collect();
    rows.sort_by(|x, y| y.1.ms.total_cmp(&x.1.ms));
    for (k, v) in &rows {
        if v.calls == 0 { continue; }
        let gf = if v.flops > 0.0 { v.flops / (v.ms / 1e3) / 1e9 } else { f64::NAN };
        let gb = if v.bytes > 0.0 { v.bytes / (v.ms / 1e3) / 1e9 } else { f64::NAN };
        println!("  {k:<30} {:>9.1} {:>7} {:>9.1} {:>8.1} {:>6.1}%", v.ms, v.calls, gf, gb, 100.0 * v.ms / total);
    }
    println!("  {:<30} {:>9.1}", "SUM of sampled ops", total);
    println!("  sampled FLOPs {:.1} GFLOP; the model's own count is {:.1} GFLOP", flops / 1e9, model / 1e9);
    println!("  over the sampled sum: {:.1} GFLOP/s (the model's FLOPs over the sampled time)", model / (total / 1e3) / 1e9);
    Ok(())
}
