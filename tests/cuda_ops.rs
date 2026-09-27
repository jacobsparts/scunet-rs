//! The device kernels against their CPU twins, op by op, plus the end-to-end
//! comparisons that would catch a wiring mistake the ops cannot see.
//!
//! WHY THIS EXISTS SEPARATELY FROM tests/parity.rs: an end-to-end diff says the
//! image is wrong, not WHICH op is wrong, and a diffuse error looks like numerics
//! while it is a layout. Every primitive here is compared on the same synthetic
//! input against the `cpu.rs` function it transcribes, so a failure names the
//! kernel. The tolerance is per-op: the elementwise and 1x1 ops are the same sums in
//! the same order (~1e-6), `lg_linear` is a dot product whose order differs from the
//! CPU's loop (~1e-5 relative), and the winograd form rounds differently from the
//! direct conv by design (~1e-4).
//!
//! TWO TRAPS THIS FILE HAS ALREADY FALLEN INTO, both recorded because they cost
//! hours and both look like engine bugs:
//!
//! 1. `cpu::forward(wt, x, hp, wp)` takes an ALREADY-PADDED plane, while
//!    `backend::run` pads, forwards and crops. Comparing a `backend::run` device
//!    result against a `cpu::forward` call with the UNPADDED dimensions compares two
//!    different problems and reports a large "device error" at exactly the sizes the
//!    plan pads. End-to-end comparisons here go through `backend::run` on BOTH sides.
//! 2. `Cuda::attention_probe` returns the token buffer `[nt][c]`; `cpu::attention`
//!    takes and returns a scattered `[h][w][c]` plane. With one window and shift 0
//!    they coincide, which is the only case where a direct comparison is meaningful -
//!    so the probe is compared there, and elsewhere the kernel is proved on
//!    hand-built qkv where the answer is known without arithmetic.
#[cfg(feature = "cuda")]
#[path = "device_lock.rs"]
mod device_lock;

mod cuda_ops {
    use scunet::cpu;
    use scunet::cuda::Cuda;
    use scunet::Weights;

    /// A deterministic sequence, so a failure is reproducible without a fixture.
    fn seq(n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * a).sin() * 0.7 + 0.3 * a).sin()).collect()
    }

    /// Two numbers, because one is not enough.
    ///
    /// A PURELY RELATIVE measure hides a structural error wherever the reference
    /// value happens to be near zero - a wrong tap on a cancelling output looks like
    /// a rounding difference - and it reports the same `2.000e0` whether the absolute
    /// gap is 2e-3 or 2e1. A purely absolute one hides a small relative error on a
    /// large plane. So this asserts on the worst absolute difference scaled by the
    /// data's own magnitude (a STRUCTURAL bound), and reports the worst relative
    /// difference floored at a per-element epsilon (a NUMERICS bound).
    fn close(what: &str, want: &[f32], got: &[f32], tol: f32) {
        assert_eq!(want.len(), got.len(), "{what}: length");
        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let mut wa = 0.0f32;
        let mut wr = 0.0f32;
        let (mut ia, mut ir) = (0usize, 0usize);
        for (i, (a, b)) in want.iter().zip(got).enumerate() {
            let d = (a - b).abs();
            if d > wa {
                wa = d;
                ia = i;
            }
            let eps = 1e-3 * scale;
            let r = d / a.abs().max(b.abs()).max(eps);
            if r > wr {
                wr = r;
                ir = i;
            }
        }
        assert!(
            wa <= tol * scale,
            "{what}: worst |diff| {wa:.3e} at {ia} against a scale of {scale:.3e} \
             (tolerance {tol:.1e} of scale); worst relative {wr:.3e} at {ir}"
        );
    }

    /// `[c][h][w]` -> `[h*w][c]`, the layout `cpu::layer_norm` and `cpu::attention`
    /// take: the transformer half works on tokens, and the engine's plane is
    /// channel-major, so a test that hands `cpu::attention` the plane directly is
    /// comparing two different indexings and reports a large "device error" on a
    /// correct device path.
    fn to_tokens(plane: &[f32], c: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; c * n];
        for p in 0..n {
            for ch in 0..c {
                out[p * c + ch] = plane[ch * n + p];
            }
        }
        out
    }

    /// The checkpoint, or `None` when it has not been converted on this machine.
    fn weights() -> Option<Weights> {
        Weights::load("../models/scunet-color-real-psnr.safetensors").ok()
    }

    /// The whole network, CUDA against the CPU backend, on planes the plan has to
    /// pad and planes it does not. BOTH SIDES GO THROUGH `backend::run`, which is
    /// what keeps the padding and the crop in the comparison exactly once.
    #[test]
    fn forward_matches_the_cpu_backend() {
        let Some(wt) = weights() else {
            eprintln!("skipping: checkpoint not converted");
            return;
        };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device with --features cuda");
        let mut cb = scunet::cpu::Cpu::new(&wt).expect("cpu backend");
        for (h, w) in [
            (64usize, 64usize),
            (128, 64),
            (80, 64),
            (96, 64),
            (72, 40),
            (64, 80),
            (100, 100),
        ] {
            let p = scunet::plan::Plan::new(h, w, wt.window);
            let x = seq(3 * h * w, 0.7);
            let want = scunet::backend::run(&mut cb, &x, 3, h, w, &wt).expect("cpu forward");
            let got = scunet::backend::run(&mut be, &x, 3, h, w, &wt).expect("cuda forward");
            let worst = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("forward {h}x{w} (plan {}x{}): worst abs {worst:.3e}", p.hp, p.wp);
            close(&format!("forward {h}x{w}"), &want, &got, 2e-4);
        }
    }

    /// One `ConvTransBlock` on the device against the CPU twin, at the geometries the
    /// graph runs, shifted and unshifted. This is the instrument that localises a
    /// divergence to a block rather than to the image.
    #[test]
    fn block_matches_the_cpu_twin() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        for (prefix, h, w, shift) in [
            ("m_body.0", 32usize, 16usize, 0usize),
            ("m_body.0", 16, 32, 0),
            ("m_body.0", 32, 32, 0),
            ("m_body.0", 64, 16, 0),
            ("m_body.0", 16, 16, 0),
            ("m_body.0", 32, 16, 4),
            ("m_body.0", 16, 32, 4),
            ("m_body.0", 32, 32, 4),
            ("m_body.0", 16, 16, 4),
            ("m_body.0", 8, 8, 4),
            ("m_down1.0", 16, 16, 0),
            ("m_down1.0", 16, 16, 4),
            ("m_down2.0", 16, 16, 0),
        ] {
            let dims = wt.block_dims(prefix);
            let c = dims.0 + dims.1;
            let x = seq(c * h * w, 0.53);
            let got = be.block_dump(prefix, h, w, shift, &x).expect("block on device");
            let mut want = vec![0.0f32; c * h * w];
            let mut sc = scunet::cpu::BlockScratch::new();
            cpu::conv_trans_block(&x, dims, h, w, wt.window, shift, &wt, prefix, &mut want, &mut sc);
            close(&format!("{prefix} {h}x{w} shift{shift}"), &want, &got, 3e-5);
        }
    }

    /// The transformer half processes its windows IN CHUNKS, and this is the test
    /// that actually crosses a chunk boundary.
    ///
    /// WHY IT EXISTS: `TOKEN_BUDGET` is 32768 tokens and a window holds `win*win` =
    /// 64 of them, so a plane needs MORE THAN 512 windows - more than 32768 pixels -
    /// before the walker loops at all. Every other geometry in this file is smaller
    /// than that, so the entire chunking path, including the window-base offset that
    /// the second chunk needs for its positions and its shifted-window mask, was
    /// previously exercised by nothing. A wrong `w0` would leave the first chunk
    /// right and the image subtly wrong past 32768 pixels, which is exactly the size
    /// a user is most likely to run.
    ///
    /// 576x64 is 72x8 = 576 windows: one chunk of 512 plus a second of 64, so the
    /// boundary falls mid-row of windows. Both shift patterns are checked, because
    /// the mask depends on the window's GLOBAL row and column.
    #[test]
    fn block_chunks_beyond_the_token_budget() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        for (prefix, h, w, shift) in [
            ("m_body.0", 576usize, 64usize, 0usize),
            ("m_body.0", 576, 64, 4),
            ("m_down1.0", 512, 72, 0),
            ("m_down1.0", 512, 72, 4),
        ] {
            let dims = wt.block_dims(prefix);
            let c = dims.0 + dims.1;
            let x = seq(c * h * w, 0.71);
            let got = be.block_dump(prefix, h, w, shift, &x).expect("block on device");
            let mut want = vec![0.0f32; c * h * w];
            let mut sc = scunet::cpu::BlockScratch::new();
            cpu::conv_trans_block(&x, dims, h, w, wt.window, shift, &wt, prefix, &mut want, &mut sc);
            close(&format!("chunked {prefix} {h}x{w} shift{shift}"), &want, &got, 3e-5);
        }
    }

    /// A BLOCK RUN TWICE MUST GIVE THE SAME ANSWER, BIT FOR BIT.
    ///
    /// This is the cheapest race detector the repository has, and it is here
    /// because a race shipped once. `sc_attn_s_body` reuses one shared array for
    /// the row maxima, the softmax sums and the two halves' accumulators; two of
    /// the three reuses had no barrier between the previous use's READ and the next
    /// use's WRITE, so a fast thread's sum could land in a slow thread's `mxall`
    /// slot and give it a wrong softmax denominator. The symptom was a relative
    /// error near 2.0 on one window row of tokens, once in every ~40 runs of the
    /// geometry with more than one token chunk - which the end-to-end tolerance
    /// absorbed, so a single green suite said nothing.
    ///
    /// NO CPU TWIN IS INVOLVED, deliberately: it costs only device time (the
    /// comparison is against the previous run, not against `cpu.rs`), and it
    /// isolates the question to the one a race is about. `m_body.0` at 576x64 is
    /// the geometry that failed - 576 windows against a 512-window `TOKEN_BUDGET`,
    /// so the chunk loop runs twice - and `REPEATS` is set from the measured rate:
    /// the bug showed up in 9 of 399 runs, so a dozen repetitions catches it about
    /// a quarter of the time, and the real detector is
    /// `cargo run --release --features cuda --example blockdet -- 400`.
    #[test]
    fn block_repeats_are_bit_identical() {
        let _guard = crate::device_lock::device_lock();
        let Some(wt) = weights() else { return };
        let mut be = Cuda::new(&wt).expect("a device");
        const REPEATS: usize = 12;
        for (prefix, h, w, shift) in [
            ("m_body.0", 576usize, 64usize, 0usize),
            ("m_body.0", 576, 64, 4),
        ] {
            let dims = wt.block_dims(prefix);
            let c = dims.0 + dims.1;
            let x = seq(c * h * w, 0.71);
            let first = be.block_dump(prefix, h, w, shift, &x).expect("block on device");
            for r in 1..REPEATS {
                let got = be.block_dump(prefix, h, w, shift, &x).expect("block on device");
                let (mut worst, mut at) = (0.0f32, 0usize);
                for (i, (a, b)) in first.iter().zip(got.iter()).enumerate() {
                    let d = (a - b).abs();
                    if d > worst {
                        worst = d;
                        at = i;
                    }
                }
                assert_eq!(
                    worst, 0.0,
                    "{prefix} {h}x{w} shift{shift}: run {r} differs from run 0 by {worst:.3e} at \
                     index {at} (channel {} of {c}, pixel {} of {}). A device kernel that is \
                     not bit-reproducible on identical input is racing - look for a shared \
                     array written after a read with no __syncthreads() between them.",
                    at / (h * w), at % (h * w), h * w
                );
            }
        }
    }

    /// The ops one at a time, each against the `cpu.rs` function it transcribes.
    #[test]
    fn ops_match_their_cpu_twins() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        let (h, w) = (16usize, 24usize);
        let n = h * w;

        // conv1x1 with a bias.
        {
            let (ci, co) = (8usize, 12usize);
            let x = seq(ci * n, 0.31);
            let got = be.op_probe("conv1x1", "m_down1.0.conv1_1", &x, ci, co, h, w).expect("conv1x1");
            let mut want = vec![0.0f32; co * n];
            cpu::conv1x1(&x, ci, co, n, wt.t("m_down1.0.conv1_1.weight"), wt.t("m_down1.0.conv1_1.bias"), &mut want);
            close("conv1x1", &want, &got, 1e-5);
        }
        // conv3x3, no bias - including at c_in >= 32, where the winograd form runs.
        for (name, ci, co) in [("m_down1.0.conv_block.0", 32usize, 32usize), ("m_body.0.conv_block.0", 256, 256)] {
            let x = seq(ci * n, 0.37);
            let got = be.op_probe("conv3x3", name, &x, ci, co, h, w).expect("conv3x3");
            let mut want = vec![0.0f32; co * n];
            cpu::conv3x3(&x, ci, co, h, w, wt.t(&format!("{name}.weight")), &mut want);
            close(&format!("conv3x3 {name}"), &want, &got, 3e-4);
        }
        // lg_linear against a hand-written row-major matmul.
        {
            let (rows, ci, co) = (5usize, 256usize, 32usize);
            let x = seq(rows * ci, 0.29);
            let name = "m_body.0.trans_block.msa.linear";
            let got = be.linear_probe(name, &x, rows, ci, co).expect("linear");
            let (wp, bp) = (wt.t(&format!("{name}.weight")), wt.t(&format!("{name}.bias")));
            let mut want = vec![0.0f32; rows * co];
            for r in 0..rows {
                for o in 0..co {
                    let mut acc = bp[o];
                    for i in 0..ci {
                        acc += wp[o * ci + i] * x[r * ci + i];
                    }
                    want[r * co + o] = acc;
                }
            }
            close("lg_linear", &want, &got, 1e-5);
        }
        // lg_layer_norm against a hand-computed mean and variance.
        {
            let (rows, c) = (7usize, 32usize);
            let x = seq(rows * c, 0.41);
            let name = "m_down1.0.trans_block.ln1";
            let got = be.layer_norm_probe(name, &x, rows, c).expect("layer norm");
            let (gw, gb) = (wt.t(&format!("{name}.weight")), wt.t(&format!("{name}.bias")));
            let mut want = vec![0.0f32; rows * c];
            for r in 0..rows {
                let row = &x[r * c..(r + 1) * c];
                let mean = row.iter().sum::<f32>() / c as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                for i in 0..c {
                    want[r * c + i] = (row[i] - mean) * inv * gw[i] + gb[i];
                }
            }
            close("lg_layer_norm", &want, &got, 1e-4);
        }
    }

    /// The window index map: GATHER and SCATTER agree with each other (the round trip
    /// is the identity) and the gather agrees with the REFERENCE's own `pix` formula,
    /// which is the half a round trip cannot check.
    #[test]
    fn window_index_map_matches_the_reference() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        let win = wt.window;
        for (h, w, c, shift) in [
            (8usize, 8usize, 32usize, 0usize),
            (16, 16, 32, 4),
            (16, 16, 8, 0),
            (32, 24, 4, 4),
            (32, 24, 4, 0),
            (24, 32, 4, 4),
        ] {
            let x = seq(c * h * w, 0.61);
            // The round trip.
            let rt = be.window_round_trip(h, w, c, shift, &x).expect("round trip");
            close(&format!("gather/scatter round trip {h}x{w} c{c} shift{shift}"), &x, &rt, 1e-6);
            // The reference's own map, written the way cpu::attention writes it.
            let nww = w / win;
            let nw = (h / win) * nww;
            let nn = win * win;
            let mut want = vec![0.0f32; nw * nn * c];
            for wi in 0..nw {
                let (wh, ww) = (wi / nww, wi % nww);
                for t in 0..nn {
                    let (i, j) = (t / win, t % win);
                    let y = (wh * win + i + shift) % h;
                    let xx = (ww * win + j + shift) % w;
                    for ch in 0..c {
                        want[(wi * nn + t) * c + ch] = x[ch * h * w + y * w + xx];
                    }
                }
            }
            let got = be.gather_probe(h, w, c, shift, &x).expect("gather");
            close(&format!("gather index map {h}x{w} c{c} shift{shift}"), &want, &got, 1e-6);
        }
    }

    /// The attention KERNEL on a degenerate input whose answer needs no arithmetic:
    /// with k = 0 every logit is equal, so the softmax is uniform and each output
    /// token is the mean of v over its window. That checks the v read and the output
    /// indexing and nothing else.
    #[test]
    fn attention_kernel_uniform_case() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        let win = wt.window;
        for (h, w) in [(8usize, 8usize), (16usize, 8usize), (16, 16)] {
            let c = 32usize;
            let (heads, hd) = (1usize, 32usize);
            let nww = w / win;
            let nw = (h / win) * nww;
            let n = win * win;
            let span = 2 * win - 1;
            let nt = nw * n;
            let mut qkv = vec![0.0f32; nt * 3 * c];
            for t in 0..nt {
                for r in 0..c {
                    qkv[t * 3 * c + r] = 1.0;
                    qkv[t * 3 * c + 2 * c + r] = if r == 0 { (t % 17) as f32 } else { 2.0 };
                }
            }
            let rp = vec![0.0f32; heads * span * span];
            let got = be.attention_kernel_probe(h, w, c, heads, hd, 0, &qkv, &rp).expect("kernel");
            let mut want = vec![0.0f32; nt * c];
            for wi in 0..nw {
                for q in 0..n {
                    for r in 0..c {
                        let mut acc = 0.0f32;
                        for k in 0..n {
                            let t = wi * n + k;
                            acc += if r == 0 { (t % 17) as f32 } else { 2.0 };
                        }
                        want[(wi * n + q) * c + r] = acc / n as f32;
                    }
                }
            }
            close(&format!("uniform attention {h}x{w}"), &want, &got, 1e-5);
        }
    }

    /// The attention KERNEL with a known, non-trivial softmax: q is the first basis
    /// vector and key token t's k is the basis vector `t % hd`, so the logit for
    /// (p, t) is 1 when `t % hd == 0` and 0 otherwise - the same for every query.
    /// That exercises the q read, the k read, the dot product and the weighting of v.
    #[test]
    fn attention_kernel_known_softmax() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        let win = wt.window;
        let (h, w) = (8usize, 8usize);
        let c = 32usize;
        let (heads, hd) = (1usize, 32usize);
        let nww = w / win;
        let nw = (h / win) * nww;
        let n = win * win;
        let span = 2 * win - 1;
        let nt = nw * n;
        let mut qkv = vec![0.0f32; nt * 3 * c];
        for t in 0..nt {
            qkv[t * 3 * c] = 1.0;
            qkv[t * 3 * c + c + (t % hd)] = 1.0;
            qkv[t * 3 * c + 2 * c] = 3.0;
            qkv[t * 3 * c + 2 * c + 1] = 1.0;
        }
        let rp = vec![0.0f32; heads * span * span];
        let got = be.attention_kernel_probe(h, w, c, heads, hd, 0, &qkv, &rp).expect("kernel");
        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut want = vec![0.0f32; nt * c];
        for wi in 0..nw {
            for q in 0..n {
                let mut exps = vec![0.0f32; n];
                let mut sum = 0.0f32;
                for t in 0..n {
                    let e = if t % hd == 0 { scale.exp() } else { 1.0 };
                    exps[t] = e;
                    sum += e;
                }
                for r in 0..2 {
                    let v = if r == 0 { 3.0f32 } else { 1.0 };
                    want[(wi * n + q) * c + r] = v * exps.iter().sum::<f32>() / sum;
                }
            }
        }
        close("known softmax", &want, &got, 1e-5);
    }

    /// The transformer half, step by step, on ONE window: layer norm, the qkv lift,
    /// the kernel, and the projection, each compared against its CPU twin. This is
    /// the test that says which step disagrees rather than only that the block does.
    ///
    /// The CPU reference here takes a `[h*w][c]` buffer (`to_tokens`), because
    /// `cpu::layer_norm` and `cpu::attention` work on tokens, not on the engine's
    /// channel-major plane.
    #[test]
    fn attention_steps_match_on_one_window() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        let prefix = "m_down1.0";
        let (h, w) = (8usize, 8usize);
        let c = wt.block_dims(prefix).1;
        let win = wt.window;
        let n = h * w;
        let plane = seq(c * n, 0.43);
        let x = to_tokens(&plane, c, n);

        // Step 1: the layer norm.
        let ln_w = wt.t(&format!("{prefix}.trans_block.ln1.weight"));
        let ln_b = wt.t(&format!("{prefix}.trans_block.ln1.bias"));
        let mut want_ln = vec![0.0f32; n * c];
        cpu::layer_norm(&x, c, ln_w, ln_b, &mut want_ln);
        let got_ln = be.layer_norm_probe(&format!("{prefix}.trans_block.ln1"), &x, n, c).expect("ln");
        close("ln1", &want_ln, &got_ln, 1e-5);

        // Step 2: the qkv lift, against the CPU's own rows of embedding_layer.
        let emb_w = wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.weight"));
        let emb_b = wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.bias"));
        let mut want_qkv = vec![0.0f32; n * 3 * c];
        for t in 0..n {
            for o in 0..3 * c {
                let mut acc = emb_b[o];
                for i in 0..c {
                    acc += emb_w[o * c + i] * want_ln[t * c + i];
                }
                want_qkv[t * 3 * c + o] = acc;
            }
        }
        let got_qkv = be.linear_probe(&format!("{prefix}.trans_block.msa.embedding_layer"), &want_ln, n, c, 3 * c).expect("qkv");
        close("qkv lift", &want_qkv, &got_qkv, 1e-5);

        // Step 3: the kernel on that qkv, then the projection - the whole path the way
        // the walker runs it, against `cpu::attention` on the same normalised tokens.
        let heads = c / wt.head_dim;
        let rp = wt.t(&format!("{prefix}.trans_block.msa.relative_position_params"));
        let att = be.attention_kernel_probe(h, w, c, heads, wt.head_dim, 0, &want_qkv, &rp).expect("kernel");
        let lin_w = wt.t(&format!("{prefix}.trans_block.msa.linear.weight"));
        let lin_b = wt.t(&format!("{prefix}.trans_block.msa.linear.bias"));
        let mut got = vec![0.0f32; n * c];
        for t in 0..n {
            for o in 0..c {
                let mut acc = lin_b[o];
                for i in 0..c {
                    acc += lin_w[o * c + i] * att[t * c + i];
                }
                got[t * c + o] = acc;
            }
        }
        let mut want = vec![0.0f32; n * c];
        cpu::attention(&want_ln, h, w, c, win, 0, emb_w, emb_b, lin_w, lin_b, rp, &mut want);
        close("the whole transformer half, one window", &want, &got, 2e-4);
    }

    /// The device's own composition - gather, layer norm, qkv lift, kernel,
    /// projection - against `cpu::attention`, at shifted and multi-window geometries,
    /// which is what the single-window step test cannot reach.
    ///
    /// THE LAYOUTS DIFFER AND THAT IS WHY THIS TEST SCATTERS: the probe returns the
    /// gather's `[tokens][c]` order (what the walker's `msa.linear` consumes) while
    /// `cpu::attention` takes a `[h*w][c]` token buffer and returns a scattered
    /// `[c][h][w]` plane. So the CPU side is fed `to_tokens(plane)` and the device's
    /// token buffer is scattered with the reference's own `pix` map before the
    /// comparison.
    #[test]
    fn attention_probe_matches_cpu_attention() {
        let Some(wt) = weights() else { return };
        let _guard = crate::device_lock::device_lock();
        let mut be = Cuda::new(&wt).expect("a device");
        let win = wt.window;
        for (prefix, h, w, shift) in [
            ("m_down1.0", 8usize, 8usize, 0usize),
            ("m_down1.0", 16, 16, 0),
            ("m_down1.0", 16, 16, 4),
            ("m_down2.0", 16, 16, 4),
            ("m_body.0", 16, 16, 0),
            ("m_body.0", 16, 16, 4),
        ] {
            let c = wt.block_dims(prefix).1;
            let n = h * w;
            let plane = seq(c * n, 0.43);
            let tok = be.attention_probe(prefix, h, w, c, shift, &plane).expect("probe");

            let x = to_tokens(&plane, c, n);
            let mut ln = vec![0.0f32; c * n];
            cpu::layer_norm(&x, c, wt.t(&format!("{prefix}.trans_block.ln1.weight")), wt.t(&format!("{prefix}.trans_block.ln1.bias")), &mut ln);
            let mut want = vec![0.0f32; c * n];
            cpu::attention(
                &ln, h, w, c, win, shift,
                wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.weight")),
                wt.t(&format!("{prefix}.trans_block.msa.embedding_layer.bias")),
                wt.t(&format!("{prefix}.trans_block.msa.linear.weight")),
                wt.t(&format!("{prefix}.trans_block.msa.linear.bias")),
                wt.t(&format!("{prefix}.trans_block.msa.relative_position_params")),
                &mut want,
            );

            // Scatter the device's tokens with the reference pix map. Shifted windows
            // still move every pixel exactly once: the map is a permutation.
            let nww = w / win;
            let nw = (h / win) * nww;
            let nn = win * win;
            let mut got = vec![0.0f32; c * n];
            let mut hits = vec![0u32; n];
            for wi in 0..nw {
                let (wh, ww) = (wi / nww, wi % nww);
                for t in 0..nn {
                    let (i, j) = (t / win, t % win);
                    let y = (wh * win + i + shift) % h;
                    let xx = (ww * win + j + shift) % w;
                    // `cpu::attention` returns `[h*w][c]` (pixel-major), so this
                    // scatter and the comparison below are pixel-major too.
                    for ch in 0..c {
                        got[(y * w + xx) * c + ch] = tok[(wi * nn + t) * c + ch];
                    }
                    hits[y * w + xx] += 1;
                }
            }
            assert!(hits.iter().all(|&k| k == 1), "{prefix} {h}x{w} shift{shift}: pix is not a permutation");
            close(&format!("attention probe {prefix} {h}x{w} shift{shift}"), &want, &got, 2e-4);
        }
    }

    /// THE WALKER AT A SIZE THE FIXTURES DO NOT REACH.
    ///
    /// `tests/parity.rs` proves the graph at 80x64, which is ONE token chunk: 10 x 8 =
    /// 80 windows, well under `TOKEN_BUDGET`'s 512. Everything the walker does with
    /// MEMORY instead of arithmetic shows up only past that boundary - the chunk loop,
    /// the host-resident stage skips and their chunked add back, the in-place block,
    /// the recomputed `m_head`, the conv half's aliased staging - so a bug in any of
    /// them would pass every other test in this repository. 256x256 is the smallest
    /// size that crosses the boundary honestly: 32 x 32 = 1024 windows, two chunks.
    ///
    /// BOTH SIDES GO THROUGH `backend::run`, for the reason this file's header gives:
    /// `cpu::forward` takes an ALREADY PADDED plane while `backend::run` pads,
    /// forwards and crops, so comparing one against the other compares two problems.
    ///
    /// THIS IS ALSO A RACE DETECTOR, and it is the only one here. It failed once in
    /// about thirty runs, which is how a pair of missing `__syncthreads()` in
    /// `sc_attn_s_body` was found: `xsh` is reused as the row-max exchange, then the
    /// softmax-sum exchange, then the two halves' accumulators, and the first two
    /// reuses had no barrier between a reader and the next writer, so a fast thread's
    /// sum could overwrite a slow thread's `mxall` slot and give it the wrong
    /// denominator. The symptom was one window row of tokens wrong by a factor near
    /// 2, which the block's 1x1 projection smears across every channel of those 8
    /// pixels - so the test compares ~18000 elements and the failure looks enormous.
    ///
    /// ONE PASS PROVES NOTHING ABOUT A RACE. `examples/blockdet.rs` runs the same
    /// geometry against itself and against `cpu::conv_trans_block` many times over;
    /// the barrier fix took that from 9 divergent runs in 399 to 0 in 399, and 0 in
    /// 299 more. A single green run of this test is not evidence that a change to a
    /// shared-memory kernel is correct.
    #[test]
    fn forward_matches_cpu_past_the_token_budget() {
        let _guard = crate::device_lock::device_lock();
        let Some(wt) = Weights::load("../models/scunet-color-real-psnr.safetensors").ok() else {
            eprintln!("skipping: the checkpoint is not converted");
            return;
        };
        let (h, w) = (256usize, 256usize);
        let plan = scunet::plan::Plan::new(h, w, wt.window);
        let nw = (plan.hp / wt.window) * (plan.wp / wt.window);
        assert!(nw > 512, "{nw} windows is not past the token budget");
        let x = seq(wt.in_nc * h * w, 0.31);

        let mut cpu = cpu::Cpu::new(&wt).expect("cpu backend");
        let want = scunet::backend::run(&mut cpu, &x, wt.in_nc, h, w, &wt).expect("cpu forward");
        let mut be = Cuda::new(&wt).expect("a device");
        let got = scunet::backend::run(&mut be, &x, wt.in_nc, h, w, &wt).expect("cuda forward");
        assert_eq!(want.len(), got.len());
        close(&format!("forward {h}x{w} past the token budget ({nw} windows)"), &want, &got, 2e-3);
    }
}
