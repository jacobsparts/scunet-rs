//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/scunet.cu` (PROJECT_KERNELS). Each gets its own fatbin and its own
//! `--entries` list, and `src/cuda.rs` loads them as separate modules, so neither
//! can shadow a name in the other.
//!
//! Both lists are checked against the source they are compiled from before nvcc
//! runs, in BOTH directions: a name that is listed but not defined fails the
//! build, and so does a name that is defined but not listed - the latter would be
//! pruned from the fatbin and fail at launch instead.

/// The toolkit's kernels this graph launches.
///
/// SCUNet is a U-Net of Swin-conv blocks: three down stages of 3x3 convs plus a
/// stride-2 2x2 conv, a body, and three up stages that are a transposed
/// convolution followed by more of the same blocks. The 3x3 family is the bulk of
/// it - `lg_conv3x3s1p1` is `m_head`, the conv half of every block's conv_block,
/// and `m_tail`. `lg_conv3x3_winograd` is the same op where `c_in` is 32 or more,
/// which is every conv_block except the first stage's; it LOSES at `c_in = 3`,
/// which is `m_head` and only `m_head`, so the first layer stays on the direct
/// kernel and the rest use F(4,3).
///
/// SIX OF THESE WERE WRITTEN IN THIS ENGINE FIRST and moved into the toolkit once
/// they proved general, which is what the notes in `cuda/scunet.cu` record. They
/// carried their measurements over:
///
/// * `lg_linear_rb` and `lg_conv1x1_rb` are ONE register-blocked GEMM in the
///   token and the plane layout. It replaces `lg_linear` (2.03x) and `lg_conv1x1`
///   (10.25x, measured over the seven real stage geometries of
///   `examples/k1x1.rs`). The plain `lg_conv1x1`/`lg_linear` are still listed
///   because `SCUNET_1X1`/`SCUNET_GEMM` A/B the two against each other.
/// * `lg_layer_norm_warp` is `ln1`/`ln2` of every Swin block: one warp per row,
///   the row in registers, 2.2x the toolkit's two-pass `lg_layer_norm` at a
///   256-wide row and 8.6x at 32 wide. `lg_layer_norm` stays listed for the A/B.
/// * `lg_conv2x2s2` and `lg_conv_t2x2` are the stride-2 pair of the down stages
///   and the transposed pair of the up stages, register-blocked over output
///   channels: 1.72-6.49x and 2.39-2.60x the elementwise forms, which stay in
///   `cuda/scunet.cu` as the `SCUNET_C2X2=elem` arm.
/// * `lg_window_gather`/`lg_window_scatter` are the Swin index map, which the
///   toolkit now shares with `rmbg-rs`. The ATTENTION between them stays here.
///
/// `lg_gelu_erf` is the MLP's activation (timm/torch's default GELU, the erf
/// form), `lg_relu` is the conv half's activation, `lg_add` is every residual and
/// every skip, and `lg_copy` is the one copy the walker needs where the CPU
/// backend uses `copy_from_slice`.
const TOOLKIT_KERNELS: &[&str] = &[
    "lg_conv3x3s1p1",
    "lg_conv3x3_winograd",
    "lg_conv1x1",
    "lg_linear",
    "lg_layer_norm",
    "lg_linear_rb",
    "lg_conv1x1_rb",
    "lg_layer_norm_warp",
    "lg_conv2x2s2",
    "lg_conv_t2x2",
    "lg_window_gather",
    "lg_window_scatter",
    "lg_noop",
    "lg_gelu_erf",
    "lg_relu",
    "lg_add",
    "lg_copy",
];

/// This engine's own kernels, in `cuda/scunet.cu`.
///
/// WHAT IS LEFT IS THE ATTENTION AND ONE COMPARISON ARM, and both are deliberate.
/// The window attention is the architecture rather than a generic op - a thread
/// owns a whole query-head row, with the axis-wide mask and the learned
/// relative-position bias - so it has no place in a toolkit. The elementwise
/// 2x2 stride-2 convolution and its transposed twin are the `SCUNET_C2X2=elem`
/// arm: the toolkit's register-blocked pair is what the graph runs, and these are
/// what it was measured against and what the CPU twin was first matched to. A
/// counterexample that cannot be re-run is not a counterexample.
///
/// The window gather/scatter, the register-blocked GEMM, the warp LayerNorm and
/// the register-blocked 2x2 pair all started here and are toolkit kernels now;
/// see TOOLKIT_KERNELS above, and `cuda/scunet.cu`'s notes for what each move
/// kept and measured.
const PROJECT_KERNELS: &[&str] = &[
    // One block per window. A thread owns a whole query-head row: the same-axis
    // shifted-window mask, the learned relative-position bias, the softmax and the
    // weighted sum. `sc_window_attn_s` keeps k and v in shared memory and
    // `sc_window_attn_r64` holds the 64 scores and 32 query values IN REGISTERS
    // (the kernel it replaces indexed runtime bounds, which ptxas puts in local
    // memory - about 8.7 GB of local traffic per forward at 256x256).
    "sc_window_attn",
    "sc_window_attn_s",
    "sc_window_attn_r64",
    // The 2x2 stride-2 convolution that halves each down stage, torch's
    // `nn.Conv2d(..., 2, 2, 0)` - weight [c_out][c_in][2][2], no padding - and
    // `nn.ConvTranspose2d(2, stride=2)`, the up-sampler before each up stage, with
    // weight [c_in][c_out][2][2] (torch's transposed-convolution layout, the
    // reverse of the conv one) and a scatter with no tap flip. Both are the
    // elementwise forms; the graph runs the toolkit's register-blocked pair.
    "sc_conv2x2s2",
    "sc_conv_t2x2",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/scunet.cu");

    // `cargo build --no-default-features` is the pure-Rust CPU build: it must not
    // need nvcc, and `src/cuda.rs` (which includes the fatbins) is not compiled.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/scunet.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/scunet.cu").expect("read cuda/scunet.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/scunet.cu (it has {})",
            defined.join(", ")
        );
    }
    for d in &defined {
        assert!(
            PROJECT_KERNELS.contains(&d.as_str()),
            "cuda/scunet.cu defines `{d}`, which PROJECT_KERNELS does not list - \
             it would be pruned from the fatbin and fail at launch"
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "scunet_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/scunet.cu",
            out_name: "scunet_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
