# scunet-rs: engineering notes

The README is the short version, for someone who wants to denoise a photograph. This
is for someone who wants to change the walker, add a kernel, or take a measurement
and compare it with one already taken.

Everything below was measured on the machine this engine was written on - an 8 GB
GTX 1080 (`sm_61`, 20 SMs) that is **shared with other tenants and thermally
throttled** - which is why comparisons are quoted as ratios taken by alternating the
two things being compared inside one measurement window. Absolute milliseconds from
different windows do not compare and are not meant to.

Reproduce any of it with the `dev` feature and the `examples/` instruments:

```
cargo build --release --features dev,cuda
cargo run --release --features dev,cuda --example bench -- --device gpu --sizes 256,512,1024 --iters 10 --warmup 5
```

| instrument | what it measures |
| ---------- | ---------------- |
| `examples/bench` | end-to-end timing and footprint, per size and device |
| `examples/budget` | one op of each class at the geometry the graph runs it at, buffers resident, times the launch count |
| `examples/budget_cpu`, `examples/stages_cpu` | the CPU forward, by op and by stage of the walker |
| `examples/cpu_mem` | the CPU path's peak live memory and the live set at the worst moment |
| `examples/blockmarg`, `examples/blockdet` | the device's margin against its CPU twin, and whether the device is deterministic |
| `examples/k1x1`, `examples/kattn`, `examples/klinear` | one kernel at one geometry with resident buffers |
| `examples/geom` | the per-stage shapes of the graph |
| `examples/probe`, `examples/dump` | localise a divergence to an op, a block or a stage |

`SCUNET_SKIP=host|device` forces where the stage skips live, and is in release builds
because it is a memory policy. The other switches - `SCUNET_PROFILE`,
`SCUNET_MEMTRACE`, `SCUNET_GEMM`, `SCUNET_1X1`, `SCUNET_ATTN`, `SCUNET_LN`,
`SCUNET_C2X2` - exist to reproduce a kernel A/B, and a build without `dev` refuses
them **by name** rather than ignoring them.

## Contents

* [Performance, and the machine it was measured on](#performance-and-the-machine-it-was-measured-on)
* [Memory](#memory)
  * [Why PyTorch needs three times the memory for the same model](#why-pytorch-needs-three-times-the-memory-for-the-same-model)
  * [What a forward actually needs, and what was removed](#what-a-forward-actually-needs-and-what-was-removed)
  * [Will it fit? The guards that answer before the allocation](#will-it-fit-the-guards-that-answer-before-the-allocation)
* [Where the device time goes](#where-the-device-time-goes)
* [What the kernels cost, and what the fixes were worth](#what-the-kernels-cost-and-what-the-fixes-were-worth)
* [An intermittent race, and how it was found](#an-intermittent-race-and-how-it-was-found)
* [The kernels this engine keeps](#the-kernels-this-engine-keeps)
* [The checkpoint, and the tools around it](#the-checkpoint-and-the-tools-around-it)
* [The demo image](#the-demo-image)


## Performance, and the machine it was measured on

Measured against upstream PyTorch on the same checkpoint, the same input and the
same card, **interleaved in one window** - see the caveat at the end of this
section for why that matters. The README has the short table.

| size | PyTorch CUDA | scunet-rs CUDA | torch CPU (24 threads) | scunet-rs CPU |
| ---- | ------------ | -------------- | ---------------------- | ------------- |
| 128  |              |                | 83-89 ms               | 381-391 ms    |
| 256  | 103-113 ms   | 129-130 ms     | 387-1834 ms            | 1250-1290 ms  |
| 512  | 411-441 ms   | 484-488 ms     | 6045-19878 ms          | 5060-5160 ms  |
| 1024 | 1679-1691 ms | 1888-1900 ms   | 19.9-20.7 s            | 21.99 s       |

So the CUDA path is **1.1-1.25x** slower than PyTorch's - 1.15x at 256x256, 1.11x
at 512 and 1.12x at 1024 in the table above, at 544-602 GFLOP/s against the card's
~7 TFLOP/s fp32 at the clock it actually holds. The register-blocked GEMM, the window
attention and the skip policy are what the ratio rests on, each measured with an
A/B in one window.

**AT 2048x2048 THERE IS NO RATIO, because PyTorch CUDA cannot run it here.** On a
momentarily idle card (8111 MiB free), upstream OOMs twice - including with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which rules out fragmentation
- asking for a 512 MiB block with 117 MiB free and 7.67 GiB already allocated.
This engine completes the same 2048x2048 denoise in 8.2-9.0 s (8978-9001 ms
median across two runs in the same window) at a 5196 MiB high-water. PyTorch's
ceiling on this 8 GB card is **1536** (1792 also OOMs); at that size, interleaved
in one window at 1822-1607 MHz, ours is 4679-4686 ms against its 4077-4081
(1.15x), where ours holds 2788 MiB against its 6856. The memory section below has
the law behind that, and the reason is upstream's attention rather than the
network's size.

**The CPU ratio is not constant with size, and the reason is cache.** Ours costs
20.0-22.8 us per pixel at EVERY size - flat from 128 to 1024 - because its working
set exceeds this box's 30 MiB L3 even at 128x128 (1.2 KB/px against 64 KiB of
image). Torch is cache-resident at 128 (5.4 us/px, about **4.2x faster than
ours**) and memory-bound by 256 (20-29 us/px), so the two cross over: at
1024x1024 torch takes 19.9-20.7 s against our 22.0 s, **only 1.07-1.13x
faster**. Ours reaches 58 GFLOP/s at 256 and holds 52-54 at 768-1024; it is
genuinely threaded, and thread scaling is near-linear (4, 8 and 24 threads give
66.0, 37.6 and 23.6 s at 1024x1024). Torch's own CPU timing is reproducible at
128 and 1024 (83-89 ms and 19.9-20.7 s) but not at 256 or 512, where it spreads
929-1834 and 6045-19878 ms across runs.

**Host memory: ours holds a fraction of torch's at every size.** Peak RSS for the
whole process, one size per process, both engines in the same window:

| size | scunet-rs CPU | torch CPU | ratio |
| ---- | ------------- | --------- | ----- |
| 128  | 108.8 MiB     | 552.1     | 5.1x  |
| 256  | 199.5         | 755.2     | 3.8x  |
| 512  | 490.1         | 1354.5    | 2.8x  |
| 1024 | 1687.8        | 3713.2    | 2.2x  |

The walker's buffers are chunked and its blocks run in place, which is where that
headroom comes from; the accounting is in the Memory section. The device
comparison at 1024 is 1132 MiB high-water against torch's 3088.6
(`max_memory_allocated`), i.e. 2.73x. Within one engine, a 1024x1024 forward costs
each about the same multiple of its own 256x256: 12.4x the time on our CPU against
our GPU, 11.8x on torch's.

Where the CPU time goes, from `examples/budget_cpu.rs`
(the arms, at the geometry the graph runs them at) and `examples/stages_cpu.rs`
(the engine's own per-stage hook, whose spans sum to its measured time): the window
attention is 30% (gather + scores + softmax + weighted sum + scatter), the two 3x3
convolutions of every block 27% at 236 GFLOP/s, the two stride-2 convolutions and
the transposed one 20% between them, the four 1x1 projections and the mlp linears
22% at 274 GFLOP/s, GELU 4% and the transposes 0.3%. `RUSTFLAGS=-C
target-cpu=native` was measured and is worth only 1-3%, so the gap is the code's
shape and not its codegen - the kernel-level explanation is in
`examples/stages_cpu.rs`, which charges the walker's own allocations and clones to
the stage that makes them.

Three fixes, all element-wise or order-preserving so no result changed, took the
CPU forward from 3695-3840 ms to 1300 at 256x256 and from 955-976 to 381-391 at
128: GELU was single-threaded (296.3 ms for 2.1 GB/s, against the 1.9 ms its `relu`
twin takes over comparable data); the NCHW -> `[h][w][c]` transpose pair at every
block boundary was single-threaded (78.5 ms together); and conv3x3 and conv1x1
were restructured so the innermost loop is a contiguous slice rather than a stride
through the channel planes - 944.8 -> 195 ms (49 -> 236 GFLOP/s) and 265 -> 37 ms
(38 -> 274) - which keeps each output element's terms in their original order and
is therefore bit-identical, not merely close.

**THE MEASUREMENTS ARE CONDITIONED ON A SHARED, THROTTLED GPU, and the numbers
above are the clean-window ones.** `nvidia-smi` reports 100% utilization and
1706 MiB in use with nothing of this project running (other tenants, whose
process names read as `[Not Found]` without root), the card sits permanently at
93 C with `SW Thermal Slowdown: Active` (threshold 96 C), and its SM clock swings
between 835 and 1843 MHz against a 1911 MHz maximum. Identical code measured
1049/1058 ms at 256x256 in one window and 2687/2807 ms in another. The
consequences are practical:

* Absolute times are not comparable across moments, only within one.
* Every comparison here was taken by alternating the two engines within a single
  window and recording the clock, and the ratios are the result - not the
  millisecond values.
* `--warmup 5 --iters 10` is the minimum for a stable number. Short runs are
  noise: `--iters 3` gave 465 ms at 256x256 for code that measures 263.


## Memory

The device footprint is reported EXACTLY, as the checkpoint plus every buffer in
the scratch pool (`Backend::footprint`), not as a free-VRAM delta - the driver's
own free-memory figure moves with the other tenants' allocations and produced
numbers that disagreed by a factor of seven between two sizes of one run.

Every size, measured with `BENCH_MEM=1`, which prints the HIGH-WATER of device
bytes held at any single moment. That is the number that decides whether an image
fits, and it is not the same as `footprint` (checkpoint + pool): the pool counts
buffers held at different moments, and the `ConvTransBlock` scratch is allocated
outside it, so `footprint` understates the requirement by 1.8x at large sizes.

Measured with `SCUNET_SKIP=auto` - the default - one size per process, three
iterations after two warmups, in one window whose clock ran 1809 down to 1607 MHz:

| size | high-water MiB | min ms | MP/s | GFLOP/s | us/px | host RSS MiB |
| ---- | -------------- | ------ | ---- | ------- | ----- | ------------ |
| 64   | 7.5            | 15.6   | 0.24 | 287     | 3.81  | 190          |
| 128  | 30.0           | 39.1   | 0.40 | 477     | 2.39  | 191          |
| 256  | 92.0           | 142.0  | 0.46 | 544     | 2.17  | 192          |
| 384  | 172.0          | 307.4  | 0.48 | 565     | 2.08  | 197          |
| 512  | 284.0          | 529.7  | 0.49 | 585     | 2.02  | 202          |
| 768  | 604.0          | 1188.7 | 0.50 | 586     | 2.02  | 217          |
| 1024 | 1132.0         | 2076.2 | 0.50 | 597     | 1.98  | 238          |
| 1536 | 2788.0         | 4674.5 | 0.50 | 596     | 1.98  | 298          |
| 2048 | 5196.0         | 8947.3 | 0.47 | 554     | 2.13  | 1181         |

The device requirement is linear in pixels - **about 1.2 KB per pixel** (1230 MiB
per megapixel, fitted) - and the time is **flat at 2.0-2.1 us per pixel at every
size**, 2048x2048 included. The host RSS column grows because the skip policy puts
the skips on the device only while they fit; the host-skip fallback measures a
little lower at each of the sizes above (92.0, 1052.0, 2588.0 and 4828.0 MiB at
256, 1024, 1536 and 2048).

TWO THINGS CAN MAKE THAT TABLE LIE, and both are about when a number is taken
rather than about the engine. The first is a `ConvTransBlock` scratch that escapes
the accounting: the walker's `slot` is a local of `forward`, so Rust frees its
device memory when the function returns, and an accounting that counts it anyway
adds one 64-float-per-pixel plane a forward - 268 MiB at 1024x1024 - with
`nvidia-smi` holding the process at 1642 MiB throughout. The high-water is
identical at 1, 2, 4, 8 and 12 forwards, which is the check that catches it. The
second is a table that mixes kernel versions: the times above belong to the
kernels of the window they were taken in, and the window-attention and skip-policy
replacements moved the same sizes by 40%.


### Why PyTorch needs three times the memory for the same model

Measured peaks, both engines on the same checkpoint: ours 1.11 KB per pixel at
1024 (1132 MiB), 1.21 at 1536 (2788 MiB), 1.24 at 2048 (5196 MiB); torch CUDA
3.02 KB per pixel at 1024 (3088.6 MiB) and 2.98 at 1536 (6855.6 MiB, from
`torch.cuda.max_memory_allocated`). That is **2.46-2.73x ours**, and it is a
straight line: 11.9 GiB at 2048x2048 against the card's 7.92 GiB, which is exactly
the out-of-memory error observed. The cause is one line of upstream
(`tools/network_scunet.py`): its attention embeds qkv for **every window in the
plane at once** (`self.embedding_layer(x)` followed by `rearrange(qkv, 'b nw np
(threeh c) -> threeh b nw np c')`), and `sim` and `attn_mask` are
`[b, heads, nw, n, n]`. At 2048 there are 65536 windows, so the qkv tensor alone
is 3072 MiB at c=64, 6144 at c=128 and **12288 MiB at c=256**, before the two
1024 MiB score/mask tensors and a 256 MiB bool mask. This engine's walker takes
the window index in chunks of `TOKEN_BUDGET = 32768` tokens (512 windows), so it
never holds more than one chunk's tokens and the qkv buffer is a constant 6 MiB
regardless of image size. The chunking costs nothing arithmetically: attention is
per window and every residual is per token.

**2048x2048 runs**, at 8.2-9.0 s and 5196 MiB of device memory, and the caveat is
worth stating plainly: 5.2 GB of an 8 GB card, plus whatever the other tenants
hold, means a 2048 forward fits only when they are quiet. One attempt at 4.8 GB
free failed at the first allocation with `CUDA_ERROR_OUT_OF_MEMORY`; another at
7.1 GB free ran. `nvidia-smi` reported 12 MiB used before the run above and 1 MiB
after, so nothing of ours is left behind.


### What a forward actually needs, and what was removed

The high-water at 2048x2048 is 5196 MiB, from six changes, all arithmetically
exact. (Do not quote the `footprint` column instead: it reads 3716.5 MiB at that
size and is a different quantity - see the note above the table.)

* **Blocks run in place.** `conv_trans_block` writes its result over its input,
  which is well defined because its only read of the input is the final residual
  add. That removed a second full-resolution plane and a whole-plane copy per
  block.
* **`m_head` is recomputed rather than held.** Its output is needed only at the
  very end, so it is recomputed from the 3-channel input for `m_tail`: 0.3% of
  the network's FLOPs, in exchange for a full-resolution plane (1073 MiB at 2048).
* **The conv half's staging lives inside the projection's buffer.** The two
  staging planes are dead weight: the conv half finishes long before `conv1_2`
  writes, and 2*(c - trans) <= c holds for every released checkpoint, so the fit
  is exact and the pair costs nothing.
* **The stage skips live in HOST RAM.** A skip-connected network holds the down
  pass's outputs for the whole up pass, and SCUNet's three skips are 56 floats
  per input pixel - 896 MiB at 2048x2048. They are written once and read once, so
  they are streamed back in 16 MiB chunks when the matching up stage consumes
  them. This is exact: the add is element-wise over disjoint ranges, so every
  output element is added once and no sum is reordered.
* **A replaced scratch is dropped before its replacement is allocated**, instead
  of both being alive across one allocation.
* **The pool is best-fit**, so a small stage can borrow a larger stage's plane
  rather than each size keeping its own set resident.

TWO OF THESE HAVE A SHARP EDGE. The stage skip's `cur` must be given back
ONLY AFTER ITS REPLACEMENT IS TAKEN: give it back first and the pool hands the
same buffer straight back, which turns the copy that separates input from output
into a self-copy. And the pool's retention is what makes `footprint` a misleading
number - if you are deciding whether an image fits, read the high-water.


### Will it fit? The guards that answer before the allocation

A backend that does not ask fails in one of two ways, and neither is one a caller
can act on.

The CPU path has **no fallible allocation anywhere** - no `try_reserve`, every
buffer a `vec![0.0; n]` - so a pass larger than the machine does not return an
error, it ABORTS: under `ulimit -v 2097152` a 1024x1024 forward exits with code
134 (SIGABRT) and the single line `memory allocation of 268435456 bytes failed`,
naming no image size, no stage and no remedy. The CUDA path reports a
failed allocation well (`Cuda::buf_at` names the request, the high-water and the
pool), but only after the card has already refused - and on a card this engine
shares, that is after another tenant has taken the bytes a policy was counting on.

`src/memguard.rs` asks first, and answers against numbers this engine has
**measured** rather than against a sum over the graph's buffer names. The family's
other engines each have one (`ifan-rs`, `maxim-rs`, `nafnet-rs`, `nightenh-rs`);
this is the same shape, fitted to this walker.

| | model | measured peak | |
| - | ----- | ------------- | - |
| CPU | `(6*dim + in_nc)` floats a PADDED pixel | 97.5 / 390.0 / 1560.0 MiB at 256 / 512 / 1024 | within 1%, same figure per pixel at every size |
| CUDA | 1210 bytes a pixel + 32 MiB | 92.0 / 1052.0 / 2588.0 / 4828.0 MiB at 256 / 1024 / 1536 / 2048 | over by 1-18%, never under |

The CPU figure is DERIVED, not tuned: the worst moment is `m_tail`, whose live set
is `cur` (dim) + `head` (dim, recomputed rather than held) + `out` (in_nc) + the
block scratch `y + proj + tok + bt` = 3*dim - so `6*dim + in_nc` = 387 floats =
1548 bytes a pixel. The device figure is the SMALLEST a forward can need, because
the skip policy only ever moves the skips further onto the device, never off it -
and it is the same function the skip policy itself calls
(`crate::memguard::device_need`), so policy and guard cannot disagree about what a
size costs. A guard built on the default `auto` measurements instead - 1132 / 2788
/ 5196 - would refuse every image between the two figures.

`check_cpu` compares against the SMALLER of `/proc/meminfo`'s `MemAvailable` and
this process's remaining address space from `/proc/self/limits` + `/proc/self/statm`,
because `RLIMIT_AS` does not move `MemAvailable` and `ulimit -v` is exactly the
case above. `MemAvailable` rather than `MemFree`: free memory excludes reclaimable
page cache, and a guard that refuses passes which fit is an outage that does not
look like one. Both checks FAIL OPEN when the figure cannot be read, and both are
called from `backend::run` - the one entry point the CLI, `tests/` and the fixture
checker share - before the padding allocation.

```
$ ( ulimit -v 2097152; scunet -m model.safetensors -i 1024.png -o o.png --device cpu )
scunet: not enough HOST memory for a 1024x1024 pass on the CPU
scunet: the padded plane is 1024x1024 pixels, so it needs about 2.00 GiB (1.51 GiB
of activations held at the worst moment + 32 MiB of workspace, plus 25% for
allocator slack, and the 69 MiB checkpoint)
scunet: 376 MiB is available of 32.00 GiB, and the machine's swap is not counted -
a pass that goes there does not fail, it stops being usable
scunet: that figure is a MODEL of this walker's buffers fitted to measured peak
RSS, not a live measurement
scunet: a smaller image, the CUDA engine if this build has one, or more memory is
what fits
$ echo $?
1
```

The 376 MiB is the address-space term doing the work - `MemAvailable` alone would
have said 32 GiB. The other half of a guard's contract is not becoming an outage:
128x128 under the SAME cap still runs, and uncapped the same 1024x1024 pass runs
in 21.99 s. Both directions are unit-tested (`cargo test --lib`), including the
refusal firing and naming its numbers and remedies, and a pass that fits not being
refused - the failure mode a guard that is too aggressive would present.


## Where the device time goes

`Cuda::run` is the single launch choke point, so with `SCUNET_PROFILE=1` it
records a CUDA-event pair around every launch and `BENCH_PROFILE=1` prints the
totals - one synchronise at the end, so the pipeline is never drained. On this
machine `ncu` is unusable for perf work (GPU performance counters are denied,
`ERR_NVGPUCTRPERM`), which is why the breakdown lives in the launch path.

**READ THAT PROFILE FOR RANKING ONLY, NOT FOR ABSOLUTE TIME.** An event pair
costs ~4 us of measured span around a kernel that may take 2 us, so the profile
inflates short kernels by 1.3-1.6x, and at 256x256 that is most of them. It
assigned `lg_linear` 52 ms per forward where timing the same 28 launches with
resident buffers gives 33.8 ms.

`examples/budget.rs` is the number to optimize against: it times one op of each
class at the geometry the graph runs it at, with buffers resident and the pipeline
drained once at the end, and multiplies by the launch count. At 256x256, with the
clock at 1835 MHz:

| op | ms | launches | GFLOP/s | GB/s | % of sum |
| -- | -- | -------- | ------- | ---- | -------- |
| conv_block 3x3 x2 | 18.9 | 56 | 2426 | | 22.1 |
| attention (window) | 7.9 | 28 | | | 9.2 |
| linear mlp.2 | 6.4 | 32 | 1598 | | 7.5 |
| linear mlp.0 | 6.1 | 32 | 1674 | | 7.1 |
| conv2x2s2 (stride 2) | 5.8 | 3 | 559 | | 6.7 |
| conv1_1 (1x1 c->c) | 5.6 | 28 | 1826 | 57 | 6.5 |
| conv1_2 (1x1 c->c) | 5.6 | 28 | 1828 | 57 | 6.5 |
| linear qkv | 4.9 | 32 | 1574 | | 5.7 |
| m_head 3x3 x2 | 4.6 | 2 | 99 | | 5.3 |
| conv_t2x2 | 4.3 | 3 | 747 | | 5.0 |
| gelu | 2.8 | 32 | | 224 | 3.2 |
| layer_norm x2 | 2.4 | 64 | | 262 | 2.8 |
| linear msa.linear | 2.1 | 32 | 1188 | | 2.5 |
| scatter / gather | 1.7 each | 28 | | 90 | 2.0 each |
| the three adds | 0.8-1.6 | | | 300-330 | 4.3 |
| relu | 0.7 | 28 | | 210 | 0.9 |
| **sum of sampled ops** | **85.5** | 577 | | | |

The launch floor is not a term: `lg_noop` measures 2.1 us a launch, so 577
launches are 1.2 ms. The sum is below the measured 142 ms forward because each
arm is timed with only its own buffers in the pool - a slightly colder cache than
the real forward gives it - and because the three stage skips cross PCIe. The
remaining targets in that table are the two 3x3 convolutions (22%, already the
toolkit's winograd), `m_head`'s two 3x3 at 5.3% but only **99 GFLOP/s** (a
three-channel input, so it is layout-bound rather than compute-bound), and the
streaming kernels - GELU at 224 GB/s, LayerNorm at 262 and gather/scatter at 90 -
against the card's ~320 GB/s.


## What the kernels cost, and what the fixes were worth

Each of the kernels below was found and verified by timing it ALONE with its
buffers resident (`examples/budget.rs`, `Cuda::bench_op`) rather than through a
forward pass. That distinction is not pedantry: a probe that uploads its input and
downloads its output on every call moves 16.8 MB each way at the first stage's
geometry, around a launch that takes 4.6 ms, and one reported 23 ms for that
kernel. A measurement dominated by PCIe is worse than no measurement.

Each replacement is A/B-able against the form it replaced. Most of them are now
TOOLKIT kernels, promoted out of this engine once they proved general: the
toolkit carries both members of each pair, so the switch chooses between two
kernels that live in `lightgpu/cuda/kernels.cu`, and the engine keeps only the
window attention and the two elementwise 2x2 forms below.

* **The GEMM** (`lg_linear_rb` for the token linears and `lg_conv1x1_rb` for the
  1x1 convolutions, replacing `lg_linear` and `lg_conv1x1`; 21% end to end,
  `SCUNET_GEMM=plain` and `SCUNET_1X1=plain` select the plain forms). The
  toolkit's 1x1 reads each activation once per output channel; the tiled
  `lg_linear` fixed the traffic but left the tile at 16x16 with one accumulator
  per thread. A register tile is the obvious next step and the first attempt was
  27% SLOWER: an 8x8 tile needs 240 registers, so 256 threads a block exceed the
  SM's 65536 and it runs ONE BLOCK PER SM. The chosen shape is 4x4 at 124
  registers, two blocks an SM. Register counts against 31 for the old kernel: 79
  (2x4), 102-124 (4x4), 127 (2x8), 164 (4x8), 190 (8x4), 240 (8x8).
  **Occupancy, not the shared-load ratio, is what binds at this register file
  size.** Two further details mattered: the shared tile stride must be `BK + 1`
  (odd), because at stride 20 with a 4-row tile the addresses are 80 apart and
  80 mod 32 = 16 puts sixteen threads on two banks; and the thread-role assignment
  has to be layout-dependent, with rows on `threadIdx.x` for the plane layout so
  `out[o * rows + r]` coalesces. ONE KERNEL BODY NOW SERVES BOTH LAYOUTS
  (`lg_rb_body<4, 4, 16, PLANE, BIAS_FIRST>`), because a 1x1 convolution over
  `[c_in][pixels]` and a linear over `[rows][c_in]` are the same matrix multiply.
  Summed over the seven geometries `examples/k1x1.rs` walks, the register-blocked
  GEMM is 10.25x the plain 1x1 and 2.03x the plain token linear.
* **The window attention** (`sc_window_attn_r64`; 16% end to end,
  `SCUNET_ATTN=runtime`). This was the largest single cost - 24.6 ms of a 102 ms
  sampled forward, at 316 GFLOP/s, under 4% of peak - and neither global nor shared
  bandwidth explained it, because every shared read in it is a broadcast.
  `nvcc -Xptxas -v` gave the answer: `sc_window_attn_s` had a **512-byte stack
  frame** with 48 registers and zero spills, so `qr[hd]` and `logits[n]`, indexed by
  RUNTIME bounds, lived in local memory - about 8.7 GB of local traffic per
  forward. Templating on (n=64, hd=32) makes the bounds compile-time (0 stack
  frame, 0 spills, 186 registers), and because templating alone still spilled at
  255 registers the score vector is split across two threads, `q = tid & 63` and
  `half = tid >> 6`, which keeps every lane of a warp on the same k/v element so
  the broadcast survives. Dot products, bias, mask and `__expf` are bit-identical;
  only the softmax denominator and the weighted sum are reassociated into two
  32-term halves. THIS ONE STAYS: it is the architecture, not a generic op.
* **The two 2x2 convolutions** (`lg_conv2x2s2`, `lg_conv_t2x2`, replacing the
  elementwise `sc_conv2x2s2`/`sc_conv_t2x2`; 9.7%, `SCUNET_C2X2=elem` selects the
  elementwise forms). Register-blocked over OUTPUT CHANNELS, eight accumulators a
  thread. With stride 2 and a 2x2 kernel there is no spatial halo to reuse, so the
  reuse is over channels. The two elementwise forms stay because they are the
  comparison arm and because the CPU twin was first matched to them.
* **LayerNorm** (`lg_layer_norm_warp`, replacing `lg_layer_norm`; 2.6%,
  `SCUNET_LN=block` selects the block-per-row form). One warp per row with
  `__shfl_down_sync` reductions and the row held in registers, up to 128 wide.
  At a 256-wide row it is 2.2x the block form and at 32 wide 8.6x: the rows a
  transformer normalizes are narrow, where a shared-memory reduction tree has
  nothing to amortise its two barriers against.
* **The window index maps** (`lg_window_gather`/`lg_window_scatter`, shared with
  `rmbg-rs`). One thread per (window, token, channel) with the cyclic shift as a
  modulo wrap, so the same pair serves every engine's window assembly. They
  measured 90 GB/s against the card's ~320, and the parity fixture is what proves
  the map.
* **The skip policy** (8.2%): see the Memory section. `SCUNET_SKIP=auto|host|device`,
  a per-stage decision against measured free VRAM. At 256x256 auto measures
  129.3-129.5 ms against forced-host 140.9-142.0 in alternating pairs at
  1847/1835 MHz.

The engine's own register-blocked 1x1, `sc_conv1x1_rb`, is the negative result that
made the above possible and is recorded here rather than kept in the tree: its
arithmetic was bit-identical to the tiled `sc_conv1x1_t` and it was 19x slower at
every geometry, which is what showed the problem was the memory access pattern
rather than the load count. `examples/k1x1.rs` then measured the plane layout of
the GEMM above against both of them: 1.345 ms a forward against 3.075 for the
tiled form and 55.723 for the register-blocked 1x1.

Two layout decisions in the walker are load-bearing rather than stylistic, and
neither is visible at 256x256:

* **The transformer half takes the window index in chunks** of `TOKEN_BUDGET =
  32768` tokens. Gathering every window in the plane at once makes the token
  buffers grow with the image - at 1024x1024 the first stage would hold 4*64 = 256
  floats per pixel for the MLP hidden layer alone, 1024 MiB, and 4352 MiB of
  scratch - and 960x960 is where that stops fitting on an 8 GB card. Chunking
  costs nothing arithmetically, since attention is per window and every residual
  is per token.
* **Pixel tiles go on `grid.x` and channels on `grid.y`**, because `grid.y` is
  capped at 65535 and a 1024x1024 plane is exactly 65536 tiles - one tile past
  the limit, which is a launch failure rather than a slow path.


## An intermittent race, and how it was found

`tests/cuda_ops.rs::block_chunks_beyond_the_token_budget` failed ONCE in about
thirty runs and then passed fifteen suites in a row. A test that does that is
either a knife-edge tolerance or a real race, and the two are told apart by the
MARGIN, which the assertion did not print - so `examples/blockmarg.rs` now does.
The margin was 1.2e-6 to 2.7e-6 of scale against a 3e-5 tolerance, eleven to
twenty-five times inside it, and bit-identical across repetitions. Not a
tolerance problem.

`examples/blockdet.rs` compares the device against ITSELF over hundreds of
repetitions, and that is what settles it: the device is the nondeterministic side
and the CPU twin is bit-stable. Only `m_body.0` at 576x64 - the one geometry in the test with more than one
token chunk - ever diverged, and only its `shift 0` case appeared to fail; removing the fix again showed `shift 4` failing too, 1 run in 399 against 8
for `shift 0`, so the shifted case is rarer rather than immune and a rate
difference between two configurations is not evidence that one of them is sound.
Decomposing the bad elements as
NCHW - `element i` is channel `i / n` at pixel `i % n`, and dividing by `c`
instead reports channel and pixel ranges that name nothing - gave the signature
that identified the kernel: **eight consecutive pixels in one row with all 512
channels wrong**. One window's row of tokens had
gone bad, and the block's 1x1 projection then smeared it across every channel.

That is `sc_attn_s_body`, shared by `sc_window_attn_r64` (the default) and
`sc_window_attn_s` (the runtime-bounds fallback) - and NOT by `sc_window_attn`,
the `row` variant, which never failed. It reuses one shared array, `xsh`, three
times: for the row maxima, then the softmax sums, then the two halves'
accumulators. Every thread READS both halves of each use (`xsh[q]` and
`xsh[N + q]`) while every thread WRITES only its own slot, so each reuse needed a
barrier between the previous use's read and the next use's write - and two of
those barriers were missing. A fast thread's sum could land in a slow thread's
`mxall` slot, giving that thread the wrong softmax denominator, which is why the
observed relative error was close to 2.0: a value divided by roughly the wrong
normaliser.

Two `__syncthreads()` added, and the reproduction goes from **9 divergent runs in
399 to 0 in 399**, then 0 in 299 more. That was checked in both directions rather
than once: removing the two barriers again brings the failure straight back, 8 of
399 and 1 of 399 on the two geometries, so the barriers are the cause and not a
change in timing. The cost is not measurable - 512x512 measured 529.0-529.9 ms
against 528.8-529.8 before the fix.

`tests/cuda_ops.rs::block_repeats_are_bit_identical` is the regression test, and
it is deliberately cheap: it runs one block a dozen times and compares the device
against ITSELF, with no CPU twin in the loop. It catches this bug about a quarter
of the time - a race cannot be caught deterministically by a short test - so the
detector of record is `cargo run --release --features cuda --example blockdet --
400`, which is what found it.

The lesson is not about barriers. It is that **a single green run of a parity
test says nothing about a shared-memory kernel**, and that the arithmetic a race
produces here is not noise - it is a plausible-looking number that a 2e-3
end-to-end tolerance absorbs. A sweep of the project's other shared-memory sites
finds them correct: the toolkit's tiled 1x1 and its register-blocked GEMM each
bracket their tile with write-barrier-read-barrier, `lg_conv1x1_rb`/
`lg_linear_rb` do the same around their staged tiles, and `sc_window_attn` stages
k and v once and only reads them afterwards.


## The kernels this engine keeps

Almost nothing, and that is the point. The general kernels are the toolkit's: the
winograd and direct 3x3 convolutions, the 1x1 convolution, the tiled linear and
its register-blocked form, channel LayerNorm and its warp form, the stride-2 2x2
convolution and its transposed twin, the window index maps, erf GELU, ReLU,
residual add and copy.

What `cuda/scunet.cu` still holds, and `PROJECT_KERNELS` in `src/cuda.rs` is the
list the build checks, is the TEMPLATED WINDOW ATTENTION (`sc_window_attn_r64`,
64 tokens by 32-wide heads, two threads per query token, with `sc_window_attn` and
`sc_window_attn_s` as its runtime-bounds fallbacks) and the two ELEMENTWISE 2x2
convolutions (`sc_conv2x2s2`, `sc_conv_t2x2`), which are the `SCUNET_C2X2=elem`
comparison arm.

The other five families were written here first and are toolkit kernels now - the
window gather/scatter, the register-blocked GEMM, the warp LayerNorm and the
register-blocked 2x2 pair - and each was measured against what it replaced before
it moved. The measurement is why they moved rather than the fact that they work:
a kernel that is 2-10x an op the toolkit already has belongs where every engine
can reach it. The switches that reproduce those A/Bs
(`SCUNET_1X1=plain`, `SCUNET_GEMM=plain`, `SCUNET_ATTN=runtime|row`,
`SCUNET_C2X2=elem`, `SCUNET_LN=block`) now select between two kernels of the
TOOLKIT, except for the two elementwise 2x2 forms, which are still this engine's
and are the reason `PROJECT_KERNELS` is not empty.

`build.rs` checks both name lists against the source in both directions, so a
kernel that is defined but not listed fails the build instead of disappearing from
the fatbin.


## The checkpoint, and the tools around it

`../models/scunet-color-real-psnr.safetensors` - the colour real-PSNR model,
`in_nc 3`, `dim 64`, `window 8`, `config [4,4,4,4,4,4,4]`, head width 32.
`tools/convert.py` builds it from the official `.pth`.

* `tools/reference.py` - the CPU reference the fixture checker and `tests/`
  compare against.
* `tools/compare.py` - the authority on torch agreement. It runs the vendored
  upstream module itself, hooks the nine named stages (plus the trailing
  stride-2 convolution as `m_down*_blocks`, the leading `ConvTranspose2d` as
  `m_up*_up` and the input to `m_tail` as `m_tail_in`) and diffs them against
  `examples/dump.rs`'s planes, plane by plane:

  ```
  cargo run --release --example dump -- color_real_psnr_64.bin /tmp/rustdump
  /home/jacob/torchenv311/bin/python tools/compare.py --dump /tmp/rustdump
  ```

  On the 64x64 fixture every plane agrees to **4.0e-05** and the whole forward to
  **1.3e-06**; on the 80x64 one to **1.6e-06**. That figure is the one
  `tests/parity.rs` and `src/cpu.rs` quote, and until this script existed nothing
  in the tree could reproduce it.
* `tools/bench_torch.py` - times upstream on CPU or CUDA from the same input,
  rebuilding the model per size because `input_resolution` is baked into the
  window attention. **`input_resolution` is a TRAP**: both `ConvTransBlock` and
  `Block` do `if self.input_resolution <= self.window_size: self.type = 'W'`, so
  a stage whose resolution collapses to one window silently loses its shift - at
  64x64 the body sits at 8x8 and passing the padded size forces all four body
  blocks unshifted, which is a 4.4 disagreement that looks like an engine bug and
  is not one. Upstream's own test scripts leave the 256 default, and so does
  `tools/compare.py`. `tools/network_scunet.py` is the vendored upstream with one
  documented edit (its `timm` import, replaced by `tools/timm_shim.py`).
* `examples/bench.rs` - end-to-end timing and footprint. `examples/k1x1.rs`,
  `kattn.rs` and `klinear.rs` time one kernel at one geometry with resident
  buffers; `examples/geom.rs` prints the per-stage shapes; `examples/probe.rs`
  and `dump.rs` localise a divergence to an op or a block.

```
cargo run --release --features cuda --example bench -- --device gpu --sizes 256,512,1024 --iters 10 --warmup 5
```


## The demo image

`docs/before-after.png` is built from a clean frame, so its PSNR figures are
measurements rather than claims. The recipe, so it can be rebuilt:

* a 512x512 crop of `/home/jacob/fox.png` taken from `(448, 60)`;
* noise added in sRGB from numpy's default RNG seeded with `20260926`, with a
  shadow-weighted luminance term (`sigma = 7 + 16 * (1 - luminance)`) plus a
  half-strength chroma term, which measures **23.11 dB** against the clean crop;
* the same checkpoint run by this engine and by upstream PyTorch, both at
  **33.14 dB** against the clean crop, and **94.08 dB against each other** - a
  1-LSB maximum deviation, so the figure doubles as an end-to-end check. The CPU
  and GPU outputs of this engine agree to 93.11 dB.

The noise itself, in full - the rest is rendering:

```python
rng = np.random.default_rng(20260926)
c = np.asarray(clean).astype(np.float64)          # the 512x512 crop
lum = c.mean(axis=2, keepdims=True) / 255.0
sigma = 7.0 + 16.0 * (1.0 - lum)                 # noisy in the shadows
noise = rng.normal(0.0, 1.0, c.shape) * sigma
chroma = rng.normal(0.0, 1.0, c.shape[:2] + (1,)) * (sigma * 0.5)
noisy = np.clip(c + noise * [1.0, 0.95, 1.05] + chroma * [-0.5, 1.0, -0.5], 0, 255)
```

The two panels are rendered at 1:1, and the zoom pair at 2x nearest-neighbour so
the noise and what is left of it are visible rather than averaged away.
