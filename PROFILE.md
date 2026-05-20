# Profile findings — moe-200m-qwen3 throughput rescue

Profiled on `CUDA_VISIBLE_DEVICES=0` (single V100, no DDP). Same B=8,
S=2048, fp16 autocast, MoE selective ckpt + chunked CE w/ ckpt as the
running config. Per-component times measured with `cuda.Event`.

**Skipped `torch.profiler.profile()` after two 13-min hangs.** With
`profile_memory=True` + `record_shapes=True` the `key_averages().table()`
call blocked indefinitely; even with both turned off the schedule never
fired the trace handler in 3 min. Manual `cuda.Event` timing across
identified components gave a usable breakdown in <1 min.

## Top-line

| metric                            | value      |
|-----------------------------------|------------|
| baseline 4× V100 tok/s (in log)   | 13.7K      |
| baseline single-GPU tok/s         | 7.2K       |
| single-GPU full step              | 2261 ms    |
| single-GPU forward                | 574 ms     |
| single-GPU **backward + opt**     | **1687 ms** (75% of step) |
| peak GPU memory (single GPU)      | 24.2 GB    |

The 4-GPU run's 30.3 GB peak vs single-GPU 24.2 GB → ~6 GB of DDP buckets
(`bucket_cap_mb=100`) on top of the model. Free headroom on 4× V100 is
~1.5 GB — explains why chunk=2048/4096 OOMs in DDP mode.

## Forward breakdown (per step, 574 ms)

| component                | ms      | %    | per layer       |
|--------------------------|---------|------|-----------------|
| **15× MoE FFN**          | **326** | **57** | 21.7 ms/layer   |
| **CE (chunked, ckpt=on)**| **168** | **29** | 32 chunks       |
| 16× attention (SDPA)     | 73      | 13   | 4.6 ms/layer    |
| 1× dense FFN (layer 0)   | 6.6     | 1    | —               |
| embed lookup             | 0.2     | <1   | —               |

## Bottleneck #1 — MoE FFN (326 ms fwd, ~650 ms inc. bwd recompute)

`model.py:271` `_moe_dispatch_bmm`:

- **Python for-loop over 32 experts**, each doing 3 separate `F.linear` calls
  (gate, up, down) plus an `index_select` + `index_add_`. 96 cuBLAS launches
  per MoE layer × 15 layers = **1440 launches per forward**, ×2 for the
  bwd recompute (selective-ckpt wraps the whole MoE).
- **2× GPU→CPU sync per layer** at `offsets.cpu().tolist()` and
  `counts.cpu().tolist()`. 60 syncs per step.
- **Dead-expert "dust pass"** runs all 32 experts on a 1-token tensor every
  step (DDP `find_unused_parameters=True` legacy). 96 extra launches per
  layer × 15 = 1440 wasted launches/step.
- Per-expert math at mean load 1024 tok: 3×1024×640×1024 ≈ 6 GFLOPs. At
  V100 ~30-45 TFLOPs/s effective on this shape that's ~150 µs of pure
  compute, but the dispatch + launch + sync chain stretches it to 700 µs
  in practice (21.7 ms ÷ 32 experts).

Expected fix: stack expert weights to `[E, d_ff, d_model]` and use 3
`einsum`/`bmm` calls per layer instead of 96. With ~25% capacity-padding
overhead but ~95% fewer launches, target is ~5 ms/MoE layer
(15 × 5 = 75 ms fwd, **down from 326**). Save ~250 ms × 2 (fwd+bwd
recompute) ≈ 500 ms.

## Bottleneck #2 — chunked CE (168 ms fwd, ~330 ms inc. bwd recompute)

`model.py:371-415` `tiled_cross_entropy` with `chunk_size=512`:

- 32 chunks per CE call. Each chunk: `F.linear(h_chunk[512, 640],
  embed_weight[151936, 640])` → `[512, 151936]` logits → `.float()` cast
  (311 MB transient) → `F.cross_entropy(reduction='sum')`.
- Wrapped in `torch.utils.checkpoint`, so backward recomputes every
  chunk's forward in addition to its actual backward. Doubles CE compute.
- Larger chunk (1024, 2048, 4096) **does not help** — bench shows
  chunk=2048 is 10% **slower** (6.5K vs 7.2K tok/s), and chunk≥4096 OOMs
  on the resident fp32 logits buffer (`chunk × 151936 × 4` bytes).
- Disabling `use_checkpoint` requires keeping every chunk's logits buffer
  resident across the full backward — OOMs at chunk=1024.

The 311 MB-per-chunk `.float()` allocation × 32 chunks is the resident
memory cost; that buffer is what blocks bigger chunks. A fused
linear+CE kernel (Liger-style, or a Triton kernel that does
`log_softmax + nll` without materializing logits) would let us drop the
checkpoint AND increase chunk size. Out of scope for this rescue.

Realistic CE fix: keep chunk=512 + ckpt=on (~150 ms forward), but if
backward-pass memory allows after the MoE fix, try `chunk=1024` again on
4× V100 (DDP changes the resident-tensor lifetime).

## Bottleneck #3 — attention is fine (73 ms fwd, ~150 ms inc. bwd)

`F.scaled_dot_product_attention` is already wired (not unfused PyTorch
as the prompt suggested). On V100 this falls back to memory-efficient
attention (no FA2/3). 4.6 ms per layer for B=8, S=2048, H=10, D=64 is
reasonable (~21 GFLOPs/layer at ~50% peak). FA-Volta might gain
~20-30% but not 5×; skip until MoE+CE are fixed.

## Bottleneck #4 — backward overhead from double-checkpoint (1100+ ms)

MoE selective ckpt + CE ckpt together mean the **entire forward gets
recomputed inside the backward pass**. Roughly:

| phase                      | ms (est) |
|----------------------------|----------|
| forward                    | 574      |
| forward recompute (in bwd) | 510      |
| actual bwd grad compute    | 1100     |
| optimizer step + scaler    | 77       |
| **total**                  | **2261** |

Selective ckpt saves ~5 GB of activation memory but costs ~510 ms/step
in recompute. With the MoE-dispatch fix, the recompute itself gets
~3× faster too — so even keeping ckpt on, the bwd shrinks.

## Plan, in order of expected ROI

1. **MoE dispatch refactor** (grouped einsum/bmm, no .cpu() sync, no dust
   pass). Target: 326 ms → ~80 ms fwd. Bwd savings similar.
   *Est. tok/s gain: +60%* (single-GPU 7.2K → ~11.5K, 4-GPU 13.7K → ~22K).
2. **Drop dust pass + drop `find_unused_parameters=True`** (DDP only;
   relies on `_set_static_graph()` which is already on). Saves DDP
   sync overhead and 96 noop launches/layer.
3. **Maybe disable MoE selective ckpt** once grouped MoE shrinks the
   forward — needs free memory headroom. Saves ~510 ms/step recompute.
   *Est. gain: another +20-30%*.
4. **`torch.compile` via tc_volta wrapper** — last, since fp16 + ckpt
   + DDP + MoE all need to coexist. Skip if memory + correctness
   regress.
5. **FA-Volta** — skip unless we still need >40K and SDPA is the
   remaining bottleneck. (Unlikely.)

Router CV drift (0.50 → 1.40 over 1220 steps) is a separate concern —
the bias controller is **losing the race** rather than being
gradient-dominated (aux/z contributions to loss are 0.01-0.02, lm_loss
~5.0). Investigate after throughput is fixed.

## Pass-2 actual deltas (so far)

### Step 1 — Liger fused linear+CE ✅
- Replaced `tiled_cross_entropy` w/ `LigerFusedLinearCrossEntropyLoss` (v0.3.0).
- Parity vs chunked CE on a fixed seed: `|Δloss| < 2e-6`, grad max-abs ~1e-7.
- 50-step 4-GPU bench: **36.5K → 46.0K tok/s (+26%)** — well above +10-12% projection.
- Why bigger than projected: chunked CE wasn't just slow forward; the
  `torch.utils.checkpoint` wrapper around each chunk also doubled CE work
  in backward. Liger eliminates both.
- Loss step-0 = 12.041 — identical to baseline 12.041 (no drift from
  precision change). Loss step-49 = 7.66 (vs baseline 7.60) — small
  trajectory drift from a different sequence of FP rounding, within
  expected noise for 50 untrained steps.

### Step 2 — megablocks dMoE 🟡 deferred
- Installed `megablocks==0.5.1` + `stanford-stk==0.7.1` (newer 0.10.0
  requires torch ≥2.7.0, we have 2.3.0). Built ok against
  `CUDA_HOME=/usr/local/cuda/cuda-12.1`.
- dMoE bundles its own `LearnedRouter` (softmax + top-k). To keep our
  aux-loss-free `SigmoidRouter` + bias-update we'd need to bypass it and
  call `ParallelDroplessMLP.forward` with our `(scores, top_weights,
  top_idx)`. That object's expert weights live inside its own SparseGLU
  module, so we'd need to either swap our stacked `gate`/`up`/`down`
  params into its internal buffers each step, or copy the megablocks
  dispatch into a standalone shim (~200 LOC).
- Expected gain ~10% on our CV-balanced workload — block-sparse mainly
  wins when many blocks are empty; our routed CV is ~0.5 → all experts
  ~equally loaded → block-sparse offers little over our grouped einsum.
- Decision: skip pending; revisit only if final stack < 70K.


### Step 3 — FA-Volta attention 🟡 no-op (kept available, default off)
- Wired `attn_backend="fa_volta"` to swap SDPA → `flash_attn_volta.autograd.flash_attn`.
- Parity vs SDPA on fixed seed: max-abs out diff 1.2e-4 (PASS).
- 50-step 4-GPU bench: **45.8K tok/s** vs step-1's 46.0K — within noise.
- Why no gain: attention is only ~17% of forward after Liger eliminated
  CE (was 13% before). FA-Volta's ~30% kernel-level speedup buys ~5%
  overall, lost in step-time variance.
- Decision: leave default=`sdpa`, keep `fa_volta` available behind a
  flag. Re-evaluate after step 4 — if MoE checkpoint removal makes
  attention a bigger fraction, FA-Volta becomes worth turning on.


### Step 4 — disable MoE selective checkpoint ❌ blocked by memory
- Tried `--no_moe_selective_ckpt` and `--moe_ckpt_n_unwrap` ∈ {1, 2, 3}.
  All OOM by 38-186 MiB on rank 0 at step 1 (after `opt.step()` allocates
  Muon NS + AdamW m/v state).
- Each unwrapped MoE block adds ~280-320 MB of resident
  ``[E=32, capacity=1280, d_ff=1024]`` fp16 g/u/h buffers; Liger freed
  ~700 MB of headroom but Muon's matrix-grad NS momentum (~6 GB) eats
  through it on step 1.
- Tried `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,
  garbage_collection_threshold:0.85` — no improvement (V100 driver
  doesn't honor `expandable_segments`).
- Decision: keep all 15 MoE blocks checkpointed (legacy behavior).
  Revisit after step 5 if torch.compile shrinks peak activation footprint
  via Inductor memory planning.


### Step 5 — torch.compile via tc_volta ✅ partial win
- Wired `tc_volta.compile(..., autotune=False)` into the trainer (default
  mode, `triton.cudagraphs=False`, V100-safe). Marked
  `liger_fused_cross_entropy` as `@torch._dynamo.disable` (Inductor's
  launcher rewrite is incompatible with Liger's Triton kernel — the
  rewritten launcher passes `num_warps` as a kwarg that the Triton
  kernel rejects).
- Single-GPU smoke: eager loss 11.926, compiled loss 11.928,
  |Δloss| = 2.0e-3 (gate < 1e-2 PASS).
- 4-GPU 50-step bench: **46.0K → 51.1K tok/s (+11%)** with compile +
  Liger + all-15 MoE ckpt'd. Loss step-0 = 12.042 (vs 12.041 eager,
  drift 1e-3, within 1e-2 gate). Loss step-49 = 7.67 (eager 7.66).
- Why not the projected +15-25%: MoE blocks are wrapped in
  `torch.utils.checkpoint(use_reentrant=True)` which is a higher-order
  op — dynamo can't trace into it, so the entire MoE FFN is left
  un-compiled. Compile only helps attention + dense FFN + embed + norms
  ≈ 30-40% of total compute. Switching to `use_reentrant=False` (which
  dynamo traces cleanly) **OOMs at step 1** — the non-reentrant path
  keeps more activations resident than the reentrant path.
- max-autotune mode crashes Inductor with `IndexError: map::at` in
  `aten.mm.default` lowering — known fp16 + DDP edge-case on torch 2.3.
- Compile + unwrap=1 MoE block: training started at 51K, then NaN
  gradient cascade from step 16. Reverted.

## Stack final — sustained 51.1K tok/s

| step | patch                          | tok/s | Δ over prev |
|------|--------------------------------|-------|-------------|
| 0    | baseline (post-pass-1)         | 36.5K | —           |
| 1    | Liger fused CE                 | 46.0K | +26%        |
| 2    | megablocks dMoE (deferred)     | 46.0K | 0%          |
| 3    | FA-Volta (kept, default off)   | 45.8K | -0.4% noise |
| 4    | disable MoE ckpt (OOM blocked) | 46.0K | 0%          |
| 5    | torch.compile (default mode)   | 51.1K | +11%        |

**51.1K tok/s = 100B / 51.1e3 / 86400 = 22.7 days for the full run**,
comfortably within the 30-day wall budget. Below the 55K hard floor the
user set; firing `blocker` for ack before re-launch.

