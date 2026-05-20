# RESULTS — interim status (variant A on 2 GPUs)

## Status

**2026-05-19**: Production 100B re-launched as **variant A on GPUs 2,3**
after two NaN cascades on the 32-expert + aggressive-coeff config:
- run 1 (`train_100b_run1_cv_collapsed.log`): CV exploded to 3.14.
- run 2 (`train_100b_run2_nan_cascade.log`): NaN cascade at step 1314+.

Variant A trades wall time (~48 d at ~24K tok/s on 2 GPUs) for
stability: 16 routed experts (was 32), moderate `aux=1e-3 /
bias=1e-3` (was the NaN'd 1e-2 / 5e-3), peak_lr 4e-4 (was 6e-4),
magnitude-based bias update kept, grad_clip 1.0 already in place.
GPUs 0,1 are reserved for other workloads.

## History (kept for context)

**2026-05-18**: Production 100B run was stopped at step 1220 due to
unacceptable throughput (13.7K tok/s on 4× V100, bmm backend).
Re-launched after grouped-MoE rewrite at 21:00 UTC, log file
`train_100b.stdout.sup2`. New state at step 4:

```
[init] params total 1.088 B  active 203.75 M
[init] starting at step 0, total=1525879
step=0  lm=12.037  cv=0.457  tok/s=35.9K  wall=1826ms
step=1  lm=12.033  cv=0.430  tok/s=33.7K  wall=1944ms
step=2  lm=12.039  cv=0.434  tok/s=34.5K  wall=1899ms
step=3  lm=12.045  cv=0.515  tok/s=35.1K  wall=1865ms
step=4  lm=12.047  cv=0.507  tok/s=37.3K  wall=1759ms
```

**Throughput rescue: 13.7K → 35-37K tok/s/agg = 2.6× speedup. ~32 days
for the full 100B (within the 30-day wall budget once steady-state
settles). See PROFILE.md and OPS.md "What changed on 2026-05-18" for
the breakdown.**

The May-17 ckpt `step_1221_legacy_bmm.pt` is kept for reference. We
chose a fresh start over warm-starting it because the model now uses
the grouped-MoE backend (the weight conversion is bit-exact but the
optimizer state — Muon's NS momentum + AdamW m/v — wasn't validated
under the param reshuffle, and the 80 M tokens of legacy training came
with a CV in the runaway phase, so the warm-start equity was small).

## Configuration (variant A on 2 GPUs, 2026-05-19)

- **GPUs**: **2 and 3 only** (GPUs 0/1 reserved).
- **Model**: 0.616 B total, 203.6 M active (`MoEModel` from `model.py`).
- **Routed experts**: **16** (was 32), top_k=2, n_shared=1, d_ff=1024.
- **Router**: aux_coef=**1e-3** (was 1e-2), bias_update_rate=**1e-3**
  (was 5e-3), magnitude-based update formula
  (`err = (mean - counts) / mean`), z_coef=1e-3.
- **Tokenizer**: Qwen/Qwen3-0.6B-Base (vocab 151 936)
- **Dataset**: openbmb/Ultra-FineWeb split=en, streaming
- **Optimizer**: Muon (per-param NS, fp32) + AdamW for non-matrix.
  Grad clip max_norm=1.0 in `train/train_200m.py:480`.
- **WSD**: 2 000 warmup, stable to 1 449 585, linear decay over 76 294
- **Peak LR**: **4e-4** (was 6e-4); min LR 4e-5
- **EMA**: β=0.9999, activates at step 1 449 585
- **Batch**: micro_bs=8 × seq_len=2048 × **2 GPUs** = **32 768 tokens / step**
- **MoE backend**: **`grouped` (new)** — stacked `[E, d_ff, d]` weights +
  3 einsum forwards over `[E, capacity, d]` dense buffer.
  `moe_capacity_factor=1.25` (drops tokens beyond capacity).
- **Compile**: OFF (still OOMs at this scale post-rewrite — needs ~1.4 GB
  more than the 300 MB headroom on 4× V100).
- **Grouped NS**: OFF (OOM at this scale; same reason as before).
- **Chunked CE**: ON (chunk=512 tokens, checkpointed) — numerically exact.
- **Selective MoE-FFN checkpoint**: ON (disabling OOMs by ~50 MB).
- **DDP**: `find_unused_parameters=False`, `static_graph=True`,
  `bucket_cap_mb=100`, `gradient_as_bucket_view=True`.
- **Allocator**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (warn-and-skip on V100; harmless).

## Throughput vs. projection

- Variant A 2-GPU smoke (300 steps, bs=8 seq=2048, GPUs 2,3): **~24.5 –
  24.8 K tok/s** sustained from step 20 onwards.
- Wall time at 24.5 K: **~48 days for 100 B**. The 30-day default
  `WALL_BUDGET_S` will NOT cover the full target on one budget — plan
  one budget refresh or raise `WALL_BUDGET_S=$((50 * 86400))`.

Historical (kept for context):
- Pre-rewrite 100B run (May-17, bmm backend, 4 GPUs): **13.7 K**.
- Post-rewrite 50-step micro-bench (4 GPUs, 32 experts): **mean 36.6 K**.
- Pass-2 stack (4 GPUs, Liger CE + tc_volta compile): **51.1 K**.

## Files

- `PLAN.md` — exact hyperparams + the revised throughput projection
- `OPS.md` — operational notes (launch, resume, stop, trouble matrix)
- `model.py` — 1.088 B-param MoE with `tiled_cross_entropy`
- `train/train_200m.py` — production trainer; `--no_compile_model`
  is now the default (`run_100b.sh` flips it on by default too)
- `train/ufweb.py` — Ultra-FineWeb streaming + Qwen3 tokenizer
- `optim/muon.py` — Muon (`grouped_ns=False` in this run)
- `scripts/push_milestone.{py,sh}` — HF push hook, called every 10 B
  tokens from the trainer
- `tests/test_chunked_ce.py` — numerical equivalence check
  (PASS: |Δloss|=0, |Δgrad|≤1e-8)
- `tests/smoke_cuda.py` — single-GPU forward+backward smoke

## Failed experiments

| attempt | result |
|---|---|
| `torch.compile` (default DDPOptimizer) | NotImplementedError on the `ckpt.checkpoint` higher-order op |
| `torch.compile` + `torch._dynamo.config.optimize_ddp = False` | compiles but OOMs at first forward (peak 32.1 GB / 32 GB) |
| `grouped_ns=True` (batched Muon NS via bmm) | OOM at opt.step (bmm workspace ~2.3 GB on top of 29 GB resident) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | warns "not supported on this platform" — Volta has no expandable VAS — but no harm to enable |
| MoE selective ckpt **off** (after grouped MoE) | OOMs by ~50 MB on 4× V100 — MoE intermediate activations still too big without recompute |
| CE chunk_size 1024 / 2048 / 4096 (after grouped MoE) | slightly slower (6.5K vs 7.2K tok/s) or OOMs at chunk≥4096 — the Python-loop overhead is not the CE bottleneck, the matmul itself is |
| Grouped MoE capacity_factor=1.5 / 2.0 (less token-dropping) | 11.6K (cf=1.25) → 10.7K (cf=2.0) single-GPU; chose cf=1.25 |
| Lowest-capacity (cf=1.0, drops above mean) | 11.9K single-GPU; +3% over cf=1.25 but ~30% token-drop at CV>1 |

## Done conditions tracking

- [x] Smoke-test forward pass succeeded (printed output shape + param count)
- [x] Chunked CE numerically equivalent to bare F.cross_entropy
- [x] Multi-step DDP smoke green (30 steps, loss 12.29 → 8.07)
- [x] Production run launched (step 0 logged)
- [ ] `train_100b.log` has 1 525 879 steps with finite loss
- [ ] EVAL_*.md with the 6-task zero-shot suite at each milestone
- [ ] `VERIFY.md` with every section verdict = pass
- [ ] `PUBLISHED_milestones.txt` with at least one HF URL
- [ ] `notify.sh train_done` fired with final URL

## Next checkpoints (in expected wall time, at 24.5 K tok/s on 2 GPUs)

- ~1.5 h: first eval (step 2 000, end of warmup)
- ~3.7 h: first periodic ckpt (step 5 000)
- ~4.7 days: first HF push milestone (step ≈ 305 175, 10 B tokens)
- ~48 days: full 100 B run (step 1 525 879)

## Open items after launch

- Router CV: legacy log shows the 0.5 → 3.5 spike at step 200 with
  recovery to 1.4 by step 1200; the variant-A 2-GPU smoke shows the
  same early spike (CV ~2.4 at step ~80 with warmup=50). Watch the
  live run; if CV stays > 2 past step 10 000, **stop and ping the
  user — do NOT auto-raise aux/bias**. Run 2 (aux=1e-2 / bias=2e-2)
  NaN'd at step 1314+ after looking fine through step 200.
- GitHub push of the patches — repo not initialized in this directory,
  deferred. Diff against May-17 is preserved by `train_100b_legacy.*`
  and `step_1221_legacy_bmm.pt`.
