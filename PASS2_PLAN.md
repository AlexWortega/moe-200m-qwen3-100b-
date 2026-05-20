# PASS-2 optimization plan — 36.5K → 70K tok/s on 4× V100

Baseline (4-GPU microbench `notes/bench/final-bench/`): mean 36.5K tok/s
over 50 steps. Build state: torch 2.3.0+cu121, triton 2.3.0, V100 SM 7.0.

Goal: stack five V100-compatible optimizations, each gated by its own
micro-bench. Apply in order, revert any step that regresses, do not
relaunch 100B at <55K tok/s without user re-approval.

## Stack

| # | Patch                          | Expected | Cumulative target | Gate                              |
|---|--------------------------------|----------|-------------------|-----------------------------------|
| 1 | Liger fused linear+CE          | +10-12%  | ≥ 40K             | loss-diff < 1e-3 vs current       |
| 2 | megablocks dMoE block-sparse   | +10-12%  | ≥ 45K             | router_cv stable, no OOM          |
| 3 | FA-Volta SDPA replacement      |  +5-7%   | ≥ 50K             | attn-out max-abs < 1e-3 vs SDPA   |
| 4 | Disable MoE selective ckpt     | +20-25%  | ≥ 58K             | no OOM at micro_bs=8              |
| 5 | torch.compile (tc_volta)       | +15-25%  | ≥ 60K             | loss within 0.01 of uncompiled    |

After all 5: target 70-75K (or land at 60K if step 5 fails).

## Implementation notes

### Step 1 — Liger fused linear+CE

- Replaces `tiled_cross_entropy` in `model.py`. Wire behind a new config
  flag `use_liger_ce` (default True when available, falls back to
  chunked-CE if Liger import fails).
- For tied embed: call
  `LigerFusedLinearCrossEntropyLoss(reduction='mean')(lin_weight=embed.weight, _input=h*mup_scale, target=labels)`.
  Pre-scaling `h` by `mup_scale` is exact since `F.linear((h*s), W) = F.linear(h, W) * s`.
- No `torch.utils.checkpoint` wrapper around CE anymore — Liger never
  materializes logits, so resident memory drops ~0.7 GB.
- Bench: `bash scripts/microbench_4gpu.sh`, target ≥ 40K mean, loss step-0
  within ±0.005 of the baseline log row.

### Step 2 — megablocks dMoE

- Install `megablocks` for py3.10 (`python3.10 -m pip install megablocks --no-deps`,
  then add `stk` if needed). Verify Triton 2.3 path compiles on SM 7.0.
- Replace `_moe_dispatch_grouped` body with a thin shim that takes the
  current stacked `[E, d_ff, d]` weights and calls `dmoe.dMoE.forward`-style
  kernel. Keep our `SigmoidRouter` + bias-update — pass our `(topk_idx,
  topk_weight)` into a custom dispatch that uses megablocks' bsmm
  kernels rather than our einsum.
- Caveat: megablocks expects column-major expert weight stacks. May need a
  `.permute(0,2,1).contiguous()` view. Doing this on weight init avoids
  per-step transpose.
- Gate: 50-step bench ≥ 45K, router_cv similar trajectory.

### Step 3 — FA-Volta in attention

- `from flash_attn_volta.autograd import flash_attn`
- Drop-in replacement in `GQAAttention.forward`: replace
  `F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)` with
  `flash_attn(q4, k4, v4, causal=True)` where tensors are reshaped to
  `(B, S, H, D)` (FA-Volta layout) instead of `(B, H, S, D)`.
- Add an `attn_backend="fa_volta"` config flag. Keep "sdpa" path for
  unit tests / non-fp16.
- Gate: 50-step bench ≥ 50K AND `max|out_fa - out_sdpa| < 1e-3` on a
  fixed-seed forward.

### Step 4 — disable MoE selective ckpt

- `train_200m.py:213` already gates this on `args.moe_selective_ckpt`.
- Add `--no_moe_selective_ckpt` (or set the existing flag to False
  default) and verify forward+backward fits in 32 GB.
- If OOM: try checkpointing only odd-numbered MoE blocks (8 of 15 instead
  of all 15). If still OOM: revert and accept step-3 throughput.
- Gate: 50-step bench ≥ 58K, no OOM, peak GPU mem < 31 GB.

### Step 5 — torch.compile via tc_volta

- Single-GPU smoke first: import `tc_volta`, call
  `tc_compile(model_inner, autotune=False)` on the inner module
  (post-DDP wrapping), forward+backward on a real batch, confirm loss
  matches eager ± 0.01.
- Then 4-GPU bench. If graph breaks dominate (more than 4-5), set
  `fullgraph=False` and accept the partial gain.
- Known fail modes: HOPs (chunked CE checkpoint) — Liger removed those.
  Dynamic shapes — Liger removed those. DDP+compile aliasing — already
  handled by `_dynamo.config.optimize_ddp=False`.
- If compile fails or regresses: disable, keep step-4 throughput, fire
  `blocker`, document in PROFILE.md, ship.
- Gate: 50-step bench ≥ 60K AND first-50-steps lm_loss series within
  ±0.01 of uncompiled.

## Hard floors

- After all 5 steps, if mean tok/s < 55K → fire `blocker`, do not
  re-launch 100B without user re-approval.
- Each step that *regresses* vs the prior step → revert that one change,
  proceed to next, document in PROFILE.md.

## Files touched (planned)

- `model.py` — Liger CE branch, FA-Volta branch, megablocks dispatch hook.
  Net +200/-100 LOC.
- `train/train_200m.py` — flag wiring, default flips, no-compile knob.
  Net +30/-10 LOC.
- New: `scripts/parity_check.py` — fixed-seed forward parity for each
  patch.
- New: `RUN_PASS2_LAUNCH.md` — final live throughput report.

## Notifications cadence

`plan_ready` (this file) → 5× `code_ready` (one per gate that passes) →
`train_started` (final relaunch) → per-10B `train_done`. `blocker` only
on fail.
