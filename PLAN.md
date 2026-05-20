# PLAN — exact hyperparams for the 100B run

## Architecture (verified, see smoke results below)

| key | value |
|---|---|
| vocab_size | 151 936 (Qwen3-0.6B-Base) |
| d_model | 640 |
| n_layers | 16 |
| n_q_heads | 10 |
| n_kv_heads | 2 (GQA, ratio 5:1) |
| head_dim | 64 |
| rope_partial | 32 |
| rope_theta | 10 000 |
| d_ff (per expert) | 1024 |
| n_routed_experts | 32 |
| n_shared_experts | 1 |
| top_k | 2 |
| moe_first_layer | 1 (layer 0 is dense SwiGLU) |
| tie_embeddings | True |
| max_seq_len | 2048 |
| use_chunked_ce | True, chunk_size=512 tokens, ckpt=True |

## Numerical targets — actual at init

```
TOTAL  params: 1.088 B   (target 1.0–1.3 B)
ACTIVE params: 203.75 M  (target 180–220 M)
  embed           97.24 M
  attn            15.75 M
  router           0.31 M
  shared_expert   29.49 M
  routed_experts 943.72 M
  dense_ffn        1.97 M
  norms            0.01 M
```

Active drift +1.9 %, total drift −1.0 % — both inside spec.

## Training budget

| key | value |
|---|---|
| total_tokens | 100 B |
| micro_bs (per GPU) | 8 |
| grad_acc | 1 |
| seq_len | 2048 |
| world_size (V100s) | 4 |
| tokens / step | 4 · 8 · 2048 = 65 536 |
| total_steps | 1 525 879 (≈ 100 B / 65 536) |

If micro_bs=8 OOMs once DDP+Muon state are live: drop to micro_bs=4
grad_acc=2 (same per-step token budget, ~half-throughput inserted).

## WSD schedule (steps)

| phase | start | end | shape |
|---|---|---|---|
| warmup | 0 | 2 000 | linear 0 → peak |
| stable | 2 000 | 1 449 585 | constant peak |
| decay  | 1 449 585 | 1 525 879 | linear to min |

- peak_lr = **6e-4** (vs 3e-4 for the 100M run; bigger model + Muon
  spectral-scaled LR ⇒ optimum higher).
- min_lr  = **6e-5** (peak × 0.1)
- decay fraction = 5 %, so decay_steps = 76 294
- EMA β = 0.9999; activation step = 1 449 585 (start of decay)

## Routing / aux losses

- z_coef = 1e-3 (router logit logsumexp penalty)
- aux_coef = 1e-4 (DeepSeek-style fine-grained-routing aux loss)
- bias_update_rate = 1e-3 (DeepSeekMoE aux-loss-free balancing)

## Optimizer

- Muon for matrix params (Q/K/V/O, gate/up/down, router.w), NS-fp32
  orthogonalization, `foreach=True`, `grouped_ns=True`.
- AdamW for non-matrix (embed, norms, bias buffers, smear) with the
  same peak_lr; β₁=0.9 β₂=0.95 wd=0.01 (Muon's default in
  `optim/muon.py`).
- Loss scale: dynamic, init 2¹⁴, floor 2⁰, ceil 2²⁴, double-every-200,
  halve-on-NaN — same as 100M.

## Checkpointing

- `ckpt_every` = 5 000 steps → 305 ckpts over the run, kept last 3 +
  best.pt + final.pt + final_ema.pt
- `eval_every` = 2 000 steps × 32 batches = ~525 K eval tokens per
  evaluation, on an independent shuffle of Ultra-FineWeb-en
- HF push milestone: every 10 B tokens consumed, i.e. step
  152 587, 305 175, 457 763, … (one HF repo per milestone, named
  `AlexWortega/ml-intern-moe200m-qwen3-step{N}-{tokens}B-{stamp}`).

## Memory smoke (single V100, no DDP / Muon state yet)

```
micro_bs=8 seq=2048:  fwd peak 15.47 GB,  bwd peak 15.47 GB
micro_bs=4 seq=2048:  fwd peak 11.33 GB,  bwd peak 11.33 GB
```

With Muon momentum (~4 GB on 1.0 B matrix params, fp32) + AdamW state
(~0.7 GB on 0.09 B non-matrix params) + fp32 grad bucket (~4.4 GB) +
DDP overhead, projected per-rank peak ≈ **25 GB / 32 GB** at micro_bs=8.

## Chunked-CE numerical check

`tests/test_chunked_ce.py`:
```
bare       loss = 9.028545
chunked    loss = 9.028545      |Δloss| = 0.00e+00
grad h max abs Δ                = 1.82e-10
grad W max abs Δ                = 7.92e-09
PASS
```

Identical to fp32 reference. Both checkpointed and non-checkpointed
chunked paths match.

## Throughput target

- 100M run hit 97 K tok/s/agg on the same 4× V100 with the same micro_bs
  but seq=2048 and a much smaller MoE (8 routed experts, d_ff=1280,
  d_model=512). At 1.09 B vs ~100 M total params, theoretical FLOPs
  scale ~1.8× per step. Expected aggregate throughput: **40–55 K
  tok/s/agg**.

### Measured throughput (30-step DDP smoke, bs=8, no compile, per-param NS)

- **Sustained: 14 – 15 K tok/s/agg** at micro_bs=8, seq=2048, 4× V100.
- Step wall: 4.3 – 4.8 s.
- Peak memory: **29.3 GB** / 32 GB per rank (91% — tight but stable).
- Forward + backward + per-param Newton-Schulz + AdamW for non-matrix.

### What did NOT work (re-tested if hardware/torch upgrade)

- **`grouped_ns=True`**: OOMs during the batched Newton-Schulz `torch.bmm`
  (~2.3 GB workspace needed; 1.6 GB free at opt.step time). The 100M
  run could afford the grouped path because it had 8 experts, not 32.
- **`torch.compile`**: OOMs during the first compiled forward (peak
  pushes from 29.3 GB → 32.1 GB; inductor pre-allocates output buffers
  more eagerly than eager-mode). Compile + bs=4 + grad_acc=2 is a future
  experiment but not on the critical path.

### Revised throughput projection

| config | tok/s | 100 B wall time |
|---|---:|---:|
| bs=8, eager, per-param NS (this PR) | 15 K | ~77 days |
| bs=8, eager + grouped_ns (if we free 5 GB) | ~22 K | ~52 days |
| bs=4 grad_acc=2, compile, grouped_ns (future) | ~30 K | ~38 days |

### Plan in practice

- Launch 100B run with the **eager / per-param-NS / bs=8** config — the
  one we know stays stable for 30 consecutive steps with no NaN, no OOM,
  no DDP-static-graph drama.
- Wall budget: configurable via `WALL_BUDGET_S` (default 30 days). At
  15 K tok/s we'll cover **≈ 39 B tokens** in 30 days.
- That's well beyond the 10 B saturation point of the 100M-active sibling
  run on Ultra-FineWeb-en, so the experiment is still meaningful even
  if we don't reach the nominal 100 B target.
- HF push every 10 B tokens means we ship a usable model at each
  milestone regardless of how far we get.

## Shakedown plan (before the 100B launch)

`run_shakedown.sh`:
- 15 000 steps = 983 M tokens (~1 B)
- Same WSD: warmup 2 000, stable 12 250, decay 750
- Same LR schedule + EMA off (15 k steps doesn't reach EMA gate)
- Save ckpt at 5 k / 10 k / 15 k, eval every 1 000
- **Stop conditions** (must all be green to launch 100B):
  - loss decreases monotonically (rolling 100)
  - router_cv stays in [0.05, 0.30]
  - no >5 consecutive NaN steps
  - throughput >= 35 K tok/s/agg
  - 0 OOMs, 0 DDP static-graph asserts

## Notify milestones

- `plan_ready` — once this file is written.
- `code_ready` — once smoke test passes and `train_1b.py` is wired up.
- `train_started` — at shakedown launch; then again at 100B launch.
- `train_done` — at each 10B HF push milestone; final one when 100B
  done and EVAL_100B.md exists.

## Stop signal

`.stop_200m_qwen3` flag file in run dir → SIGTERM the trainer at next
ckpt boundary. No supervisor auto-restart at first; manual relaunch
after diagnosing each crash. Add `run_supervised.sh` only if we see
frequent crashes during shakedown.
