# Variant A on 2 GPUs — 100B launch plan (2026-05-19)

Date: 2026-05-19
Status: Smoke bench running; 100B launch pending smoke pass.

This file supersedes the 2026-05-18 pass-2 launch plan, which assumed
4-GPU + 32 experts. The pass-2 plan was held after **two consecutive
NaN cascades** on the 32+1 config:

- run 1 (`train_100b_run1_cv_collapsed.log`): router CV exploded to
  3.14, then the loss diverged.
- run 2 (`train_100b_run2_nan_cascade.log`): NaN cascade at step 1314+
  despite the aggressive `aux=1e-2 / bias=5e-3` settings that were
  supposed to stabilise the router.

The user then asked for **variant A on 2 GPUs**: half the experts,
half the GPUs, moderate router coefficients, lower peak LR. The trade
is wall time (~48 d vs ~22 d) for stability.

## Config

| key                   | value           | notes |
|-----------------------|-----------------|-------|
| GPUs                  | **2,3 only**    | GPUs 0/1 reserved for other workloads |
| NPROC                 | 2               | DDP |
| n_routed_experts      | 16 (was 32)     | half the router load = 2× easier to balance |
| n_shared_experts      | 1               | unchanged |
| top_k                 | 2               | unchanged |
| d_ff                  | 1024            | unchanged |
| router_aux_coef       | **1e-3**        | 10× lower than the NaN'd 1e-2 |
| bias_update_rate      | **1e-3**        | 5× lower than the NaN'd 5e-3 |
| bias update formula   | magnitude       | `err = (mean - counts) / mean` |
| router_z_coef         | 1e-3            | unchanged |
| peak LR               | **4e-4**        | was 6e-4 |
| min LR                | 4e-5            | scaled with peak |
| grad clip             | max_norm=1.0    | already at `train_200m.py:480` |
| total params          | 0.616 B         | was 1.07 B |
| active params         | 0.204 B         | unchanged |

## Stability hardening already in `train_200m.py`

1. `torch.nn.utils.clip_grad_norm_(params, 1.0)` between manual
   unscale and `opt.step()` (line 480).
2. Per-step NaN/Inf grad detection with loss-scale halving and step
   skip (line 466).
3. Loss-scale auto-rebuild on `n_good >= grow_every` (line 486).
4. `--nan_cap N` (default 3000 in `run_100b.sh`, 100 in bench) — bails
   if too many consecutive skips.
5. Early rollback ckpts at steps 500 / 1000 / 2000 / 3000 / 4000
   plus regular `CKPT_EVERY` cadence — so a fast NaN strike costs at
   most ~500 steps of lost work.

## Smoke gates (300-step bench)

Must all pass before 100B launch:

1. Zero NaN/Inf events in stdout.
2. router_cv ≤ 1.0 at step 300 (target 0.3–0.7).
3. router_entropy_bits ≥ 3.0 at step 300.
4. tok/s ≥ 20 K (target ~24 K — half the 4-GPU figure of ~48 K).
5. lm_loss falling: step 0 ≈ 12.0, step 300 ≤ 6.5.

If gate (1) fails: stop, fire `blocker`, write `BLOCKER.md`, **do not**
iterate to more aggressive coefficients.

## Launch (after smoke passes)

```
rm -f .stop_200m_qwen3
rm -rf ckpts_100b/step_*.pt ckpts_100b/best.pt   # FORCE_FRESH safety
FORCE_FRESH=1 nohup bash supervise_100b.sh > supervise.log 2>&1 &
```

`run_100b.sh` defaults are now `CUDA_VISIBLE_DEVICES=2,3`,
`NPROC=2`, `PEAK_LR=4e-4`. The supervisor inherits all three.

## Wall-clock projection

At sustained ~24 K tok/s:

- 1 B tokens ≈ 11.6 h.
- 10 B tokens (one HF push milestone) ≈ 4.8 days.
- 100 B target ≈ **~48 days**.

The 30-day default `WALL_BUDGET_S` will NOT cover the full 100 B target
on one budget. Either:
- Raise `WALL_BUDGET_S=$((50 * 86400))` at launch, OR
- Plan one budget refresh around day 28 (the launcher writes
  `.run_100b_start_t.txt` as the anchor; deleting it resets the clock).

## Why not the 4-GPU + compile stack from pass-2

The compile stack (Liger fused CE + tc_volta torch.compile) was
landed in `model.py` (Liger) and `train/train_200m.py` (compile via
`--compile_model`). Liger is on by default; compile is off
(`--no_compile_model` is the default in `run_100b.sh`). With the
variant-A goal of stability over speed, we keep compile off for the
100B launch — eager backward is the most-tested path and any compile
regression (recompile loop, OOM by 280 MB) costs more wall time than
the +11% it bought on 4× V100.

## Files touched 2026-05-19

- `model.py:69-70` — `router_aux_coef = 1e-3`, `bias_update_rate = 1e-3`.
- `model.py:262` — `SigmoidRouter.__init__` defaults updated.
- `run_100b.sh:43-44` — defaults `CUDA_VISIBLE_DEVICES=2,3`, `NPROC=2`.
- `run_100b.sh:67` — `PEAK_LR` default already `4e-4`.
- `scripts/microbench_2gpu.sh` — NEW. 2-GPU 300-step bench wrapper.
- `OPS.md` — updated GPU set + memory/throughput sections.
- This file.
