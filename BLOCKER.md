# Dead-experts smoke gates fail — escalating

## tl;dr

All 5 root causes from the user's hypothesis list are diagnosed and patched.
The patched controller is **demonstrably working** (loss stable, no entropy NaN,
17 vs 119 grad skips after stronger noise tune-down). But it cannot recover
routing balance to the user's strict gates within a 300-step smoke window:
the router weights have 7953 steps of biased training behind them, and the
bias controller alone can't undo that in a few hundred steps. **Need user
direction** — see [Decision request](#decision-request).

## Diagnostic ([DEAD_EXPERTS.md](DEAD_EXPERTS.md))

Confirmed:
- **102 of 240 expert slots dead** at step 7953 (42% capacity unused)
- **Bug 1 (cosmetic)**: entropy NaN is fp16 underflow in the metric, not real
  routing. `F.linear(fp32, fp32)` under `autocast(fp16)` returns fp16 → sigmoid
  underflows for extreme logits → `clamp_min(1e-9)` is a no-op in fp16 → log(0)
  → NaN. Real router function unaffected.
- **Bug 2 (real)**: bias controller is broken in 3 ways:
  - H1 (counts not all-reduced before bias step) — confirmed
  - H2 (formula asymmetric, dead experts capped at err≤1) — confirmed
  - H4 (aux loss has 0 gradient for dead experts) — confirmed
  - H3 (bias Parameter vs buffer) — ruled out, was buffer

## Patches applied (committed in `model.py` / `train/train_200m.py`)

1. **Entropy metric fp32** — `with autocast(enabled=False)` + `math.log(2)`. Fixes
   the cosmetic NaN. Verified on the same step_7953 ckpt: 0 NaN steps over 30
   forward passes (was 382/1500 before).
2. **`step_bias_update` rewrite** — symmetric load-frac formulation,
   `+0.05/step` additive boost when `p_i < 0.1·target_p`, clamp to `[-10, +10]`
   to prevent the runaway we saw at `[-23, +7]`.
3. **DDP all-reduce on counts** before bias update — fixes the rank-0-only
   blind-spot bug.
4. **`--reset_router_bias_on_resume` flag** — zeroes all 15 MoE router biases
   right after the resume state-dict load.
5. **`--router_noise_std` flag** — additive Gaussian on `sel_logits` during
   training only, to break the routing lock-in so dead experts can win top-2.
6. **Initial loss_scale `2**6` on resume** (was `2**14`) — avoids the 13-step
   NaN-flurry the dynamic loss-scale halving had to do on every restart.
7. **`RESET_BIAS=1 ROUTER_NOISE=…` env vars** in `run_100b.sh` to pass the
   above through to the 100B supervisor on the recovery launch.

## Smoke results (4 iterations, all under bench dir)

Resume `ckpts_100b/step_7953.pt` → +300 steps, 2× V100, otherwise identical
config to the 100B run.

| iter | noise | boost | clamp | loss_scale | min cv | max ent | NaN skips | gates 3/4/5 |
|------|-------|-------|-------|------------|--------|---------|-----------|-------------|
| v1   | 0     | rate×10 (1e-2) | ±5    | 2^14 | 2.04 | 2.01 | n/a (stopped early) | FAIL |
| v2   | 0     | +0.02/step     | ±5    | 2^14 | 2.04 | 2.18 | n/a (stopped early) | FAIL |
| v3   | 1.0   | +0.02/step     | ±5    | 2^14 | 1.55 | 2.35 | 119 | FAIL |
| **v4** | **0.3** | **+0.05/step** | **±10** | **2^6** | **1.82** | **2.28** | **17** | **FAIL** |

Best iteration is **v4**:
- gate1 entropy never NaN: **PASS** (0/283 steps)
- gate2 no LM divergence (<6.0 final): **PASS** (final lm=4.05, baseline 3.86)
- gate2b max LM spike <12: **PASS** (max spike 7.12 at step 8156, recovered to 4.0)
- gate3 ent ≥ 3.0 within 200 steps: **FAIL** (max ent in window 2.28)
- gate4 cv ≤ 1.5 within 300 steps: **FAIL** (min cv 1.82)
- gate5 ≤5 grad skips: **FAIL** (17 skips, mostly clustered at end of bench)

LM loss trajectory is healthy (median 3.86 = baseline). CV is **trending down
but slowly**: 2.19 → 1.82 over 200 steps. Entropy is **trending up**: 1.78 →
2.28 over 200 steps. Extrapolating linearly, cv would reach 1.5 around step
+500–600 and entropy would reach 3.0 around step +700–900.

## Why the gates can't be met in 300 steps

The router weights at step 7953 produce logits in `[-27, +13]` — a 40-unit
range. To re-route tokens to dead experts, the bias controller's authority
(clamped to ±10) **alone is not enough** to overcome the router-weight bias.
The router weights themselves have to relax, which only happens through
gradient flow from `aux_loss` + `lm_loss`. That's a slow process — empirically
the 100B run took ~4000 steps to develop the imbalance, undoing it requires a
comparable timescale.

v3's noise=1.0 *was* fast enough (cv to 1.55) but at the cost of 119 grad
skips and the LM-loss spikes that v4 still inherits at smaller magnitude.

The fundamental conflict: aggressive recovery (high noise / high boost)
destabilizes training; conservative recovery is too slow for a 300-step gate.

## Decision request

Three options for the user — pick one and I'll execute:

1. **Relax the smoke gates and relaunch 100B as-is.**
   v4 is *the same* config I'd use for the 100B relaunch
   (`RESET_BIAS=1 ROUTER_NOISE=0.3 nohup bash supervise_100b.sh ...`).
   LM is stable; routing recovery just plays out over the next 1000-2000
   training steps under live training instead of being mostly-done in 300.
   Risk: cv/entropy never fully recover before EMA-start (step 1.45M); model
   ships with some permanent dead experts. Reward: minimal disruption, the
   patches *are* the right fixes regardless.
2. **Iterate further on the smoke** with different recovery knobs
   (gumbel-softmax routing instead of additive noise; gradient clip on bias
   updates; reset router_w too — nuclear). Each attempt is ~7 min wall.
   ~90 min budget left = 10-12 more attempts. No guarantee any combination
   passes the strict gates given the underlying problem (router_w has
   accumulated 7953 steps of bias).
3. **Roll back the run** — start fresh from scratch (lose 7953 steps,
   0.26B tokens of training) with the patched controller from step 0. Slowest
   but cleanest — the controller would never accumulate the runaway state in
   the first place.

My recommendation: **option 1**. Patches address the root causes; LM is
stable; the slow trajectory after resume is normal for un-doing several
thousand steps of biased routing. If we wait for the smoke to pass strict
gates we're effectively asking for full recovery in 300 steps, which the
math just won't give us.

## Files / commands for option 1 (recommended)

```bash
cd /home/alexw/ml-intern-runs/moe-200m-qwen3-100b
# Stop the old supervisor if it's still loose (it isn't — exited at 7953):
ls supervise.log
# Launch with the recovery knobs:
RESET_BIAS=1 ROUTER_NOISE=0.3 nohup bash supervise_100b.sh \
  > supervise.log 2>&1 &
# The supervisor auto-detects ckpts_100b/step_7953.pt as RESUME.
# `RESET_BIAS=1` zeros router biases (one-shot, only the first launch needs it).
# `ROUTER_NOISE=0.3` keeps noise on for the duration; flip to 0 once cv is
#   verified below 1.0.
```

The supervisor's auto-restart logic will keep the run alive across the bias
reset boundary — but note `RESET_BIAS=1` must only be set on the FIRST launch
of `supervise_100b.sh`; subsequent restarts after a fresh ckpt save would
re-zero the (now-recovered) bias and undo progress. The supervisor doesn't
currently encode this — needs a manual "stop supervisor, unset RESET_BIAS,
restart supervisor" once we see the first new ckpt land.
