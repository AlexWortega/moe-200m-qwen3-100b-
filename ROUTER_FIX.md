# Router load-balancing fix (2026-05-19)

## Problem

100B run 1 (`train_100b_run1_cv_collapsed.log`) collapsed routing:
`router_cv = 3.14` mean over last 500 steps, `router_entropy = 1.64 bits`
— effectively ~3 of 32 routed experts active.

Root causes:
- `router_aux_coef = 1e-4` too small — aux_loss * 0.0001 ≈ 0.05 vs lm_loss ≈ 10 → no gradient pressure.
- `step_bias_update` used **sign-only** update (`sign * 1e-3` per step) — converges linearly, way too slow for 32 experts.

## Patches applied (model.py)

1. `step_bias_update` now magnitude-normalized (DeepSeek-V3 formulation):
   ```python
   counts_f = counts.float()
   mean_count = counts_f.mean().clamp_min(1.0)
   err = (mean_count - counts_f) / mean_count
   self.bias.add_(err * self.bias_update_rate)
   ```
2. `router_aux_coef`: `1e-4` → `1e-2` (100×).
3. `bias_update_rate`: `1e-3` → `2e-2` (20×).
4. `router_z_coef`: kept at `1e-3`.

## Microbench gates

| Test | bias_rate | aux_coef | step 200 CV | step 200 entropy | tok/s | gate |
|------|-----------|----------|-------------|------------------|-------|------|
| A    | 1e-3      | 1e-3     | 2.82        | 3.99             | 50.6K | FAIL |
| B    | 5e-3      | 1e-2     | 1.62        | 3.99             | 48.7K | FAIL |
| B 500| 5e-3      | 1e-2     | 1.06 (@500) | 3.47 (@500)      | 49.6K | slow |
| B+   | 2e-2      | 1e-2     | 0.44        | 4.42             | 49.4K | **PASS** |

## 100B run 2 (post-fix) — also NaN'd

Launched 2026-05-19 06:18 UTC under `supervise_100b.sh`. First 200
steps looked clean (CV 0.46 → trough 0.26, entropy 4.94 bits, tok/s
48.5K, lm 12.04 → 8.36, no NaN). But the run **NaN-cascaded at step
1314+**: `train_100b_run2_nan_cascade.log` shows repeated NaN/Inf
grad skips compounding into a divergence the loss-scale halving could
not catch up with. The aggressive aux=1e-2 + bias=2e-2 combo was too
hot once the router escaped the early warmup regime.

## Variant A retreat (2026-05-19 ~08:30 UTC, current)

User asked to drop to 16 experts on 2 GPUs and pull the coefficients
back to moderate levels (`aux=1e-3`, `bias=1e-3`) while keeping the
magnitude formula. See `RUN_PASS2_LAUNCH.md` for the full config and
launch plan. The bet: half the routed-expert count + half the DDP
all-reduce noise + 10× lower aux + 5× lower bias should land in a
basin where the router stays well-behaved without the coefficient
arms race that broke run 2.

## Files moved aside

- `train_100b_run1_cv_collapsed.log` — old run trace (kept for reference)
- `train_100b_run2_nan_cascade.log` — run 2 trace
- `supervise.log.run1` — old supervisor log
