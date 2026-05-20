# Dead-Experts Diagnostic — step_7953.pt

## Root cause (verified)

Two independent bugs, both confirmed by reading the code + reproducing
with isolated tests:

### Bug 1 — entropy NaN is fp16 underflow in the metric, not real

`SigmoidRouter.forward` runs inside `with torch.cuda.amp.autocast(fp16)`.
The router does `logits = F.linear(x_flat.float(), self.w.float())` —
but `F.linear` is in autocast's fp16 cast list, so the explicit `.float()`
gets overridden and **`logits` ends up fp16**. Then `scores = sigmoid(logits)`
is also fp16. For very negative logits (we see logit_min ≈ -27 at this
ckpt), `sigmoid(-27)` in fp16 underflows to **0**.

`p_avg = scores.mean(dim=0).clamp_min(1e-9)` cannot save it: `1e-9` is
unrepresentable in fp16 (smallest fp16 subnormal ≈ 6e-8), so `clamp_min(1e-9)`
silently becomes `clamp_min(0)`. Then `p_avg / p_avg.sum()` keeps the
zeros, `log(0) = -inf`, `0 * -inf = NaN`, entropy = NaN.

Verified: standalone `F.linear(fp32, fp32)` under autocast(fp16) returns fp16;
in the diagnose trace below, layers 1-4 have `p_avg_min = 0.0` exactly,
confirming the clamp failed. **The router itself works fine — only the
diagnostic metric is corrupt. CV is computed from integer `counts`, so CV is
correct.**

### Bug 2 — Dead experts are real (and severe): 102/240 expert slots are dead

The bias controller has run away in the early/late layers. Example layer 1:
bias range [-18.81, +7.10]. The DeepSeek-V3 magnitude formula was being
asked to control a router whose raw logits (router_w·x) had drifted into a
±27 range — the bias updates (±1e-3/step) and clamp-min-1 denominator
could not keep up. Specifically:

- **H1 (DDP counts not all-reduced before bias step) — CONFIRMED.**
  `train_200m.py:491-492` calls `target_model.step_router_biases(aux["counts_per_layer"])`
  with rank-local counts. DDP sets `broadcast_buffers=True` by default, so
  `self.bias` (a buffer) is broadcast from rank 0 to all ranks each forward;
  effectively rank-1's local bias update is overwritten every step. The
  bias controller sees only rank 0's biased local view of routing.
- **H2 (revival formula too weak) — CONFIRMED.**
  `err = (mean - c) / max(mean, 1)` gives starved experts at most `err ≤ 1`,
  so push is bounded to `1 × 1e-3 = 0.001` per step. For overloaded experts,
  err is unbounded negative (we see `c / mean ≈ 5-7`), so the push is
  asymmetric: overloaded experts get pushed down 5-10× faster than dead
  experts get pushed up. This is why we see bias ranging [-23, +7] —
  unbounded negative on overloaded, bounded positive on dead.
- **H3 (bias is a Parameter not buffer) — RULED OUT.**
  `self.register_buffer("bias", ...)` in `SigmoidRouter.__init__`.
- **H4 (aux loss alone can't revive dead) — CONFIRMED, by design.**
  `aux_loss = E·Σ f_i·P_i`; for `f_i=0` the gradient w.r.t. P_i is 0.
  Aux alone never re-routes to dead experts; the bias controller has to
  do that job, and it's broken.

Once enough router_w drift accumulates, dead experts have raw logit ≈ -10
even after the bias controller has pushed bias to +7 (sel_logit ≈ -3,
still loses top-2). The bias-controller fight is lost in layers 1-4
(`bias_min ≤ -18`) and 13-15 (`bias_min ≤ -13`).

## Fix plan

1. **model.py `SigmoidRouter.forward`** — wrap the entropy metric in an
   `autocast(enabled=False)` block + use `math.log(2)` so the clamp_min
   floor is honored. Fixes Bug 1 (entropy NaN reporting). Zero effect on
   real router function.
2. **model.py `SigmoidRouter.step_bias_update`** — switch to a symmetric
   load-fraction formulation, add a 10× rate multiplier for starved
   experts (`p_i < 0.1·target_p`), and hard-clamp `self.bias` to
   `[-5, +5]` after update. Prevents runaway in both directions and
   gives dead experts a real revival path.
3. **train/train_200m.py** — call `dist.all_reduce(counts, op=SUM)` on
   every layer's count tensor before passing to `step_router_biases`,
   so all ranks compute the same update from the global routing
   distribution. Eliminates the rank-0-only blind-spot bug.
4. **train/train_200m.py** — add `--reset_router_bias_on_resume` flag.
   Used for the smoke + 100B relaunch to start the controller from a
   clean state since the current bias values are degenerate (range -23
   to +7, with bias clamp at ±5 they'd still be skewed).

Patches verified by re-running the diagnose loop on the patched ckpt
state and watching dead-expert count + entropy.

---

- Steps: **100** forward passes (no train), single GPU (no DDP all-reduce)
- Batch: bs=8 seq_len=2048 → N=16384 tokens/step, top_k=2, E=16
- Fair-share per step: **2048.0** tokens/expert
- Fair-share over 100 steps: **204800** tokens/expert
- MoE layers: 15 (block indices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

## Verdict (auto)

- Dead expert *slots* (load < 1% of fair): **102** of 240
- Starved expert *slots* (load < 10% of fair): **123** of 240
- Layers with ≥1 dead expert: **15** / 15 (max dead per layer: 12)
- Total NaN-entropy steps (across layers): **382** of 1500

## Per-layer summary

| layer | dead | starved | logit_min | logit_max | nan_logit | nan_score | nan_entropy_steps | bias_min | bias_max | w_norm_min | w_norm_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 11 | -24.44 | 10.35 | no | no | 80/100 | -18.813 | +7.100 | 0.749 | 1.368 |
| 2 | 11 | 11 | -26.16 | 12.30 | no | no | 100/100 | -19.634 | +6.908 | 0.694 | 1.559 |
| 3 | 11 | 12 | -24.38 | 12.84 | no | no | 100/100 | -21.330 | +5.392 | 0.708 | 1.461 |
| 4 | 12 | 12 | -27.50 | 12.86 | no | no | 100/100 | -23.380 | +5.280 | 1.005 | 1.807 |
| 5 | 8 | 10 | -24.09 | 3.36 | no | no | 1/100 | -2.465 | +3.220 | 1.166 | 1.844 |
| 6 | 10 | 11 | -23.66 | 3.31 | no | no | 1/100 | -1.684 | +2.863 | 1.283 | 2.045 |
| 7 | 8 | 9 | -21.97 | 5.16 | no | no | 0/100 | -1.692 | +1.788 | 1.218 | 1.839 |
| 8 | 4 | 6 | -18.92 | 5.04 | no | no | 0/100 | -1.085 | +1.884 | 1.133 | 1.704 |
| 9 | 2 | 2 | -14.88 | 6.98 | no | no | 0/100 | -0.517 | +0.457 | 0.891 | 1.480 |
| 10 | 3 | 7 | -20.44 | 6.70 | no | no | 0/100 | -1.669 | +1.598 | 1.181 | 1.879 |
| 11 | 6 | 6 | -21.44 | 8.72 | no | no | 0/100 | -1.605 | +0.974 | 1.068 | 1.843 |
| 12 | 5 | 7 | -19.41 | 5.49 | no | no | 0/100 | -2.122 | +1.494 | 1.040 | 1.592 |
| 13 | 3 | 5 | -19.12 | 9.77 | no | no | 0/100 | -13.066 | +2.734 | 0.843 | 1.467 |
| 14 | 5 | 5 | -22.14 | 11.64 | no | no | 0/100 | -16.663 | +2.963 | 0.918 | 1.641 |
| 15 | 4 | 9 | -21.58 | 11.18 | no | no | 0/100 | -15.203 | +1.967 | 0.881 | 1.471 |

## Per-expert per-layer detail

Each layer: ``expert_i: load_frac=X% (mean_count/step) bias=B w_norm=N``.
``DEAD`` marker if load_frac < 1% of fair share.

### Layer 1
- e00: tot=    148 mean/step=    1.5 min/step=    0 max/step=   10 load_frac=  0.07% bias=+1.1626 w_norm=1.0073 **DEAD**
- e01: tot=     33 mean/step=    0.3 min/step=    0 max/step=    3 load_frac=  0.02% bias=+6.8094 w_norm=1.3479 **DEAD**
- e02: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+3.4551 w_norm=1.1048 **DEAD**
- e03: tot= 479400 mean/step= 4794.0 min/step= 2429 max/step= 6504 load_frac=234.08% bias=-4.7220 w_norm=0.7486
- e04: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+3.6870 w_norm=1.1140 **DEAD**
- e05: tot=   2133 mean/step=   21.3 min/step=    4 max/step=   54 load_frac=  1.04% bias=+2.5365 w_norm=1.0466 STARVED
- e06: tot=     13 mean/step=    0.1 min/step=    0 max/step=    3 load_frac=  0.01% bias=+1.6254 w_norm=1.0348 **DEAD**
- e07: tot= 466683 mean/step= 4666.8 min/step= 3488 max/step= 7191 load_frac=227.87% bias=-18.4102 w_norm=0.7513
- e08: tot=     10 mean/step=    0.1 min/step=    0 max/step=    2 load_frac=  0.00% bias=+6.6271 w_norm=1.3620 **DEAD**
- e09: tot=      1 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+3.8893 w_norm=1.1065 **DEAD**
- e10: tot= 401679 mean/step= 4016.8 min/step= 3206 max/step= 5428 load_frac=196.13% bias=-14.9652 w_norm=0.8100
- e11: tot=    346 mean/step=    3.5 min/step=    0 max/step=   13 load_frac=  0.17% bias=+6.6396 w_norm=1.3180 **DEAD**
- e12: tot=    660 mean/step=    6.6 min/step=    0 max/step=  124 load_frac=  0.32% bias=+6.5108 w_norm=1.3156 **DEAD**
- e13: tot= 715271 mean/step= 7152.7 min/step= 5391 max/step=10090 load_frac=349.25% bias=-18.8125 w_norm=0.7747
- e14: tot=    322 mean/step=    3.2 min/step=    0 max/step=   20 load_frac=  0.16% bias=+7.1001 w_norm=1.3680 **DEAD**
- e15: tot=1210101 mean/step=12101.0 min/step= 8355 max/step=13397 load_frac=590.87% bias=+6.8649 w_norm=1.1974

### Layer 2
- e00: tot=     31 mean/step=    0.3 min/step=    0 max/step=    3 load_frac=  0.02% bias=-3.5313 w_norm=0.7871 **DEAD**
- e01: tot=1141908 mean/step=11419.1 min/step= 9111 max/step=12428 load_frac=557.57% bias=+0.9265 w_norm=0.9303
- e02: tot=    203 mean/step=    2.0 min/step=    0 max/step=   10 load_frac=  0.10% bias=+6.9079 w_norm=1.4940 **DEAD**
- e03: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+4.6190 w_norm=1.3274 **DEAD**
- e04: tot= 906271 mean/step= 9062.7 min/step= 7539 max/step= 9925 load_frac=442.52% bias=-7.5718 w_norm=0.6974
- e05: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+1.4595 w_norm=1.0727 **DEAD**
- e06: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+2.3386 w_norm=1.1962 **DEAD**
- e07: tot=     51 mean/step=    0.5 min/step=    0 max/step=    8 load_frac=  0.02% bias=+3.6747 w_norm=1.2360 **DEAD**
- e08: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+3.2492 w_norm=1.2002 **DEAD**
- e09: tot= 453723 mean/step= 4537.2 min/step= 3644 max/step= 6403 load_frac=221.54% bias=-16.1583 w_norm=0.6941
- e10: tot=      2 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+6.4493 w_norm=1.5586 **DEAD**
- e11: tot=     18 mean/step=    0.2 min/step=    0 max/step=    2 load_frac=  0.01% bias=+4.6419 w_norm=1.3388 **DEAD**
- e12: tot= 176226 mean/step= 1762.3 min/step=  842 max/step= 2323 load_frac= 86.05% bias=+6.8160 w_norm=1.4110
- e13: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+3.8917 w_norm=1.2470 **DEAD**
- e14: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+1.9199 w_norm=1.1225 **DEAD**
- e15: tot= 598367 mean/step= 5983.7 min/step= 4810 max/step= 8058 load_frac=292.17% bias=-19.6342 w_norm=0.8749

### Layer 3
- e00: tot= 611292 mean/step= 6112.9 min/step= 5250 max/step= 7516 load_frac=298.48% bias=-15.6536 w_norm=0.7079
- e01: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+2.9634 w_norm=1.3200 **DEAD**
- e02: tot=1621721 mean/step=16217.2 min/step=15991 max/step=16337 load_frac=791.86% bias=-0.3525 w_norm=0.8973
- e03: tot=      5 mean/step=    0.1 min/step=    0 max/step=    1 load_frac=  0.00% bias=-3.1524 w_norm=0.9584 **DEAD**
- e04: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+4.3420 w_norm=1.4412 **DEAD**
- e05: tot=     98 mean/step=    1.0 min/step=    0 max/step=    7 load_frac=  0.05% bias=+3.3820 w_norm=1.3250 **DEAD**
- e06: tot=      1 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+3.9770 w_norm=1.3737 **DEAD**
- e07: tot=    849 mean/step=    8.5 min/step=    1 max/step=   36 load_frac=  0.41% bias=+5.3918 w_norm=1.4614 **DEAD**
- e08: tot= 713360 mean/step= 7133.6 min/step= 5387 max/step= 8613 load_frac=348.32% bias=-21.3304 w_norm=0.9498
- e09: tot=     17 mean/step=    0.2 min/step=    0 max/step=    2 load_frac=  0.01% bias=+2.0408 w_norm=1.3143 **DEAD**
- e10: tot=    193 mean/step=    1.9 min/step=    0 max/step=    6 load_frac=  0.09% bias=+0.0425 w_norm=1.1918 **DEAD**
- e11: tot= 308942 mean/step= 3089.4 min/step= 2287 max/step= 3918 load_frac=150.85% bias=+4.5676 w_norm=1.3337
- e12: tot=      8 mean/step=    0.1 min/step=    0 max/step=    1 load_frac=  0.00% bias=+4.1667 w_norm=1.4000 **DEAD**
- e13: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+1.1684 w_norm=1.2188 **DEAD**
- e14: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+3.4026 w_norm=1.3321 **DEAD**
- e15: tot=  20314 mean/step=  203.1 min/step=  124 max/step=  360 load_frac=  9.92% bias=+5.0432 w_norm=1.4506 STARVED

### Layer 4
- e00: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+0.2413 w_norm=1.3900 **DEAD**
- e01: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+2.9888 w_norm=1.6420 **DEAD**
- e02: tot=     16 mean/step=    0.2 min/step=    0 max/step=    1 load_frac=  0.01% bias=+5.2798 w_norm=1.8068 **DEAD**
- e03: tot=1119957 mean/step=11199.6 min/step=10197 max/step=11969 load_frac=546.85% bias=-0.1344 w_norm=1.2496
- e04: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-1.1426 w_norm=1.3465 **DEAD**
- e05: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-1.4631 w_norm=1.3073 **DEAD**
- e06: tot=      2 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+4.3909 w_norm=1.7585 **DEAD**
- e07: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+4.6709 w_norm=1.7758 **DEAD**
- e08: tot=      1 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+3.1516 w_norm=1.6502 **DEAD**
- e09: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-0.3073 w_norm=1.3894 **DEAD**
- e10: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-0.1843 w_norm=1.3889 **DEAD**
- e11: tot=1032515 mean/step=10325.1 min/step= 8957 max/step=11589 load_frac=504.16% bias=+4.3832 w_norm=1.6540
- e12: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+3.3408 w_norm=1.6798 **DEAD**
- e13: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-2.7310 w_norm=1.2105 **DEAD**
- e14: tot=  44858 mean/step=  448.6 min/step=  283 max/step=  598 load_frac= 21.90% bias=+0.8950 w_norm=1.6160
- e15: tot=1079451 mean/step=10794.5 min/step= 9642 max/step=12369 load_frac=527.08% bias=-23.3801 w_norm=1.0053

### Layer 5
- e00: tot=     65 mean/step=    0.7 min/step=    0 max/step=    7 load_frac=  0.03% bias=-1.2372 w_norm=1.4162 **DEAD**
- e01: tot=1636363 mean/step=16363.6 min/step=16345 max/step=16378 load_frac=799.01% bias=-0.1933 w_norm=1.1656
- e02: tot=    675 mean/step=    6.8 min/step=    0 max/step=  334 load_frac=  0.33% bias=-0.2343 w_norm=1.4565 **DEAD**
- e03: tot= 345774 mean/step= 3457.7 min/step= 2625 max/step= 5390 load_frac=168.83% bias=-0.3933 w_norm=1.8444
- e04: tot=   4353 mean/step=   43.5 min/step=   18 max/step=  118 load_frac=  2.13% bias=+1.8092 w_norm=1.6331 STARVED
- e05: tot=    604 mean/step=    6.0 min/step=    0 max/step=   52 load_frac=  0.29% bias=-2.4654 w_norm=1.2866 **DEAD**
- e06: tot=    670 mean/step=    6.7 min/step=    0 max/step=  114 load_frac=  0.33% bias=-1.1488 w_norm=1.3798 **DEAD**
- e07: tot=     10 mean/step=    0.1 min/step=    0 max/step=    5 load_frac=  0.00% bias=-0.7355 w_norm=1.4537 **DEAD**
- e08: tot= 347355 mean/step= 3473.6 min/step= 2710 max/step= 5014 load_frac=169.61% bias=-0.1233 w_norm=1.6077
- e09: tot=      4 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=-0.0099 w_norm=1.4775 **DEAD**
- e10: tot= 595984 mean/step= 5959.8 min/step= 4498 max/step= 6720 load_frac=291.01% bias=+3.2200 w_norm=1.7325
- e11: tot=  21200 mean/step=  212.0 min/step=  118 max/step=  373 load_frac= 10.35% bias=-1.4559 w_norm=1.3401
- e12: tot= 312328 mean/step= 3123.3 min/step= 2080 max/step= 4071 load_frac=152.50% bias=-0.4927 w_norm=1.7360
- e13: tot=    635 mean/step=    6.3 min/step=    1 max/step=   18 load_frac=  0.31% bias=+2.5138 w_norm=1.7184 **DEAD**
- e14: tot=  10738 mean/step=  107.4 min/step=   37 max/step=  413 load_frac=  5.24% bias=+1.7384 w_norm=1.6460 STARVED
- e15: tot=     42 mean/step=    0.4 min/step=    0 max/step=    8 load_frac=  0.02% bias=-0.7916 w_norm=1.4163 **DEAD**

### Layer 6
- e00: tot=1249336 mean/step=12493.4 min/step=11574 max/step=14585 load_frac=610.03% bias=-1.6637 w_norm=1.2829
- e01: tot= 103008 mean/step= 1030.1 min/step=  259 max/step= 1621 load_frac= 50.30% bias=-0.2628 w_norm=1.8049
- e02: tot=      1 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=-1.2234 w_norm=1.4437 **DEAD**
- e03: tot= 131550 mean/step= 1315.5 min/step=  555 max/step= 2151 load_frac= 64.23% bias=-0.0295 w_norm=1.8697
- e04: tot=     38 mean/step=    0.4 min/step=    0 max/step=    3 load_frac=  0.02% bias=+2.3334 w_norm=1.8080 **DEAD**
- e05: tot=      4 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=-0.3162 w_norm=1.5479 **DEAD**
- e06: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-1.6836 w_norm=1.4057 **DEAD**
- e07: tot=1634277 mean/step=16342.8 min/step=16298 max/step=16379 load_frac=797.99% bias=+0.5922 w_norm=1.3790
- e08: tot=     32 mean/step=    0.3 min/step=    0 max/step=    3 load_frac=  0.02% bias=+0.9685 w_norm=1.6745 **DEAD**
- e09: tot=      5 mean/step=    0.1 min/step=    0 max/step=    2 load_frac=  0.00% bias=-1.2199 w_norm=1.4250 **DEAD**
- e10: tot=    104 mean/step=    1.0 min/step=    0 max/step=    6 load_frac=  0.05% bias=+0.5628 w_norm=1.6390 **DEAD**
- e11: tot=   1702 mean/step=   17.0 min/step=    4 max/step=   50 load_frac=  0.83% bias=-1.3656 w_norm=1.4087 **DEAD**
- e12: tot=  14931 mean/step=  149.3 min/step=   50 max/step=  267 load_frac=  7.29% bias=+2.8634 w_norm=1.7994 STARVED
- e13: tot= 141812 mean/step= 1418.1 min/step=  900 max/step= 1796 load_frac= 69.24% bias=-0.1120 w_norm=2.0452
- e14: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+0.2064 w_norm=1.5841 **DEAD**
- e15: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+0.3503 w_norm=1.5744 **DEAD**

### Layer 7
- e00: tot= 170240 mean/step= 1702.4 min/step= 1044 max/step= 2850 load_frac= 83.12% bias=-0.4434 w_norm=1.7142
- e01: tot=     14 mean/step=    0.1 min/step=    0 max/step=    3 load_frac=  0.01% bias=-0.0043 w_norm=1.5034 **DEAD**
- e02: tot=      1 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+0.5287 w_norm=1.5715 **DEAD**
- e03: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-1.6917 w_norm=1.3297 **DEAD**
- e04: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=+0.7013 w_norm=1.5688 **DEAD**
- e05: tot=1111488 mean/step=11114.9 min/step=10267 max/step=12343 load_frac=542.72% bias=-1.2054 w_norm=1.2185
- e06: tot= 129383 mean/step= 1293.8 min/step=  827 max/step= 1686 load_frac= 63.18% bias=-0.1653 w_norm=1.7418
- e07: tot=      6 mean/step=    0.1 min/step=    0 max/step=    1 load_frac=  0.00% bias=-0.4744 w_norm=1.4357 **DEAD**
- e08: tot=     88 mean/step=    0.9 min/step=    0 max/step=    6 load_frac=  0.04% bias=+1.0788 w_norm=1.5867 **DEAD**
- e09: tot= 132638 mean/step= 1326.4 min/step= 1064 max/step= 1660 load_frac= 64.76% bias=-0.3513 w_norm=1.8392
- e10: tot=   5867 mean/step=   58.7 min/step=   13 max/step=  160 load_frac=  2.86% bias=+1.7854 w_norm=1.6675 STARVED
- e11: tot=1403735 mean/step=14037.4 min/step=12872 max/step=14956 load_frac=685.42% bias=+1.7884 w_norm=1.5378
- e12: tot= 146519 mean/step= 1465.2 min/step=  888 max/step= 2565 load_frac= 71.54% bias=-0.3546 w_norm=1.7757
- e13: tot=     20 mean/step=    0.2 min/step=    0 max/step=    2 load_frac=  0.01% bias=-0.6586 w_norm=1.4311 **DEAD**
- e14: tot=      0 mean/step=    0.0 min/step=    0 max/step=    0 load_frac=  0.00% bias=-0.4296 w_norm=1.4520 **DEAD**
- e15: tot= 176801 mean/step= 1768.0 min/step=  885 max/step= 2896 load_frac= 86.33% bias=-0.1039 w_norm=1.8015

### Layer 8
- e00: tot= 252807 mean/step= 2528.1 min/step= 1638 max/step= 3288 load_frac=123.44% bias=-0.6772 w_norm=1.5995
- e01: tot=   3852 mean/step=   38.5 min/step=   11 max/step=  449 load_frac=  1.88% bias=+0.5291 w_norm=1.3959 STARVED
- e02: tot=    256 mean/step=    2.6 min/step=    0 max/step=    9 load_frac=  0.12% bias=+0.7118 w_norm=1.4483 **DEAD**
- e03: tot= 243348 mean/step= 2433.5 min/step= 1170 max/step= 3742 load_frac=118.82% bias=-0.4650 w_norm=1.4778
- e04: tot= 248384 mean/step= 2483.8 min/step= 1615 max/step= 3682 load_frac=121.28% bias=-0.6294 w_norm=1.7040
- e05: tot=  75345 mean/step=  753.5 min/step=  316 max/step= 1725 load_frac= 36.79% bias=+1.4236 w_norm=1.4884
- e06: tot=  34022 mean/step=  340.2 min/step=  145 max/step=  711 load_frac= 16.61% bias=+1.8836 w_norm=1.5117
- e07: tot= 135095 mean/step= 1351.0 min/step=  687 max/step= 3523 load_frac= 65.96% bias=-1.0846 w_norm=1.2355
- e08: tot= 251469 mean/step= 2514.7 min/step= 1512 max/step= 4329 load_frac=122.79% bias=-0.6839 w_norm=1.6386
- e09: tot=   5461 mean/step=   54.6 min/step=   15 max/step=  126 load_frac=  2.67% bias=-0.4080 w_norm=1.3201 STARVED
- e10: tot= 213651 mean/step= 2136.5 min/step= 1420 max/step= 2599 load_frac=104.32% bias=-0.6128 w_norm=1.6733
- e11: tot= 193206 mean/step= 1932.1 min/step= 1116 max/step= 5167 load_frac= 94.34% bias=+0.7104 w_norm=1.4797
- e12: tot=    522 mean/step=    5.2 min/step=    0 max/step=   19 load_frac=  0.25% bias=+0.4521 w_norm=1.4313 **DEAD**
- e13: tot=      2 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=-0.8468 w_norm=1.3011 **DEAD**
- e14: tot=      1 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=-0.8715 w_norm=1.2932 **DEAD**
- e15: tot=1619379 mean/step=16193.8 min/step=15820 max/step=16356 load_frac=790.71% bias=+0.5688 w_norm=1.1333

### Layer 9
- e00: tot=1105564 mean/step=11055.6 min/step= 9303 max/step=12853 load_frac=539.83% bias=+0.0127 w_norm=0.9119
- e01: tot= 212833 mean/step= 2128.3 min/step=  864 max/step= 3230 load_frac=103.92% bias=+0.0775 w_norm=1.3921
- e02: tot= 182383 mean/step= 1823.8 min/step= 1042 max/step= 3559 load_frac= 89.05% bias=+0.3673 w_norm=0.9927
- e03: tot=  20795 mean/step=  207.9 min/step=   96 max/step=  323 load_frac= 10.15% bias=-0.0288 w_norm=0.9403
- e04: tot= 191913 mean/step= 1919.1 min/step= 1174 max/step= 2680 load_frac= 93.71% bias=-0.1064 w_norm=1.3348
- e05: tot= 194629 mean/step= 1946.3 min/step= 1007 max/step= 2373 load_frac= 95.03% bias=+0.1913 w_norm=1.4041
- e06: tot= 185993 mean/step= 1859.9 min/step= 1100 max/step= 2252 load_frac= 90.82% bias=-0.3010 w_norm=1.4259
- e07: tot= 169622 mean/step= 1696.2 min/step=  852 max/step= 2301 load_frac= 82.82% bias=-0.1243 w_norm=1.4077
- e08: tot= 163058 mean/step= 1630.6 min/step=  788 max/step= 2892 load_frac= 79.62% bias=-0.0376 w_norm=1.2166
- e09: tot= 198994 mean/step= 1989.9 min/step= 1188 max/step= 2677 load_frac= 97.17% bias=-0.2061 w_norm=1.4800
- e10: tot= 213255 mean/step= 2132.6 min/step= 1285 max/step= 3225 load_frac=104.13% bias=+0.2732 w_norm=1.4223
- e11: tot= 204142 mean/step= 2041.4 min/step= 1042 max/step= 3072 load_frac= 99.68% bias=-0.1331 w_norm=1.3512
- e12: tot=  62760 mean/step=  627.6 min/step=  355 max/step= 1054 load_frac= 30.64% bias=-0.5165 w_norm=0.8912
- e13: tot= 168804 mean/step= 1688.0 min/step=  530 max/step= 2363 load_frac= 82.42% bias=+0.1474 w_norm=1.2589
- e14: tot=    585 mean/step=    5.8 min/step=    0 max/step=   16 load_frac=  0.29% bias=-0.0725 w_norm=0.9903 **DEAD**
- e15: tot=   1470 mean/step=   14.7 min/step=    2 max/step=   56 load_frac=  0.72% bias=+0.4570 w_norm=1.0529 **DEAD**

### Layer 10
- e00: tot= 249433 mean/step= 2494.3 min/step= 1790 max/step= 3798 load_frac=121.79% bias=-0.5003 w_norm=1.6763
- e01: tot= 398339 mean/step= 3983.4 min/step= 2459 max/step= 5470 load_frac=194.50% bias=-1.6689 w_norm=1.1955
- e02: tot= 185406 mean/step= 1854.1 min/step= 1291 max/step= 2319 load_frac= 90.53% bias=-0.7682 w_norm=1.7366
- e03: tot=    452 mean/step=    4.5 min/step=    0 max/step=   13 load_frac=  0.22% bias=+0.1723 w_norm=1.4481 **DEAD**
- e04: tot= 219099 mean/step= 2191.0 min/step= 1556 max/step= 3150 load_frac=106.98% bias=-0.0336 w_norm=1.6646
- e05: tot= 230557 mean/step= 2305.6 min/step= 1381 max/step= 4179 load_frac=112.58% bias=-0.7415 w_norm=1.6966
- e06: tot=    586 mean/step=    5.9 min/step=    0 max/step=   19 load_frac=  0.29% bias=-1.0420 w_norm=1.2988 **DEAD**
- e07: tot= 103361 mean/step= 1033.6 min/step=  540 max/step= 1503 load_frac= 50.47% bias=+1.5981 w_norm=1.5336
- e08: tot=   3024 mean/step=   30.2 min/step=    0 max/step=   94 load_frac=  1.48% bias=+1.1760 w_norm=1.5550 STARVED
- e09: tot=     31 mean/step=    0.3 min/step=    0 max/step=    2 load_frac=  0.02% bias=-0.2770 w_norm=1.4290 **DEAD**
- e10: tot=  49647 mean/step=  496.5 min/step=  170 max/step= 2318 load_frac= 24.24% bias=+0.3268 w_norm=1.4450
- e11: tot=   6767 mean/step=   67.7 min/step=    8 max/step=  150 load_frac=  3.30% bias=+0.3725 w_norm=1.4506 STARVED
- e12: tot=   2049 mean/step=   20.5 min/step=    6 max/step=   48 load_frac=  1.00% bias=+1.3561 w_norm=1.5859 STARVED
- e13: tot=1551556 mean/step=15515.6 min/step=14008 max/step=15886 load_frac=757.60% bias=-0.1121 w_norm=1.1805
- e14: tot= 272144 mean/step= 2721.4 min/step= 2158 max/step= 3919 load_frac=132.88% bias=+0.2839 w_norm=1.8794
- e15: tot=   4349 mean/step=   43.5 min/step=    1 max/step=  105 load_frac=  2.12% bias=-0.1421 w_norm=1.4024 STARVED

### Layer 11
- e00: tot=     27 mean/step=    0.3 min/step=    0 max/step=    5 load_frac=  0.01% bias=+0.8216 w_norm=1.4606 **DEAD**
- e01: tot= 183470 mean/step= 1834.7 min/step= 1301 max/step= 2731 load_frac= 89.58% bias=+0.8335 w_norm=1.8429
- e02: tot=    271 mean/step=    2.7 min/step=    0 max/step=   53 load_frac=  0.13% bias=-0.4276 w_norm=1.4365 **DEAD**
- e03: tot= 155698 mean/step= 1557.0 min/step= 1086 max/step= 2622 load_frac= 76.02% bias=-0.4002 w_norm=1.5749
- e04: tot=  25359 mean/step=  253.6 min/step=   65 max/step=  811 load_frac= 12.38% bias=+0.9736 w_norm=1.3884
- e05: tot= 236144 mean/step= 2361.4 min/step= 1356 max/step= 4672 load_frac=115.30% bias=+0.0699 w_norm=1.5662
- e06: tot=    675 mean/step=    6.8 min/step=    0 max/step=   22 load_frac=  0.33% bias=+0.5836 w_norm=1.4123 **DEAD**
- e07: tot=  37445 mean/step=  374.4 min/step=   53 max/step=  860 load_frac= 18.28% bias=-0.8945 w_norm=1.2266
- e08: tot= 203622 mean/step= 2036.2 min/step= 1045 max/step= 2878 load_frac= 99.42% bias=-0.2339 w_norm=1.6623
- e09: tot= 929425 mean/step= 9294.2 min/step= 7529 max/step=10808 load_frac=453.82% bias=-1.6050 w_norm=1.0683
- e10: tot=1203323 mean/step=12033.2 min/step= 9748 max/step=12781 load_frac=587.56% bias=+0.5834 w_norm=1.2293
- e11: tot=     17 mean/step=    0.2 min/step=    0 max/step=    3 load_frac=  0.01% bias=-0.0627 w_norm=1.3419 **DEAD**
- e12: tot=    352 mean/step=    3.5 min/step=    0 max/step=   17 load_frac=  0.17% bias=-0.0498 w_norm=1.3447 **DEAD**
- e13: tot=    852 mean/step=    8.5 min/step=    0 max/step=   60 load_frac=  0.42% bias=+0.2482 w_norm=1.3536 **DEAD**
- e14: tot= 135423 mean/step= 1354.2 min/step= 1001 max/step= 2705 load_frac= 66.12% bias=-0.2288 w_norm=1.6061
- e15: tot= 164697 mean/step= 1647.0 min/step=  771 max/step= 3192 load_frac= 80.42% bias=-0.2112 w_norm=1.6317

### Layer 12
- e00: tot=   1355 mean/step=   13.6 min/step=    1 max/step=   46 load_frac=  0.66% bias=-1.5160 w_norm=1.0995 **DEAD**
- e01: tot= 208162 mean/step= 2081.6 min/step= 1675 max/step= 2652 load_frac=101.64% bias=+0.0170 w_norm=1.5500
- e02: tot= 220342 mean/step= 2203.4 min/step= 1336 max/step= 4317 load_frac=107.59% bias=+0.0559 w_norm=1.5633
- e03: tot= 351552 mean/step= 3515.5 min/step= 2132 max/step= 6522 load_frac=171.66% bias=-0.1248 w_norm=1.4498
- e04: tot=    270 mean/step=    2.7 min/step=    0 max/step=   12 load_frac=  0.13% bias=-0.7041 w_norm=1.1816 **DEAD**
- e05: tot= 379677 mean/step= 3796.8 min/step= 2329 max/step= 5761 load_frac=185.39% bias=+0.4214 w_norm=1.2449
- e06: tot=   5630 mean/step=   56.3 min/step=    2 max/step=  181 load_frac=  2.75% bias=+0.6857 w_norm=1.3185 STARVED
- e07: tot=  18938 mean/step=  189.4 min/step=   39 max/step=  426 load_frac=  9.25% bias=+1.4945 w_norm=1.3838 STARVED
- e08: tot=  49415 mean/step=  494.1 min/step=   93 max/step= 1000 load_frac= 24.13% bias=+1.0768 w_norm=1.3035
- e09: tot= 227385 mean/step= 2273.8 min/step= 1296 max/step= 3603 load_frac=111.03% bias=+0.0526 w_norm=1.5919
- e10: tot=    416 mean/step=    4.2 min/step=    0 max/step=   10 load_frac=  0.20% bias=+1.0731 w_norm=1.3585 **DEAD**
- e11: tot=  86903 mean/step=  869.0 min/step=  206 max/step= 1904 load_frac= 42.43% bias=-2.1223 w_norm=1.0400
- e12: tot=    427 mean/step=    4.3 min/step=    0 max/step=   17 load_frac=  0.21% bias=-0.3982 w_norm=1.1949 **DEAD**
- e13: tot=    410 mean/step=    4.1 min/step=    0 max/step=   18 load_frac=  0.20% bias=+0.7420 w_norm=1.3319 **DEAD**
- e14: tot= 188410 mean/step= 1884.1 min/step=  196 max/step= 5499 load_frac= 92.00% bias=-0.5286 w_norm=1.5515
- e15: tot=1537508 mean/step=15375.1 min/step=14693 max/step=15931 load_frac=750.74% bias=-0.2250 w_norm=1.0916

### Layer 13
- e00: tot=   2045 mean/step=   20.4 min/step=    3 max/step=   53 load_frac=  1.00% bias=+1.6103 w_norm=1.1671 **DEAD**
- e01: tot=   8901 mean/step=   89.0 min/step=   16 max/step=  197 load_frac=  4.35% bias=+0.1019 w_norm=1.0280 STARVED
- e02: tot=  82437 mean/step=  824.4 min/step=  280 max/step= 2500 load_frac= 40.25% bias=+0.8574 w_norm=1.3083
- e03: tot= 217054 mean/step= 2170.5 min/step= 1287 max/step= 2983 load_frac=105.98% bias=+0.1125 w_norm=1.3761
- e04: tot=  39835 mean/step=  398.4 min/step=  138 max/step= 2857 load_frac= 19.45% bias=+0.8212 w_norm=1.1117
- e05: tot= 211974 mean/step= 2119.7 min/step= 1013 max/step= 3070 load_frac=103.50% bias=+0.5660 w_norm=1.4665
- e06: tot=1546660 mean/step=15466.6 min/step=14688 max/step=15980 load_frac=755.21% bias=+0.1525 w_norm=0.8587
- e07: tot= 399637 mean/step= 3996.4 min/step= 1716 max/step= 9369 load_frac=195.14% bias=-13.0663 w_norm=0.8432
- e08: tot=    682 mean/step=    6.8 min/step=    0 max/step=   23 load_frac=  0.33% bias=+1.2001 w_norm=1.1597 **DEAD**
- e09: tot=   1932 mean/step=   19.3 min/step=    2 max/step=   50 load_frac=  0.94% bias=+0.2413 w_norm=1.0705 **DEAD**
- e10: tot=   2442 mean/step=   24.4 min/step=    2 max/step=  123 load_frac=  1.19% bias=+0.6203 w_norm=1.0678 STARVED
- e11: tot= 260796 mean/step= 2608.0 min/step= 1213 max/step= 4500 load_frac=127.34% bias=+0.6231 w_norm=1.3608
- e12: tot= 312369 mean/step= 3123.7 min/step=  787 max/step= 4564 load_frac=152.52% bias=+0.0868 w_norm=1.4035
- e13: tot=  49366 mean/step=  493.7 min/step=  203 max/step=  686 load_frac= 24.10% bias=+0.7594 w_norm=1.0582
- e14: tot=  75626 mean/step=  756.3 min/step=  346 max/step= 1664 load_frac= 36.93% bias=+2.5790 w_norm=1.2868
- e15: tot=  65044 mean/step=  650.4 min/step=  155 max/step= 2681 load_frac= 31.76% bias=+2.7344 w_norm=1.2860

### Layer 14
- e00: tot= 315611 mean/step= 3156.1 min/step= 1691 max/step= 5825 load_frac=154.11% bias=-16.6626 w_norm=0.9605
- e01: tot=    145 mean/step=    1.4 min/step=    0 max/step=   11 load_frac=  0.07% bias=+0.5660 w_norm=1.0054 **DEAD**
- e02: tot= 195881 mean/step= 1958.8 min/step=  800 max/step= 3606 load_frac= 95.65% bias=+1.1063 w_norm=1.2468
- e03: tot=      4 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+0.9275 w_norm=1.1375 **DEAD**
- e04: tot= 127557 mean/step= 1275.6 min/step=  617 max/step= 2222 load_frac= 62.28% bias=+1.9508 w_norm=1.0784
- e05: tot=   1032 mean/step=   10.3 min/step=    1 max/step=   30 load_frac=  0.50% bias=+0.0173 w_norm=0.9929 **DEAD**
- e06: tot=  86421 mean/step=  864.2 min/step=  353 max/step= 1603 load_frac= 42.20% bias=+1.0233 w_norm=1.1750
- e07: tot= 131091 mean/step= 1310.9 min/step=  740 max/step= 1726 load_frac= 64.01% bias=+2.4116 w_norm=1.6411
- e08: tot= 146345 mean/step= 1463.5 min/step=  236 max/step= 3429 load_frac= 71.46% bias=+0.7738 w_norm=1.2458
- e09: tot= 478897 mean/step= 4789.0 min/step= 2954 max/step= 7014 load_frac=233.84% bias=-0.4847 w_norm=0.9183
- e10: tot=  74870 mean/step=  748.7 min/step=  117 max/step= 1388 load_frac= 36.56% bias=+1.0169 w_norm=1.3136
- e11: tot= 116000 mean/step= 1160.0 min/step=  720 max/step= 1963 load_frac= 56.64% bias=+0.7449 w_norm=1.3280
- e12: tot=    979 mean/step=    9.8 min/step=    1 max/step=   48 load_frac=  0.48% bias=+1.3701 w_norm=1.1488 **DEAD**
- e13: tot=1548093 mean/step=15480.9 min/step=15068 max/step=15826 load_frac=755.90% bias=+2.0892 w_norm=0.9473
- e14: tot=  53872 mean/step=  538.7 min/step=  193 max/step= 1983 load_frac= 26.30% bias=+2.9632 w_norm=1.1647
- e15: tot=      2 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=+0.1867 w_norm=1.0181 **DEAD**

### Layer 15
- e00: tot=    770 mean/step=    7.7 min/step=    0 max/step=   40 load_frac=  0.38% bias=+1.1251 w_norm=1.1265 **DEAD**
- e01: tot=  19375 mean/step=  193.8 min/step=   67 max/step=  793 load_frac=  9.46% bias=+1.9671 w_norm=1.2208 STARVED
- e02: tot=     18 mean/step=    0.2 min/step=    0 max/step=    4 load_frac=  0.01% bias=+0.5127 w_norm=1.1095 **DEAD**
- e03: tot=   7982 mean/step=   79.8 min/step=   15 max/step=  220 load_frac=  3.90% bias=+1.5210 w_norm=1.2007 STARVED
- e04: tot=     26 mean/step=    0.3 min/step=    0 max/step=    2 load_frac=  0.01% bias=+0.6369 w_norm=1.1245 **DEAD**
- e05: tot=   2421 mean/step=   24.2 min/step=    0 max/step=  196 load_frac=  1.18% bias=+0.4424 w_norm=1.1129 STARVED
- e06: tot=   7234 mean/step=   72.3 min/step=   17 max/step=  342 load_frac=  3.53% bias=+1.1202 w_norm=1.1126 STARVED
- e07: tot=      1 mean/step=    0.0 min/step=    0 max/step=    1 load_frac=  0.00% bias=-1.4118 w_norm=1.0261 **DEAD**
- e08: tot= 132076 mean/step= 1320.8 min/step=  785 max/step= 1901 load_frac= 64.49% bias=+0.9162 w_norm=1.3146
- e09: tot= 181216 mean/step= 1812.2 min/step=  499 max/step= 3505 load_frac= 88.48% bias=+1.8891 w_norm=1.2245
- e10: tot=   2145 mean/step=   21.4 min/step=    2 max/step=  122 load_frac=  1.05% bias=+1.4878 w_norm=1.1827 STARVED
- e11: tot= 502793 mean/step= 5027.9 min/step= 2556 max/step= 7375 load_frac=245.50% bias=+1.9200 w_norm=1.3546
- e12: tot= 377483 mean/step= 3774.8 min/step= 2062 max/step= 5041 load_frac=184.32% bias=+1.7151 w_norm=1.4706
- e13: tot=1365328 mean/step=13653.3 min/step=12354 max/step=14606 load_frac=666.66% bias=-0.1761 w_norm=0.9245
- e14: tot= 354241 mean/step= 3542.4 min/step= 2027 max/step= 5210 load_frac=172.97% bias=-15.2034 w_norm=0.8810
- e15: tot= 323691 mean/step= 3236.9 min/step= 1233 max/step= 7119 load_frac=158.05% bias=+1.5376 w_norm=1.2042
