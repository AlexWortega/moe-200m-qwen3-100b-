# OPS — operational notes for the 200M-qwen3 100B run

## Launch

Variant A on 2 GPUs (current):
```
rm -f .stop_200m_qwen3
rm -rf ckpts_100b/step_*.pt ckpts_100b/best.pt   # FORCE_FRESH safety
FORCE_FRESH=1 nohup bash supervise_100b.sh > supervise.log 2>&1 &
```

Defaults pin `CUDA_VISIBLE_DEVICES=2,3 NPROC=2 PEAK_LR=4e-4`. The
supervisor re-launches on non-zero exit up to `MAX_RESTART=15` times.

Legacy (4-GPU):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 nohup bash run_100b.sh > train_100b.stdout.sup1 2>&1 &
```

State files in run-dir:
- `train_100b.log` — JSONL, one row per training step
- `eval_100b.log` — JSONL, one row per eval slice
- `ckpts_100b/step_*.pt` — rotated, keep_last=3
- `ckpts_100b/best.pt` — best rolling-100 lm_loss
- `ckpts_100b/final.pt` — graceful-stop final ckpt
- `ckpts_100b/final_ema.pt` — written iff EMA was active at exit
- `.run_100b_start_t.txt` — anchor for wall-budget bookkeeping
- `PUBLISHED_milestones.txt` — one line per HF push (step / tokens / URL)
- `MILESTONE_STEP_<N>.txt` — per-push URL marker
- `train_100b.stdout.sup1`, `…sup2`, … — stdout of each launch attempt

## Graceful stop

```
touch .stop_200m_qwen3        # next-step boundary: ckpt + exit
# OR
kill -TERM <train_pid>        # same — handler catches it
```

## Resume

`run_100b.sh` auto-detects `ckpts_100b/step_*.pt` and resumes with full
state (model, opt, sched, EMA, RNG). Just re-launch the script. **The
2026-05-18 grouped-MoE rewrite is backward-compatible** with the
May-17 legacy `step_1221.pt` checkpoint — `MoEModel.load_state_dict`
auto-stacks the per-expert `routed_experts.{i}.{gate,up,down}.weight`
keys into the new `gate`/`up`/`down` tensors (verified bit-exact).

## Wall budget

Default 30 days (`WALL_BUDGET_S=2592000`). Anchor is set at first
launch and persisted in `.run_100b_start_t.txt`. The launcher itself
enforces the budget — when remaining <= 120s, it refuses to start and
fires `error` notify.

Override (e.g. tighter):
```
WALL_BUDGET_S=$((20 * 86400)) bash run_100b.sh   # 20-day cap
```

## HF push

Every 10 B tokens at the next ckpt boundary, the trainer spawns
`scripts/push_milestone.sh <ckpt_path> <tokens_seen>` in the
background. Each push creates a new HF repo:
```
AlexWortega/ml-intern-moe200m-qwen3-step{N}-{tokens}B-{stamp}
```
with model.safetensors, config.json, README.md, model.py,
tokenizer.json, load_test.py, plus TASK/PLAN/RESEARCH/train+eval logs.

The push is best-effort — it logs to `ckpts_100b/push_step_<N>.log`
and updates `PUBLISHED_milestones.txt` on success.

## GPU set (variant A, 2026-05-19)

The run uses **only GPUs 2 and 3**. GPUs 0 and 1 are reserved for other
workloads. Launcher defaults (`run_100b.sh`):
`CUDA_VISIBLE_DEVICES=2,3` and `NPROC=2`. Do not lift the default to
include GPU 0 or 1 without explicit user instruction.

## Memory

Per-rank peak: **~30.7 GB / 32 GB** at micro_bs=8 on 4× V100 (legacy
config). On the 2-GPU variant-A config (16 experts vs 32, total
0.616 B vs 1.07 B params) per-rank memory is lower; DDP buckets are
also half the size with 2 ranks. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launcher
defensively against fragmentation over long runs.

If memory drifts up (fragmentation over weeks), the trainer will OOM
during a step. Recovery: SIGTERM the run, wait for ckpt, relaunch.
The supervisor pattern (not enabled yet — drop in
`run_100b_supervised.sh` modeled on the 100M run if needed) does this
automatically.

## Throughput expectation

Variant A on 2 V100 (GPUs 2,3), 16-expert config, smoke step 20
measured **24.2 K tok/s** at micro_bs=8. Projection at sustained
24 K tok/s/agg:

- 1 B tokens ≈ 41.7 K seconds ≈ **11.6 h** of wall clock.
- 10 B tokens (one HF push milestone) ≈ **4.8 days**.
- 100 B target ≈ **~48 days** of wall clock.

This is roughly half the 4-GPU rate from the 2026-05-18 grouped-MoE
rewrite (~36 K tok/s/agg). The trade is wall time (48 d vs ~24 d) for
stability — half the DDP all-reduce noise and half the routed-expert
count make the router much easier to balance, which is what cost us
the two prior NaN cascades (run 1 CV→3.14, run 2 NaN at step 1314+).

If a 48-d wall is unacceptable, raise `WALL_BUDGET_S` or lift NPROC=2
to 4 (which requires releasing GPU 0/1 first).

## What changed on 2026-05-18

PROFILE.md identified MoE FFN as 57 % of forward (326 ms of 574 ms)
and CE as 29 %. Backward dominated the step at 75 % of wall, half of
which was MoE selective-ckpt recompute. The dispatch was running 96
cuBLAS launches/MoE-layer × 15 layers × 2 (fwd + recompute) = 2 880
launches/step plus 60 GPU→CPU syncs (offsets.cpu() × counts.cpu() per
layer × 2 passes).

Three changes, all in `model.py` + `train/train_200m.py`:

1. **Grouped MoE dispatch** (`_moe_dispatch_grouped` +
   `MoEFFN(backend="grouped")`). Stacks the 32 experts' SwiGLU
   weights into `[E, d_ff, d]` / `[E, d, d_ff]` tensors and runs the
   forward as 3 `einsum`s over a `[E, capacity, d]` dense buffer. No
   `.cpu()` sync, no per-expert dust-pass. Capacity-padded to
   `ceil(N·K/E · moe_capacity_factor)` — tokens beyond capacity are
   silently dropped (contribute 0). Default `cf=1.25`.
   The legacy `bmm` backend stays on `moe_backend="bmm"` for parity
   tests.

2. **DDP `find_unused_parameters=False`** (was `True`). With every
   expert weight now touched by the same 3 bmm calls, no params are
   ever "unused"; `static_graph=True` carries the rest.

3. **CLI: `--no_moe_selective_ckpt`** to switch off the
   `torch.utils.checkpoint` wrapper around each MoE block. Default
   stays on — disabling it OOMs at micro_bs=8 + chunk=512 on 4× V100.

Skipped after trying:

- `torch.compile(mode='default')` — adds ~1.4 GB peak, OOMs at
  micro_bs=8 (300+ MB short). Keep `--no_compile_model` for now.
- Larger CE `chunk_size` (1024, 2048, 4096) — slightly slower or
  OOMs. Stayed at 512.
- FA-Volta — attention is only 13 % of fwd and SDPA's mem-efficient
  kernel already handles it. Re-evaluate only if MoE+CE shrink
  further.

## What to watch in the first 24 h

- `tail -F train_100b.log | jq '.lm_loss, .router_cv, .tok_per_s'`
- LR should ramp 0 → **4e-4** linearly over steps 0–2 000 (variant A).
- Loss should drop from ~12 (random init, vocab 151 936 ≈ ln 151 936
  = 11.93) toward ~6–7 within the first 5 000 steps.
- `router_cv` may spike to ~2 around steps 100–200 then recover. With
  the 16-expert + magnitude-bias config it should sit ≤ 1.5 by step
  1000. If `router_cv > 2` past step 10 000, routing collapsed — stop,
  ping user (do NOT auto-iterate aux/bias higher; that's what NaN'd
  run 2 at step 1314).
- **Any NaN/Inf grad in first 1000 steps → stop, ask user.**
- Step wall jitter < ±20 %. If a single step takes > 4 s, suspect
  data-loader stall or NCCL hiccup — check `dmesg` / GPU error counts.
- Memory: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
  should stay in [30.3, 31.0] GB / rank. Drift > 1 GB indicates a leak.

## Trouble matrix

| symptom | most-likely cause | response |
|---|---|---|
| NaN/Inf in grad → repeated halve | fp16 overflow at high LR | loss-scale auto-halves; rebuilds; OK if not persistent |
| OOM at opt.step | fragmentation | save ckpt, restart |
| OOM at forward | leaked tensor | save ckpt, restart |
| DDP timeout / NCCL error | network blip | supervisor restart |
| router_cv stuck > 2 past 10 k steps | routing collapse | stop, ping user — **do NOT auto-raise aux/bias** (that's what NaN'd run 2 at step 1314) |
| any NaN/Inf in first 1000 steps (variant A) | router instability | stop, fire `blocker`, ping user |
| eval_loss flatlines for 3+ consecutive evals over ≥ 6 B tokens | data saturation | stop, push final ckpt, EVAL.md |
| trainer dies but ckpt clean | crash | `bash run_100b.sh` (auto-resume) |
| trainer dies mid-ckpt | partial write | `rm ckpts_100b/step_*.pt.tmp`, resume from prior |
| disk full | leftover bench ckpts | `rm -rf notes/bench/*/ckpts` |
