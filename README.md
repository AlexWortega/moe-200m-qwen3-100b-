# moe-200m-qwen3-100b

**200M-active / 620M-total Mixture-of-Experts pretrain on Ultra-FineWeb-en with Qwen3 tokenizer (vocab 151,936). Currently training on 2× V100 SXM2 32GB toward 100B tokens.**

This repo holds source code, configs, run reports and diagnostics. **Model checkpoints live separately** on `/mnt/storage/alexw/ml-intern-ckpts/moe-200m-qwen3-100b/` and on the Hugging Face Hub at intermediate token milestones.

---

## Architecture

| | |
|---|---|
| Active params | 203.59 M |
| Total params | 0.616 B |
| Layers | 16 (1 dense layer-0 + 15 MoE) |
| `d_model` | 640 |
| Attention | GQA, 10 q-heads / 2 kv-heads, head_dim 64 |
| RoPE | Partial (half head_dim rotated) |
| Norm | RMSNorm + QK-Norm |
| MoE per layer | 16 routed + 1 shared experts, top-2, `d_ff=1024` |
| Activation | SwiGLU |
| Tokenizer | `Qwen/Qwen3-0.6B-Base` (vocab 151 936) |
| Embedding | Tied (embed + LM head share weights) |
| Loss | Liger fused linear + cross-entropy |

Per-layer numerics:

* Router forward wrapped in `torch.cuda.amp.autocast(enabled=False)` — fp32 routing avoids fp16 overflow when bias-clamped logits reach the ±20 range.
* Magnitude-based bias controller with starved-expert boost.
* Router noise (`std=0.1`) for exploration during training.
* `dist.all_reduce(counts)` before bias update — required under DDP `broadcast_buffers=True`.

See `model.py` for the full implementation.

---

## Repo layout

### Source

| path | purpose |
|---|---|
| `model.py` | Full model: `MoEModelConfig`, `MoEModel`, `SigmoidRouter`, attention, blocks. Liger CE path. **Patched for fp32 router + DDP-safe bias control.** |
| `train/train_200m.py` | Training driver: DDP, AMP fp16, optimizer setup, ckpt save/resume, eval, HF push hook |
| `train/ufweb.py` | Ultra-FineWeb streaming loader (HF datasets, no local cache) |
| `optim/muon.py` | Muon optimizer (matrix params, Newton-Schulz iterations) |
| `optim/schedule.py` | WSD learning-rate schedule with EMA shadow |
| `tests/test_chunked_ce.py` | Numerical-parity test: chunked CE vs single-shot `F.cross_entropy` |
| `tests/smoke_cuda.py` | One-step forward+backward smoke test |

### Scripts

| path | purpose |
|---|---|
| `scripts/diagnose_router.py` | **Per-layer / per-expert dead-expert diagnostic.** Load a ckpt, run 30 forwards on real data, write Markdown with load fractions, biases, logit ranges, NaN counts. |
| `scripts/microbench_4gpu.sh` | 50–500-step throughput micro-bench on 4 GPUs (DDP). |
| `scripts/microbench_2gpu.sh` | Same on GPUs 2,3 (current production layout). |
| `scripts/compile_smoke.py` | Verify `torch.compile` via `tc_volta` doesn't change loss vs eager. |
| `scripts/parity_check.py` | DDP-vs-single-GPU forward parity. |
| `scripts/profile_step.py` | `cuda.Event` breakdown per component (attn / MoE / CE / opt). |
| `scripts/time_components.py` | Coarser component timings. |
| `scripts/push_milestone.py` / `.sh` | Convert ckpt → safetensors + push to HF Hub with model card. |

### Entrypoints

| path | purpose |
|---|---|
| `run_100b.sh` | Single training launch (one supervisor attempt). Reads `STEPS`, `WARMUP`, `EMA_START`, `ROUTER_NOISE`, `FORCE_FRESH`, etc. env knobs. Auto-detects latest ckpt and resumes unless `FORCE_FRESH=1`. |
| `supervise_100b.sh` | Restart-on-crash wrapper around `run_100b.sh`. `nohup`-friendly. Respects `.stop_200m_qwen3` flag. |
| `run_shakedown.sh` | Short pre-flight run (~15k steps) to validate config before the full 100B launch. |

### Reports

| path | what's inside |
|---|---|
| `TASK.md` | Original problem statement and unknowns. |
| `RESEARCH.md` | Reference papers, related work, design choices. |
| `PLAN.md` | Initial hyperparameter and training plan. |
| `PROFILE.md` | Detailed CUDA-event component breakdown identifying MoE dispatch + chunked-CE as the throughput bottlenecks. |
| `PASS2_PLAN.md` | Optimization stack plan: Liger CE → megablocks → FA-Volta → disable ckpt → torch.compile. |
| `OPS.md` | Operations log — launches, kills, supervisor cycles. |
| `RESULTS.md` | Training summary at each major milestone. |
| `SMOKE.md` | Pre-launch smoke-bench numbers (300 steps, 4-GPU). |
| `ROUTER_FIX.md` | Notes on router patches applied (magnitude controller, starved boost, noise). |
| `DEAD_EXPERTS.md` | Step-by-step expert-load diagnostic, per-layer per-expert state at step_7953 (run 3 pre-fix baseline). |
| `RUN_PASS2_LAUNCH.md` | Documented launch config of the pass-2 (post-optimization) run. |
| `NEXT_OPTIMIZATIONS.md` | Conditional plan for follow-up Liger / megablocks / FA-Volta integration. |
| `BLOCKER.md` | Active blockers / known issues. |

### Logs (kept for archive)

`train_100b.log`, `eval_100b*.log`, `supervise.log*`, `train_100b_legacy.stdout.sup1` — JSON-per-step training metrics and supervisor history.

Excluded by `.gitignore`: `ckpts_100b/` (symlink to `/mnt`), large `.pt.tmp` files, `archive/` (prior failed run cycles), `notes/bench/` (micro-bench artifacts), `__pycache__/`, agent stream-json logs (`*.log` > 100 KB).

---

## Hardware & training setup

| | |
|---|---|
| Host | `eva01` / `kanbaru` |
| GPUs | 2× V100 SXM2 32 GB (devices 2,3; devices 0,1 left free) |
| Precision | fp16 AMP (router forced fp32) |
| Optimizer | Muon (matrix params, NS iterations) + AdamW (embed/router/head) |
| LR | peak 4e-4, WSD schedule, warmup 2000 steps |
| Batch | per-GPU 8 × seq 2048 = 16 384 tokens; global 32 768 tokens / step |
| `total_steps` | 1 525 879 (target 100B tokens) |
| Throughput | ~27 K tokens/s sustained |
| ETA | ~43–50 days on this hardware |
| Ckpt cadence | every 500 steps until 5k, then every 5k |
| Ckpt rotation | `--ckpt_keep_last 3` + `best.pt` |
| Ckpt destination | `ckpts_100b/` (symlinked to `/mnt/storage/alexw/ml-intern-ckpts/moe-200m-qwen3-100b/`) |
| HF push cadence | every 10 B tokens (background hook) |

Launch sequence:

```bash
cd /home/alexw/ml-intern-runs/moe-200m-qwen3-100b
ROUTER_NOISE=0.1 FORCE_FRESH=1 nohup bash supervise_100b.sh > supervise.log 2>&1 &
```

Stop signal:

```bash
touch /home/alexw/ml-intern-runs/moe-200m-qwen3-100b/.stop_200m_qwen3
```

---

## Related repos

* [`AlexWortega/claude-ml-intern-skill`](https://github.com/AlexWortega/claude-ml-intern-skill) — the Claude Code skill that orchestrates pretraining runs.
* `AlexWortega/moe-100m-volta-week` (TBD) — predecessor 100M-active model trained to 21B tokens for baseline comparison.

---

## License

Apache 2.0 (matches the Qwen3 tokenizer license).
