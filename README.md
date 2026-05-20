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

See [`model.py`](./model.py) for the full implementation.

---

## Model code (key pieces)

### Architecture diagram

```
input_ids (B, S)
   │
   ▼
embed (vocab=151936, d=640, tied)
   │
   ▼
┌─ Block 0  ──────────────────────  dense layer ──┐
│  RMSNorm → GQA(10q/2kv, RoPE-partial, QK-Norm) →│
│  RMSNorm → SwiGLU FFN (d_ff=1024)               │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─ Block 1..15  ──────────────────  MoE layers ───┐
│  RMSNorm → GQA → residual                       │
│  RMSNorm → MoE block                            │
│     │                                            │
│     ├─ Router (fp32) → top-2 routing            │
│     │    └─ aux-free bias controller            │
│     │       + magnitude update                  │
│     │       + starved-expert boost              │
│     │       + ε noise (training)                │
│     │                                            │
│     ├─ 16 routed SwiGLU experts (d_ff=1024)     │
│     └─ 1 shared SwiGLU expert (always-on)       │
│  → weighted sum → residual                       │
└─────────────────────────────────────────────────┘
   │
   ▼
RMSNorm → tied LM head → Liger fused linear+CE
   │
   ▼
loss (scalar) + aux dict {z_loss, aux_loss, counts, cv, entropy}
```

### Config (`MoEModelConfig`)

```python
@dataclass
class MoEModelConfig:
    vocab_size: int = 151936          # Qwen3 tokenizer
    d_model: int = 640
    n_layers: int = 16
    n_q_heads: int = 10
    n_kv_heads: int = 2               # GQA
    head_dim: int = 64
    rope_partial: int = 32            # half head_dim rotated
    rope_theta: float = 10000.0
    d_ff: int = 1024
    n_routed_experts: int = 16        # variant A (dropped from 32 after NaN cascade)
    n_shared_experts: int = 1
    top_k: int = 2
    moe_first_layer: int = 1          # layer 0 is dense
    router_z_coef: float = 1e-3
    router_aux_coef: float = 1e-3
    router_noise_std: float = 0.0     # set to 0.1 during training for exploration
    bias_update_rate: float = 1e-3
    max_seq_len: int = 2048
    tie_embeddings: bool = True
```

### SigmoidRouter (the patched, DDP-safe router)

```python
class SigmoidRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k, ...):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_experts, d_model))
        nn.init.normal_(self.w, std=0.02)
        self.register_buffer("bias", torch.zeros(n_experts))   # not a Parameter — manual updates
        ...

    def forward(self, x_flat):
        # Entire routing path in fp32 to avoid fp16 overflow at extreme logits.
        with torch.cuda.amp.autocast(enabled=False):
            logits = F.linear(x_flat.float(), self.w.float())
            scores = torch.sigmoid(logits)
            sel_logits = logits + self.bias.float().unsqueeze(0)

            # Training-time exploration noise (breaks load-imbalance lock-in).
            if self.training and self.noise_std > 0:
                sel_logits = sel_logits + torch.randn_like(sel_logits) * self.noise_std

            topk_sel, topk_idx = torch.topk(sel_logits, k=self.top_k, dim=-1)
            topk_weight = scores.gather(-1, topk_idx)
            topk_weight = topk_weight / (topk_weight.sum(-1, keepdim=True) + 1e-9)

            # ST-MoE z-loss + DeepSeek aux loss (small contribution; bias does heavy lifting)
            lse = torch.logsumexp(logits, dim=-1)
            z_loss = (lse ** 2).mean()
            one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=1)
            p_i = scores.mean(dim=0)
            aux_loss = self.n_experts * (one_hot.float().mean(0) * p_i).sum()

            # Routing health metrics
            counts = one_hot.sum(0).float()
            cv = counts.std() / counts.mean().clamp_min(1.0)
            p_avg = scores.mean(0).clamp_min(1e-9)
            p_avg = p_avg / p_avg.sum()
            entropy = -(p_avg * p_avg.log()).sum() / math.log(2.0)
        return topk_idx, topk_weight, {"z_loss": z_loss, "aux_loss": aux_loss,
                                        "counts": counts, "router_cv": cv,
                                        "router_entropy_bits": entropy}

    @torch.no_grad()
    def step_bias_update(self, counts):
        """Magnitude-based bias controller with starved-expert boost.

        `counts` MUST be all-reduced across ranks before this call (see
        train_200m.py). Otherwise DDP's `broadcast_buffers=True` overwrites
        rank-1's bias view with rank-0's after every forward, silently losing
        half the update signal.
        """
        counts_f = counts.float()
        total = counts_f.sum().clamp_min(1.0)
        p_i = counts_f / total
        target_p = 1.0 / self.n_experts
        err = target_p - p_i                              # positive => underloaded
        # 10× boost for starved experts (load < 10% of fair)
        starved = (p_i < 0.1 * target_p).float()
        err = err + starved * 0.5
        self.bias.add_(err * self.bias_update_rate)
        self.bias.clamp_(-10.0, 10.0)
```

Caller side (in `train/train_200m.py`):

```python
# Critical: all-reduce counts across DDP ranks BEFORE step_router_biases,
# otherwise the bias buffer drifts per-rank then gets clobbered by
# broadcast_buffers=True.
if world > 1:
    for layer_counts in aux["counts_per_layer"]:
        dist.all_reduce(layer_counts, op=dist.ReduceOp.SUM)
target_model.step_router_biases(aux["counts_per_layer"])
```

### Loss path (Liger fused linear+CE)

```python
# train/train_200m.py — never materialize full (B, T, vocab=151936) logits.
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss as LinCE,
)
loss = LinCE(reduction="mean")(
    _input=h_scaled,             # (B*T, d_model) hidden states
    lin_weight=embed_weight,     # (vocab, d_model) tied embedding
    target=labels.view(-1),
    bias=None,
    ignore_index=-100,
)
```

This saves ~4.7 GB peak per GPU vs chunked CE-with-checkpoint, freeing room for keeping MoE activations resident (no selective checkpoint needed → +20% throughput, was blocked only by Muon's NS momentum eating the headroom).

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
