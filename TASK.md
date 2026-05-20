# TASK — MoE 200M-active / ~1.2B-total pretrain on Ultra-FineWeb-en (Qwen3 tokenizer)

Fresh **from-scratch** pretrain of a sparse-MoE LM scaled ~2× from the
previous `moe-100m-volta-week` run. The 100M-active run plateaued at
held-out `eval_loss ≈ 2.91` somewhere around 6-10 B tokens consumed
(see `~/ml-intern-runs/moe-100m-volta-week/EVAL_21B.md`) — data-saturation
at that active-param count is exhausted on Ultra-FineWeb-en. Bumping
active params to ~200 M and swapping the custom 48 k SentencePiece for
the Qwen3-0.6B-Base BPE tokenizer (151 936 vocab) gives the model more
capacity *and* more granular token boundaries.

## Architecture target

- d_model = 640
- n_layers = 16
- n_q_heads = 10, n_kv_heads = 2, head_dim = 64 (GQA, ratio 5:1)
- partial RoPE on the same head-dim fraction as the 100M run (32/64)
- QK-Norm
- 32 routed experts + 1 shared, top-2 routing
- layer 0 dense (SwiGLU FFN), layers 1–15 MoE
- d_ff = 1024 per expert (SwiGLU = 3 matrices: gate/up/down)
- tied embed / lm_head
- **Tokenizer = `Qwen/Qwen3-0.6B-Base`, vocab=151 936**
- max_seq_len = 2048

## Hard numerical targets

- Active params: 180–220 M  (target ~200 M)
- Total  params: 1.0–1.3 B
- 33 experts per MoE layer (1 shared + 32 routed); top-2 routing → 3
  experts active per token per MoE layer

## Training budget

- 100 B tokens on Ultra-FineWeb-en (split=`en`, column=`content`),
  streaming, no replay across epochs (we won't reach an epoch boundary).
- 4× V100 SXM2 32 GB on eva01 (`kanbaru`).
- Estimated wall time: 22–26 days at 40-50 K tok/s.

## Reuse

Lifted as-is from `~/ml-intern-runs/moe-100m-volta-week/`:
- `optim/muon.py`, `optim/schedule.py`, `optim/__init__.py`
- skeleton of `train/train_1b.py` (DDP + ckpt + EMA + SIGTERM + resume)
- skeleton of `train/ufweb.py` (Ultra-FineWeb streaming) — tokenizer swap
- HF push pattern from `scripts/push_step131778.py`
- `run_100b.sh` + `run_100b_supervised.sh` patterns

Rewritten for the larger scale:
- `model.py` — bigger config, **chunked CE loss** (151 k vocab × 8 ×
  2048 = 4.6 GB of fp16 logits — must be tiled).
- `train/ufweb.py` — Qwen3 tokenizer, no vocab clipping (we own the full
  151 936 ids).

## Done conditions

- VERIFY.md all-pass at 100 B tokens (or at the saturation point if we
  flatline earlier and user approves stopping).
- Final ckpt pushed to HF Hub as `AlexWortega/ml-intern-moe200m-qwen3-…`.
- EVAL_100B.md with the 6-task zero-shot suite vs. our-21B / qwen2.5-0.5B
  / LFM2-350M baselines.
