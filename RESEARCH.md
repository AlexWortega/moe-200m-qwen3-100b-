# RESEARCH — references and key design choices

## Tokenizer
- `Qwen/Qwen3-0.6B-Base` — config: `https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json`
- Declared `vocab_size = 151 936` (multiple of 128 for kernel alignment),
  with 151 643 base BPE tokens + 26 special tokens above. We adopt the
  full 151 936 so that the embedding matrix matches the standard Qwen3
  size — unused rows just stay near init.
- Fast tokenizer is available (`is_fast=True`); `eos_token_id = 151643`.

## Architecture lineage
- DeepSeek-V3 / DeepSeekMoE pattern: 1 shared + N routed experts, fine-
  grained routing, top-k. Used as basis for the 100M run.
- We keep the same Muon + AdamW split, partial-RoPE, QK-Norm,
  fp16-normalized NS-5 orthogonalization, aux-loss-free routing with a
  tiny aux-loss + z-loss regularizer.

## Throughput / memory constraints
- V100 SXM2 32 GB, fp16 (no bf16 on Volta). FA-Volta-compatible torch 2.3.
- Param count: ~1.09 B total ⇒ fp32 master (4.4 GB) + fp32 grads (4.4 GB)
  + Muon momentum on matrix params (~4 GB) + AdamW state on non-matrix
  (<1 GB) = ~14 GB optimizer/grad/state on each rank (DDP, not FSDP).
  Leaves ~16 GB for activations + workspace per rank → feasible if we
  selective-checkpoint MoE FFNs and tile the CE loss.

## Logits memory blow-up
- At vocab=151 936, micro_bs=8, seq=2048: fp16 logits = 8·2048·151936·2
  ≈ 4.66 GB *per micro-batch*, and another 4.66 GB for softmax intermediates.
  Single-shot `F.cross_entropy` would OOM before the backward even starts.
- Solution: tile the matmul `h @ embed.T` along the (B·S) axis in chunks
  of ~512 tokens; wrap each chunk in `torch.utils.checkpoint` so the
  backward recomputes the per-chunk logits on demand. Peak resident logit
  buffer = 512·151936·4 (fp32) ≈ 0.3 GB.

## References for chunked / tiled CE
- `nanotron` and `megatron-lm` both implement vocab-parallel tiled CE
  for the same reason (vocab ≥100k makes single-shot impractical).
- `apex.contrib.xentropy.SoftmaxCrossEntropyLoss` is a CUDA kernel that
  fuses softmax+CE, but it doesn't tile and isn't bundled with our torch
  2.3 / FA-Volta wheels — we roll our own seq-tile version.

## Dataset
- `openbmb/Ultra-FineWeb`, `split=en`, `column=content`. Same source as
  the 100M run so EVAL comparisons stay valid. Streaming + per-rank
  `split_dataset_by_node` for the 4-GPU setup.

## Hyperparam reasoning
- LR 6e-4 (was 3e-4 for 100M). Muon scales LR by spectral-norm of the
  ortho update, so the optimal LR varies sub-linearly with width. We bump
  by 2× because d_model is bigger and per-step gradient signal is less
  noisy.
- Batch 4·8·2048 = 65 536 tok / step. Same per-step budget as the 100M
  run for direct loss-curve comparison.
- WSD: 2 k warmup, 5 % decay tail. EMA β=0.9999 (unchanged).
- Total 100 B / 65 536 ≈ **1 525 879 steps**.

## What's deliberately *not* changing from 100M → 200M
- attention backend = SDPA (Flash-Attn-Volta gave us +5% on 100M but
  also a lot of debug pain — keep SDPA for the production run)
- moe backend = `bmm` (sorted-permute + per-expert matmul)
- selective checkpoint on MoE FFN
- DDP + static_graph + bucket_cap_mb=100
