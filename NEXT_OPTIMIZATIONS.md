# Next-step optimizations (post-CV-fix, post-100B-launch)

Queued by user 2026-05-18 after DeepEP triage (DeepEP not viable on V100 SM 7.0 — needs Hopper TMA / Ampere `cp.async` / NVSHMEM, none of which we have).

If 100B launches but sustained throughput is **< 40K tok/s**, dispatch a follow-up optimization round on these targets, in order of expected ROI on Volta SM 7.0:

## 1. Liger-Kernel fused chunked CE (V100-compatible)

- Repo: `https://github.com/linkedin/Liger-Kernel`
- Target: replace our current `_ce_chunk_forward` (with `torch.utils.checkpoint`) by Liger's `LigerFusedLinearCrossEntropyLoss`.
- Why: Liger's CE is a single Triton kernel that fuses `linear(h, embed.T) → softmax → NLL` without ever materializing the full logit tensor. On V100 + Triton 2.3 it works (verified by Liger's CI on T4 which is same generation).
- Current baseline cost: **168 ms / forward** for chunked CE at chunk=512 + ckpt. Liger should land at **50-70 ms** with **less** memory (no recompute needed).
- Expected throughput gain: **+15-20%** wall-clock (saves ~100 ms / step out of current ~1500 ms).
- Caveat: Liger expects `lm_head` to be a standalone `nn.Linear`. Our model ties embed + LM head, so we'll need the `LigerFusedLinearCrossEntropyLoss` flavour that takes `weight` directly. Both flavours exist in their API.

## 2. megablocks block-sparse MoE (BSMM)

- Repo: `https://github.com/databricks/megablocks`
- Target: replace agent's grouped-bmm MoE dispatch with megablocks' `dmoe` block-sparse path.
- Why: megablocks builds a single CSR-style sparse matrix of (tokens × expert_layer_weights) and runs **one** `block_sparse_matmul` instead of per-expert routing. On V100 with Triton 2.3 it compiles (their CI covers SM 7.0).
- Current baseline cost (after agent's grouped GEMM fix): **~80 ms / forward** for MoE FFN.
- Expected gain: **+10-15%** (megablocks bsmm is ~1.2× faster than naive grouped bmm at top-2 routing on small d_ff, per their paper figure 6).
- Caveat: integration is ~200 LOC; megablocks Layer takes its own dispatch tensor, we'd swap our `_moe_dispatch_bmm` for `mb.layers.dmoe.dMoE`.

## 3. Custom FA-Volta forward+backward (last resort)

- Repo: `~/flash-attn-volta/` (built earlier this month for this exact hardware)
- Target: replace SDPA with FA-Volta in the attention block.
- Why: SDPA on V100 falls back to memory-efficient attention (no FA2/3). FA-Volta is ~1.3× faster.
- Current baseline cost: **73 ms / forward** for attention.
- Expected gain: **+5-7%** wall-clock — modest.
- Caveat: only do this if combined Liger + megablocks still leave us short of 40K tok/s. Otherwise overkill.

## Skipping list (do not attempt — wasted effort on V100)

- **DeepEP** — Hopper-only kernels, NVSHMEM dependency, designed for inter-node EP. We are single-node DDP, no Hopper.
- **TransformerEngine / FP8** — Hopper-only.
- **Expert parallelism (sharded experts)** — overkill for 4 GPU; experts already fit per-GPU. Communication isn't our bottleneck.
- **Mixtral / Switch-Transformer dispatch** — algorithmically same as what we already have, no kernel win.

## Decision threshold

- Throughput ≥ 40K tok/s sustained over 1000 steps → ship as-is, don't add more optimization complexity.
- 30-40K → only do Liger CE (step 1). Skip rest. 100B in 24-30 days is acceptable.
- < 30K → do Liger + megablocks. If still < 30K, last-resort FA-Volta.
