"""Numerically validate `tiled_cross_entropy` matches a bare
`F.cross_entropy` (single-shot) on identical inputs.

Tested invariants:
  - loss value: |chunked - bare| < 1e-4 (fp32 hidden, fp32 embed)
  - hidden-state gradients: max abs diff < 1e-4
  - embed-weight gradients: max abs diff < 1e-4

The chunked path tiles along (B·S) with `ce_chunk_tokens` chunk size and
wraps each chunk in `torch.utils.checkpoint`. The bare path computes the
full (B·S, V) logits matrix once and runs `F.cross_entropy` on it.

Vocab is kept smaller than production (8192 vs 151 936) so the bare
single-shot path fits in memory of the test harness; the tile-size is
chosen to *not* divide N evenly so we cover the ragged-last-chunk case.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import tiled_cross_entropy


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    V = 8192
    D = 256
    B = 3
    S = 137  # deliberately not a multiple of CHUNK
    CHUNK = 64
    MUP = 0.7

    h = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)
    W = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True) * 0.02

    labels = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
    # punch a few ignore-index holes to test the n_valid path
    ignore_mask = torch.rand(B, S, device=device) < 0.1
    labels = labels.masked_fill(ignore_mask, -100)

    # ---- bare reference ----
    h_a = h.detach().clone().requires_grad_(True)
    W_a = W.detach().clone().requires_grad_(True)
    logits = F.linear(h_a.reshape(B * S, D), W_a) * MUP
    loss_a = F.cross_entropy(logits.float(), labels.reshape(-1).long(),
                             ignore_index=-100, reduction="mean")
    loss_a.backward()
    grad_h_a = h_a.grad.detach().clone()
    grad_W_a = W_a.grad.detach().clone()

    # ---- chunked + checkpoint ----
    h_b = h.detach().clone().requires_grad_(True)
    W_b = W.detach().clone().requires_grad_(True)
    loss_b = tiled_cross_entropy(
        h_b.reshape(B * S, D), W_b, labels.reshape(-1).long(),
        mup_scale=MUP, chunk_size=CHUNK, use_checkpoint=True,
    )
    loss_b.backward()
    grad_h_b = h_b.grad.detach().clone()
    grad_W_b = W_b.grad.detach().clone()

    # ---- chunked without checkpoint ----
    h_c = h.detach().clone().requires_grad_(True)
    W_c = W.detach().clone().requires_grad_(True)
    loss_c = tiled_cross_entropy(
        h_c.reshape(B * S, D), W_c, labels.reshape(-1).long(),
        mup_scale=MUP, chunk_size=CHUNK, use_checkpoint=False,
    )
    loss_c.backward()
    grad_h_c = h_c.grad.detach().clone()
    grad_W_c = W_c.grad.detach().clone()

    dloss_b = (loss_a - loss_b).abs().item()
    dloss_c = (loss_a - loss_c).abs().item()
    dh_b = (grad_h_a - grad_h_b).abs().max().item()
    dh_c = (grad_h_a - grad_h_c).abs().max().item()
    dW_b = (grad_W_a - grad_W_b).abs().max().item()
    dW_c = (grad_W_a - grad_W_c).abs().max().item()

    print(f"bare loss        = {loss_a.item():.6f}")
    print(f"chunked+ckpt loss= {loss_b.item():.6f}  |Δ|={dloss_b:.2e}")
    print(f"chunked no-ckpt  = {loss_c.item():.6f}  |Δ|={dloss_c:.2e}")
    print(f"grad h max abs Δ (ckpt)   = {dh_b:.2e}")
    print(f"grad h max abs Δ (no-ckpt)= {dh_c:.2e}")
    print(f"grad W max abs Δ (ckpt)   = {dW_b:.2e}")
    print(f"grad W max abs Δ (no-ckpt)= {dW_c:.2e}")

    tol_loss = 1e-4
    tol_grad = 1e-4
    failures = []
    for name, val in [("loss(ckpt)", dloss_b), ("loss(no-ckpt)", dloss_c)]:
        if val > tol_loss:
            failures.append(f"{name} {val} > {tol_loss}")
    for name, val in [("grad_h(ckpt)", dh_b), ("grad_h(no-ckpt)", dh_c),
                       ("grad_W(ckpt)", dW_b), ("grad_W(no-ckpt)", dW_c)]:
        if val > tol_grad:
            failures.append(f"{name} {val} > {tol_grad}")
    if failures:
        print("FAIL — " + "; ".join(failures))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
