"""Fixed-seed parity check for the pass-2 patches.

Runs both paths (baseline + new) on the same hidden state / labels and
reports max-abs / rel-err per checked module.

Usage:
    python3.10 scripts/parity_check.py liger      # Liger CE vs tiled CE
    python3.10 scripts/parity_check.py fa_volta   # FA-Volta vs SDPA
    python3.10 scripts/parity_check.py megablocks # megablocks dMoE vs grouped
"""
from __future__ import annotations
import argparse
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import model as M


def parity_liger():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    N, D, V = 8 * 2048, 640, 151936
    h = torch.randn(N, D, device=device, dtype=dtype) * 0.5
    embed = torch.randn(V, D, device=device, dtype=dtype) * 0.02
    labels = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    labels[::197] = -100  # sprinkle ignore_index
    mup = math.sqrt(512.0 / 640.0)

    # Tiled CE reference.
    h1 = h.detach().clone().requires_grad_(True)
    e1 = embed.detach().clone().requires_grad_(True)
    loss_tiled = M.tiled_cross_entropy(h1, e1, labels, mup,
                                       chunk_size=512, use_checkpoint=False)
    loss_tiled.backward()

    # Liger.
    h2 = h.detach().clone().requires_grad_(True)
    e2 = embed.detach().clone().requires_grad_(True)
    loss_liger = M.liger_fused_cross_entropy(h2, e2, labels, mup)
    loss_liger.backward()

    diff_loss = (loss_liger - loss_tiled).abs().item()
    diff_dh = (h1.grad - h2.grad).abs().max().item()
    rel_dh = diff_dh / h1.grad.abs().max().clamp_min(1e-8).item()
    diff_de = (e1.grad - e2.grad).abs().max().item()
    rel_de = diff_de / e1.grad.abs().max().clamp_min(1e-8).item()

    print(f"loss_tiled  = {loss_tiled.item():.6f}")
    print(f"loss_liger  = {loss_liger.item():.6f}")
    print(f"|Δloss|     = {diff_loss:.3e}")
    print(f"|Δgrad_h|max= {diff_dh:.3e}  rel = {rel_dh:.3e}")
    print(f"|Δgrad_e|max= {diff_de:.3e}  rel = {rel_de:.3e}")
    # Pass criteria.
    ok = diff_loss < 1e-3 and rel_dh < 5e-2 and rel_de < 5e-2
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["liger", "fa_volta", "megablocks"])
    args = p.parse_args()
    if args.which == "liger":
        sys.exit(parity_liger())
    else:
        print(f"parity check for {args.which} not yet wired")
        sys.exit(2)
