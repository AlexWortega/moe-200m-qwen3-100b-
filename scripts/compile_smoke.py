"""Single-GPU torch.compile smoke test for the MoE model.

Goal: verify that compiling the model on V100 produces a working
forward+backward pass and that loss matches the uncompiled path within
±0.01. Run before the 4-GPU bench.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3.10 scripts/compile_smoke.py
"""
from __future__ import annotations
import sys
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/alexw/ml-intern-runs/torch-compile-volta-cp-16k")

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from model import MoEModel, small_config


def _wrap_moe_ckpt(model):
    for blk in model.blocks:
        if not getattr(blk, "is_moe", False):
            continue
        moe = blk.ffn
        orig = moe.forward
        def make(o):
            def w(x):
                return ckpt.checkpoint(lambda y: o(y), x, use_reentrant=True)
            return w
        moe.forward = make(orig)


def main():
    torch.manual_seed(0)
    device = "cuda"
    cfg = small_config(attn_backend="sdpa", moe_backend="grouped",
                       moe_capacity_factor=1.25)

    # Reference: uncompiled
    m_ref = MoEModel(cfg).to(device)
    _wrap_moe_ckpt(m_ref)
    # Match state with compiled copy
    sd = m_ref.state_dict()

    m_cmp = MoEModel(cfg).to(device)
    _wrap_moe_ckpt(m_cmp)
    m_cmp.load_state_dict(sd)

    # Compile
    try:
        from tc_volta import compile as tc_compile
        m_cmp_c = tc_compile(m_cmp, autotune=False)
        compiled_via = "tc_volta"
    except Exception as e:
        print(f"tc_volta unavailable ({e}); falling back to torch.compile")
        import torch._dynamo as _dynamo
        _dynamo.config.optimize_ddp = False
        import torch._inductor.config as _ind_cfg
        _ind_cfg.triton.cudagraphs = False
        m_cmp_c = torch.compile(m_cmp, mode="default", dynamic=False, fullgraph=False)
        compiled_via = "torch.compile/default"

    B, S = 4, 1024  # small for memory headroom
    ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)
    labels = ids.clone()

    print(f"[smoke] backend={compiled_via}")
    print(f"[smoke] running eager forward+backward …")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out_ref = m_ref(ids, labels=labels)
    _, loss_ref, _ = out_ref
    loss_ref.backward()
    print(f"[smoke] eager   loss = {loss_ref.item():.6f}")

    print(f"[smoke] running compiled forward+backward …")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out_cmp = m_cmp_c(ids, labels=labels)
    _, loss_cmp, _ = out_cmp
    loss_cmp.backward()
    print(f"[smoke] compile loss = {loss_cmp.item():.6f}")
    diff = abs(loss_ref.item() - loss_cmp.item())
    print(f"[smoke] |Δloss|     = {diff:.4e}")
    if diff < 1e-2:
        print(f"[smoke] VERDICT: PASS (gate < 1e-2)")
        return 0
    else:
        print(f"[smoke] VERDICT: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
