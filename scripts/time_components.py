"""Time the major components of one forward+backward step using cuda events.

No torch.profiler — just manual timing on each block & loss kernel.
"""
from __future__ import annotations
import os, sys, time, json
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from model import MoEModel, small_config


def time_block(fn, n=5, warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def main():
    device = "cuda"
    cfg = small_config()
    cfg.ce_chunk_tokens = 512
    cfg.ce_checkpoint_chunks = True
    model = MoEModel(cfg).to(device).train()
    # selective ckpt on MoE
    import torch.utils.checkpoint as ckpt
    for blk in model.blocks:
        if not blk.is_moe: continue
        orig = blk.ffn.forward
        def mw(o):
            def w(x): return ckpt.checkpoint(lambda x: o(x), x, use_reentrant=True)
            return w
        blk.ffn.forward = mw(orig)

    B, S = 8, 2048
    V = cfg.vocab_size
    ids = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
    lbl = ids

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.cuda.amp.GradScaler(init_scale=2.0**14)

    # warmup full step
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, loss, _ = model(ids, labels=lbl)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()

    print("=== Full step ===")
    def step_full():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, loss, _ = model(ids, labels=lbl)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    t_full = time_block(step_full, n=5, warmup=2)
    print(f"full step: {t_full:.1f} ms  tok/s={B*S/(t_full/1000):.0f}")

    print("\n=== Forward only (no backward) ===")
    def step_fwd():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, loss, _ = model(ids, labels=lbl)
    t_fwd = time_block(step_fwd, n=5, warmup=2)
    print(f"fwd: {t_fwd:.1f} ms")

    # skip "no labels" forward — would materialize full [B,S,V] logits and OOM.

    print("\n=== Block-by-block forward ===")
    # Time embed + N blocks + final_norm + CE separately
    with torch.cuda.amp.autocast(dtype=torch.float16):
        def t_embed():
            x = model.embed(ids)
        t_e = time_block(t_embed, n=10, warmup=3)
        x0 = model.embed(ids)
        def t_block0():
            x_local = x0.clone()
            for blk in model.blocks:
                out = blk(x_local)
                x_local = out[0] if isinstance(out, tuple) else out
        t_all_blocks = time_block(t_block0, n=3, warmup=1)
        # Per-block (dense layer 0)
        def t_block_dense():
            _ = model.blocks[0](x0.clone())
        t_dense = time_block(t_block_dense, n=10, warmup=3)
        # MoE block
        def t_block_moe():
            _ = model.blocks[1](x0.clone())
        t_moe = time_block(t_block_moe, n=10, warmup=3)
        # MoE attn-only (attn + attn_norm)
        x_after_blk0 = model.blocks[0](x0)
        x1 = x_after_blk0[0] if isinstance(x_after_blk0, tuple) else x_after_blk0
        def t_attn_only():
            _ = model.blocks[1].attn(model.blocks[1].attn_norm(x1))
        t_attn = time_block(t_attn_only, n=10, warmup=3)
        # MoE ffn-only (router + dispatch + experts)
        x_post_attn = x1 + model.blocks[1].attn(model.blocks[1].attn_norm(x1))
        def t_ffn_only():
            _ = model.blocks[1].ffn(model.blocks[1].ffn_norm(x_post_attn))
        t_ffn = time_block(t_ffn_only, n=10, warmup=3)
        # CE alone — skip (OOM without backward path holding refs); estimate as fwd - sum(blocks)
        t_ce_v = max(0.0, t_fwd - (t_e + t_dense + 15*t_moe))
    print(f"embed:            {t_e:.1f} ms")
    print(f"all 16 blocks:    {t_all_blocks:.1f} ms")
    print(f"  dense block (layer 0): {t_dense:.1f} ms")
    print(f"  MoE block (layer 1):   {t_moe:.1f} ms")
    print(f"    attn alone:    {t_attn:.1f} ms")
    print(f"    MoE FFN alone: {t_ffn:.1f} ms")
    print(f"CE (derived = fwd - blocks):     {t_ce_v:.1f} ms")
    print()
    print(f"=== Expected fwd breakdown (per step) ===")
    print(f"   embed:                 {t_e:.1f} ms")
    print(f"   1 dense block:         {t_dense:.1f} ms")
    print(f"   15 MoE blocks:         {t_moe*15:.1f} ms  ({t_moe:.1f}/each)")
    print(f"     attn:                {t_attn*15:.1f} ms  ({t_attn:.1f}/each)")
    print(f"     ffn:                 {t_ffn*15:.1f} ms  ({t_ffn:.1f}/each)")
    print(f"   ce:                    {t_ce_v:.1f} ms")
    print(f"   sum:                   {t_e + t_dense + 15*t_moe + t_ce_v:.1f} ms")
    print(f"   actual fwd:            {t_fwd:.1f} ms")


if __name__ == "__main__":
    main()
