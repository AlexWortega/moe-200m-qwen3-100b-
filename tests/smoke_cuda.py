"""Single-GPU forward+backward smoke test at production micro-batch shape.

Goal: confirm peak memory < 32 GB so a single V100 can hold one DDP rank's
share with margin. Measures:

  - peak GPU mem after forward
  - peak GPU mem after backward
  - param count breakdown
  - one finite loss value
  - one finite grad norm

Two configurations are tested in sequence:

  A) micro_bs=8 seq_len=2048 — the production target.
  B) micro_bs=4 seq_len=2048 — fallback if A blows up. With grad_acc=2
     the effective per-step token budget is the same.

For each, with selective MoE checkpointing on (matches the production
loop). The chunked-CE path is unconditional.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MoEModel, small_config


def selective_ckpt(model: MoEModel):
    for blk in model.blocks:
        if not blk.is_moe:
            continue
        moe = blk.ffn
        orig_forward = moe.forward
        def make_wrapped(orig):
            def wrapped(x):
                def _fn(x):
                    return orig(x)
                return ckpt.checkpoint(_fn, x, use_reentrant=True)
            return wrapped
        moe.forward = make_wrapped(orig_forward)


def run_one(B, S, label):
    cfg = small_config()
    model = MoEModel(cfg).to("cuda")
    selective_ckpt(model)
    n_total = model.num_parameters(only_active=False)
    n_active = model.num_parameters(only_active=True)
    print(f"\n== {label}  B={B} S={S}  total={n_total/1e9:.3f}B active={n_active/1e6:.1f}M ==")

    ids = torch.randint(0, cfg.vocab_size, (B, S), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        _, lm_loss, aux = model(ids, labels=ids)
    loss = lm_loss + cfg.router_z_coef * aux["z_loss"] + cfg.router_aux_coef * aux["aux_loss"]
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    mem_fwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  fwd: loss={loss.item():.3f}  router_cv={aux['router_cv'].item():.3f}  "
          f"wall={t_fwd*1000:.0f}ms  peak={mem_fwd:.2f}GB")

    t0 = time.perf_counter()
    (loss * 16384.0).backward()
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0
    mem_bwd = torch.cuda.max_memory_allocated() / 1e9
    # grad norm
    gn2 = 0.0
    for p in model.parameters():
        if p.grad is None: continue
        gn2 += float(p.grad.data.norm().item() ** 2)
    gn = gn2 ** 0.5
    print(f"  bwd: wall={t_bwd*1000:.0f}ms  peak={mem_bwd:.2f}GB  grad_norm={gn:.2f}")

    # cleanup
    del model, ids, lm_loss, aux, loss
    torch.cuda.empty_cache()
    return mem_fwd, mem_bwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["A", "B"], help="run only A or B")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA — abort")
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"total mem: {cap:.1f} GB")

    if args.only != "B":
        try:
            mf, mb = run_one(B=8, S=2048, label="A: micro_bs=8 seq=2048")
            if mb >= cap * 0.92:
                print("  WARN: peak ≥ 92% of GPU; not safe headroom for compile/DDP overhead")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at A: {e}")
            torch.cuda.empty_cache()
    if args.only != "A":
        mf, mb = run_one(B=4, S=2048, label="B: micro_bs=4 seq=2048")
        print(f"  (with grad_acc=2 this matches the per-step token budget of A)")


if __name__ == "__main__":
    main()
