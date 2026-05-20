"""50-step torch.profiler trace of one DDP-rank forward+backward+opt step.

Runs single-GPU (rank 0 standalone) — we don't need 4-GPU to identify the
single-GPU bottlenecks; DDP all-reduce is overlapped with backward and
contributes <5% at this scale per prior runs. Per-GPU compute is what's
slow.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3.10 scripts/profile_step.py [--ce_chunk N] [--no_ckpt_ce] [--bench_only]

Writes:
    notes/profile_trace.json     (chrome trace)
    notes/profile_summary.txt    (top-30 ops by CUDA time)
    notes/profile_walls.txt      (per-step wall-time stats)
"""
from __future__ import annotations
import argparse, os, sys, time, json
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from model import MoEModel, small_config


def make_batch(B, S, V, device):
    ids = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
    lbl = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
    return ids, lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--ce_chunk", type=int, default=None,
                    help="override cfg.ce_chunk_tokens")
    ap.add_argument("--no_ckpt_ce", action="store_true",
                    help="disable ce checkpoint recompute")
    ap.add_argument("--no_moe_ckpt", action="store_true",
                    help="disable selective MoE checkpoint wrapper")
    ap.add_argument("--bench_only", action="store_true",
                    help="skip profiler, just bench 50 steps")
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_active", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "notes"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    cfg = small_config(attn_backend="sdpa", moe_backend="bmm")
    if args.ce_chunk is not None:
        cfg.ce_chunk_tokens = args.ce_chunk
    if args.no_ckpt_ce:
        cfg.ce_checkpoint_chunks = False

    model = MoEModel(cfg).to(device)
    if not args.no_moe_ckpt:
        # mimic train_200m.maybe_selective_ckpt
        import torch.utils.checkpoint as ckpt
        for blk in model.blocks:
            if not blk.is_moe: continue
            moe = blk.ffn
            orig = moe.forward
            def make_wrap(o):
                def w(x):
                    return ckpt.checkpoint(lambda x: o(x), x, use_reentrant=True)
                return w
            moe.forward = make_wrap(orig)

    if args.compile:
        torch._inductor.config.triton.cudagraphs = False
        model = torch.compile(model, mode="default", dynamic=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.cuda.amp.GradScaler(init_scale=2.0**14)

    V = cfg.vocab_size
    ids, lbl = make_batch(args.batch_size, args.seq_len, V, device)

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(ids, labels=lbl)
        if isinstance(out, tuple) and len(out) == 3:
            _, lm_loss, aux = out
        else:
            _, lm_loss = out[0], out[1]; aux = None
        loss = lm_loss
        if aux is not None:
            loss = loss + cfg.router_z_coef * aux["z_loss"] + cfg.router_aux_coef * aux["aux_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return float(loss.item())

    # warmup (don't time)
    for _ in range(args.n_warmup):
        one_step()
    torch.cuda.synchronize()

    if args.bench_only:
        walls = []
        for i in range(args.n_active):
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
        toks = args.batch_size * args.seq_len
        tok_per_s = toks / (sum(walls) / len(walls))
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        msg = (f"bench_only n={args.n_active} avg_wall={sum(walls)/len(walls)*1000:.1f}ms "
               f"med_wall={sorted(walls)[len(walls)//2]*1000:.1f}ms "
               f"tok_per_s={tok_per_s:.0f} peak_mem={mem_gb:.2f}GB "
               f"ce_chunk={cfg.ce_chunk_tokens} ce_ckpt={cfg.ce_checkpoint_chunks} "
               f"moe_ckpt={not args.no_moe_ckpt} compile={args.compile}")
        print(msg)
        with open(os.path.join(args.out_dir, "bench.txt"), "a") as f:
            f.write(msg + "\n")
        return

    sched = schedule(wait=2, warmup=args.n_warmup, active=args.n_active, repeat=1)

    def _on_trace_ready(p):
        # write summary
        with open(os.path.join(args.out_dir, "profile_summary.txt"), "w") as f:
            f.write(p.key_averages().table(
                sort_by="cuda_time_total", row_limit=40))
            f.write("\n\n=== by self_cuda_time ===\n\n")
            f.write(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40))
        # chrome trace
        p.export_chrome_trace(os.path.join(args.out_dir, "profile_trace.json"))

    n_total = args.n_warmup + args.n_active + 2  # wait + warmup + active
    walls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sched,
                 on_trace_ready=_on_trace_ready,
                 record_shapes=False, with_stack=False,
                 profile_memory=False) as prof:
        for i in range(n_total):
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
            prof.step()

    toks = args.batch_size * args.seq_len
    bench_walls = walls[args.n_warmup + 2 : args.n_warmup + 2 + args.n_active]
    avg = sum(bench_walls) / max(1, len(bench_walls))
    tok_per_s = toks / avg
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    msg = (f"profile n_active={len(bench_walls)} avg_wall={avg*1000:.1f}ms "
           f"tok_per_s={tok_per_s:.0f} peak_mem={mem_gb:.2f}GB "
           f"ce_chunk={cfg.ce_chunk_tokens} ce_ckpt={cfg.ce_checkpoint_chunks} "
           f"moe_ckpt={not args.no_moe_ckpt} compile={args.compile}")
    print(msg)
    with open(os.path.join(args.out_dir, "profile_walls.txt"), "w") as f:
        f.write(msg + "\n")
        f.write("walls_ms = " + json.dumps([round(w*1000, 2) for w in walls]) + "\n")


if __name__ == "__main__":
    main()
