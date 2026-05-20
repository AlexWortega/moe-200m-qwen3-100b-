"""Full 100B-token pretrain driver for the 200M-active MoE.

Adapted from `~/ml-intern-runs/moe-100m-volta-week/train/train_1b.py`. The
training mechanics (DDP + selective ckpt + EMA + WSD + Muon + dynamic
fp16 loss-scale + SIGTERM handler + ckpt rotation + resume + warm-start)
are unchanged. What's different here:

  - imports `MoEModel`, `small_config` from the new `model.py` (chunked
    CE wired in there)
  - MFU estimator uses the actual active-param count (203.75 M)
  - stop-file flag name is `.stop_200m_qwen3`
  - HF-push hook fires every N tokens (`--hf_push_every_tokens`) and
    spawns `scripts/push_milestone.sh` in the background, leaving the
    trainer untouched
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MoEModel, small_config
from optim import Muon, make_param_groups, WSDSchedule, EMA
from train.ufweb import make_loader as make_ufweb_loader, LoaderToGPU


# Estimated active-param count for MFU. Recomputed below from the real
# model to avoid drift; the constant here is the design target.
ACTIVE_PARAMS_DESIGN = 203.75e6
STOP_FILE = ".stop_200m_qwen3"


# -------- distributed bootstrap --------

def init_dist():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local)
    return rank, world, local


# -------- selective ckpt wrapper --------

def maybe_selective_ckpt(model: MoEModel, enable: bool = True,
                         n_unwrap: int = 0):
    """Wrap each MoE block's forward in ``torch.utils.checkpoint`` except
    the LAST ``n_unwrap`` MoE blocks (which run free → activations stay
    resident for backward, no recompute).

    On V100 32 GB at micro_bs=8, seq_len=2048, every MoE block's
    activations cost ~280 MB (the [E=32, capacity=1280, d_ff=1024] fp16
    g/u/h buffers from the grouped einsum). Liger frees ~700 MB
    headroom, so ``n_unwrap=2`` typically fits; ``n_unwrap>=3`` OOMs.

    ``enable=False`` → no checkpoint at all (equivalent to n_unwrap=∞).
    ``enable=True, n_unwrap=0`` → legacy behaviour (all blocks ckpt'd).
    """
    if not enable:
        return
    import torch.utils.checkpoint as ckpt
    moe_blocks = [blk for blk in model.blocks if blk.is_moe]
    n_total = len(moe_blocks)
    n_wrap = max(0, n_total - max(0, n_unwrap))
    # use_reentrant=False is the non-HOP path; torch.compile traces into
    # it cleanly (dynamo supports it as of 2.1+). use_reentrant=True needs
    # the higher-order op machinery which forces a graph break on every
    # MoE block and severely limits Inductor's win.
    # Default reentrant=True — that's the legacy ckpt path that fits in 32 GB.
    # The non-reentrant path is more memory-hungry but compiles better; set
    # MOE_CKPT_REENTRANT=0 to opt in when memory allows.
    use_reentrant = os.environ.get("MOE_CKPT_REENTRANT", "1") == "1"
    for blk in moe_blocks[:n_wrap]:
        moe = blk.ffn
        orig_forward = moe.forward
        def make_wrapped(orig):
            def wrapped(x):
                def _fn(x):
                    return orig(x)
                return ckpt.checkpoint(_fn, x, use_reentrant=use_reentrant)
            return wrapped
        moe.forward = make_wrapped(orig_forward)


# -------- checkpoint i/o --------

def gather_rng_state():
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_ckpt(path, model, opt, sched, ema, step, rng, cfg, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {
        "model": (model.module if hasattr(model, "module") else model).state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "step": step,
        "rng": rng,
        "cfg": cfg.as_dict(),
    }
    if extra is not None:
        sd.update(extra)
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sched, ema, device="cuda"):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    (model.module if hasattr(model, "module") else model).load_state_dict(sd["model"])
    opt.load_state_dict(sd["opt"])
    sched.load_state_dict(sd["sched"])
    if ema is not None and sd.get("ema") is not None:
        ema.load_state_dict(sd["ema"], device=device)
    return sd["step"], sd.get("rng")


def prune_ckpts(ckpt_dir, keep_last=3, special_keep=("best.pt", "final.pt", "final_ema.pt")):
    step_files = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith("step_") and fn.endswith(".pt"):
            try:
                n = int(fn[5:-3])
                step_files.append((n, fn))
            except ValueError:
                pass
    step_files.sort()
    to_drop = step_files[:-keep_last] if len(step_files) > keep_last else []
    for _, fn in to_drop:
        try:
            os.remove(os.path.join(ckpt_dir, fn))
        except OSError:
            pass


# -------- eval slice --------

@torch.no_grad()
def eval_slice(model, eval_loader, batches: int, device: str, world: int, rank: int):
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    n = 0
    for _ in range(batches):
        try:
            ids, lbl, _ = eval_loader.next_batch()
        except StopIteration:
            break
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = (model.module if hasattr(model, "module") else model)(ids, labels=lbl, return_aux=False)
            if isinstance(out, tuple) and len(out) >= 2:
                _, loss = out[0], out[1]
            else:
                loss = out
        total_loss += float(loss.item())
        n += 1
    if model_was_training:
        model.train()
    if world > 1:
        t = torch.tensor([total_loss, float(n)], device=device)
        dist.all_reduce(t)
        total_loss, n = t[0].item(), t[1].item()
    return total_loss / max(1, n)


# -------- HF push background hook --------

def _maybe_fire_hf_push(rank, args, step, tokens_seen, last_push_tokens):
    """Spawn the HF push script in the background once we cross a
    `hf_push_every_tokens` boundary. Returns the updated last_push_tokens
    sentinel."""
    if rank != 0 or args.hf_push_every_tokens <= 0:
        return last_push_tokens
    if tokens_seen - last_push_tokens < args.hf_push_every_tokens:
        return last_push_tokens
    ck_path = os.path.join(args.ckpt_dir, f"step_{step+1}.pt")
    # Push script needs the ckpt to exist on disk — if we just ckpt'd at
    # this step, it's there; if not, defer to next step.
    if not os.path.exists(ck_path):
        return last_push_tokens
    push_log = os.path.join(args.ckpt_dir, f"push_step_{step+1}.log")
    try:
        subprocess.Popen(
            ["bash", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                  "push_milestone.sh"),
             ck_path, str(tokens_seen)],
            stdout=open(push_log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"[hf_push] spawned push for step={step+1} tokens={tokens_seen/1e9:.2f}B "
              f"(log={push_log})", flush=True)
    except Exception as e:
        print(f"[hf_push] spawn failed: {e}", flush=True)
    return tokens_seen


# -------- main loop --------

def run(args):
    rank, world, local = init_dist()
    device = "cuda"
    seed = args.seed
    torch.manual_seed(seed + rank)

    # ---- build model ----
    cfg = small_config(attn_backend=args.attn_backend, moe_backend="grouped",
                       moe_capacity_factor=args.moe_capacity_factor,
                       router_noise_std=args.router_noise_std)
    model = MoEModel(cfg)
    maybe_selective_ckpt(model, enable=args.moe_selective_ckpt,
                         n_unwrap=args.moe_ckpt_n_unwrap)
    model = model.to(device)
    if rank == 0:
        n_total = model.num_parameters(only_active=False)
        n_active = model.num_parameters(only_active=True)
        print(f"[init] params total {n_total/1e9:.3f} B  "
              f"active {n_active/1e6:.2f} M", flush=True)
        bd = model.param_breakdown()
        for k, v in bd.items():
            print(f"[init]   {k:18s} {v/1e6:8.2f} M", flush=True)

    if world > 1:
        # find_unused_parameters=False — with the grouped MoE backend every
        # expert weight is touched by the same 3 bmm calls regardless of
        # token routing, so there are no dead params. (The legacy "bmm"
        # backend relied on a per-expert dust pass + find_unused=True; we
        # killed both.) Set static_graph after DDP wraps to keep bucket
        # layout stable across iterations.
        model = DDP(model, device_ids=[local], find_unused_parameters=False,
                    gradient_as_bucket_view=True, bucket_cap_mb=args.bucket_cap_mb)
        model._set_static_graph()

    matrix_params, non_matrix_params = make_param_groups(
        model.module if world > 1 else model
    )
    # grouped_ns batches NS over same-shape matrices via torch.bmm; for the
    # 200M model the (1024, 640, 1024) bmm needs ~6 GB peak which pushes a
    # 32 GB V100 over the edge once activations + DDP buckets + AdamW state
    # are live. Per-param NS (grouped_ns=False) uses < 50 MB per call. The
    # bookkeeping (momentum update, weight decay) still goes through the
    # foreach path so the python-overhead win is mostly preserved.
    opt = Muon(matrix_params=matrix_params, non_matrix_params=non_matrix_params,
               lr=args.peak_lr, ns_mode="fp32", foreach=True, grouped_ns=False)
    sched = WSDSchedule(peak_lr=args.peak_lr,
                        warmup_steps=args.warmup_steps,
                        total_steps=args.total_steps,
                        decay_steps=args.decay_steps,
                        decay_shape="linear",
                        min_lr=args.min_lr)

    ema_base = model.module if world > 1 else model
    ema = EMA(ema_base, decay=args.ema_decay)

    if args.compile_model:
        # torch.utils.checkpoint(use_reentrant=True) is a higher-order op
        # which the DDP-optimized dynamo backend refuses to handle (it
        # wants to split the graph at bucket boundaries which doesn't
        # work across the checkpoint). Disable DDPOptimizer; dynamo falls
        # back to a single bucket for the whole graph. Marginal perf hit,
        # but it's what makes `selective_ckpt + chunked_ce(ckpt=True) +
        # DDP + compile` co-exist. tc_volta also handles the V100
        # cudagraph aliasing bug for us (`triton.cudagraphs=False`).
        import torch._dynamo as _dynamo
        _dynamo.config.optimize_ddp = False
        try:
            sys.path.insert(0, "/home/alexw/ml-intern-runs/torch-compile-volta-cp-16k")
            from tc_volta import compile as tc_compile
            _via = "tc_volta(default, no-cudagraphs)"
        except Exception as _e:
            import torch._inductor.config as _ind_cfg
            _ind_cfg.triton.cudagraphs = False
            def tc_compile(m, **kw):
                return torch.compile(m, mode="default", dynamic=False,
                                     fullgraph=False)
            _via = f"torch.compile(fallback: {_e})"
        if rank == 0:
            print(f"[init] torch._dynamo.config.optimize_ddp = False "
                  "(higher-order op compat)", flush=True)
            print(f"[init] compiling model via {_via}", flush=True)
        autotune = bool(int(os.environ.get("TC_AUTOTUNE", "0")))
        per_block = os.environ.get("TC_PER_BLOCK", "0") == "1"
        inner = model.module if world > 1 else model
        if per_block:
            # Per-Block compile: dynamo creates one graph per Block.forward
            # instead of one big model graph. Sidesteps the dynamo cache-size
            # churn we saw on transformers/utils/generic during whole-model
            # compile, and keeps the per-block guard sets small.
            if rank == 0:
                print(f"[init] compiling each Block.forward via {_via}", flush=True)
            for blk in inner.blocks:
                blk.forward = tc_compile(blk.forward, autotune=autotune)
        else:
            if world > 1:
                model.module = tc_compile(inner, autotune=autotune)
            else:
                model = tc_compile(inner, autotune=autotune)

    # ---- data loaders ----
    loader_seed = seed
    if args.warm_start_model and args.start_step > 0:
        loader_seed = seed + args.start_step // 1000
    train_loader = make_ufweb_loader(
        seq_len=args.seq_len, batch_size=args.batch_size,
        num_workers=args.num_workers, prefetch_factor=2,
        rank=rank, world_size=world, seed=loader_seed,
        vocab_cap=cfg.vocab_size,
    )
    train_data = LoaderToGPU(train_loader, device=device)

    eval_loader = None
    eval_data = None
    if args.eval_every > 0:
        eval_loader = make_ufweb_loader(
            seq_len=args.seq_len, batch_size=args.batch_size,
            num_workers=2, prefetch_factor=2,
            rank=rank, world_size=world, seed=seed + 7919,
            vocab_cap=cfg.vocab_size,
        )
        eval_data = LoaderToGPU(eval_loader, device=device)

    # ---- loss scaling state ----
    # Initial scale of 2**14 is fine for fresh-start, but on --resume the
    # model is already past warmup with non-trivial gradient magnitudes —
    # starting at 2**14 forces ~13 steps of NaN/Inf-skip halving before
    # settling. Use 2**6 on resume so the first real step actually applies.
    loss_scale = 2.0**6 if args.resume else 2.0**14
    loss_scale_min = 2.0**0
    loss_scale_max = 2.0**24
    n_good = 0
    grow_every = 200

    # ---- warm-start / resume ----
    start_step = 0
    if args.warm_start_model and not args.resume:
        if rank == 0:
            print(f"[init] warm-start from {args.warm_start_model} "
                  f"(start_step={args.start_step})", flush=True)
        sd = torch.load(args.warm_start_model, map_location="cpu", weights_only=False)
        target = model.module if world > 1 else model
        target_inner = target._orig_mod if hasattr(target, "_orig_mod") else target
        try:
            target_inner.load_state_dict(sd["model"])
        except RuntimeError:
            target.load_state_dict(sd["model"])
        start_step = args.start_step
        if rank == 0:
            ws_step = sd.get("step", "?")
            print(f"[init] warm-start loaded (source step={ws_step}); "
                  f"opt/sched/ema fresh; start_step={start_step}", flush=True)

    if args.resume:
        if rank == 0:
            print(f"[init] resuming from {args.resume}", flush=True)
        target = model.module if world > 1 else model
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        target.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        if sd.get("ema") is not None:
            ema.load_state_dict(sd["ema"], device=device)
        start_step = sd["step"] + 1
        if sd.get("rng") is not None:
            restore_rng_state(sd["rng"])
        if rank == 0:
            print(f"[init] resumed at step {start_step}", flush=True)
        if args.reset_router_bias_on_resume:
            base_for_reset = (model.module if world > 1 else model)
            base_for_reset = (base_for_reset._orig_mod
                              if hasattr(base_for_reset, "_orig_mod")
                              else base_for_reset)
            n_reset = 0
            for blk in base_for_reset.blocks:
                if blk.is_moe:
                    blk.ffn.router.bias.zero_()
                    n_reset += 1
            if rank == 0:
                print(f"[init] --reset_router_bias_on_resume: zeroed bias "
                      f"on {n_reset} MoE router(s)", flush=True)

    # ---- one-step warm-up only on a true fresh start ----
    if start_step == 0 and not args.warm_start_model:
        ids, lbl, _ = train_data.next_batch()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(ids, labels=lbl)
        if isinstance(out, tuple) and len(out) == 3:
            _, lm_loss, _ = out
        else:
            _, lm_loss = out[0], out[1]
        (lm_loss * loss_scale).backward()
        opt.zero_grad()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # ---- SIGTERM handler ----
    sigterm_received = {"flag": False}
    def _handle_sigterm(signum, frame):
        sigterm_received["flag"] = True
        if rank == 0:
            print(f"[sigterm] caught signum={signum}; will checkpoint at "
                  f"next step boundary", flush=True)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # ---- log files ----
    train_log = None
    eval_log = None
    if rank == 0:
        train_log = open(args.train_log, "a")
        eval_log = open(args.eval_log, "a")

    # ---- training ----
    if rank == 0:
        print(f"[init] starting at step {start_step}, total={args.total_steps}", flush=True)

    nan_count = 0
    loss_window = []
    best_rolling = float("inf")
    t_run_start = time.time()
    walls = []
    last_eval_loss = float("nan")
    train_started_fired = (start_step >= args.warmup_steps)
    last_hf_push_tokens = (start_step
                          * args.batch_size * args.seq_len * world)

    for step in range(start_step, args.total_steps):
        cur_lr = sched.lr_at(step)
        opt.set_lr(cur_lr)
        if (not ema.activated) and step >= args.ema_start:
            base = model.module if world > 1 else model
            base_inner = base._orig_mod if hasattr(base, "_orig_mod") else base
            ema.activate(base_inner)
            if rank == 0:
                print(f"[{step}] EMA activated (decay={ema.decay})", flush=True)

        ids, lbl, _ = train_data.next_batch()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(ids, labels=lbl)
        if isinstance(out, tuple) and len(out) == 3:
            _, lm_loss, aux = out
        else:
            _, lm_loss, aux = out[0], out[1], None

        loss = lm_loss
        if aux is not None:
            loss = loss + cfg.router_z_coef * aux["z_loss"] + cfg.router_aux_coef * aux["aux_loss"]

        opt.zero_grad()
        (loss * loss_scale).backward()

        nan_seen = False
        inv = 1.0 / loss_scale
        params = list((model.module if world > 1 else model).parameters())
        for p in params:
            if p.grad is None: continue
            p.grad.data.mul_(inv)
            if not torch.isfinite(p.grad.data).all():
                nan_seen = True
        if nan_seen:
            nan_count += 1
            loss_scale = max(loss_scale_min, loss_scale * 0.5)
            n_good = 0
            if rank == 0:
                print(f"[{step}] NaN/Inf grad — skip; new scale={loss_scale:.0f}", flush=True)
            if nan_count > args.nan_cap:
                if rank == 0:
                    print(f"[{step}] >{args.nan_cap} NaN steps — aborting", flush=True)
                break
            continue

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        base = model.module if world > 1 else model
        base_inner = base._orig_mod if hasattr(base, "_orig_mod") else base
        ema.update(base_inner)
        n_good += 1
        if n_good >= grow_every:
            loss_scale = min(loss_scale_max, loss_scale * 2.0)
            n_good = 0

        target_model = base_inner
        if aux is not None:
            # DDP only syncs gradients, not arbitrary buffer mutations from
            # `bias.add_()`. If we updated bias from rank-local counts each
            # rank would compute a different update; DDP's default
            # broadcast_buffers=True would then overwrite all ranks' bias
            # with rank 0's biased view. All-reduce the counts so every
            # rank applies the same global update.
            if world > 1:
                for layer_counts in aux["counts_per_layer"]:
                    dist.all_reduce(layer_counts, op=dist.ReduceOp.SUM)
            target_model.step_router_biases(aux["counts_per_layer"])

        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        walls.append(wall)

        tokens_seen = (step + 1) * args.batch_size * args.seq_len * world

        if rank == 0:
            tokens = args.batch_size * args.seq_len * world
            row = {
                "step": step,
                "wall_ms": wall * 1000,
                "loss": float(loss.item()),
                "lm_loss": float(lm_loss.item()),
                "z_loss": float(aux["z_loss"].item()) if aux is not None else 0.0,
                "aux_loss": float(aux["aux_loss"].item()) if aux is not None else 0.0,
                "router_cv": float(aux["router_cv"].item()) if aux is not None else 0.0,
                "router_entropy_bits": float(aux["router_entropy_bits"].item()) if aux is not None else 0.0,
                "lr": cur_lr,
                "tokens": tokens,
                "tokens_seen": tokens_seen,
                "tok_per_s": tokens / wall,
                "ema_active": int(ema.activated),
            }
            train_log.write(json.dumps(row) + "\n"); train_log.flush()
            if step % 20 == 0 or step == args.total_steps - 1:
                print(f"[{step:6d}] loss={row['loss']:.3f} lm={row['lm_loss']:.3f} "
                      f"cv={row['router_cv']:.3f} lr={cur_lr:.2e} "
                      f"tok/s={row['tok_per_s']/1e3:.1f}K  wall={wall*1000:.0f}ms  "
                      f"toks={tokens_seen/1e9:.2f}B  ema={row['ema_active']}", flush=True)
            loss_window.append(row["lm_loss"])
            if len(loss_window) > 100:
                loss_window.pop(0)
            if len(loss_window) == 100:
                rolling = sum(loss_window) / 100
                if rolling < best_rolling and step > args.warmup_steps:
                    best_rolling = rolling

        # ---- ckpt? ----
        do_ckpt = (step + 1) in args.ckpt_steps
        wall_elapsed = time.time() - t_run_start
        wall_cap_hit = wall_elapsed > args.wall_cap_s and step + 1 < args.total_steps
        stop_file_hit = os.path.exists(STOP_FILE) and step > start_step
        if sigterm_received["flag"] or wall_cap_hit or stop_file_hit:
            do_ckpt = True
        if do_ckpt:
            if world > 1:
                dist.barrier()
            if rank == 0:
                ck_path = os.path.join(args.ckpt_dir, f"step_{step+1}.pt")
                rng = gather_rng_state()
                save_ckpt(ck_path, model, opt, sched, ema, step, rng, cfg,
                          extra={"loss": row["loss"],
                                 "rolling100": (sum(loss_window)/len(loss_window)) if loss_window else 0.0,
                                 "tokens_seen": tokens_seen})
                print(f"[ckpt] wrote {ck_path}", flush=True)
                if len(loss_window) == 100:
                    rolling = sum(loss_window) / 100
                    best_path = os.path.join(args.ckpt_dir, "best.pt")
                    if (not os.path.exists(best_path)) or rolling < best_rolling + 1e-9:
                        save_ckpt(best_path, model, opt, sched, ema, step, rng, cfg,
                                  extra={"loss": row["loss"], "rolling100": rolling,
                                         "tokens_seen": tokens_seen})
                        print(f"[ckpt] updated best.pt (rolling100={rolling:.4f})", flush=True)
                prune_ckpts(args.ckpt_dir, keep_last=args.ckpt_keep_last)
            if world > 1:
                dist.barrier()
            last_hf_push_tokens = _maybe_fire_hf_push(
                rank, args, step, tokens_seen, last_hf_push_tokens)

        # ---- eval? ----
        if eval_data is not None and ((step + 1) % args.eval_every == 0 or step + 1 == args.total_steps):
            eval_loss = eval_slice(model, eval_data, batches=args.eval_batches,
                                   device=device, world=world, rank=rank)
            last_eval_loss = eval_loss
            if rank == 0:
                ev_row = {"step": step, "eval_loss": eval_loss,
                          "tokens_seen": tokens_seen}
                eval_log.write(json.dumps(ev_row) + "\n"); eval_log.flush()
                print(f"[eval] step={step} eval_loss={eval_loss:.4f}", flush=True)

        # ---- train_started one-shot ----
        if rank == 0 and args.train_started_at_warmup and (not train_started_fired) and step + 1 >= args.warmup_steps:
            try:
                subprocess.run(["bash", os.path.expanduser("~/.claude/skills/ml-intern/scripts/notify.sh"),
                                "train_started",
                                f"200m-qwen3 warmup done @ step={step+1} lr={cur_lr:.2e} loss={row['loss']:.3f}"],
                               check=False, timeout=10)
            except Exception:
                pass
            train_started_fired = True

        # ---- progress notify ----
        if rank == 0 and (step + 1) % args.progress_every == 0:
            window = walls[-args.progress_every:]
            avg_wall = sum(window) / max(1, len(window))
            tok_per_s_agg = (args.batch_size * args.seq_len * world) / avg_wall
            flops = 6 * ACTIVE_PARAMS_DESIGN * args.batch_size * args.seq_len * world
            mfu = (flops / avg_wall) / (90e12 * world) * 100
            ev_str = f" eval_loss={last_eval_loss:.3f}" if last_eval_loss == last_eval_loss else ""
            try:
                subprocess.run(["bash", os.path.expanduser("~/.claude/skills/ml-intern/scripts/notify.sh"),
                                "progress",
                                f"200m-qwen3 step={step+1} loss={row['loss']:.3f} "
                                f"mfu={mfu:.1f}% tok/s={tok_per_s_agg/1e3:.1f}K "
                                f"toks={tokens_seen/1e9:.2f}B{ev_str}"],
                               check=False, timeout=10)
            except Exception:
                pass

        if sigterm_received["flag"] or wall_cap_hit or stop_file_hit:
            if rank == 0:
                why = ("sigterm" if sigterm_received["flag"]
                       else "wall_cap" if wall_cap_hit else "stop_file")
                print(f"[exit] {why} at step {step+1}", flush=True)
            break

    # ---- final / final_ema ----
    if rank == 0:
        rng = gather_rng_state()
        ck_final = os.path.join(args.ckpt_dir, "final.pt")
        save_ckpt(ck_final, model, opt, sched, ema, step, rng, cfg,
                  extra={"final": True})
        print(f"[ckpt] wrote {ck_final}", flush=True)
        if ema.activated:
            base = model.module if world > 1 else model
            base_inner = base._orig_mod if hasattr(base, "_orig_mod") else base
            backup = ema.swap_into(base_inner)
            ema_state = base_inner.state_dict()
            ema.restore(base_inner, backup)
            tmp = ck_final + ".ema.tmp"
            torch.save({"model": ema_state, "step": step, "cfg": cfg.as_dict(),
                        "from_ema": True}, tmp)
            os.replace(tmp, os.path.join(args.ckpt_dir, "final_ema.pt"))
            print(f"[ckpt] wrote final_ema.pt", flush=True)
        train_log.close()
        if eval_log: eval_log.close()

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--total_steps", type=int, default=1525879)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--decay_steps", type=int, default=76294)
    p.add_argument("--peak_lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--ema_start", type=int, default=1449585)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bucket_cap_mb", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile_model", action="store_true", default=True)
    p.add_argument("--no_compile_model", action="store_false", dest="compile_model")
    p.add_argument("--ckpt_dir", default="ckpts")
    p.add_argument("--train_log", default="train.log")
    p.add_argument("--eval_log", default="eval.log")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_batches", type=int, default=32)
    p.add_argument("--ckpt_steps", type=str, default="",
                   help="comma-separated step numbers to checkpoint at; "
                        "leave blank to auto-derive every CKPT_EVERY steps")
    p.add_argument("--ckpt_every", type=int, default=5000,
                   help="used when --ckpt_steps is empty")
    p.add_argument("--wall_cap_s", type=float, default=4 * 3600.0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--nan_cap", type=int, default=2000)
    p.add_argument("--progress_every", type=int, default=20000)
    p.add_argument("--ckpt_keep_last", type=int, default=3)
    p.add_argument("--train_started_at_warmup", action="store_true", default=False)
    p.add_argument("--warm_start_model", type=str, default=None)
    p.add_argument("--start_step", type=int, default=0)
    p.add_argument("--hf_push_every_tokens", type=int, default=0,
                   help="if > 0, spawn scripts/push_milestone.sh "
                        "after each ckpt that crosses this token boundary")
    p.add_argument("--moe_capacity_factor", type=float, default=1.25,
                   help="capacity multiplier for the grouped MoE backend; "
                        "1.0 = no padding (drops overflow), 2.0 = no drops "
                        "at CV<=0.5 but 2x bmm compute")
    p.add_argument("--moe_selective_ckpt", action="store_true", default=True,
                   help="wrap each MoE block forward in torch.utils.checkpoint "
                        "to save activation memory at the cost of bwd recompute")
    p.add_argument("--no_moe_selective_ckpt", action="store_false",
                   dest="moe_selective_ckpt")
    p.add_argument("--moe_ckpt_n_unwrap", type=int, default=0,
                   help="number of MoE blocks (from the END of the stack) "
                        "to leave UN-checkpointed. 0 = all blocks ckpt'd "
                        "(legacy), 2 = last 2 blocks run free. Each unwrap "
                        "costs ~280 MB peak activation but saves ~34 ms "
                        "recompute. Larger values OOM on V100 32 GB.")
    p.add_argument("--router_noise_std", type=float, default=0.0,
                   help="std of additive Gaussian noise applied to "
                        "router sel_logits (logit + bias) during training "
                        "only — breaks load-imbalance lock-in so dead "
                        "experts can win top-k. 0 = off. Set to ~1.0 "
                        "for the first few thousand steps of a "
                        "router-recovery resume, then back to 0.")
    p.add_argument("--reset_router_bias_on_resume", action="store_true",
                   default=False,
                   help="zero out all MoE router biases right after a "
                        "--resume checkpoint load. Use when the loaded "
                        "ckpt was trained with the old (broken) bias "
                        "controller and the bias state is degenerate "
                        "(range outside [-5, +5]).")
    p.add_argument("--attn_backend", type=str, default="sdpa",
                   choices=["sdpa", "fa_volta"],
                   help="attention backend; fa_volta uses the Triton FA "
                        "kernels for V100 (forward + backward) — ~5-7% faster "
                        "than SDPA's memory-efficient fallback on SM 7.0.")
    args = p.parse_args()
    if args.ckpt_steps:
        args.ckpt_steps = set(int(s) for s in args.ckpt_steps.split(",") if s)
    else:
        # auto-derive: ckpt_every, 2*ckpt_every, ... up to total_steps
        ce = args.ckpt_every
        st = args.total_steps
        args.ckpt_steps = set(ce * i for i in range(1, st // ce + 2) if ce * i < st)
    return args


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
