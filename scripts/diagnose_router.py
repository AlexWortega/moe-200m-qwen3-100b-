"""Diagnose dead MoE experts in a checkpoint.

Loads a checkpoint, runs N single-GPU forward passes on the real
UltraFineWeb data, and reports per-layer / per-expert load + bias
statistics + router-weight / router-logit / score finiteness.

Run:
    CUDA_VISIBLE_DEVICES=2 python3.10 scripts/diagnose_router.py \\
        --ckpt ckpts_100b/step_7953.pt \\
        --steps 100 \\
        --batch_size 8 --seq_len 2048 \\
        --out DEAD_EXPERTS.md

Single-GPU only; bypasses DDP. Goal is to read the per-layer state of
``self.bias`` (which is rank-broadcast in DDP, so rank-0's view IS the
state) plus the actual count distribution under typical batches.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Repo root on sys.path so we can `import model`.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from model import MoEModel, small_config  # noqa: E402
from train.ufweb import make_loader as make_ufweb_loader, LoaderToGPU  # noqa: E402


@dataclass
class LayerStat:
    """Per-MoE-layer rolling stats over the diagnostic loop."""
    counts_sum: torch.Tensor   # [E] long — sum of per-step counts
    counts_min: torch.Tensor   # [E] long — min single-step count seen
    counts_max: torch.Tensor   # [E] long — max single-step count seen
    n_steps: int = 0
    bias_snapshot: torch.Tensor | None = None     # [E] fp32, captured once
    w_norm_per_expert: torch.Tensor | None = None  # [E] fp32, captured once
    logit_min: float = float("inf")
    logit_max: float = float("-inf")
    logit_has_nan: bool = False
    score_has_nan: bool = False
    entropy_has_nan: bool = False
    n_entropy_nan: int = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="DEAD_EXPERTS.md")
    p.add_argument("--also_dump_npz", default=None,
                   help="optional path to save raw per-layer counts as .npz")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)

    # ------- model + ckpt -------
    cfg = small_config(attn_backend="sdpa", moe_backend="grouped",
                       moe_capacity_factor=1.25)
    model = MoEModel(cfg)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_sd = sd["model"] if "model" in sd else sd
    # checkpoints may carry the DDP `module.` prefix or torch.compile's
    # `_orig_mod.` prefix; strip both before load.
    cleaned = {}
    for k, v in model_sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[diag] WARN missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[diag] WARN unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(device).eval()

    # Identify MoE blocks and snapshot bias + router-weight norms NOW (these
    # do not change during diagnostic).
    moe_blocks = [(li, blk) for li, blk in enumerate(model.blocks) if blk.is_moe]
    n_layers_moe = len(moe_blocks)
    E = cfg.n_routed_experts
    print(f"[diag] model has {n_layers_moe} MoE layers, {E} routed experts each")

    stats = []
    for li, blk in moe_blocks:
        router = blk.ffn.router
        with torch.no_grad():
            bias = router.bias.detach().float().cpu().clone()
            # per-expert L2 norm of router weight [E, d]
            w = router.w.detach().float().cpu()
            w_norm = w.pow(2).sum(dim=-1).sqrt()
        stats.append(LayerStat(
            counts_sum=torch.zeros(E, dtype=torch.long),
            counts_min=torch.full((E,), 10**9, dtype=torch.long),
            counts_max=torch.zeros(E, dtype=torch.long),
            n_steps=0,
            bias_snapshot=bias,
            w_norm_per_expert=w_norm,
        ))

    # ------- data -------
    print(f"[diag] building UltraFineWeb loader (bs={args.batch_size}, "
          f"seq_len={args.seq_len})...")
    loader = make_ufweb_loader(
        seq_len=args.seq_len, batch_size=args.batch_size,
        num_workers=2, prefetch_factor=2,
        rank=0, world_size=1, seed=args.seed,
        vocab_cap=cfg.vocab_size,
    )
    data = LoaderToGPU(loader, device=device)

    # ------- per-layer hooks to capture router internals -------
    # We can't just use the `aux` dict from forward() because we want
    # logit min/max + NaN flags + per-step count tensors which are not
    # all returned. Instead we register a forward hook on every router.
    captured = {li: None for li, _ in moe_blocks}

    def make_hook(li):
        def hook(module, inputs, outputs):
            # outputs = (topk_idx, topk_weight, aux_dict)
            topk_idx, topk_weight, aux_dict = outputs
            x_flat = inputs[0]
            with torch.no_grad():
                # Recompute logits cheaply to inspect (router.w is small,
                # F.linear over [N, d] @ [E, d] is fast).
                x32 = x_flat.float()
                logits = F.linear(x32, module.w.float())
                scores = torch.sigmoid(logits)
                lmin = logits.min().item()
                lmax = logits.max().item()
                has_nan_l = bool(torch.isnan(logits).any().item())
                has_nan_s = bool(torch.isnan(scores).any().item())
            captured[li] = {
                "counts": aux_dict["counts"].detach().to(torch.long).cpu(),
                "entropy": aux_dict["router_entropy_bits"].detach().float().cpu(),
                "logit_min": lmin,
                "logit_max": lmax,
                "logit_has_nan": has_nan_l,
                "score_has_nan": has_nan_s,
            }
        return hook

    hook_handles = []
    for li, blk in moe_blocks:
        h = blk.ffn.router.register_forward_hook(make_hook(li))
        hook_handles.append(h)

    # ------- forward loop -------
    n_entropy_nan_total = 0
    with torch.no_grad():
        for step in range(args.steps):
            ids, lbl, _ = data.next_batch()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(ids, labels=lbl, return_aux=True)
            # Aggregate per-layer captures.
            for ix, (li, _) in enumerate(moe_blocks):
                c = captured[li]
                cnt = c["counts"]
                ent_nan = bool(torch.isnan(c["entropy"]).any().item())
                stats[ix].counts_sum.add_(cnt)
                stats[ix].counts_min = torch.minimum(stats[ix].counts_min, cnt)
                stats[ix].counts_max = torch.maximum(stats[ix].counts_max, cnt)
                stats[ix].n_steps += 1
                if c["logit_min"] < stats[ix].logit_min:
                    stats[ix].logit_min = c["logit_min"]
                if c["logit_max"] > stats[ix].logit_max:
                    stats[ix].logit_max = c["logit_max"]
                stats[ix].logit_has_nan = stats[ix].logit_has_nan or c["logit_has_nan"]
                stats[ix].score_has_nan = stats[ix].score_has_nan or c["score_has_nan"]
                if ent_nan:
                    stats[ix].entropy_has_nan = True
                    stats[ix].n_entropy_nan += 1
                    n_entropy_nan_total += 1
            if (step + 1) % 10 == 0:
                print(f"[diag] step {step+1}/{args.steps}  "
                      f"entropy_nan_layers_so_far_this_step="
                      f"{sum(1 for ix in range(n_layers_moe) if stats[ix].entropy_has_nan)}",
                      flush=True)

    for h in hook_handles:
        h.remove()

    # ------- write report -------
    # Per-step fair share = N * top_k / E where N = B*S
    N = args.batch_size * args.seq_len
    fair_per_step = N * cfg.top_k / E
    fair_total = fair_per_step * args.steps

    lines = []
    lines.append(f"# Dead-Experts Diagnostic — {os.path.basename(args.ckpt)}")
    lines.append("")
    lines.append(f"- Steps: **{args.steps}** forward passes (no train), "
                 f"single GPU (no DDP all-reduce)")
    lines.append(f"- Batch: bs={args.batch_size} seq_len={args.seq_len} → "
                 f"N={N} tokens/step, top_k={cfg.top_k}, E={E}")
    lines.append(f"- Fair-share per step: **{fair_per_step:.1f}** tokens/expert")
    lines.append(f"- Fair-share over {args.steps} steps: **{fair_total:.0f}** "
                 f"tokens/expert")
    lines.append(f"- MoE layers: {n_layers_moe} (block indices: "
                 f"{[li for li,_ in moe_blocks]})")
    lines.append("")
    lines.append("## Verdict (auto)")
    lines.append("")
    # dead = any expert getting < 1% of fair share over the entire 100 steps.
    dead_threshold = 0.01 * fair_total
    starved_threshold = 0.10 * fair_total
    dead_count_per_layer = []
    starved_count_per_layer = []
    for ix in range(n_layers_moe):
        s = stats[ix]
        dead_count_per_layer.append(int((s.counts_sum < dead_threshold).sum()))
        starved_count_per_layer.append(int((s.counts_sum < starved_threshold).sum()))
    total_dead_slots = sum(dead_count_per_layer)
    total_starved_slots = sum(starved_count_per_layer)
    max_dead = max(dead_count_per_layer)
    layers_with_dead = sum(1 for d in dead_count_per_layer if d > 0)
    lines.append(f"- Dead expert *slots* (load < 1% of fair): "
                 f"**{total_dead_slots}** of {n_layers_moe*E}")
    lines.append(f"- Starved expert *slots* (load < 10% of fair): "
                 f"**{total_starved_slots}** of {n_layers_moe*E}")
    lines.append(f"- Layers with ≥1 dead expert: "
                 f"**{layers_with_dead}** / {n_layers_moe} "
                 f"(max dead per layer: {max_dead})")
    lines.append(f"- Total NaN-entropy steps (across layers): "
                 f"**{n_entropy_nan_total}** of {args.steps * n_layers_moe}")
    lines.append("")
    lines.append("## Per-layer summary")
    lines.append("")
    lines.append("| layer | dead | starved | logit_min | logit_max | nan_logit | nan_score | nan_entropy_steps | bias_min | bias_max | w_norm_min | w_norm_max |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ix, (li, _) in enumerate(moe_blocks):
        s = stats[ix]
        d = dead_count_per_layer[ix]
        st = starved_count_per_layer[ix]
        lines.append(
            f"| {li} | {d} | {st} | {s.logit_min:.2f} | {s.logit_max:.2f} | "
            f"{'YES' if s.logit_has_nan else 'no'} | "
            f"{'YES' if s.score_has_nan else 'no'} | "
            f"{s.n_entropy_nan}/{args.steps} | "
            f"{s.bias_snapshot.min().item():+.3f} | "
            f"{s.bias_snapshot.max().item():+.3f} | "
            f"{s.w_norm_per_expert.min().item():.3f} | "
            f"{s.w_norm_per_expert.max().item():.3f} |"
        )
    lines.append("")
    lines.append("## Per-expert per-layer detail")
    lines.append("")
    lines.append("Each layer: ``expert_i: load_frac=X% (mean_count/step) bias=B w_norm=N``.")
    lines.append("``DEAD`` marker if load_frac < 1% of fair share.")
    lines.append("")
    for ix, (li, _) in enumerate(moe_blocks):
        s = stats[ix]
        lines.append(f"### Layer {li}")
        for e in range(E):
            tot = int(s.counts_sum[e].item())
            mn = int(s.counts_min[e].item())
            mx = int(s.counts_max[e].item())
            mean_per_step = tot / s.n_steps
            load_frac = mean_per_step / fair_per_step
            tag = ""
            if tot < dead_threshold:
                tag = " **DEAD**"
            elif tot < starved_threshold:
                tag = " STARVED"
            lines.append(
                f"- e{e:02d}: tot={tot:>7d} mean/step={mean_per_step:7.1f} "
                f"min/step={mn:5d} max/step={mx:5d} "
                f"load_frac={load_frac*100:6.2f}% "
                f"bias={s.bias_snapshot[e].item():+.4f} "
                f"w_norm={s.w_norm_per_expert[e].item():.4f}{tag}"
            )
        lines.append("")

    out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[diag] wrote {out_path}")
    print(f"[diag] dead slots total = {total_dead_slots} "
          f"(of {n_layers_moe*E}); starved = {total_starved_slots}; "
          f"layers_with_dead = {layers_with_dead}/{n_layers_moe}; "
          f"entropy_nan_steps_total = {n_entropy_nan_total}")

    if args.also_dump_npz:
        import numpy as np
        npz = {}
        for ix, (li, _) in enumerate(moe_blocks):
            s = stats[ix]
            npz[f"layer{li}_counts_sum"] = s.counts_sum.numpy()
            npz[f"layer{li}_counts_min"] = s.counts_min.numpy()
            npz[f"layer{li}_counts_max"] = s.counts_max.numpy()
            npz[f"layer{li}_bias"] = s.bias_snapshot.numpy()
            npz[f"layer{li}_w_norm"] = s.w_norm_per_expert.numpy()
        np.savez_compressed(args.also_dump_npz, **npz)
        print(f"[diag] dumped raw arrays to {args.also_dump_npz}")


if __name__ == "__main__":
    main()
