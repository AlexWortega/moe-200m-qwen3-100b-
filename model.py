"""200M-active / ~1.2B-total MoE model — fresh-pretrain target.

Architecture: same DeepSeekMoE-style 1-shared + 32-routed top-2 MoE,
GQA attention with QK-Norm and partial RoPE, tied embed/lm_head.

Two material differences from the 100M-active sibling at
`~/ml-intern-runs/moe-100m-volta-week/model.py`:

  1. Bigger config defaults (vocab=151 936, d_model=640, n_layers=16,
     n_q_heads=10, n_kv_heads=2, head_dim=64, d_ff=1024,
     n_routed_experts=32, top_k=2, moe_first_layer=1).

  2. **Tiled cross-entropy loss.** Vocab=151 936 × micro_bs=8 ×
     seq_len=2048 in fp16 ≈ 4.7 GB just for the logits, again the same
     for softmax intermediates. Instead we never materialize the full
     logit tensor: we tile the post-final-norm hidden state into
     `seq_chunk_size` slices along (B·S), do the per-slice
     `F.linear(h, embed.T) → logits` and `F.cross_entropy(...)` inside a
     `torch.utils.checkpoint`, and sum the partial CE values. Peak
     resident logit buffer is `seq_chunk_size · vocab · 4` (fp32 for
     numerical safety) ≈ 0.3 GB at chunk=512.

The chunked-CE path is mathematically equivalent to a single
`F.cross_entropy` on the full `(N, V)` logits with `reduction='mean'`,
to fp32 precision — verified by `tests/test_chunked_ce.py`.

The single-shot reference path is kept gated behind
`MoEModelConfig.use_chunked_ce=False` for the verification test only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


# ============================ Config ============================

@dataclass
class MoEModelConfig:
    vocab_size: int = 151936
    d_model: int = 640
    n_layers: int = 16
    n_q_heads: int = 10
    n_kv_heads: int = 2
    head_dim: int = 64
    rope_partial: int = 32
    rope_theta: float = 10000.0
    d_ff: int = 1024
    # Variant A (router-stability rescue): dropped 32 -> 16 routed experts
    # after two NaN cascades on 32+1. d_ff stays at 1024; active stays at
    # ~200M, total drops 1.07B -> ~620M. Half the router load = 2x easier
    # to balance under fp16 on V100 (no bf16 tensor cores), which lets us
    # use moderate aux/bias rather than the aggressive ones that NaN'd.
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    top_k: int = 2
    moe_first_layer: int = 1
    router_z_coef: float = 1e-3
    # Additive Gaussian noise applied to ``sel_logits`` (logit + bias) during
    # training, before the top-k pick. Breaks routing lock-in so dead experts
    # can occasionally win top-2 and the bias controller has something to
    # work with. Set non-zero during a router-recovery resume. 0.0 = noise
    # off (standard inference + post-recovery training). Eval is always
    # noise-free regardless of this value.
    router_noise_std: float = 0.0
    # Variant A on 2 GPUs (CUDA_VISIBLE_DEVICES=2,3): moderate coeffs.
    # 1e-3 aux + 1e-3 bias is 10x lower than the aux=1e-2 / bias=5e-3 that
    # NaN'd on 32+1 experts, and we have 16 experts now (2x easier to
    # balance). Half DDP all-reduce noise from 2 vs 4 GPUs further helps
    # router stability. Magnitude-based bias formula kept (err=(mean-c)/mean).
    router_aux_coef: float = 1e-3
    bias_update_rate: float = 1e-3
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    rms_eps: float = 1e-6
    init_std: float = 0.02
    mup_base_d: int = 512
    attn_backend: str = "sdpa"  # "sdpa" or "fa_volta" (Triton FA fwd+bwd on V100)
    moe_backend: str = "grouped"  # "bmm" = per-expert for-loop (legacy);
                                  # "grouped" = stacked-weight bmm (fast)
    moe_capacity_factor: float = 1.25  # only used when moe_backend="grouped".
                                       # 1.0 = no padding (drops overflow);
                                       # 1.25 = ~6% drops at CV=0.5 (acceptable);
                                       # 2.0 = no drops up to CV≈0.5 but 2x bmm work.
    smear_gate: bool = True
    use_chunked_ce: bool = True
    ce_chunk_tokens: int = 512  # per-chunk token count for tiled CE
    ce_checkpoint_chunks: bool = True
    # Pass-2 optimization: route CE through Liger's fused linear+CE kernel
    # which never materializes the (N, V) logit tensor. Falls back to the
    # chunked CE path above if liger_kernel is not importable. Mathematically
    # equivalent to `cross_entropy(F.linear(h, embed.T) * mup, labels)` to
    # fp32 precision (verified by tests/test_liger_ce.py).
    use_liger_ce: bool = True

    def as_dict(self):
        return asdict(self)


def small_config(**overrides) -> MoEModelConfig:
    cfg = MoEModelConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg


# ============================ Norms ============================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x32 * rms).to(dtype) * self.weight


class QKNorm(nn.Module):
    def __init__(self, n_heads, head_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_heads, head_dim))
        self.gain = nn.Parameter(torch.ones(n_heads, 1))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        out = (x32 * rms).to(dtype)
        return out * self.weight.view(1, 1, *self.weight.shape) * \
               self.gain.view(1, 1, *self.gain.shape)


# ============================ RoPE ============================

def _build_cos_sin(seq_len, dim, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def _rotate_half_pairs(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class PartialRoPE(nn.Module):
    def __init__(self, head_dim, rope_dim, max_seq_len, theta=10000.0):
        super().__init__()
        assert rope_dim <= head_dim and rope_dim % 2 == 0
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = _build_cos_sin(max_seq_len, rope_dim, theta, "cpu", torch.float32)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q, k, position_ids=None):
        S = q.size(1)
        if position_ids is None:
            cos = self.cos_cached[:S].to(q.dtype)
            sin = self.sin_cached[:S].to(q.dtype)
        else:
            cos = self.cos_cached[position_ids].to(q.dtype)
            sin = self.sin_cached[position_ids].to(q.dtype)
        if cos.dim() == 2:
            cos = cos.view(1, S, 1, self.rope_dim)
            sin = sin.view(1, S, 1, self.rope_dim)
        else:
            cos = cos.view(cos.size(0), S, 1, self.rope_dim)
            sin = sin.view(sin.size(0), S, 1, self.rope_dim)
        def _apply(x):
            x_rot = x[..., :self.rope_dim]
            x_pass = x[..., self.rope_dim:]
            x_rot = x_rot * cos + _rotate_half_pairs(x_rot) * sin
            return torch.cat([x_rot, x_pass], dim=-1)
        return _apply(q), _apply(k)


# ============================ Attention ============================

def _repeat_kv(x, n_rep):
    if n_rep == 1: return x
    B, S, H, D = x.shape
    return x[:, :, :, None, :].expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


class GQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_q, self.n_kv, self.d_h = cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim
        assert self.n_q % self.n_kv == 0
        self.n_rep = self.n_q // self.n_kv
        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.n_q * self.d_h, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.d_h, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.d_h, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.d_h, d, bias=False)
        self.q_norm = QKNorm(self.n_q, self.d_h, eps=cfg.rms_eps)
        self.k_norm = QKNorm(self.n_kv, self.d_h, eps=cfg.rms_eps)
        self.rope = PartialRoPE(self.d_h, cfg.rope_partial, cfg.max_seq_len, cfg.rope_theta)
        if cfg.smear_gate:
            self.smear = nn.Parameter(torch.ones(self.n_kv))
        else:
            self.smear = None

    def forward(self, x, attn_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_q, self.d_h)
        k = self.k_proj(x).view(B, S, self.n_kv, self.d_h)
        v = self.v_proj(x).view(B, S, self.n_kv, self.d_h)
        q = self.q_norm(q); k = self.k_norm(k)
        q, k = self.rope(q, k)
        if self.smear is not None:
            v = v * self.smear.view(1, 1, self.n_kv, 1)
        backend = getattr(self.cfg, "attn_backend", "sdpa")
        # FA-Volta path is only correct in fp16/bf16 (Triton kernel is
        # half-precision only) and only with the kv-repeat layout it
        # expects: (B, S, H, D). SDPA path keeps the legacy (B, H, S, D).
        use_fa = (backend == "fa_volta") and q.dtype in (torch.float16, torch.bfloat16)
        if use_fa:
            from flash_attn_volta.autograd import flash_attn
            k_rep = _repeat_kv(k, self.n_rep)
            v_rep = _repeat_kv(v, self.n_rep)
            out = flash_attn(q.contiguous(), k_rep.contiguous(), v_rep.contiguous(),
                             causal=(attn_mask is None))
            out = out.contiguous().view(B, S, self.n_q * self.d_h)
        else:
            qh = q.transpose(1, 2)
            kh = _repeat_kv(k, self.n_rep).transpose(1, 2)
            vh = _repeat_kv(v, self.n_rep).transpose(1, 2)
            out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=(attn_mask is None))
            out = out.transpose(1, 2).contiguous().view(B, S, self.n_q * self.d_h)
        return self.o_proj(out)


# ============================ Experts / Router / MoE ============================

class SwiGLUExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SigmoidRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k,
                 z_coef=1e-3, aux_coef=1e-3, bias_update_rate=1e-3,
                 noise_std=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.w = nn.Parameter(torch.zeros(n_experts, d_model))
        nn.init.normal_(self.w, std=0.02)
        self.register_buffer("bias", torch.zeros(n_experts))
        self.z_coef = z_coef
        self.aux_coef = aux_coef
        self.bias_update_rate = bias_update_rate
        self.noise_std = noise_std

    def forward(self, x_flat):
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x_flat.float()
            logits = F.linear(x32, self.w.float())
            scores = torch.sigmoid(logits)
            sel_logits = logits + self.bias.float().unsqueeze(0)
            if self.training and self.noise_std > 0:
                # Additive Gaussian noise breaks load-imbalance lock-in.
                sel_logits = sel_logits + torch.randn_like(sel_logits) * self.noise_std
            topk_sel, topk_idx = torch.topk(sel_logits, k=self.top_k, dim=-1)
            topk_weight = scores.gather(-1, topk_idx)
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-9)
            lse = torch.logsumexp(logits, dim=-1)
            z_loss = (lse ** 2).mean()
            with torch.no_grad():
                one_hot = F.one_hot(topk_idx, num_classes=self.n_experts).sum(dim=1)
            p_i = scores.mean(dim=0)
            f_i_grad = one_hot.float().mean(dim=0)
            aux_loss = self.n_experts * (f_i_grad * p_i).sum()
            with torch.no_grad():
                counts = one_hot.sum(dim=0).float()
                cv = counts.std() / counts.mean().clamp_min(1.0)
                # Entropy metric — whole router is in autocast(enabled=False)
                # so plain fp32 ops are safe.
                scores_fp32 = scores.float()
                p_avg = scores_fp32.mean(dim=0).clamp_min(1e-9)
                p_avg = p_avg / p_avg.sum()
                entropy = -(p_avg * p_avg.log()).sum() / math.log(2.0)
        return topk_idx, topk_weight, {"z_loss": z_loss, "aux_loss": aux_loss,
                                       "counts": counts, "router_cv": cv,
                                       "router_entropy_bits": entropy}

    @torch.no_grad()
    def step_bias_update(self, counts):
        """Symmetric load-balance bias update with starved-expert boost.

        Old formulation used ``err = (mean - c) / max(mean, 1)`` which was
        asymmetric — overloaded experts got pushed down with unbounded
        magnitude (err can be large negative when c >> mean) while starved
        experts could push up by at most +1 (err ≤ 1). After 7953 steps
        on the 100B run, this drove biases to range [-23, +7], with 102 /
        240 expert slots completely dead. See ``DEAD_EXPERTS.md``.

        New formulation uses fractional load (`p_i = c_i / total`) vs
        uniform target, gives a 10× rate boost for starved experts
        (`p_i < 0.1 · target_p`), and hard-clamps the bias to [-5, +5]
        to prevent runaway in either direction.

        Caller MUST all-reduce ``counts`` across ranks before calling this
        in a DDP setting — otherwise different ranks compute different
        updates from local-view counts, and DDP's default
        ``broadcast_buffers=True`` then overwrites all ranks' bias with
        rank 0's (biased) view.
        """
        counts_f = counts.float()
        total = counts_f.sum().clamp_min(1.0)
        p_i = counts_f / total
        target_p = 1.0 / self.n_experts
        err = target_p - p_i                       # positive = underloaded
        update = err * self.bias_update_rate
        # Constant additive boost for starved experts (load < 10 % of
        # fair share). Rate-multiplier alone is too weak; the original
        # 100B run drove unclamped bias to -23/+7, so the natural control
        # range is wide — clamp at ±10 (not ±5) so the controller can
        # actually compete with router_w logits in the ±10 range we see
        # at this ckpt. 0.05/step boost reaches the +10 clamp in 200 steps.
        starved = (p_i < 0.1 * target_p).float()
        update = update + starved * 0.05
        self.bias.add_(update)
        self.bias.clamp_(min=-10.0, max=10.0)


def _moe_dispatch_bmm(x, topk_idx, topk_weight, experts):
    """Per-expert dispatch — kept for parity / unit tests.

    Issues a single GPU->CPU sync (via ``offsets.cpu().tolist()``) at the
    start of each MoE forward so the python for-loop can slice the sorted
    token list with integer offsets. Also runs a 1-token "dust" pass on
    every expert each step so DDP's ``find_unused_parameters=True`` path
    sees all expert grads. Slow but correct; superseded by
    ``_moe_dispatch_grouped`` on the fast path.
    """
    N, K = topk_idx.shape
    flat_expert = topk_idx.reshape(-1)
    flat_weight = topk_weight.reshape(-1).to(x.dtype)
    flat_token  = torch.arange(N, device=x.device).repeat_interleave(K)
    order = torch.argsort(flat_expert, stable=False)
    flat_expert_s = flat_expert[order]
    flat_token_s  = flat_token[order]
    flat_weight_s = flat_weight[order]
    n_experts = len(experts)
    counts = torch.bincount(flat_expert_s, minlength=n_experts)
    offsets = torch.cumsum(counts, dim=0)
    out = torch.zeros_like(x)
    offsets_cpu = offsets.cpu().tolist()
    counts_cpu = counts.cpu().tolist()
    start = 0
    x_dust = x[:1]
    for e in range(n_experts):
        y_dust = experts[e](x_dust)
        out.index_add_(0, flat_token[:1], (y_dust * 0.0).to(out.dtype))
        end = offsets_cpu[e]
        if counts_cpu[e] == 0:
            start = end; continue
        tok_idx = flat_token_s[start:end]
        w = flat_weight_s[start:end].unsqueeze(-1)
        x_e = x.index_select(0, tok_idx)
        y_e = experts[e](x_e)
        out.index_add_(0, tok_idx, (y_e * w).to(out.dtype))
        start = end
    return out


def _moe_dispatch_grouped(x, topk_idx, topk_weight,
                           gate_w, up_w, down_w,
                           capacity_factor: float = 1.5):
    """Token-permuted, capacity-padded grouped-bmm MoE dispatch.

    Inputs:
        x:           [N, d]
        topk_idx:    [N, K]   long
        topk_weight: [N, K]
        gate_w, up_w: [E, d_ff, d]
        down_w:       [E, d, d_ff]
        capacity_factor: pad each expert's slot count to
            ``ceil(N * K / E * capacity_factor)``. Tokens beyond capacity
            are dropped (contribute 0); their topk weight is wasted but the
            router still receives gradient through the still-routed top-k
            partner. A factor of 1.5 leaves slack for CV up to ~1.5.

    Returns:
        out: [N, d]  - sum over the top-k expert outputs, each multiplied
                       by the matching topk_weight, with dropped tokens
                       contributing zero.

    Stays GPU-resident throughout — no .cpu() / .item() sync. Issues
    3 batched-bmm kernels (gate, up, down) plus 1 sort + 1 bincount +
    index_select/scatter, regardless of expert count.
    """
    N, K = topk_idx.shape
    E, d_ff, d = gate_w.shape
    NK = N * K
    capacity = max(1, int(math.ceil(NK / E * capacity_factor)))

    flat_e = topk_idx.reshape(-1)              # [NK]
    flat_t = (torch.arange(N, device=x.device, dtype=torch.long)
              .repeat_interleave(K))           # [NK]
    flat_w = topk_weight.reshape(-1).to(x.dtype)  # [NK]

    order = torch.argsort(flat_e, stable=False)
    sorted_e = flat_e[order]                   # [NK]
    sorted_t = flat_t[order]
    sorted_w = flat_w[order]

    # Per-expert slot index (0..count[e]-1). Tokens with slot >= capacity
    # are dropped.
    counts = torch.bincount(sorted_e, minlength=E)     # [E]
    expert_start = counts.cumsum(0) - counts            # [E]
    global_pos = torch.arange(NK, device=x.device, dtype=torch.long)
    slot_in_expert = global_pos - expert_start.index_select(0, sorted_e)
    keep = slot_in_expert < capacity
    slot_idx = sorted_e * capacity + slot_in_expert    # flat [E*capacity]

    kept_slot = slot_idx[keep]
    kept_tok  = sorted_t[keep]
    kept_w    = sorted_w[keep]

    # Gather tokens into [E*capacity, d] dense buffer (zero where unused).
    x_grouped = x.new_zeros((E * capacity, d))
    x_kept = x.index_select(0, kept_tok)               # [n_kept, d]
    x_grouped.index_copy_(0, kept_slot, x_kept)
    x_grouped = x_grouped.view(E, capacity, d)

    # All-expert SwiGLU forward via 3 batched matmuls.
    # gate_w: [E, d_ff, d]; x_grouped: [E, capacity, d]
    g = torch.einsum("etd,efd->etf", x_grouped, gate_w)
    u = torch.einsum("etd,efd->etf", x_grouped, up_w)
    h = F.silu(g) * u                                  # [E, capacity, d_ff]
    y = torch.einsum("etf,edf->etd", h, down_w)        # [E, capacity, d]

    # Scatter back with topk weights.
    y_flat = y.view(E * capacity, d)
    y_kept = y_flat.index_select(0, kept_slot) * kept_w.unsqueeze(-1)
    out = x.new_zeros((N, d))
    out.index_add_(0, kept_tok, y_kept)
    return out


class MoEFFN(nn.Module):
    """MoE FFN with two dispatch backends.

    ``cfg.moe_backend``:
      * ``"bmm"``        — legacy per-expert python for-loop. Kept for
                           the chunked-CE unit test and as a fallback.
      * ``"grouped"``    — token-permuted, capacity-padded grouped-bmm
                           dispatch (`_moe_dispatch_grouped`). Stacked
                           expert weights as ``self.gate`` / ``self.up``
                           / ``self.down`` (shape ``[E, d_ff, d]`` and
                           ``[E, d, d_ff]`` for down). The per-expert
                           ``routed_experts`` ModuleList is **not**
                           built in this mode — state-dict keys are
                           flat ``gate``/``up``/``down``. Conversion
                           from a legacy ckpt is handled by
                           :func:`MoEModel.load_state_dict` (auto-stacks
                           ``routed_experts.i.{gate,up,down}.weight``
                           into the new tensors).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_routed = cfg.n_routed_experts
        self.n_shared = cfg.n_shared_experts
        self.backend = getattr(cfg, "moe_backend", "bmm")
        self.capacity_factor = getattr(cfg, "moe_capacity_factor", 1.5)
        if self.backend == "grouped":
            d = cfg.d_model; d_ff = cfg.d_ff; E = self.n_routed
            self.gate = nn.Parameter(torch.empty(E, d_ff, d))
            self.up   = nn.Parameter(torch.empty(E, d_ff, d))
            self.down = nn.Parameter(torch.empty(E, d, d_ff))
            self.routed_experts = None
        else:
            self.routed_experts = nn.ModuleList(
                [SwiGLUExpert(cfg.d_model, cfg.d_ff) for _ in range(self.n_routed)]
            )
            self.gate = self.up = self.down = None
        self.shared_expert = SwiGLUExpert(cfg.d_model, cfg.d_ff) if self.n_shared > 0 else None
        self.router = SigmoidRouter(d_model=cfg.d_model, n_experts=self.n_routed,
                                    top_k=cfg.top_k, z_coef=cfg.router_z_coef,
                                    aux_coef=cfg.router_aux_coef,
                                    bias_update_rate=cfg.bias_update_rate,
                                    noise_std=getattr(cfg, "router_noise_std", 0.0))

    def forward(self, x):
        B, S, d = x.shape
        x_flat = x.reshape(B * S, d)
        topk_idx, topk_weight, aux = self.router(x_flat)
        if self.backend == "grouped":
            y_routed = _moe_dispatch_grouped(
                x_flat, topk_idx, topk_weight,
                self.gate, self.up, self.down,
                capacity_factor=self.capacity_factor,
            )
        else:
            y_routed = _moe_dispatch_bmm(x_flat, topk_idx, topk_weight, self.routed_experts)
        if self.shared_expert is not None:
            y = y_routed + self.shared_expert(x_flat)
        else:
            y = y_routed
        return y.view(B, S, d), aux


# ============================ Block ============================

class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.is_moe = layer_idx >= cfg.moe_first_layer
        if self.is_moe:
            self.ffn = MoEFFN(cfg)
        else:
            self.ffn = SwiGLUExpert(cfg.d_model, cfg.d_ff)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        if self.is_moe:
            y, aux = self.ffn(self.ffn_norm(x))
            return x + y, aux
        else:
            return x + self.ffn(self.ffn_norm(x)), None


# ============================ Tiled CE loss ============================

def _ce_chunk_forward(h_chunk: torch.Tensor,
                      embed_weight: torch.Tensor,
                      labels_chunk: torch.Tensor,
                      mup_scale: float,
                      reduction: str = "sum") -> torch.Tensor:
    """Compute CE on a single (N_chunk, D) slice. Returns a scalar tensor
    representing the *sum* of CE over the chunk's non-ignored positions
    (or 'mean' if reduction='mean')."""
    logits = F.linear(h_chunk, embed_weight)
    logits = (logits * mup_scale).float()
    return F.cross_entropy(logits, labels_chunk,
                           ignore_index=-100, reduction=reduction)


def tiled_cross_entropy(h_flat: torch.Tensor,
                        embed_weight: torch.Tensor,
                        labels_flat: torch.Tensor,
                        mup_scale: float,
                        chunk_size: int = 512,
                        use_checkpoint: bool = True) -> torch.Tensor:
    """Mean cross-entropy over `labels_flat`, computed in chunks along the
    token dimension. Gradient flows back into `h_flat` and `embed_weight`.

    Equivalent (to fp32 precision) to:
        logits = F.linear(h_flat, embed_weight) * mup_scale
        F.cross_entropy(logits.float(), labels_flat, ignore_index=-100,
                        reduction='mean')

    Memory: peak resident logit buffer is `chunk_size · vocab · 4` bytes
    (one chunk at a time, no full (N, V) materialization).

    With `use_checkpoint=True`, each chunk's forward (linear + CE) is
    wrapped in `torch.utils.checkpoint`, so backward recomputes the chunk
    instead of holding logits + softmax intermediates resident across the
    full backward pass. Cost: one extra forward per chunk during backward.
    """
    N = h_flat.size(0)
    total_sum = h_flat.new_zeros((), dtype=torch.float32)
    valid_mask = labels_flat != -100
    n_valid = valid_mask.sum().clamp_min(1).to(torch.float32)
    for i in range(0, N, chunk_size):
        h_i = h_flat[i:i + chunk_size]
        lbl_i = labels_flat[i:i + chunk_size]
        if use_checkpoint and h_i.requires_grad:
            # Float values cannot be passed into checkpoint as a Tensor arg
            # would be — wrap as a 0-d tensor so autograd treats it cleanly.
            mup = torch.tensor(mup_scale, device=h_i.device, dtype=torch.float32)
            def _fn(h_c, w_c, lbl_c, mup_c):
                logits = F.linear(h_c, w_c)
                logits = (logits * mup_c).float()
                return F.cross_entropy(logits, lbl_c, ignore_index=-100,
                                       reduction="sum")
            ce_sum_i = ckpt.checkpoint(_fn, h_i, embed_weight, lbl_i, mup,
                                        use_reentrant=True)
        else:
            ce_sum_i = _ce_chunk_forward(h_i, embed_weight, lbl_i, mup_scale,
                                         reduction="sum")
        total_sum = total_sum + ce_sum_i
    return total_sum / n_valid


# ============================ Liger fused linear+CE ============================

_LIGER_AVAILABLE = None
_LIGER_LOSS_FN = None


def _try_import_liger():
    """Resolve the Liger fused linear+CE loss class lazily and cache it.

    Returns the class object on success, ``None`` on import failure. The
    module-level cache means the import + class lookup happens once per
    process even though we may instantiate the loss many times.
    """
    global _LIGER_AVAILABLE, _LIGER_LOSS_FN
    if _LIGER_AVAILABLE is False:
        return None
    if _LIGER_LOSS_FN is not None:
        return _LIGER_LOSS_FN
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )
        _LIGER_LOSS_FN = LigerFusedLinearCrossEntropyLoss
        _LIGER_AVAILABLE = True
        return _LIGER_LOSS_FN
    except Exception:
        _LIGER_AVAILABLE = False
        return None


def _maybe_disable_dynamo(fn):
    """Mark ``fn`` opaque to ``torch._dynamo`` if dynamo is importable.

    Liger's Triton kernel is incompatible with Inductor's launcher rewrite
    (it gets called with ``num_warps`` as a kwarg that the rewritten
    launcher does not accept). We don't *want* Inductor to inline this
    call anyway — the whole point of Liger is that its hand-tuned kernel
    is already faster than anything dynamo would synthesize.
    """
    try:
        import torch._dynamo as _dynamo
        return _dynamo.disable(fn)
    except Exception:
        return fn


@_maybe_disable_dynamo
def liger_fused_cross_entropy(h_flat: torch.Tensor,
                              embed_weight: torch.Tensor,
                              labels_flat: torch.Tensor,
                              mup_scale: float) -> torch.Tensor:
    """Liger fused linear+CE — single Triton kernel that computes
    ``cross_entropy(F.linear(h, embed_weight) * mup_scale, labels)`` without
    ever materializing the (N, V) logit tensor.

    Equivalence to ``tiled_cross_entropy``:
        F.linear(h * mup_scale, embed_weight)
            = F.linear(h, embed_weight) * mup_scale            (linearity)
    so pre-scaling ``h_flat`` by ``mup_scale`` and feeding it as ``_input``
    matches the original ``logits * mup_scale`` semantics exactly.

    Args:
        h_flat:        [N, D] hidden states (typically fp16 in training).
        embed_weight:  [V, D] tied embed / lm_head weight (fp16).
        labels_flat:   [N] long tensor; -100 entries are ignored.
        mup_scale:     scalar; multiplies hidden state pre-linear.

    Returns:
        scalar fp32 loss (mean over non-ignored positions).
    """
    cls = _try_import_liger()
    if cls is None:
        raise RuntimeError(
            "liger_kernel not importable — install with "
            "`python3.10 -m pip install liger-kernel==0.3.0 --no-deps`."
        )
    loss_fn = cls(ignore_index=-100, reduction="mean")
    h_scaled = h_flat * mup_scale
    # Inside autocast, Liger reads ``torch.get_autocast_gpu_dtype()`` to decide
    # the internal logits dtype but accumulates ``grad_weight`` in
    # ``weight.dtype`` (fp32) and ``_input_chunk`` in ``_input.dtype`` (fp32 if
    # the upstream RMSNorm promoted), so addmm sees mat1=Half and mat2=Float
    # and rejects. Fix by casting both _input and lin_weight to autocast dtype
    # before the call; autograd's .to() handles the gradient cast back to fp32
    # at parameter accumulation time.
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_gpu_dtype()
        h_scaled = h_scaled.to(dt)
        embed_in = embed_weight.to(dt)
    else:
        embed_in = embed_weight
    return loss_fn(lin_weight=embed_in, _input=h_scaled,
                   target=labels_flat, bias=None)


# ============================ Top-level model ============================

_ROUTED_PARAM_SUFFIXES = (".gate", ".up", ".down")


def _is_routed_expert_param(name: str) -> bool:
    """True if the parameter name belongs to the routed-expert FFN stack.

    Covers both layouts:
      * legacy:   ``blocks.{i}.ffn.routed_experts.{e}.{gate,up,down}.weight``
      * grouped:  ``blocks.{i}.ffn.{gate,up,down}``  (stacked [E, *, *])

    The grouped-tensor names collide with the shared-expert's
    ``blocks.{i}.ffn.shared_expert.{gate,up,down}.weight`` — that case is
    filtered out by the ``shared_expert`` clause earlier in the caller's
    classification chain, so this function only needs to ID the routed
    stack vs everything else.
    """
    if "routed_experts" in name:
        return True
    # Grouped layout: blocks.{i}.ffn.{gate,up,down} (no further suffix)
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blocks" and parts[2] == "ffn":
        if parts[3] in ("gate", "up", "down") and len(parts) == 4:
            return True
    return False


class MoEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.mup_scale = math.sqrt(cfg.mup_base_d / cfg.d_model)
        self.apply(self._init_weights)
        # Initialize stacked MoE weights (the apply() walk above only sees
        # nn.Linear / nn.Embedding modules; raw nn.Parameter tensors on the
        # grouped backend need explicit init.).
        if getattr(cfg, "moe_backend", "bmm") == "grouped":
            self._init_grouped_moe()

    def _init_weights(self, m):
        cfg = self.cfg
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                nn.init.orthogonal_(m.weight)
                fan_in = m.weight.size(1)
                m.weight.mul_(1.0 / math.sqrt(fan_in) * math.sqrt(m.weight.size(0)))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=cfg.init_std)

    def _init_grouped_moe(self):
        """Match the per-expert SwiGLUExpert init: per-expert orthogonal,
        then ``1/sqrt(fan_in) * sqrt(fan_out)`` rescale. Applied independently
        per expert so the stacked tensor is statistically equivalent to a
        ModuleList of independently-initialized experts."""
        for blk in self.blocks:
            if not blk.is_moe: continue
            moe = blk.ffn
            if moe.backend != "grouped": continue
            for w in (moe.gate, moe.up, moe.down):
                with torch.no_grad():
                    # w: [E, out, in]
                    fan_in = w.size(-1)
                    fan_out = w.size(-2)
                    for e in range(w.size(0)):
                        nn.init.orthogonal_(w[e])
                        w[e].mul_(1.0 / math.sqrt(fan_in) * math.sqrt(fan_out))

    def _convert_legacy_moe_keys(self, state_dict):
        """If state_dict carries per-expert ``routed_experts.{i}.{gate,up,down}.weight``
        and the model is in grouped backend, stack them into the new
        ``gate``/``up``/``down`` tensors and drop the per-expert keys.

        No-op if either the model is in bmm backend or the state_dict already
        uses the stacked keys.
        """
        if getattr(self.cfg, "moe_backend", "bmm") != "grouped":
            return state_dict
        new_sd = dict(state_dict)
        for li, blk in enumerate(self.blocks):
            if not blk.is_moe: continue
            prefix = f"blocks.{li}.ffn"
            legacy_key = f"{prefix}.routed_experts.0.gate.weight"
            if legacy_key not in new_sd: continue
            E = self.cfg.n_routed_experts
            for which, attr in [("gate", "gate"), ("up", "up"), ("down", "down")]:
                stack = []
                for e in range(E):
                    k = f"{prefix}.routed_experts.{e}.{which}.weight"
                    stack.append(new_sd.pop(k))
                new_sd[f"{prefix}.{attr}"] = torch.stack(stack, dim=0)
        return new_sd

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = self._convert_legacy_moe_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _lm_head_weight(self):
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def forward(self, input_ids, labels=None, return_aux=True,
                return_logits: bool = False):
        """Forward pass.

        If `labels` is provided, returns `(logits_or_None, loss, aux_total)`
        — and by default `logits` is `None` (we never materialize the full
        (B,S,V) tensor in training to keep peak memory low). Pass
        `return_logits=True` only for eval / generation paths that fit.

        If `labels` is None, returns `(logits, None, aux_total)` with the
        full logit tensor — only safe at small B·S.
        """
        x = self.embed(input_ids)
        aux_total = {"z_loss": 0.0, "aux_loss": 0.0,
                     "router_cv_sum": 0.0, "router_entropy_sum": 0.0,
                     "n_moe": 0, "counts_per_layer": []}
        for blk in self.blocks:
            x, aux = blk(x)
            if aux is not None:
                aux_total["z_loss"] = aux_total["z_loss"] + aux["z_loss"]
                aux_total["aux_loss"] = aux_total["aux_loss"] + aux["aux_loss"]
                aux_total["router_cv_sum"] = aux_total["router_cv_sum"] + aux["router_cv"].detach()
                aux_total["router_entropy_sum"] = aux_total["router_entropy_sum"] + aux["router_entropy_bits"].detach()
                aux_total["n_moe"] += 1
                aux_total["counts_per_layer"].append(aux["counts"].detach())
        x = self.final_norm(x)
        head_w = self._lm_head_weight()

        loss = None
        logits = None
        if labels is not None:
            B, S, D = x.shape
            h_flat = x.reshape(B * S, D)
            lbl_flat = labels.reshape(-1).long()
            use_liger = getattr(self.cfg, "use_liger_ce", False) and \
                        _try_import_liger() is not None and \
                        not return_logits
            if use_liger:
                loss = liger_fused_cross_entropy(
                    h_flat, head_w, lbl_flat, self.mup_scale,
                )
            elif self.cfg.use_chunked_ce:
                loss = tiled_cross_entropy(
                    h_flat, head_w, lbl_flat, self.mup_scale,
                    chunk_size=self.cfg.ce_chunk_tokens,
                    use_checkpoint=self.cfg.ce_checkpoint_chunks,
                )
            else:
                logits_full = F.linear(h_flat, head_w) * self.mup_scale
                loss = F.cross_entropy(logits_full.float(), lbl_flat,
                                       ignore_index=-100, reduction="mean")
                if return_logits:
                    logits = logits_full.view(B, S, -1)
            if return_logits and logits is None:
                # Materialize once for eval/gen if caller insists. Tile to
                # avoid the single-shot allocation in the chunked path.
                logits_full = F.linear(h_flat, head_w) * self.mup_scale
                logits = logits_full.view(B, S, -1)
        else:
            logits = F.linear(x, head_w) * self.mup_scale

        if return_aux:
            n_moe = max(1, aux_total["n_moe"])
            aux_total["router_cv"] = aux_total["router_cv_sum"] / n_moe
            aux_total["router_entropy_bits"] = aux_total["router_entropy_sum"] / n_moe
            return logits, loss, aux_total
        return logits, loss

    @torch.no_grad()
    def step_router_biases(self, counts_per_layer):
        i = 0
        for blk in self.blocks:
            if blk.is_moe:
                blk.ffn.router.step_bias_update(counts_per_layer[i])
                i += 1

    def num_parameters(self, only_active=False):
        if not only_active:
            return sum(p.numel() for p in self.parameters())
        total = 0
        for n, p in self.named_parameters():
            if _is_routed_expert_param(n):
                total += int(p.numel() * self.cfg.top_k / self.cfg.n_routed_experts)
            else:
                total += p.numel()
        return total

    def param_breakdown(self):
        """Return a dict with named param-count buckets — useful for
        comparing against the design target."""
        b = {"embed": 0, "attn": 0, "router": 0,
             "shared_expert": 0, "routed_experts": 0,
             "dense_ffn": 0, "norms": 0, "lm_head": 0, "other": 0}
        for n, p in self.named_parameters():
            num = p.numel()
            if "embed" in n:
                b["embed"] += num
            elif "lm_head" in n:
                b["lm_head"] += num
            elif "attn" in n or "q_proj" in n or "k_proj" in n or "v_proj" in n or "o_proj" in n or "q_norm" in n or "k_norm" in n or "rope" in n or "smear" in n:
                b["attn"] += num
            elif "router" in n:
                b["router"] += num
            elif "shared_expert" in n:
                b["shared_expert"] += num
            elif _is_routed_expert_param(n):
                b["routed_experts"] += num
            elif "ffn" in n and ("gate" in n or "up" in n or "down" in n):
                b["dense_ffn"] += num
            elif "norm" in n:
                b["norms"] += num
            else:
                b["other"] += num
        return b


if __name__ == "__main__":
    # quick standalone smoke
    cfg = small_config()
    m = MoEModel(cfg)
    print(f"params total {m.num_parameters()/1e6:.2f} M  "
          f"active {m.num_parameters(only_active=True)/1e6:.2f} M")
    bd = m.param_breakdown()
    for k, v in bd.items():
        print(f"  {k}: {v/1e6:.2f} M")
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss, aux = m(ids, labels=ids)
    print(f"logits {logits}  loss {loss.item():.3f}  router_cv {aux['router_cv'].item():.3f}")
