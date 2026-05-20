"""Muon optimizer with fp16-normalized Newton-Schulz, plus AdamW for the
non-matrix params (embed, router, head, biases, norms).

We expose a single Optimizer-like wrapper that holds both Muon and AdamW
internally; from the training-loop point of view it acts like one optimizer
with two param groups (so the LR schedule can scale both at once).

Newton-Schulz inner loop is the standard 5-step polynomial from Keller
Jordan's Muon:
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.T
        X = a*X + b*(A @ X) + c*((A @ A) @ X)

In FP32 fallback we cast `X` to fp32 inside the loop. In the Dao Lab fp16-
normalized recipe we instead:
    1. Normalize: X = X / (||X||_2 + 1e-7)
    2. Apply a 1.02 safety factor on coefficients
    3. Keep X in fp16 throughout
After NS the orthogonalized direction is scaled by the spectral norm of the
original gradient (here approximated by ||grad||_2 / sqrt(min(rows,cols))).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


# ---------------- Newton-Schulz orthogonalization ----------------

def _ns_fp32(grad: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Standard NS-5 in fp32, à la Keller Jordan.

    We transpose to put the longer dimension as rows so that A = X @ X.T is
    the smaller (cols × cols) intermediate, then transpose back.
    """
    transposed = grad.size(-2) > grad.size(-1)
    X = grad.float()
    if transposed:
        X = X.T
    norm = X.norm() + 1e-7
    X = X / norm
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(n_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(grad.dtype)


def _ns_fp16_normalized(grad: torch.Tensor, n_iters: int = 5,
                        safety: float = 1.02) -> torch.Tensor:
    """Dao-Lab fp16-normalized NS recipe — keep computation in fp16 throughout,
    rely on the explicit per-iter norm and 1.02 coefficient safety factor."""
    transposed = grad.size(-2) > grad.size(-1)
    X = grad.to(torch.float16)
    if transposed:
        X = X.T
    norm32 = X.float().norm() + 1e-7
    X = (X.float() / norm32).to(torch.float16)
    a = 3.4445 / safety
    b = -4.7750 / safety
    c = 2.0315 / safety
    for _ in range(n_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(grad.dtype)


def _orthogonalize(grad: torch.Tensor, mode: str = "fp32") -> torch.Tensor:
    if mode == "fp16":
        return _ns_fp16_normalized(grad)
    return _ns_fp32(grad)


# ---------- Grouped NS over a batch of same-shape matrices ----------
#
# When many parameters share a shape (e.g. all 200 (1280, 512) MoE expert
# weights) we batch their NS into a single bmm path:
#   - one tensor X of shape (B, m, n)
#   - per-batch normalization
#   - bmm(X, X.transpose) for A
#   - bmm chains for the polynomial step
# This replaces 200 × 15 = 3000 individual sgemm launches per param-group
# per step with 15 bmm launches, each over (200, m, n). On V100 the bmm
# kernels amortize launch latency across the batch and reach higher
# achieved TFLOPS than the same-volume separate gemm sequence.

def _ns_fp32_batched(grads: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Batched NS-5 in fp32 over (B, m, n) where m <= n.

    All matrices in the batch must share a shape and orientation (m <= n
    already enforced by `_group_and_orient`).
    """
    X = grads.float()                                        # (B, m, n)
    # per-matrix Frobenius norm; flatten last two dims for norm
    norm = X.flatten(1).norm(dim=-1).view(-1, 1, 1) + 1e-7
    X = X / norm
    a, b, c = 3.4445, -4.7750, 2.0315
    Xt = X.transpose(-1, -2)
    for _ in range(n_iters):
        A = torch.bmm(X, Xt)                                  # (B, m, m)
        AA = torch.bmm(A, A)
        B_tens = b * A + c * AA                               # (B, m, m)
        X = a * X + torch.bmm(B_tens, X)                      # (B, m, n)
        Xt = X.transpose(-1, -2)
    return X


def _ns_fp16_batched(grads: torch.Tensor, n_iters: int = 5,
                     safety: float = 1.02) -> torch.Tensor:
    X = grads.to(torch.float16)                                # (B, m, n)
    norm32 = X.float().flatten(1).norm(dim=-1).view(-1, 1, 1) + 1e-7
    X = (X.float() / norm32).to(torch.float16)
    a = 3.4445 / safety
    b = -4.7750 / safety
    c = 2.0315 / safety
    Xt = X.transpose(-1, -2)
    for _ in range(n_iters):
        A = torch.bmm(X, Xt)
        AA = torch.bmm(A, A)
        B_tens = b * A + c * AA
        X = a * X + torch.bmm(B_tens, X)
        Xt = X.transpose(-1, -2)
    return X


def _group_and_orient(params, grads):
    """Group (param, grad) pairs by oriented shape (m, n) with m <= n.

    Returns dict: oriented_shape -> list of (param, grad, transposed_flag).
    `transposed_flag=True` means the grad was transposed to enforce m <= n
    and the orthogonalized result must be transposed back before applying.
    """
    out = {}
    for p, g in zip(params, grads):
        transposed = g.size(-2) > g.size(-1)
        if transposed:
            g_o = g.transpose(-1, -2)
        else:
            g_o = g
        shape = tuple(g_o.shape)
        out.setdefault(shape, []).append((p, g_o, transposed, g))
    return out


# ---------------- Muon proper ----------------

class Muon:
    """Muon over 2-D weight matrices. AdamW for the rest.

    Caller passes already-split param lists. The matrix list goes through the
    momentum buffer + Newton-Schulz; the non-matrix list goes through AdamW.

    LR is interpreted as the *Muon* lr; AdamW gets `lr * adam_lr_scale`
    (defaults to 1.0 since the spec uses the same lr).

    A single `.step()` call advances both internal optimizers.
    """

    def __init__(self, matrix_params: Iterable[torch.nn.Parameter],
                 non_matrix_params: Iterable[torch.nn.Parameter],
                 lr: float = 3e-4,
                 momentum: float = 0.95,
                 ns_mode: str = "fp32",
                 ns_iters: int = 5,
                 weight_decay: float = 0.01,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 adam_lr_scale: float = 1.0,
                 nesterov: bool = True,
                 foreach: bool = False,
                 grouped_ns: bool = False):
        self.matrix_params: List[torch.nn.Parameter] = [p for p in matrix_params if p.requires_grad]
        self.non_matrix_params: List[torch.nn.Parameter] = [p for p in non_matrix_params if p.requires_grad]
        self.lr = lr
        self.momentum = momentum
        self.ns_mode = ns_mode
        self.ns_iters = ns_iters
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.foreach = foreach
        # grouped_ns implies foreach (it uses the foreach path for the
        # outer momentum + decay book-keeping and just substitutes the per-
        # param NS loop for the batched-per-shape loop).
        self.grouped_ns = grouped_ns
        if grouped_ns:
            self.foreach = True

        self._mu_state = {id(p): {"buf": torch.zeros_like(p, memory_format=torch.preserve_format)}
                          for p in self.matrix_params}
        self._mu_buf_list = [self._mu_state[id(p)]["buf"] for p in self.matrix_params]

        # AdamW for the rest. `foreach=True` is the fast multi-tensor path on
        # CUDA — collapses per-param python loops into one launch per op.
        self.adam_lr_scale = adam_lr_scale
        self._adam = torch.optim.AdamW(self.non_matrix_params, lr=lr * adam_lr_scale,
                                       betas=betas, eps=eps, weight_decay=weight_decay,
                                       foreach=foreach if foreach else None)

    def zero_grad(self, set_to_none: bool = True):
        for p in self.matrix_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_().zero_()
        self._adam.zero_grad(set_to_none=set_to_none)

    def set_lr(self, lr: float):
        self.lr = lr
        for g in self._adam.param_groups:
            g["lr"] = lr * self.adam_lr_scale

    @torch.no_grad()
    def step(self):
        if self.foreach:
            self._step_foreach()
        else:
            self._step_eager()
        self._adam.step()

    @torch.no_grad()
    def _step_eager(self):
        for p in self.matrix_params:
            if p.grad is None:
                continue
            g = p.grad
            st = self._mu_state[id(p)]
            buf = st["buf"]
            buf.mul_(self.momentum).add_(g)
            if self.nesterov:
                d = g.add(buf, alpha=self.momentum)
            else:
                d = buf
            d2 = _orthogonalize(d, mode=self.ns_mode)
            if self.weight_decay != 0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            rows, cols = d2.shape
            scale = 0.2 * math.sqrt(max(rows, cols))
            p.add_(d2, alpha=-self.lr * scale)

    @torch.no_grad()
    def _step_foreach(self):
        params, grads, bufs = [], [], []
        for p, buf in zip(self.matrix_params, self._mu_buf_list):
            if p.grad is None:
                continue
            params.append(p); grads.append(p.grad); bufs.append(buf)
        if not params:
            return
        # Momentum buffer update across all matrix params (foreach).
        torch._foreach_mul_(bufs, self.momentum)
        torch._foreach_add_(bufs, grads)
        if self.nesterov:
            ds = torch._foreach_add(grads, bufs, alpha=self.momentum)
        else:
            ds = list(bufs)
        # Weight decay across all params (foreach).
        if self.weight_decay != 0:
            torch._foreach_mul_(params, 1.0 - self.lr * self.weight_decay)

        if self.grouped_ns:
            # Grouped Newton-Schulz: bucket directions by oriented shape,
            # run a single batched NS per shape via torch.bmm, then unstack
            # + apply. Replaces ~3000 per-param sgemm launches per step
            # (200 (1280,512) + 100 (512,1280) + 24 (512,512) + 24 (128,512)
            # matrices × 5 NS iters × 3 matmuls) with ~15 bmm launches,
            # each over the whole shape-batch.
            groups = _group_and_orient(params, ds)
            ns_fn = _ns_fp16_batched if self.ns_mode == "fp16" else _ns_fp32_batched
            for oriented_shape, items in groups.items():
                stacked = torch.stack([g for _, g, _, _ in items], dim=0)
                stacked_dtype = items[0][3].dtype
                X = ns_fn(stacked, self.ns_iters)                          # (B, m, n)
                X = X.to(stacked_dtype)
                m_, n_ = oriented_shape
                scale = 0.2 * math.sqrt(max(m_, n_))
                for (p, _, transposed, _), x_b in zip(items, X.unbind(0)):
                    d2 = x_b.transpose(-1, -2) if transposed else x_b
                    p.add_(d2, alpha=-self.lr * scale)
        else:
            # Foreach-only path: per-param NS, but momentum + decay are
            # batched. The python-overhead win is ~5-10× on the bookkeeping
            # ops, but the dominant per-param NS launches are unchanged.
            for p, d in zip(params, ds):
                d2 = _orthogonalize(d, mode=self.ns_mode)
                rows, cols = d2.shape
                scale = 0.2 * math.sqrt(max(rows, cols))
                p.add_(d2, alpha=-self.lr * scale)

    def state_dict(self):
        # Save Muon momentum buffers as a positional list (same order as
        # self.matrix_params). Indexing by id(p) does not survive process
        # restarts (id() is the param's memory address, not stable).
        return {"mu_buffers": [self._mu_state[id(p)]["buf"].detach().cpu().clone()
                               for p in self.matrix_params],
                "adam": self._adam.state_dict(),
                "lr": self.lr, "ns_mode": self.ns_mode}

    def load_state_dict(self, sd):
        if "mu_buffers" in sd:
            buffers = sd["mu_buffers"]
            assert len(buffers) == len(self.matrix_params), \
                f"muon ckpt has {len(buffers)} bufs, model has {len(self.matrix_params)} matrix params"
            for p, buf in zip(self.matrix_params, buffers):
                self._mu_state[id(p)]["buf"].copy_(buf.to(p.device, dtype=p.dtype))
        elif "mu_state" in sd:
            # Legacy id-keyed format — only works if reloaded in the same
            # process, which never happens in practice. Best-effort: iterate
            # by save order. This branch is for backward-compat with already-
            # written ckpts.
            buffers = list(sd["mu_state"].values())
            for p, buf in zip(self.matrix_params, buffers):
                self._mu_state[id(p)]["buf"].copy_(buf.to(p.device, dtype=p.dtype))
        self._adam.load_state_dict(sd["adam"])
        self.lr = sd["lr"]
        self.ns_mode = sd["ns_mode"]


def make_param_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split parameters into (matrix, non-matrix).

    Matrix params: 2-D weights of nn.Linear in attention/expert/MoE blocks.
    Non-matrix: embeddings, RMSNorm weights, QK gains, smear-gate, router.w
    (which is 2-D but is in the router — kept on AdamW since it's tiny and
    has its own statistics), biases (none in this model), final-norm.

    Returns two lists of `nn.Parameter`.
    """
    matrix, non_matrix = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_router = "router.w" in name
        is_embed = name.endswith("embed.weight")
        is_norm = name.endswith("_norm.weight") or name.endswith("final_norm.weight")
        is_qk = ".q_norm." in name or ".k_norm." in name
        is_smear = name.endswith(".smear")
        is_2d = (p.dim() == 2)
        if is_router or is_embed or is_norm or is_qk or is_smear or not is_2d:
            non_matrix.append(p)
        else:
            matrix.append(p)
    return matrix, non_matrix
