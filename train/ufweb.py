"""Ultra-FineWeb streaming dataloader for the 200M-Qwen3 run.

Differences from the 100M-volta-week version:
  - Tokenizer is `Qwen/Qwen3-0.6B-Base` (vocab 151 936). No id clipping —
    our model owns the full vocab. Fallback to `Qwen/Qwen2.5-0.5B` if
    Qwen3 is gated (same tokenizer + vocab).
  - EOS = `tok.eos_token_id` (151 643 for Qwen3).
  - All other plumbing (per-rank `split_dataset_by_node`, per-worker
    striding, document-start mask, packed seq_len+1 chunks for label
    shift, async H2D `LoaderToGPU`) is the same.

Streamed dataset: `openbmb/Ultra-FineWeb`, split=`en`, column=`content`.
"""
from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info


_TOKENIZER_PRIMARY = "Qwen/Qwen3-0.6B-Base"
_TOKENIZER_FALLBACK = "Qwen/Qwen2.5-0.5B"


def _load_tokenizer():
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(_TOKENIZER_PRIMARY, use_fast=True)
    except Exception as e:
        print(f"[ufweb] primary tokenizer load failed ({e}); "
              f"falling back to {_TOKENIZER_FALLBACK}")
        tok = AutoTokenizer.from_pretrained(_TOKENIZER_FALLBACK, use_fast=True)
    tok.model_max_length = 10**9  # silence the long-doc warning
    if tok.eos_token_id is None:
        tok.eos_token_id = tok.vocab_size - 1
    return tok


class UltraFineWebStream(IterableDataset):
    """Stream Ultra-FineWeb, tokenize, pack to (seq_len+1, ) chunks.

    Each `__iter__` yields a tuple (ids, labels, doc_starts) where
      ids:        (seq_len,)  int64
      labels:     (seq_len,)  int64  (next-token shifted)
      doc_starts: (seq_len,)  bool   (True at first token of a new doc)
    """

    def __init__(self,
                 seq_len: int = 2048,
                 vocab_cap: int = 151936,
                 split: str = "en",
                 buffer_initial: int = 4,
                 seed: int = 0,
                 score_min: float = 0.0,
                 rank: int = 0,
                 world_size: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_cap = vocab_cap
        self.split = split
        self.buffer_initial = buffer_initial
        self.seed = seed
        self.score_min = score_min
        self.rank = rank
        self.world_size = world_size

    def _iter_raw(self) -> Iterator[dict]:
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node
        ds = load_dataset("openbmb/Ultra-FineWeb",
                          split=self.split, streaming=True)
        if self.world_size > 1:
            ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)
        if self.score_min > 0.0:
            ds = ds.filter(lambda r: r["score"] is not None and r["score"] >= self.score_min)
        wi = get_worker_info()
        n_workers = wi.num_workers if wi else 1
        worker_id = wi.id if wi else 0
        ds = ds.shuffle(seed=self.seed + self.rank * 16 + worker_id, buffer_size=1024)
        it = iter(ds)
        if n_workers > 1:
            def strided():
                for i, row in enumerate(it):
                    if i % n_workers == worker_id:
                        yield row
            return strided()
        return it

    def __iter__(self):
        tok = _load_tokenizer()
        cap = self.vocab_cap - 1
        eos = min(tok.eos_token_id, cap)
        seq_len = self.seq_len
        pack_len = seq_len + 1  # need one extra for label shift

        raw = self._iter_raw()
        buffer = []
        starts = []
        while True:
            while len(buffer) < pack_len:
                try:
                    row = next(raw)
                except StopIteration:
                    return
                text = row.get("content") or row.get("text") or ""
                if not text:
                    continue
                ids = tok.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                marks = [1] + [0] * (len(ids) - 1)
                buffer.extend(ids)
                starts.extend(marks)
                buffer.append(eos)
                starts.append(0)

            chunk_ids = buffer[:pack_len]
            chunk_marks = starts[:pack_len]
            del buffer[:seq_len]
            del starts[:seq_len]

            t_ids = torch.tensor(chunk_ids[:seq_len], dtype=torch.long)
            t_lbl = torch.tensor(chunk_ids[1:seq_len + 1], dtype=torch.long)
            t_ds = torch.tensor(chunk_marks[1:seq_len + 1], dtype=torch.bool)
            yield t_ids, t_lbl, t_ds


def make_loader(seq_len: int, batch_size: int, *,
                num_workers: int = 4,
                prefetch_factor: int = 2,
                rank: int = 0,
                world_size: int = 1,
                score_min: float = 0.0,
                seed: int = 0,
                vocab_cap: int = 151936):
    from torch.utils.data import DataLoader
    ds = UltraFineWebStream(seq_len=seq_len,
                            vocab_cap=vocab_cap,
                            rank=rank, world_size=world_size,
                            seed=seed, score_min=score_min)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return loader


class LoaderToGPU:
    """Async H2D-copy wrapper around a DataLoader."""

    def __init__(self, loader, device: str = "cuda"):
        self.loader = loader
        self.device = device
        self._it = iter(loader)
        self._next = self._prefetch_one()

    def _prefetch_one(self):
        try:
            ids, lbl, ds_mask = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            ids, lbl, ds_mask = next(self._it)
        ids = ids.to(self.device, non_blocking=True)
        lbl = lbl.to(self.device, non_blocking=True)
        ds_mask = ds_mask.to(self.device, non_blocking=True)
        return ids, lbl, ds_mask

    def next_batch(self):
        cur = self._next
        self._next = self._prefetch_one()
        return cur
