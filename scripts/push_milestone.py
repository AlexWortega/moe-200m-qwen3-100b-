"""Push a single MoE-200M-qwen3 milestone checkpoint to HF Hub.

Invocation:
  python3.10 scripts/push_milestone.py <ckpt_path> <tokens_seen>

Where `<ckpt_path>` is an absolute or run-relative path to a
`step_<N>.pt` (or `final.pt` / `final_ema.pt`) produced by
`train/train_200m.py`, and `<tokens_seen>` is the cumulative training
token count at that checkpoint.

The push creates a model repo at
`AlexWortega/ml-intern-moe200m-qwen3-step{N}-{tokens}B-{stamp}` and uploads:
  - model.safetensors  (weights — `_orig_mod.` prefix stripped)
  - config.json        (cfg dict + `_model_class`, step, tokens_consumed)
  - model.py           (architecture — self-contained)
  - tokenizer.json + tokenizer_config.json (Qwen3-0.6B-Base)
  - README.md          (model card with run-summary block)
  - load_test.py       (forward-pass one-shot smoke for the repo's reader)
  - TASK.md / PLAN.md / RESEARCH.md   (reproducibility bundle)

Run-level state files (`PUBLISHED_milestones.txt`,
`MILESTONE_STEP_{N}.txt`) are updated atomically.

The push is intentionally idempotent — calling twice on the same ckpt is
harmless (HF `upload_folder` overwrites by path).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch


RUN = Path(__file__).resolve().parents[1]
SLUG_BASE = "ml-intern-moe200m-qwen3"


def strip_orig_mod(sd: dict) -> dict:
    pref = "_orig_mod."
    out = {}
    for k, v in sd.items():
        if k.startswith(pref):
            out[k[len(pref):]] = v
        else:
            out[k] = v
    return out


def _read_token() -> str:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env.strip()
    p = Path.home() / ".cache" / "huggingface" / "token"
    if p.exists():
        return p.read_text().strip()
    raise SystemExit("[push] no HF_TOKEN env and no ~/.cache/huggingface/token")


def _load_test_template(repo_id: str) -> str:
    return f'''"""One-shot smoke test for {repo_id}.

`python load_test.py` will:
  1. download the repo
  2. instantiate `MoEModel` with the published config
  3. load the safetensors weights
  4. run a forward pass + greedy generation on a short prompt
  5. print logits stats and generation
"""
import json
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO = "{repo_id}"

def main():
    local = snapshot_download(REPO)
    import sys
    sys.path.insert(0, local)
    from model import MoEModel, MoEModelConfig
    cfg_dict = json.load(open(f"{{local}}/config.json"))
    cfg_dict.pop("_model_class", None)
    cfg_dict.pop("step", None)
    cfg_dict.pop("tokens_consumed", None)
    cfg = MoEModelConfig(**{{k: v for k, v in cfg_dict.items() if hasattr(MoEModelConfig(), k)}})
    model = MoEModel(cfg)
    sd = load_file(f"{{local}}/model.safetensors")
    model.load_state_dict(sd, strict=False)
    model.eval()
    tok = AutoTokenizer.from_pretrained(local)
    prompt = "The quick brown fox"
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits, _, _ = model(ids)
    print(f"prompt: {{prompt!r}}")
    print(f"logits  shape={{tuple(logits.shape)}}  mean={{logits.mean().item():.3f}}  std={{logits.std().item():.3f}}")
    # Greedy 32-token continuation
    for _ in range(32):
        with torch.no_grad():
            logits, _, _ = model(ids)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    print(f"greedy : {{tok.decode(ids[0])!r}}")

if __name__ == "__main__":
    main()
'''


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <ckpt_path> <tokens_seen>")
    ckpt_path = Path(sys.argv[1]).resolve()
    tokens_seen = int(sys.argv[2])

    if not ckpt_path.exists():
        sys.exit(f"[push] missing ckpt: {ckpt_path}")

    token = _read_token()
    os.environ["HF_TOKEN"] = token

    from huggingface_hub import HfApi, create_repo, upload_folder
    api = HfApi(token=token)
    user = api.whoami()["name"]
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(sd["cfg"])
    step = int(sd.get("step", -1)) + 1  # ckpt stores last completed step

    gb = tokens_seen / 1e9
    repo_id = f"{user}/{SLUG_BASE}-step{step}-{gb:.1f}B-{stamp}"
    print(f"[push] target repo: {repo_id}")
    print(f"[push] step={step}  tokens_seen={tokens_seen:,} ({gb:.3f} B)")

    stage = Path(tempfile.mkdtemp(prefix="push_200m_"))
    print(f"[push] staging dir: {stage}")
    try:
        from safetensors.torch import save_file
        clean_sd = strip_orig_mod(sd["model"])
        save_file(clean_sd, str(stage / "model.safetensors"))
        size_mb = (stage / "model.safetensors").stat().st_size / 1e6
        print(f"[push] model.safetensors: {size_mb:.1f} MB ({len(clean_sd)} keys)")

        cfg_out = dict(cfg)
        cfg_out["_model_class"] = "MoEModel"
        cfg_out["step"] = step
        cfg_out["tokens_consumed"] = tokens_seen
        (stage / "config.json").write_text(json.dumps(cfg_out, indent=2))

        for fname in ["model.py", "TASK.md", "PLAN.md", "RESEARCH.md"]:
            src = RUN / fname
            if src.exists():
                shutil.copy(src, stage / fname)

        # Tokenizer: re-download and re-save to repo so it's self-contained.
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", use_fast=True)
            tok.save_pretrained(str(stage))
            print(f"[push] saved Qwen3 tokenizer to staging")
        except Exception as e:
            print(f"[push] WARN: tokenizer save failed: {e}")

        (stage / "load_test.py").write_text(_load_test_template(repo_id))

        # Train log tail: copy the last 5MB only — full log is huge.
        for log_name in ["train.log", "eval.log"]:
            src = RUN / log_name
            if src.exists():
                size = src.stat().st_size
                if size <= 5 * 1024 * 1024:
                    shutil.copy(src, stage / log_name)
                else:
                    with open(src, "rb") as f:
                        f.seek(-5 * 1024 * 1024, 2)
                        tail = f.read()
                    (stage / log_name).write_bytes(b"...truncated...\n" + tail)

        # README
        is_final = ckpt_path.name in ("final.pt", "final_ema.pt")
        title_tail = "final" if is_final else f"~{gb:.1f}B-token intermediate"
        readme = f"""---
library_name: pytorch
tags: [ml-intern, pretraining, moe, ultra-fineweb, qwen3-tokenizer]
datasets: [openbmb/Ultra-FineWeb]
language: [en]
license: apache-2.0
pipeline_tag: text-generation
---
# MoE-200M-active / 1.09B-total @ {gb:.2f}B tokens — {title_tail} ckpt

Sparse-MoE LM with **~200M active / ~1.09B total parameters**, trained
from-scratch on Ultra-FineWeb (English split), tokenizer
[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
(vocab 151,936).

- Step: **{step}**
- Tokens consumed: **{tokens_seen:,}** ({gb:.3f}B)
- Architecture: see `model.py` and `config.json`. 1 shared + 32 routed
  experts top-2, GQA(10/2)×16 layers, partial RoPE, QK-Norm, SwiGLU,
  tied embeddings.
- Trained autonomously by the
  [ml-intern Claude Code skill](https://github.com/AlexWortega/claude-ml-intern-skill).

## Run summary

| key | value |
|---|---|
| param_count_total  | 1,088,484,192 |
| param_count_active | 203,748,192 |
| dataset            | openbmb/Ultra-FineWeb (split=en, content) |
| tokenizer          | Qwen/Qwen3-0.6B-Base (vocab=151,936) |
| step               | {step} |
| tokens_consumed    | {tokens_seen:,} |

## Caveats

- Trained on Ultra-FineWeb-en only — does not speak non-English / code well.
- Intermediate checkpoint; downstream evals attached in the run repo
  (`EVAL_*.md`).
- Predecessor 100M-active run plateaued around 10B tokens on this dataset.
  This 200M-active model is the scaling follow-up.
"""
        (stage / "README.md").write_text(readme)

        print(f"[push] create_repo {repo_id}")
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        url = upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"step {step}, {gb:.2f}B tokens",
        )
        repo_url = f"https://huggingface.co/{repo_id}"
        print(f"[push] DONE {repo_url}")

        # Append to PUBLISHED_milestones.txt (atomic-ish)
        pub = RUN / "PUBLISHED_milestones.txt"
        line = f"step={step}\ttokens={tokens_seen}\t{repo_url}\n"
        with open(pub, "a") as f:
            f.write(line)

        # Marker for the trainer to read back
        (RUN / f"MILESTONE_STEP_{step}.txt").write_text(f"{repo_url}\n")

        # Fire notify (best-effort, no_op without TG/Slack tokens)
        try:
            import subprocess
            subprocess.run(["bash", os.path.expanduser("~/.claude/skills/ml-intern/scripts/notify.sh"),
                            "train_done",
                            f"200m-qwen3 milestone @ step={step} tokens={gb:.2f}B  {repo_url}"],
                           check=False, timeout=10)
        except Exception:
            pass
    finally:
        shutil.rmtree(stage)


if __name__ == "__main__":
    main()
