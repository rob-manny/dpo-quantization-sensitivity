"""Shared config, model/dataset loading, and result helpers for the DPO learning project.

Everything is sized for a 16GB T4 (Colab free tier).
"""
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# Config — change MODEL_ID here and every script/notebook picks it up.
# ----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B"          # ungated base model
FALLBACK_MODEL_ID = "TinyLlama/TinyLlama-1.1B"

DPO_DATASET_ID = "argilla/dpo-mix-7k"   # small preference dataset (chosen/rejected)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ADAPTER_DIR = PROJECT_ROOT / "dpo_adapter"       # LoRA adapter saved by stage 2
MERGED_DIR = PROJECT_ROOT / "dpo_merged"          # LoRA merged into base, used by stages 3-4

# Small, fast MMLU slice for before/after comparison (full MMLU is overkill here)
MMLU_TASKS = [
    "mmlu_high_school_mathematics",
    "mmlu_elementary_mathematics",
    "mmlu_college_computer_science",
]
MMLU_LIMIT = 30  # questions per subject

# Fixed open-ended prompts — identical pre/post training so the difference is visible
OPEN_ENDED_PROMPTS = [
    "Explain why the sky is blue in one short paragraph.",
    "Write a polite email declining a meeting invitation.",
    "What are three tips for debugging a slow Python program?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Is it safe to share my password with tech support? Explain briefly.",
]

DTYPE = torch.float16  # T4 has no bf16 support; A100+ boxes can switch to bfloat16


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_tokenizer(model_id: str = MODEL_ID) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Qwen2.5 has no pad token by default; DPO batching needs one.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(model_id: str = MODEL_ID):
    """Load the base causal LM in fp16 on GPU (or CPU fallback)."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )


def _as_text(msgs, tok) -> str:
    """Normalize a chosen/rejected field (string or chat-message list) to text."""
    if isinstance(msgs, str):
        return msgs.strip()
    # list of {"role": ..., "content": ...}
    return tok.apply_chat_template(msgs, tokenize=False).strip()


def _split_prompt(conv):
    """Split a conversation into (prompt_messages, response_messages).

    dpo-mix-7k has no separate `prompt` column — the prompt is the shared
    prefix of the chosen/rejected conversations (everything but the last
    assistant message). Plain strings have no separable prompt.
    """
    if isinstance(conv, str):
        return [], conv
    if isinstance(conv, list) and len(conv) > 1:
        return conv[:-1], conv[-1:]
    return [], conv


def prepare_dpo_dataset(tok, split: str = "train", max_rows: int | None = None):
    """Load argilla/dpo-mix-7k and normalize to prompt/chosen/rejected text columns.

    DPOTrainer expects `prompt`, `chosen`, `rejected`. dpo-mix-7k stores
    full conversations, so the prompt is recovered as the shared prefix
    (all messages except the final response) and everything is rendered
    to strings with the chat template.
    """
    ds = load_dataset(DPO_DATASET_ID, split=split)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))

    def _fmt(batch):
        prompts, chosens, rejecteds = [], [], []
        for c, r in zip(batch["chosen"], batch["rejected"]):
            p_c, c_last = _split_prompt(c)
            p_r, _ = _split_prompt(r)
            prompt_msgs = p_c if p_c else p_r
            prompts.append(_as_text(prompt_msgs, tok) if prompt_msgs else "")
            chosens.append(_as_text(c_last, tok))
            rejecteds.append(_as_text(_split_prompt(r)[1], tok))
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    ds = ds.map(_fmt, batched=True, remove_columns=ds.column_names)
    return ds


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
