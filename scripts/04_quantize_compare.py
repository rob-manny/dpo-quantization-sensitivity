"""Stage 4 — 4-bit quantization sensitivity: base vs DPO checkpoint.

The question: does the ALIGNED checkpoint degrade differently under 4-bit
quantization than the BASE? We use standard bitsandbytes NF4 (weight-only,
the most widely reproduced 4-bit recipe). For each checkpoint (base fp16,
DPO fp16) we measure:
  1. Perplexity (WikiText-2, with fallback to held-out preference text)
     in fp16 vs NF4.
  2. A multiple-choice MMLU slice scored by log-likelihood over A/B/C/D
     (same idea as lm-eval), in fp16 vs NF4, run in-process on each model.

A larger degradation on the DPO checkpoint would mean preference fine-tuning
changed the weight distributions in ways quantization is sensitive to —
i.e. the alignment step introduced a new fragility mode. A null result
(base and DPO degrade alike) means the two compose without surprises, at
least at this scale.

Outputs: results/nf4_compare.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from common import (
    MERGED_DIR,
    MMLU_LIMIT,
    MMLU_TASKS,
    MODEL_ID,
    RESULTS_DIR,
    ensure_dirs,
    get_tokenizer,
    save_json,
    _as_text,
)

BNB_NF4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# lm-eval task names -> cais/mmlu dataset configs
SUBJECTS = [t.replace("mmlu_", "") for t in MMLU_TASKS]


def ppl_corpus(tok, n_chars: int = 200_000) -> str:
    """Perplexity text: WikiText-2 test, falling back to held-out preference
    text if the wikitext dataset ID fails to resolve in this environment."""
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])[:n_chars]
    except Exception as e:
        print(f"wikitext load failed ({type(e).__name__}); falling back to dpo-mix-7k test text")
        ds = load_dataset("argilla/dpo-mix-7k", split="test")
        return "\n\n".join(_as_text(c, tok) for c in ds["chosen"][:400])[:n_chars]


@torch.inference_mode()
def perplexity(model, tok, stride: int = 512) -> float:
    """Token-level perplexity, strided like the classic eval."""
    ids = tok(ppl_corpus(tok), return_tensors="pt")["input_ids"].to(model.device)
    nlls = []
    for i in tqdm(range(0, ids.size(1) - stride, stride), desc="ppl", leave=False):
        chunk = ids[:, i : i + stride]
        logits = model(chunk).logits
        nlls.append(
            torch.nn.functional.cross_entropy(
                logits[0, :-1].reshape(-1, logits.size(-1)),
                chunk[0, 1:].reshape(-1),
            )
        )
    return torch.exp(torch.stack(nlls).mean()).item()


@torch.inference_mode()
def mmlu_manual(model, tok, limit: int = MMLU_LIMIT) -> dict:
    """Multiple-choice MMLU scored by log-likelihood over A/B/C/D.

    Same scoring idea as the lm-eval harness: for each question, the model's
    log-probability of each choice token as a continuation is compared and
    the highest-scoring choice wins. Run in-process so it works on the
    quantized model object directly.
    """
    acc = {}
    for subj in SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split="test").select(range(limit))
        correct = 0
        for ex in tqdm(ds, desc=f"mmlu/{subj}", leave=False):
            prompt = (
                "Question: " + ex["question"] + "\n"
                + "\n".join(f"{chr(65 + i)}. {ch}" for i, ch in enumerate(ex["choices"]))
                + "\nAnswer:"
            )
            inp = tok(prompt, return_tensors="pt").to(model.device)
            L = inp["input_ids"].size(1)
            scores = []
            for letter in "ABCD":
                cont = tok(" " + letter, return_tensors="pt")["input_ids"].to(model.device)
                full = torch.cat([inp["input_ids"], cont], dim=1)
                logits = model(full).logits[0]
                lp = torch.log_softmax(logits[L - 1 : -1], dim=-1)
                scores.append(lp.gather(-1, cont[0].unsqueeze(-1)).sum().item())
            correct += int(torch.tensor(scores).argmax()) == ex["answer"]
        acc[subj] = correct / len(ds)
    return acc


def load_model(path: str, quantize: bool):
    if quantize:
        return AutoModelForCausalLM.from_pretrained(
            path, quantization_config=BNB_NF4, device_map="auto", trust_remote_code=True
        ).eval()
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    ).eval()


def main():
    ensure_dirs()
    if not MERGED_DIR.exists():
        raise SystemExit("dpo_merged/ not found — run 02_dpo_train.py first.")

    report = {}
    for name, path in [("base", MODEL_ID), ("dpo", str(MERGED_DIR))]:
        print(f"\n===== {name} ({path}) =====")
        tok = get_tokenizer(path)
        report[name] = {}
        for fmt, quant in [("fp16", False), ("nf4", True)]:
            model = load_model(path, quant)
            ppl = perplexity(model, tok)
            acc = mmlu_manual(model, tok)
            report[name][fmt] = {"ppl": ppl, "mmlu": acc}
            print(f"[{name}/{fmt}] ppl={ppl:.2f} mmlu={ {k: round(v, 3) for k, v in acc.items()} }")
            del model
            torch.cuda.empty_cache()

    print("\n===== degradation vs fp16 =====")
    for name in report:
        dp = 100 * (report[name]["nf4"]["ppl"] - report[name]["fp16"]["ppl"]) / report[name]["fp16"]["ppl"]
        dm = {k: round(report[name]["nf4"]["mmlu"][k] - report[name]["fp16"]["mmlu"][k], 3) for k in SUBJECTS}
        print(f"[{name}/nf4] ppl +{dp:.1f}%   mmlu delta {dm}")

    save_json(report, RESULTS_DIR / "nf4_compare.json")
    print("\nTakeaway: compare the nf4 degradation between base and dpo.")
    print("If they degrade alike, DPO introduced no new quantization fragility.")


if __name__ == "__main__":
    main()
