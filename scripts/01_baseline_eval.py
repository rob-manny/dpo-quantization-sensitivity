"""Stage 1 — Baseline evaluation of the BASE model (before DPO).

Runs:
  1. A small MMLU slice via lm-eval (multiple choice accuracy).
  2. Fixed open-ended prompts, generations saved verbatim.

Outputs (results/):
  - baseline_mmlu.json        (per-task accuracy + per-sample records)
  - baseline_generations.json (prompt -> generated text)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm_eval
import torch
from common import (
    MMLU_LIMIT,
    MMLU_TASKS,
    OPEN_ENDED_PROMPTS,
    RESULTS_DIR,
    ensure_dirs,
    get_tokenizer,
    load_base_model,
    load_json,
    save_json,
    MODEL_ID,
)


def run_mmlu(model_id: str = MODEL_ID):
    print(f"[mmlu] evaluating {model_id} on {MMLU_TASKS} (limit={MMLU_LIMIT}) ...")
    # If a task name 404s on your lm-eval version, list valid names with:
    #   python -c "import lm_eval.tasks as t; print([n for n in t.TaskManager().all_tasks if n.startswith('mmlu_h')][:5])"
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_id},dtype=float16,trust_remote_code=True",
        tasks=MMLU_TASKS,
        limit=MMLU_LIMIT,
        log_samples=True,
        batch_size=8,
    )
    summary = {
        task: {
            "acc": results["results"][task].get("acc,none"),
            "acc_stderr": results["results"][task].get("acc_stderr,none"),
        }
        for task in MMLU_TASKS
    }
    save_json(summary, RESULTS_DIR / "baseline_mmlu.json")
    save_json(results.get("samples", {}), RESULTS_DIR / "baseline_mmlu_samples.json")
    print("[mmlu] summary:", summary)
    return summary


@torch.inference_mode()
def run_generations():
    print("[gen] generating open-ended baselines ...")
    tok = get_tokenizer()
    model = load_base_model()
    model.eval()
    out = []
    for prompt in OPEN_ENDED_PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id
        )
        text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"prompt": prompt, "generation": text})
        print(f"\nPROMPT: {prompt}\nGEN: {text}\n{'-' * 60}")
    save_json(out, RESULTS_DIR / "baseline_generations.json")
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ensure_dirs()
    run_mmlu()
    run_generations()
    print("\n[done] baseline artifacts in", RESULTS_DIR)


if __name__ == "__main__":
    main()
