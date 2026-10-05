"""Stage 3 — Re-evaluation of the DPO checkpoint (identical harness as stage 1).

Runs the SAME evals on dpo_merged/:
  1. Same MMLU slice -> compare accuracy vs results/baseline_mmlu.json (the "alignment tax").
  2. Same open-ended prompts -> side-by-side with baseline generations.
  3. Preference win-rate on held-out pairs: DPO model vs base model, judged by
     implicit reward (logprob margin). Win-rate > 50% = alignment worked.

Outputs (results/):
  - dpo_mmlu.json, dpo_generations.json, comparison.json, win_rate.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm_eval
import torch
from common import (
    MERGED_DIR,
    MMLU_LIMIT,
    MMLU_TASKS,
    OPEN_ENDED_PROMPTS,
    RESULTS_DIR,
    ensure_dirs,
    get_tokenizer,
    load_base_model,
    load_json,
    save_json,
)
from transformers import AutoModelForCausalLM


def run_mmlu(model_path: str):
    print(f"[mmlu] evaluating {model_path} ...")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=float16,trust_remote_code=True",
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
    save_json(summary, RESULTS_DIR / "dpo_mmlu.json")
    return summary


@torch.inference_mode()
def run_generations(model_path: str):
    print("[gen] generating post-DPO responses ...")
    tok = get_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    out = []
    for prompt in OPEN_ENDED_PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id
        )
        text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"prompt": prompt, "generation": text})
    save_json(out, RESULTS_DIR / "dpo_generations.json")
    del model
    torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def preference_win_rate(n_pairs: int = 200):
    """Head-to-head: does the DPO model assign higher implicit reward to `chosen`?

    Implicit reward of a response y for prompt x under DPO:
        r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))
    We approximate the win-rate by comparing mean logprob margins; a full
    implementation would score with both models, but the margin direction is
    what matters for the learning signal.
    """
    from common import MODEL_ID, prepare_dpo_dataset

    print(f"[winrate] scoring {n_pairs} held-out preference pairs ...")
    tok = get_tokenizer()
    ds = prepare_dpo_dataset(tok).shuffle(seed=0).select(range(n_pairs))

    def mean_logprob(model, prompt, response):
        full = prompt + response
        enc = tok(full, return_tensors="pt").to(model.device)
        p_enc = tok(prompt, return_tensors="pt")
        prompt_len = p_enc["input_ids"].shape[1]
        logits = model(**enc).logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_ids = enc["input_ids"][0][1:]
        token_logprobs = logprobs[:-1].gather(1, token_ids.unsqueeze(1)).squeeze(1)
        resp_logprobs = token_logprobs[prompt_len - 1:]
        return resp_logprobs.mean().item()

    base = load_base_model(MODEL_ID)
    dpo = AutoModelForCausalLM.from_pretrained(
        str(MERGED_DIR), torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    wins = 0
    for row in ds:
        base_margin = mean_logprob(base, row["prompt"], row["chosen"]) - mean_logprob(
            base, row["prompt"], row["rejected"]
        )
        dpo_margin = mean_logprob(dpo, row["prompt"], row["chosen"]) - mean_logprob(
            dpo, row["prompt"], row["rejected"]
        )
        if dpo_margin > base_margin:
            wins += 1
    result = {"n_pairs": n_pairs, "dpo_win_rate": wins / n_pairs}
    save_json(result, RESULTS_DIR / "win_rate.json")
    print("[winrate]", result)
    return result


def main():
    ensure_dirs()
    if not MERGED_DIR.exists():
        raise SystemExit("dpo_merged/ not found — run 02_dpo_train.py first.")

    dpo_mmlu = run_mmlu(str(MERGED_DIR))
    baseline_mmlu = load_json(RESULTS_DIR / "baseline_mmlu.json")
    comparison = {
        task: {
            "baseline_acc": baseline_mmlu[task]["acc"],
            "dpo_acc": dpo_mmlu[task]["acc"],
            "delta": (dpo_mmlu[task]["acc"] or 0) - (baseline_mmlu[task]["acc"] or 0),
        }
        for task in MMLU_TASKS
    }
    save_json(comparison, RESULTS_DIR / "comparison.json")
    print("\n[comparison] MMLU deltas (negative delta = alignment tax):")
    for task, row in comparison.items():
        print(f"  {task}: {row['baseline_acc']:.3f} -> {row['dpo_acc']:.3f} (Δ {row['delta']:+.3f})")

    dpo_gens = run_generations(str(MERGED_DIR))
    base_gens = load_json(RESULTS_DIR / "baseline_generations.json")
    print("\n[side-by-side] generations (base vs DPO):")
    for b, d in zip(base_gens, dpo_gens):
        print(f"\nPROMPT: {b['prompt']}\nBASE: {b['generation'][:200]}\nDPO:  {d['generation'][:200]}")

    preference_win_rate()
    print("\n[done] stage-3 artifacts in", RESULTS_DIR)


if __name__ == "__main__":
    main()
