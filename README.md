# Does post-training change how a model quantizes?

## TL;DR — the concrete finding

Perplexity change vs. fp16 (v2 harness; lower is better):

| Precision | Base ppl Δ | DPO ppl Δ |
| --- | --- | --- |
| INT8 (bitsandbytes LLM.int8) | +0.7% | +0.6% |
| BFP16 (block 64, 7-bit mantissa) | +0.2% | +0.3% |
| BFP12 (block 32, 3-bit mantissa) | +36.7% | +35.0% |
| BFP identity check (23-bit mantissa) | +0.0% | −0.0% |

**The DPO checkpoint degrades like the base under reduced precision: no added quantization fragility.**
Across INT8, BFP16, and BFP12, base and DPO perplexity moves within ~2 points of each other.
BFP16 is effectively lossless here (better than INT8), while BFP12 costs ~35–37% perplexity.
MMLU deltas stayed within ±1–3 questions per subject (noise at n=30; see Limits).
Full numbers: `results/bfp_compare.json`.

- **DPO itself worked:** the reward margin (chosen-vs-rejected implicit-reward gap) rose ~0 → ~1.25 over 500 steps.
- **No capability collapse:** MMLU deltas after DPO were 0.000 / −0.033 / −0.067 (0, −1, −2 questions at n=30), reproduced exactly on a second training run.

## Correction: v1 BFP results were wrong

The first version of this study reported BFP perplexity increases of +231% (BFP16) and +1878% (BFP12). Those numbers came from a bug in the simulated quantizer: the step size was one exponent bit too small, so the largest weight in every block was clipped by up to ~50%.

How it was found and fixed:
1. The v1 magnitudes were implausible for an 8-bit-class format.
2. **Identity test:** quantizing with a 23-bit mantissa should change nothing. v2 gives exactly 0.0 max weight difference and +0.0% perplexity.
3. **Per-layer weight error:** v2 BFP16 gives 0.96% relative error on layer 0 `down_proj` (BFP12: 13.5%).
4. **INT8 reference:** a widely used, known-good format (bitsandbytes LLM.int8) lands at +0.6–0.7%, confirming the evaluation harness itself is sound.

The fix uses `step = 2^(exp + 1 − mantissa_bits)` with a symmetric clamp, so each block's maximum fits without clipping. The v1 relative conclusion (base and DPO degrade alike) holds in v2; the magnitudes did not.

## The question

The model that ships is never the base model — it's the post-trained one. But quantization studies are usually run on the base. If preference fine-tuning shifts weight distributions (outlier channels, sharper activations), the aligned model could be more fragile under quantization than the base suggests. This project tests that directly, at small scale.

## Method

- **Base model:** Qwen2.5-0.5B (ungated), fp16
- **Post-training:** DPO via TRL `DPOTrainer` + LoRA (r=16), β=0.1, 500 steps, effective batch 8 — sized for a free Colab T4 (16GB)
- **Data:** `argilla/dpo-mix-7k` — 6,750 preference pairs. The dataset has no `prompt` column, so the prompt is recovered as the shared prefix of each chosen/rejected conversation
- **Eval, two tracks:**
  - *Track A (quantitative):* MMLU slice — 3 subjects × 30 questions, multiple-choice scored by log-likelihood over A/B/C/D
  - *Track B (qualitative):* 5 fixed open-ended prompts, generations saved verbatim before/after for side-by-side comparison
- **Stage 4 (quantization):** for both checkpoints, compare fp16, INT8 (bitsandbytes LLM.int8), and simulated block floating point (weight-only, all linear layers). BFP is the precision family closest to real AI-accelerator numerics.
  - `lm_head` is excluded: Qwen2.5 ties it to the input embeddings (`tie_word_embeddings=True`), so both stay at fp16.
  - Sanity checks run before the full comparison (identity test, per-layer weight error).

## Findings

- **The aligned checkpoint quantizes like the base (null result, with numbers).** INT8: +0.7% vs +0.6%. BFP16: +0.2% vs +0.3%. BFP12: +36.7% vs +35.0%. The base↔DPO gap does not widen under quantization.
- **BFP16 beat INT8** on perplexity here (+0.2–0.3% vs +0.6–0.7%), consistent with a shared per-block exponent preserving more dynamic range than per-tensor scaling.
- **BFP12 is the cliff:** with 3 magnitude bits per value, perplexity rises ~35–37%. MMLU moved by at most 3 questions per subject, which this slice cannot distinguish from noise.
- **The preference was learned.** The DPO reward margin rose from ~0 to ~1.25 over 500 steps, flattening toward the end — 500 steps was about right; more would be memorizing noise.
- **No MMLU regression from DPO, with a caveat.** Deltas were 0.000, −0.033, −0.067 across the three subjects (0, −1, −2 questions). Noise, not signal; the base model is near chance on these subjects, so this slice can't resolve a small alignment tax.
- **4-bit NF4 (earlier run):** base and DPO degraded alike; the gap did not widen. Stage-4 NF4 outputs from that run were not saved.
- **Visible behavior barely moved** — greedy generations on generic prompts were light paraphrases of the base. DPO nudges probabilities; small nudges rarely flip the argmax. This is why post-training is evaluated with win-rates over held-out pairs, not eyeballed prompts.

## Limits — read before citing

- 0.5B parameters, 500 DPO steps, one dataset, one seed. A study, not research.
- BFP is simulated, weight-only, and an approximation of the format family, not a specific hardware spec. Activations stay in fp16.
- Perplexity was measured on dpo-mix-7k test text: WikiText-2 failed to load in this run's environment. All formats use the same text, so comparisons between formats are valid, but absolute perplexity is not comparable to published WikiText numbers. The notebook now loads `Salesforce/wikitext`.
- The MMLU slice (n=30/subject) is too small and too close to chance for fine-grained claims. Stage 3 (lm-eval) and Stage 4 (in-process scorer) use different prompt formatting, so compare MMLU only within a stage.
- LoRA adapters, not full fine-tuning; DPO, not PPO/GRPO.

## Layout

```
├── README.md               # this file
├── requirements.txt
├── dpo_learning.ipynb      # full pipeline, Colab-ready (T4 GPU, run top to bottom)
├── scripts/
│   ├── common.py           # shared config, model/dataset loading, result saving
│   ├── 01_baseline_eval.py # stage 1: baseline MMLU + generations
│   ├── 02_dpo_train.py     # stage 2: DPO fine-tuning + LoRA merge
│   ├── 03_reeval.py        # stage 3: re-eval, alignment-tax table
│   └── 04_quantize_compare.py  # stage 4 (script version): fp16 vs 4-bit NF4
├── results/                # bfp_compare.json and other outputs
└── assets/                 # reward-margin plot
```

## Quickstart

Open `dpo_learning.ipynb` in Colab (T4 GPU) and run top to bottom. Stage 4a prints the quantizer sanity checks; run Stage 4b only after they pass.

`dpo_adapter/` and `dpo_merged/` are written by stage 2 and are reproducible from the notebook; they are not committed.

## What this demonstrates

End-to-end post-training literacy — preference-data preparation, DPO mechanics (reference policy, KL budget β, reward margins), alignment-tax measurement — and quantization-sensitivity analysis of aligned vs. base checkpoints across INT8 and block floating point, including catching, diagnosing, and correcting a bug in my own quantizer.
