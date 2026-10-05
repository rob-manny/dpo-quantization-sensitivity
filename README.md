# Does post-training change how a model quantizes?

## TL;DR — the concrete finding

| Precision | Base ppl Δ vs fp16 | DPO ppl Δ vs fp16 | Base MMLU Δ (3 subjects) | DPO MMLU Δ (3 subjects) |
|---|---|---|---|---|
| bfp16_64 | +231% | +236% | −0.167 / −0.100 / −0.067 | −0.133 / −0.067 / 0.000 |
| bfp12 | +1878% | +1940% | −0.200 / −0.133 / −0.100 | −0.133 / −0.200 / −0.100 |

**The DPO checkpoint degrades like the base under reduced precision — no added
quantization fragility.** Perplexity and MMLU move together across bfp16_64 and
bfp12; the base↔DPO gap does not widen under quantization. Full numbers:
`results/bfp_compare.json` (run 2026-10-05; perplexity measured on dpo-mix-7k
test text after WikiText-2 failed to load in that environment).

- **DPO itself worked:** the reward margin (chosen-vs-rejected implicit-reward gap)
  rose ~0 → ~1.25 over 500 steps. MMLU deltas were 0.000 / −0.033 / −0.067 — at
  n=30 per subject each question is worth 0.033, so that's 0, −1, −2 questions:
  **noise, not signal.** No capability collapse; the slice can't resolve a small
  alignment tax.
- The scripted stage 4 (`scripts/04_quantize_compare.py`) covers fp16 vs 4-bit NF4
  (bitsandbytes, weight-only); the BFP table above was a follow-up run on the same
  base-vs-DPO harness.

An independent, single-evening study. Take a small base model, run DPO
preference fine-tuning end-to-end, measure what changed — then ask the
question production teams actually care about: **does the aligned checkpoint
quantize differently than the base?**

## The question

The model that ships is never the base model — it's the post-trained one.
But quantization studies are usually run on the base. If preference
fine-tuning shifts weight distributions (outlier channels, sharper
activations), the aligned model could be more fragile under quantization
than the base suggests. This project tests that directly, at small scale.

## Method

- **Base model:** Qwen2.5-0.5B (ungated), fp16
- **Post-training:** DPO via TRL `DPOTrainer` + LoRA (r=16), β=0.1,
  500 steps, effective batch 8 — sized for a free Colab T4 (16GB)
- **Data:** `argilla/dpo-mix-7k` — 6,750 preference pairs. The dataset has
  no `prompt` column, so the prompt is recovered as the shared prefix of
  each chosen/rejected conversation (everything but the final response)
- **Eval, two tracks:**
  - *Track A (quantitative):* MMLU slice — 3 subjects × 30 questions,
    multiple-choice scored by log-likelihood over A/B/C/D
  - *Track B (qualitative):* 5 fixed open-ended prompts, generations saved
    verbatim before/after for side-by-side comparison
- **Stage 4:** fp16 vs 4-bit NF4 (bitsandbytes, weight-only) on **both**
  checkpoints — perplexity plus an in-process log-likelihood MMLU scorer,
  so the quantized model is evaluated directly

## Findings

- **The DPO checkpoint quantizes like the base (null result, with numbers).**
  Under bfp16_64, base perplexity rose 231% vs 236% for DPO; under bfp12, +1878%
  vs +1940%. MMLU deltas per subject track within a question or two of each other
  (n=30, so ±0.033 per question — noise-dominated individually, but the *pattern*
  of base and DPO moving together is the finding). No evidence that this DPO run
  introduced a new quantization fragility mode. Null results are results: the two
  compose without surprises, at least at this scale. Block floating point is also
  the precision family closest to real AI-accelerator numerics, which is what
  motivated this follow-up.


- **The preference was learned.** The DPO reward margin (chosen-vs-rejected
  implicit-reward gap) rose from ~0 to ~1.25 over 500 steps, flattening
  toward the end — 500 steps was about right; more would be memorizing noise.
- **No MMLU regression — with a caveat.** Deltas were 0.000, −0.033, −0.067
  across the three subjects. At n=30 per subject each question is worth
  0.033, so that's 0, −1, −2 questions: **noise, not signal.** The honest
  claim is narrower: no capability collapse. The base model was near chance
  on these subjects anyway, so this slice can't resolve a small alignment tax.
- **The aligned checkpoint quantizes like the base (null result).** Under
  4-bit NF4, base and DPO checkpoints degraded alike — the gap did not widen
  under quantization. (Qualitative result from the Sept 21 run; stage-4 outputs
  were not saved. See the BFP table in the TL;DR for the same question answered
  with numbers.)
- **Visible behavior barely moved** — greedy-decoded generations on generic
  prompts were light paraphrases of the base. DPO nudges probabilities; small
  nudges rarely flip the argmax. The learning is statistical (see the margin
  curve), not a personality transplant. This is why post-training is
  evaluated with win-rates over held-out pairs, not eyeballed prompts.

## Limits — read before citing

- 0.5B parameters, 500 DPO steps, one dataset, one seed. A study, not research.
- 4-bit NF4 is weight-only via bitsandbytes — a standard recipe, not hardware.
- The BFP perplexity numbers were measured on dpo-mix-7k test text (WikiText-2
  failed to load in that run's environment), not a standard ppl corpus — treat
  the MMLU deltas as the cleaner signal.
- The MMLU slice (n=30/subject) is too small and too close to chance for
  fine-grained claims. It rules out collapse; it can't measure a small tax.
- Single-evening scope: LoRA adapters, not full fine-tuning; DPO, not PPO/GRPO.

## Layout

```
├── README.md               # this file
├── requirements.txt
├── dpo_learning.ipynb      # full pipeline, Colab-ready (open in Colab, T4 GPU, run top to bottom)
├── scripts/
│   ├── common.py           # shared config, model/dataset loading, result saving
│   ├── 01_baseline_eval.py # stage 1: baseline MMLU + generations
│   ├── 02_dpo_train.py     # stage 2: DPO fine-tuning + LoRA merge
│   ├── 03_reeval.py        # stage 3: re-eval, alignment-tax table, win-rate
│   └── 04_quantize_compare.py  # stage 4: fp16 vs 4-bit NF4, base vs DPO
├── results/                # populated by running (see results/README.md)
└── assets/                 # reward-margin plot lives here
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/01_baseline_eval.py   # ~15 min on T4
python scripts/02_dpo_train.py       # ~30 min on T4
python scripts/03_reeval.py          # ~15 min on T4
python scripts/04_quantize_compare.py
```

Or open `dpo_learning.ipynb` in Colab and run top to bottom.

`dpo_adapter/` and `dpo_merged/` are written to the repo root by stage 2 —
add them to `.gitignore`; they're reproducible from the scripts.

## What this demonstrates

End-to-end post-training literacy: preference-data preparation, DPO
mechanics (reference policy, KL budget β, reward margins), alignment-tax
measurement, and quantization-sensitivity analysis of aligned vs base
checkpoints — with the experimental honesty to report a null result.
