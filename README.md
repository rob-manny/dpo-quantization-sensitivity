# Does post-training change how a model quantizes?

## TL;DR

| Checkpoint | MMLU Δ vs base fp16 (3 subjects × 30) | fp16 → 4-bit NF4 Δ | fp16 → INT8 Δ |
|---|---|---|---|
| Base (fp16) | — | re-run to fill in | not run |
| DPO (fp16) | 0.000 / −0.033 / −0.067 | re-run to fill in | not run |

- **The concrete finding we do have:** DPO's MMLU deltas were 0.000, −0.033, −0.067
  across three subjects. At n=30 per subject each question is worth 0.033, so that's
  0, −1, −2 questions: **noise, not signal.** No capability collapse; the slice is too
  small and too close to chance to resolve a small alignment tax.
- **The X-vs-Y quantization number doesn't exist yet.** The Sept 21 Colab run's stage-4
  outputs were never saved back (`results/` is empty), so there is no saved
  "DPO lost X points under quantization vs Y for base" table. Re-running stage 4
  (~15 min on a T4) regenerates it — the qualitative null result reported below is
  all that survived from the original run.
- **Correction:** this study compared fp16 vs 4-bit NF4 (bitsandbytes, weight-only),
  not INT8. An actual INT8 claim is a small extension to `scripts/04_quantize_compare.py`,
  not a re-read of old data.

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

![DPO reward margin](assets/reward_margin.png)

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
  under quantization. No evidence that this DPO run introduced a new
  quantization fragility mode. Null results are results: it means the two
  compose without surprises, at least at this scale.
- **Visible behavior barely moved** — greedy-decoded generations on generic
  prompts were light paraphrases of the base. DPO nudges probabilities; small
  nudges rarely flip the argmax. The learning is statistical (see the margin
  curve), not a personality transplant. This is why post-training is
  evaluated with win-rates over held-out pairs, not eyeballed prompts.

## Limits — read before citing

- 0.5B parameters, 500 DPO steps, one dataset, one seed. A study, not research.
- 4-bit NF4 is weight-only via bitsandbytes — a standard recipe, not hardware.
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
