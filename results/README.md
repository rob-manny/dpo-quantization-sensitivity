# results/

This directory is populated by running the pipeline — nothing here is
committed with fabricated numbers. Run the stages and these files appear:

| File | Produced by | Contents |
|---|---|---|
| `baseline_mmlu.json` | stage 1 | base-model MMLU accuracy per subject (+ stderr) |
| `baseline_mmlu_samples.json` | stage 1 | per-sample MMLU records (auditability) |
| `baseline_generations.json` | stage 1 | verbatim generations for the 5 fixed prompts (base) |
| `dpo_train_log.json` | stage 2 | full TRL trainer log history (loss, `rewards/margins`, …) |
| `dpo_mmlu.json` | stage 3 | DPO-checkpoint MMLU accuracy per subject |
| `dpo_generations.json` | stage 3 | verbatim generations for the 5 fixed prompts (DPO) |
| `comparison.json` | stage 3 | base → DPO MMLU deltas per subject (the "alignment tax" table) |
| `win_rate.json` | stage 3 | DPO-vs-base preference win-rate on held-out pairs |
| `nf4_compare.json` | stage 4 | fp16 vs 4-bit-NF4 perplexity + MMLU for base and DPO checkpoints |
| `bfp_compare.json` | follow-up run 2026-10-05 | fp16 vs bfp16_64/bfp12 perplexity + MMLU deltas for base and DPO checkpoints (transcribed from run log; ppl on dpo-mix-7k test text) |

Model artifacts (`dpo_adapter/`, `dpo_merged/`) are written to the repo root
by stage 2 and are git-ignored — they're large and reproducible from the
scripts.
