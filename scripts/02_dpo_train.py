"""Stage 2 — DPO fine-tuning with TRL + LoRA.

What happens here, conceptually:
  - The policy model (base + trainable LoRA adapter) learns from (prompt, chosen, rejected) triples.
  - The reference model is a FROZEN copy of the base weights. With LoRA, TRL reuses the
    base weights with the adapter disabled as the reference — no second copy in VRAM.
  - DPO loss:  -log sigmoid( beta * [(logpi(y_w) - logref(y_w)) - (logpi(y_l) - logref(y_l))] )
    beta = how far the policy is allowed to drift from the reference (KL budget).
  - Watch `rewards/margins` in the logs: chosen-vs-rejected implicit-reward gap.
    Rising margin = preference learning is working.

Sized for a 16GB T4: LoRA r=16, batch 2 x accum 4, max_length 512, 500 steps, fp16.

Outputs:
  - dpo_adapter/            (LoRA adapter + tokenizer)
  - dpo_merged/             (adapter merged into base weights, fp16 — used by stages 3-4)
  - results/dpo_train_log.json (final trainer log history)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from common import (
    ADAPTER_DIR,
    MERGED_DIR,
    MODEL_ID,
    RESULTS_DIR,
    ensure_dirs,
    get_tokenizer,
    load_base_model,
    prepare_dpo_dataset,
    save_json,
)

# Keep the run short and cheap: 500 steps is enough to see margins move on 0.5B.
MAX_STEPS = 500
BETA = 0.1  # KL budget knob — try 0.05 vs 0.2 in a second run to feel the difference


def main():
    ensure_dirs()
    tok = get_tokenizer()

    print("[data] loading preference dataset ...")
    train_ds = prepare_dpo_dataset(tok)  # prompt / chosen / rejected text columns
    print(f"[data] {len(train_ds)} preference pairs")

    print("[model] loading base model ...")
    model = load_base_model()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = DPOConfig(
        output_dir=str(ADAPTER_DIR),
        beta=BETA,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 8
        max_steps=MAX_STEPS,
        max_length=512,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        save_steps=250,
        bf16=False,
        fp16=True,                       # T4: no bf16 support
        gradient_checkpointing=True,     # trades compute for VRAM — needed on 16GB
        remove_unused_columns=False,     # keep prompt/chosen/rejected columns intact
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # None -> frozen copy of base (with LoRA: base weights, adapter disabled)
        args=args,
        train_dataset=train_ds,
        processing_class=tok,    # `tokenizer=` is deprecated in trl>=0.15
        peft_config=peft_config,
    )

    print("[train] starting DPO — watch rewards/margins ...")
    trainer.train()

    print("[save] saving adapter ...")
    trainer.save_model(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    save_json(trainer.state.log_history, RESULTS_DIR / "dpo_train_log.json")

    # Merge LoRA into base weights so stages 3-4 can eval a plain HF checkpoint.
    print("[merge] merging adapter into base weights ...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).merge_and_unload()
    merged.save_pretrained(str(MERGED_DIR))
    tok.save_pretrained(str(MERGED_DIR))

    print(f"\n[done] adapter -> {ADAPTER_DIR}\n[done] merged  -> {MERGED_DIR}")
    print("Tip: plot rewards/margins from results/dpo_train_log.json — rising margin = learning.")


if __name__ == "__main__":
    main()
