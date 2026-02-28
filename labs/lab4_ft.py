"""
lora_vs_full.py
Compare full fine-tuning vs LoRA on a small instruction dataset.
Tracks: trainable parameters, peak GPU memory, training time, train loss.

Requirements:
    pip install transformers peft trl datasets accelerate bitsandbytes
"""

import os, time, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ── config ────────────────────────────────────────────────────────────────────

MODEL_ID    = "Qwen/Qwen2.5-1.5B"
DATASET_ID  = "yahma/alpaca-cleaned"
MAX_SAMPLES = 1000
MAX_SEQ_LEN = 512
EPOCHS      = 1
BATCH_SIZE  = 4
GRAD_ACCUM  = 4

LORA_R      = 8
LORA_ALPHA  = 16
LORA_TARGET = ["q_proj", "v_proj"]

OUTPUT_DIR  = "checkpoints"

# ── helpers ───────────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_example(example):
    instruction = example["instruction"].strip()
    inp         = example["input"].strip()
    output      = example["output"].strip()
    if inp:
        text = (f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}")
    else:
        text = (f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}")
    return {"text": text}


def load_data():
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.shuffle(seed=42).select(range(MAX_SAMPLES))
    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds


def peak_memory_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def run_experiment(mode, dataset, tokenizer):
    assert mode in ("full", "lora")
    print(f"\n{'='*60}")
    print(f"  Mode: {mode.upper()} FINE-TUNING")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    if mode == "lora":
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET,
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, lora_cfg)

    total, trainable = count_parameters(model)
    print(f"  Total parameters    : {total/1e6:.1f} M")
    print(f"  Trainable parameters: {trainable/1e6:.3f} M  "
          f"({100*trainable/total:.2f}%)")

    out_dir = os.path.join(OUTPUT_DIR, mode)

    # SFTConfig = TrainingArguments + SFT-specific args (max_seq_length, etc.)
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4 if mode == "lora" else 2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
    )

    start   = time.time()
    result  = trainer.train()
    elapsed = time.time() - start

    mem_gb     = peak_memory_gb()
    final_loss = result.training_loss

    print(f"\n  Training time : {elapsed:.1f} s")
    print(f"  Peak GPU mem  : {mem_gb:.2f} GB")
    print(f"  Final loss    : {final_loss:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        "mode":               mode,
        "total_params_M":     total / 1e6,
        "trainable_params_M": trainable / 1e6,
        "trainable_pct":      100 * trainable / total,
        "train_time_s":       elapsed,
        "peak_mem_gb":        mem_gb,
        "final_loss":         final_loss
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "lora", "both"],
                        default="both")
    args_cli = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_data()
    print(f"Dataset: {len(dataset)} examples loaded.")

    results = []
    modes   = ["full", "lora"] if args_cli.mode == "both" else [args_cli.mode]

    for mode in modes:
        stats = run_experiment(mode, dataset, tokenizer)
        results.append(stats)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'Metric':<28} {'Full FT':>12} {'LoRA':>12}"
    print(hdr)
    print("-" * len(hdr))

    def row(label, key, fmt=".2f"):
        vals = {r["mode"]: r for r in results}
        full = f"{vals['full'][key]:{fmt}}" if "full" in vals else "---"
        lora = f"{vals['lora'][key]:{fmt}}" if "lora" in vals else "---"
        print(f"  {label:<26} {full:>12} {lora:>12}")

    row("Total params (M)",     "total_params_M",     ".1f")
    row("Trainable params (M)", "trainable_params_M", ".3f")
    row("Trainable (%)",        "trainable_pct",      ".2f")
    row("Training time (s)",    "train_time_s",       ".1f")
    row("Peak GPU memory (GB)", "peak_mem_gb",        ".2f")
    row("Final train loss",     "final_loss",         ".4f")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
