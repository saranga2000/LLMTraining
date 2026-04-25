"""
Supervised Fine-Tuning (SFT) of SmolLM2-135M on Dolly-15k.

# 📚 LEARN: What is SFT (Supervised Fine-Tuning)?
#
# A pretrained model (like SmolLM2-135M) knows English — it was trained
# on ~1 trillion tokens of internet text. But it doesn't know how to
# follow instructions. It just predicts the next token.
#
# SFT teaches it a new behavior: "when you see an instruction, generate
# a helpful response." We do this by showing it thousands of examples
# of (instruction, response) pairs.
#
# The training loop is IDENTICAL to Phase 1:
#   forward pass → loss → backward pass → optimizer step
# The only differences:
#   1. We start from pretrained weights (not random)
#   2. The data is structured (instruction/response, not raw text)
#   3. We only compute loss on response tokens (instruction masking)
#
# This is exactly how ChatGPT was created:
#   GPT-3 (pretrained) → InstructGPT (SFT on human demonstrations)
"""

import os
import sys

import torch
import mlflow
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sft_model")
MLRUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")

# Model
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def train():
    print("\n" + "=" * 60)
    print("Phase 2: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    # Determine device and dtype
    if torch.backends.mps.is_available():
        device = "mps"
        # MPS supports bf16 on Apple Silicon
        use_bf16 = True
        print(f"Using device: MPS (Apple Silicon) with bf16")
    else:
        device = "cpu"
        use_bf16 = False
        print(f"Using device: CPU (fp32) — this will be slow!")

    # ─── Load data ───────────────────────────────────────────
    print("\nLoading formatted Dolly data...")
    train_dataset = load_from_disk(os.path.join(DATA_DIR, "train"))
    val_dataset = load_from_disk(os.path.join(DATA_DIR, "val"))
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")

    # ─── Load model and tokenizer ────────────────────────────
    print(f"\nLoading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Ensure pad token exists (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters ({total_params/1e6:.0f}M)")
    print(f"Vocab size: {tokenizer.vocab_size:,}")

    # ─── Training config ─────────────────────────────────────

    # 📚 LEARN: SFTTrainer configuration explained
    #
    # num_train_epochs=3: Pass through the data 3 times. More passes =
    #   more learning, but risk overfitting (memorizing instead of generalizing).
    #
    # per_device_train_batch_size=4: Process 4 examples at once on the GPU.
    #   Limited by GPU memory (16GB unified on your M4).
    #
    # gradient_accumulation_steps=4: Here's the trick — we want an effective
    #   batch size of 16 (4 × 4), but can't fit 16 examples in memory at once.
    #   So we accumulate gradients over 4 mini-batches before updating weights.
    #   It's like carrying groceries: can't carry 64 bags at once, so make
    #   4 trips of 16. Same math, smaller memory footprint.
    #
    # learning_rate=2e-5: Much smaller than Phase 1's 3e-4. Why? The model
    #   already has good weights — we don't want to destroy what it learned
    #   during pretraining. Small steps = gentle adaptation.
    #
    # warmup_ratio=0.03: Warm up for 3% of training steps. Same idea as
    #   Phase 1, but shorter because the model is already stable.
    #
    # max_seq_length=512: Truncate sequences longer than 512 tokens.
    #   Longer = more context but more memory. 512 is a good balance.

    # Compute warmup steps (3% of total)
    # 📚 LEARN: Gradient accumulation — virtual batch size
    # We want effective batch size = 16, but 16GB unified memory can't
    # handle even batch=2 on long sequences (backward pass stores all
    # intermediate activations). So we use batch=1, accumulate=16:
    #   1 example × 16 accumulation steps = 16 effective batch size.
    # Slowest per step, but guaranteed to fit in memory.
    steps_per_epoch = len(train_dataset) // (1 * 16)  # batch_size * grad_accum
    total_steps = steps_per_epoch * 3
    warmup_steps = int(0.03 * total_steps)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_length=512,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,
        dataloader_pin_memory=False,  # required for MPS
        report_to="none",  # we handle MLflow manually
        remove_unused_columns=False,
    )

    # ─── MLflow setup ────────────────────────────────────────
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment("sft_dolly")

    # ─── Create trainer ──────────────────────────────────────

    # 📚 LEARN: SFTTrainer — what "supervised" means here
    # The trainer takes care of:
    #   1. Tokenizing the "text" field from our dataset
    #   2. Creating input/target pairs (shifted by one, like Phase 1)
    #   3. Running the training loop (forward, loss, backward, step)
    #   4. Evaluating on the val set periodically
    #
    # Under the hood, it's doing the same thing as our Phase 1 loop!
    # The key difference: it can optionally mask instruction tokens
    # so the model only learns to generate responses, not repeat prompts.

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # ─── Train! ──────────────────────────────────────────────
    print(f"\nStarting SFT training...")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Max seq length: 512 tokens")

    total_steps = (len(train_dataset) // (int(training_args.per_device_train_batch_size) * int(training_args.gradient_accumulation_steps))) * int(training_args.num_train_epochs)
    print(f"  Total training steps: ~{total_steps}")
    print("-" * 60)

    with mlflow.start_run(run_name="smollm2_sft_v1"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "num_train_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "max_seq_length": 512,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "total_params": total_params,
            "device": device,
        })

        # Train
        train_result = trainer.train()

        # Log final metrics
        mlflow.log_metrics({
            "train_loss": train_result.metrics["train_loss"],
            "train_runtime_seconds": train_result.metrics["train_runtime"],
        })

        # Evaluate
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss": eval_metrics["eval_loss"],
        })

        print(f"\n  Final train loss: {train_result.metrics['train_loss']:.4f}")
        print(f"  Final eval loss: {eval_metrics['eval_loss']:.4f}")
        print(f"  Training time: {train_result.metrics['train_runtime']:.0f}s")

        # ─── Save model ─────────────────────────────────────
        print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # ─── Generate samples ────────────────────────────────
        print("\n" + "=" * 60)
        print("Generating samples from fine-tuned model...")
        print("=" * 60)

        model.eval()
        if device == "mps":
            model = model.to("mps")

        test_prompts = [
            "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
            "### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n",
            "### Instruction:\nWrite a short poem about the ocean.\n\n### Response:\n",
            "### Instruction:\nWhat are the three states of matter?\n\n### Response:\n",
            "### Instruction:\nHow does photosynthesis work?\n\n### Response:\n",
        ]

        samples_text = []
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples_text.append(response)

            # Print just the response part
            print(f"\n{response}")
            print("─" * 40)

        # Save samples
        samples_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "outputs", "sft_samples.txt"
        )
        with open(samples_path, "w") as f:
            f.write("\n\n".join(samples_text))
        mlflow.log_artifact(samples_path)

        print(f"\n{'=' * 60}")
        print(f"✅ Phase 2 Complete")
        print(f"   Model: {MODEL_NAME} ({total_params/1e6:.0f}M params)")
        print(f"   Train loss: {train_result.metrics['train_loss']:.4f}")
        print(f"   Eval loss: {eval_metrics['eval_loss']:.4f}")
        print(f"   Model saved: {OUTPUT_DIR}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
