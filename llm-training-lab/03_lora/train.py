"""
LoRA / PEFT Fine-Tuning on Medical Data.

# 📚 LEARN: What is LoRA (Low-Rank Adaptation)?
#
# Full fine-tuning updates ALL weights: W_new = W_old + ΔW
# where ΔW is a full-rank matrix (same size as W_old).
#
# LoRA's insight: ΔW doesn't need to be full-rank. In practice,
# the weight updates during fine-tuning have LOW RANK — meaning
# they can be approximated as: ΔW = A × B
# where A is (d × r) and B is (r × d), with r << d.
#
# Example with our model:
#   W_old is 384 × 384 = 147,456 parameters
#   Full fine-tuning: ΔW is also 384 × 384 = 147,456 parameters
#   LoRA with r=16: A is 384 × 16 = 6,144 params
#                    B is 16 × 384 = 6,144 params
#                    Total: 12,288 params (12x fewer!)
#
# Analogy: It's like compressing a photo. Full fine-tuning is the
# uncompressed TIFF. LoRA is the JPEG — r is like the quality level.
# r=1 is very compressed (loses info), r=64 is barely compressed.
# r=16 is the sweet spot for most tasks.
#
# WHY IT WORKS:
# The pretrained model already has a rich representation of language.
# Fine-tuning only needs to make small, structured adjustments —
# not rewrite everything. These adjustments naturally live in a
# low-dimensional subspace, which LoRA exploits.
"""

import os
import sys

import torch
import mlflow
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SFT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sft_model")
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "lora_adapter")
MLRUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns")


def train():
    print("\n" + "=" * 60)
    print("Phase 3: LoRA / PEFT Fine-Tuning on Medical Data")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    use_bf16 = device == "mps"
    print(f"Device: {device}")

    # ─── Load data ───────────────────────────────────────────
    print("\nLoading formatted MedAlpaca data...")
    train_dataset = load_from_disk(os.path.join(DATA_DIR, "train"))
    val_dataset = load_from_disk(os.path.join(DATA_DIR, "val"))
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")

    # ─── Load SFT model from Phase 2 ────────────────────────
    print(f"\nLoading SFT model from {SFT_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Base model: {total_params_before:,} parameters ({total_params_before/1e6:.0f}M)")

    # ─── Apply LoRA ──────────────────────────────────────────

    # 📚 LEARN: LoRA configuration explained
    #
    # r=16 (rank):
    #   The "compression level." Each adapter matrix is decomposed
    #   into two small matrices of rank 16. Higher r = more capacity
    #   but more parameters. Common values: 4, 8, 16, 32, 64.
    #   r=16 is a good balance for domain adaptation tasks.
    #
    # lora_alpha=32 (scaling factor):
    #   The adapter output is scaled by alpha/r = 32/16 = 2.
    #   This means the adapter's contribution is amplified 2x.
    #   Think of it as a volume knob for the adapter.
    #   Higher alpha = adapter has more influence on the output.
    #   Rule of thumb: alpha = 2 * r is a good starting point.
    #
    # target_modules=["q_proj", "v_proj"]:
    #   We only add LoRA adapters to the Query and Value projections
    #   in the attention layers. Why not all layers?
    #   - Q and V are the most impactful for changing model behavior
    #   - K (key) and O (output) projections add params with less benefit
    #   - MLP layers are less important for behavioral adaptation
    #   This is an empirical finding from the original LoRA paper.
    #
    # lora_dropout=0.05:
    #   Small dropout on the adapter weights to prevent overfitting.
    #   With only ~3,500 training examples, some regularization helps.

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # 📚 LEARN: get_peft_model — "freezing" weights
    # This function:
    #   1. Freezes ALL original model parameters (requires_grad=False)
    #   2. Injects small LoRA adapter matrices next to q_proj and v_proj
    #   3. Only the adapter matrices are trainable
    #
    # The result: during forward pass, the output is:
    #   output = W_frozen(x) + (alpha/r) * B(A(x))
    # where W_frozen is the original (frozen) weight, and A, B are
    # the small LoRA matrices that we train.
    model = get_peft_model(model, lora_config)

    # Print trainable vs total params — this is educational!
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    efficiency = trainable_params / total_params * 100

    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Efficiency:       {efficiency:.2f}% of model is being trained")
    print(f"  Frozen params:    {total_params - trainable_params:,} (untouched)")

    # ─── Training config ─────────────────────────────────────
    steps_per_epoch = len(train_dataset) // (1 * 8)
    total_steps = steps_per_epoch * 5
    warmup_steps = 50

    training_args = SFTConfig(
        output_dir=ADAPTER_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,       # LoRA can use higher LR — we're only training
                                  # small adapters, not the full model
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_length=256,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        per_device_eval_batch_size=1,
        bf16=use_bf16,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        remove_unused_columns=False,
    )

    # ─── MLflow setup ────────────────────────────────────────
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment("lora_medical")

    # ─── Create trainer ──────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # ─── Train! ──────────────────────────────────────────────
    print(f"\nStarting LoRA training...")
    print(f"  Epochs: 5")
    print(f"  Batch size: 1 (grad accum: 8, effective: 8)")
    print(f"  Learning rate: 3e-4 (10x higher than SFT — safe because few params)")
    print(f"  Total training steps: ~{total_steps}")
    print("-" * 60)

    with mlflow.start_run(run_name="smollm2_lora_v1"):
        mlflow.log_params({
            "base_model": "SFT from Phase 2",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_target_modules": "q_proj,v_proj",
            "lora_dropout": 0.05,
            "num_train_epochs": 5,
            "learning_rate": 3e-4,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_efficiency_pct": round(efficiency, 2),
            "device": device,
        })

        # Train
        train_result = trainer.train()

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_result.metrics["train_loss"],
            "train_runtime_seconds": train_result.metrics["train_runtime"],
            "trainable_params": trainable_params,
            "total_params": total_params,
            "param_efficiency_pct": efficiency,
        })

        # Evaluate
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss": eval_metrics["eval_loss"],
        })

        print(f"\n  Final train loss: {train_result.metrics['train_loss']:.4f}")
        print(f"  Final eval loss: {eval_metrics['eval_loss']:.4f}")
        print(f"  Training time: {train_result.metrics['train_runtime']:.0f}s")

        # ─── Save LoRA adapter (NOT full model) ─────────────
        print(f"\nSaving LoRA adapter to {ADAPTER_DIR}...")
        model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)

        # Log adapter size
        adapter_size_bytes = sum(
            os.path.getsize(os.path.join(ADAPTER_DIR, f))
            for f in os.listdir(ADAPTER_DIR)
            if f.endswith(('.safetensors', '.bin'))
        )
        adapter_size_mb = adapter_size_bytes / (1024 * 1024)
        mlflow.log_metric("adapter_size_mb", adapter_size_mb)

        print(f"  Adapter size: {adapter_size_mb:.1f} MB")
        print(f"  (Full model would be ~{total_params_before * 2 / 1024 / 1024:.0f} MB in bf16)")

        # ─── Generate samples ────────────────────────────────
        print("\n" + "=" * 60)
        print("Generating medical samples from LoRA model...")
        print("=" * 60)

        model.eval()

        test_prompts = [
            "### Medical Question:\nWhat is hypertension?\n\n### Answer:\n",
            "### Medical Question:\nWhat are the symptoms of diabetes?\n\n### Answer:\n",
            "### Medical Question:\nWhat is the function of the liver?\n\n### Answer:\n",
            "### Medical Question:\nWhat is an electrocardiogram (ECG)?\n\n### Answer:\n",
            "### Medical Question:\nWhat causes anemia?\n\n### Answer:\n",
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
            print(f"\n{response}")
            print("─" * 40)

        # Save samples
        samples_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "outputs", "lora_samples.txt"
        )
        with open(samples_path, "w") as f:
            f.write("\n\n".join(samples_text))
        mlflow.log_artifact(samples_path)

        print(f"\n{'=' * 60}")
        print(f"✅ Phase 3 Complete")
        print(f"   Trainable params: {trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M total ({efficiency:.1f}% of model)")
        print(f"   Adapter size: {adapter_size_mb:.1f} MB")
        print(f"   Train loss: {train_result.metrics['train_loss']:.4f}")
        print(f"   Eval loss: {eval_metrics['eval_loss']:.4f}")
        print(f"   Adapter saved: {ADAPTER_DIR}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
