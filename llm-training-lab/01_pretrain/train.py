"""
NanoGPT training loop — the core of all LLM training.

This is an explicit training loop: no HuggingFace Trainer, no magic.
Every LLM ever trained does exactly these steps:
  1. Sample a batch of (input, target) token sequences
  2. Forward pass: model predicts probability of each next token
  3. Loss: cross-entropy between predictions and actual next tokens
  4. Backward pass: compute gradients (how to nudge each weight)
  5. Optimizer step: nudge weights in the direction that reduces loss
  6. Repeat.
"""

import os
import sys
import math
import time
import pickle

import numpy as np
import torch
import mlflow

# Add parent dir so we can import model
sys.path.insert(0, os.path.dirname(__file__))
from model import NanoGPT, count_parameters

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# These are the knobs that control training. Changing them changes
# how fast the model learns, how much memory it uses, and how good
# the final model is.

batch_size = 32         # How many sequences to process in parallel per step
block_size = 256        # Context length — how many chars the model can see at once
max_iters = 3000        # Total training steps (~5-10 min on M4)
eval_interval = 300     # Evaluate on val set every N steps
eval_iters = 50         # How many batches to average for val loss estimate
learning_rate = 3e-4    # Step size for optimizer — how big each weight nudge is
warmup_iters = 100      # Gradually increase LR from 0 to target over this many steps
min_lr = 3e-5           # Floor for cosine decay (10% of max LR)
weight_decay = 0.1      # Regularization — penalizes large weights to prevent overfitting

# Model architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load the pre-tokenized binary data and vocab metadata."""
    train_data = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16)

    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    return train_data, val_data, meta


def get_batch(split_data):
    """
    Sample a random batch of (input, target) pairs.

    For language modeling, the target is simply the input shifted by one:
      input:  "To be or "
      target: "o be or n"

    Each input character's target is the next character in the sequence.
    The model learns to predict what comes next at every position.
    """
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(split_data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(split_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(it):
    """
    # 📚 LEARN: Cosine learning rate schedule with warmup
    # Why not just use a fixed learning rate?
    #
    # Warmup (first 100 steps):
    #   Start with a tiny LR and linearly increase to the target.
    #   Why? The model's initial weights are random, so early gradients
    #   are noisy. Big steps with noisy gradients = unstable training.
    #
    # Cosine decay (after warmup):
    #   Gradually decrease LR following a cosine curve.
    #   Why? Early in training, big steps help explore the loss landscape.
    #   Later, small steps help fine-tune into a good minimum.
    #   Cosine is smooth — no sudden jumps like step-decay schedules.
    #
    # The shape looks like:
    #   LR ▲
    #      │    /‾‾‾‾\
    #      │   /      ‾‾‾\
    #      │  /            ‾‾‾‾
    #      │ /
    #      └──────────────────→ steps
    #      warmup    cosine decay
    """
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate train and val loss by averaging over multiple batches."""
    model.eval()
    out = {}
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split_data)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split_name] = sum(losses) / len(losses)
    model.train()
    return out


def generate_sample(model, meta, prompt_text="", max_tokens=200, temperature=0.8):
    """Generate a text sample from the model."""
    model.eval()
    stoi = meta["stoi"]
    itos = meta["itos"]

    if prompt_text:
        context = torch.tensor(
            [stoi.get(c, 0) for c in prompt_text],
            dtype=torch.long, device=device
        ).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
    text = "".join([itos[i] for i in generated[0].tolist()])
    model.train()
    return text


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    print("\n" + "=" * 60)
    print("NanoGPT Training — Phase 1: Pre-training from Scratch")
    print("=" * 60)

    # Load data
    train_data, val_data, meta = load_data()
    vocab_size = meta["vocab_size"]
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")

    # Create model
    model = NanoGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model params: {total_params:,} ({total_params/1e6:.1f}M)")

    # 📚 LEARN: AdamW optimizer
    # Adam = Adaptive Moment Estimation. It tracks two things:
    #   - Momentum (m): a running average of past gradients. This smooths
    #     out noisy gradients — like a ball rolling downhill with inertia.
    #   - Adaptive learning rate (v): a per-parameter learning rate based on
    #     how much each parameter's gradient has varied. Parameters with
    #     consistently large gradients get smaller steps (and vice versa).
    #
    # AdamW adds weight decay: it slightly shrinks all weights toward zero
    # each step. This prevents any single weight from getting too large,
    # acting as regularization (like a complexity penalty).
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # MLflow tracking
    mlflow.set_tracking_uri(os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns"))
    mlflow.set_experiment("pretrain_shakespeare")

    best_val_loss = float("inf")
    start_time = time.time()

    with mlflow.start_run(run_name="nanogpt_v1"):
        # Log hyperparameters
        mlflow.log_params({
            "batch_size": batch_size,
            "block_size": block_size,
            "max_iters": max_iters,
            "learning_rate": learning_rate,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "total_params": total_params,
            "device": str(device),
        })

        print(f"\nStarting training for {max_iters} iterations...")
        print("-" * 60)

        # 📉 TRAINING PROGRESS explanation (printed once):
        print(f"\n📉 What the loss numbers mean:")
        print(f"   Random guessing: loss ≈ {math.log(vocab_size):.2f} "
              f"(ln({vocab_size}) — uniform probability over all chars)")
        print(f"   loss < 2.0 = learning letter frequencies and common words")
        print(f"   loss < 1.5 = model has learned something real")
        print(f"   loss < 1.2 = starting to sound fluent")
        print()

        for iter_num in range(max_iters):
            # Set learning rate for this step
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Sample a batch
            x, y = get_batch(train_data)

            # Forward pass: predict next tokens, compute loss
            logits, loss = model(x, y)

            # Backward pass: compute gradients
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Optimizer step: update weights
            optimizer.step()

            # Progress logging every 100 steps
            if iter_num % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  iter {iter_num:4d} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | time {elapsed:.1f}s")

            # Evaluation every eval_interval steps
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                losses = estimate_loss(model, train_data, val_data)
                train_loss = losses["train"]
                val_loss = losses["val"]

                print(f"\n  ▸ Eval @ iter {iter_num}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Log to MLflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": lr,
                }, step=iter_num)

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(OUTPUT_DIR, "pretrain_best.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": {
                            "vocab_size": vocab_size,
                            "block_size": block_size,
                            "n_layer": n_layer,
                            "n_head": n_head,
                            "n_embd": n_embd,
                            "dropout": dropout,
                        },
                    }, checkpoint_path)
                    print(f"  ✓ New best val_loss={val_loss:.4f} — saved checkpoint")

                print()

        # Training complete
        elapsed = time.time() - start_time
        mlflow.log_metric("training_time_seconds", elapsed)
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Generate samples
        print("\n" + "=" * 60)
        print("Generating samples from trained model...")
        print("=" * 60)

        prompts = [
            "To be or not to be",
            "ROMEO:",
            "What is",
        ]

        samples_text = []
        for prompt in prompts:
            sample = generate_sample(model, meta, prompt_text=prompt, max_tokens=200)
            samples_text.append(f"Prompt: '{prompt}'\n{sample}\n{'─' * 40}")
            print(f"\nPrompt: '{prompt}'")
            print(sample)
            print("─" * 40)

        # Save samples as MLflow artifact
        samples_path = os.path.join(OUTPUT_DIR, "pretrain_samples.txt")
        with open(samples_path, "w") as f:
            f.write("\n\n".join(samples_text))
        mlflow.log_artifact(samples_path)

        print(f"\n{'=' * 60}")
        print(f"✅ Phase 1 Complete")
        print(f"   Model params: {total_params/1e6:.1f}M")
        print(f"   Best val loss: {best_val_loss:.4f} (iter best)")
        print(f"   Training time: {elapsed:.0f}s")
        print(f"   Checkpoint: {OUTPUT_DIR}/pretrain_best.pt")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
