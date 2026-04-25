"""
Databricks Dolly-15k data loader for Supervised Fine-Tuning.

Dolly-15k contains 15,000 instruction/response pairs written by
Databricks employees. Categories include: brainstorming, classification,
closed QA, generation, information extraction, open QA, and summarization.

We format each example into a prompt template so the model learns:
  "When you see '### Instruction:', read the task.
   When you see '### Response:', generate a helpful answer."
"""

import os
import random

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def format_example(example):
    """
    Format a Dolly example into our prompt template.

    # 📚 LEARN: Why the template matters
    # A raw pretrained model just predicts the next token — it doesn't
    # know what an "instruction" is. The template creates a clear structure:
    #   - "### Instruction:" signals "here's what the user wants"
    #   - "### Context:" (optional) provides reference material
    #   - "### Response:" signals "now generate your answer"
    #
    # After seeing thousands of these, the model learns:
    #   "When I see ### Response:, I should start being helpful."
    # This is literally how ChatGPT-style models were created —
    # OpenAI did exactly this with InstructGPT.
    """
    instruction = example["instruction"].strip()
    context = example.get("context", "").strip()
    response = example["response"].strip()

    if context:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    return {"text": text}


def load_and_format_dolly(num_examples=3000, seed=42):
    """
    Load Dolly-15k, sample a subset, format, and split.

    We use 3,000 examples (not all 15k) for speed — on a 135M param
    model with 16GB RAM, this gives us meaningful learning in ~30 min.
    """
    print("Downloading Databricks Dolly-15k...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    print(f"Full dataset: {len(dataset)} examples")

    # Sample a subset for speed
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    dataset = dataset.select(indices)
    print(f"Using {len(dataset)} examples (seed={seed})")

    # Format into prompt template
    dataset = dataset.map(format_example)

    # Show example
    print(f"\n--- Example formatted prompt ---")
    print(dataset[0]["text"][:500])
    print("...")
    print(f"--- End example ---\n")

    # Train/val split: 90/10
    split = dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")

    # Save to disk for reuse
    os.makedirs(DATA_DIR, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(DATA_DIR, "train"))
    val_dataset.save_to_disk(os.path.join(DATA_DIR, "val"))
    print(f"Saved to {DATA_DIR}/train and {DATA_DIR}/val")

    # Compute some stats
    text_lengths = [len(ex["text"]) for ex in train_dataset]
    avg_len = sum(text_lengths) / len(text_lengths)
    max_len = max(text_lengths)
    print(f"\nText length stats: avg={avg_len:.0f} chars, max={max_len} chars")

    print(f"\n→ Next, we fine-tune SmolLM2-135M on these instruction/response pairs.")
    return train_dataset, val_dataset


if __name__ == "__main__":
    load_and_format_dolly()
