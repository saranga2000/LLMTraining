"""
MedAlpaca Medical Flashcards data loader for LoRA fine-tuning.

MedAlpaca contains ~3,500 medical question/answer pairs from
medical flashcards — covering anatomy, pharmacology, pathology,
and clinical medicine.

We format each example into the same prompt template structure
so the model can leverage what it learned during SFT (Phase 2).
"""

import os

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def format_medical_example(example):
    """
    Format a MedAlpaca example into our medical prompt template.

    # 📚 LEARN: Why use a different template than Phase 2?
    # We use "### Medical Question:" instead of "### Instruction:"
    # to create a clear domain signal. The model learns:
    #   "When I see ### Medical Question:, I should respond with
    #    medical knowledge, not general knowledge."
    # This is a subtle but effective way to activate domain-specific
    # behavior in the model.
    """
    question = example["input"].strip()
    answer = example["output"].strip()

    text = (
        f"### Medical Question:\n{question}\n\n"
        f"### Answer:\n{answer}"
    )

    return {"text": text}


def load_and_format_medalpaca(num_examples=3500, seed=42):
    """
    Load MedAlpaca flashcards, sample a subset, format, and split.

    The full dataset has ~34k examples but the CLAUDE.md targets ~3,500
    for a manageable training time on Apple Silicon.
    """
    import random

    print("Downloading MedAlpaca Medical Flashcards...")
    dataset = load_dataset(
        "medalpaca/medical_meadow_medical_flashcards", split="train"
    )
    print(f"Full dataset: {len(dataset)} examples")

    # Subsample for tractable training time
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    dataset = dataset.select(indices)
    print(f"Using {len(dataset)} examples (seed={seed})")

    # Format into prompt template
    dataset = dataset.map(format_medical_example)

    # Show example
    print(f"\n--- Example formatted prompt ---")
    print(dataset[0]["text"][:500])
    print("...")
    print(f"--- End example ---\n")

    # Train/val split: 90/10
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")

    # Save to disk
    os.makedirs(DATA_DIR, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(DATA_DIR, "train"))
    val_dataset.save_to_disk(os.path.join(DATA_DIR, "val"))
    print(f"Saved to {DATA_DIR}/train and {DATA_DIR}/val")

    # Stats
    text_lengths = [len(ex["text"]) for ex in train_dataset]
    avg_len = sum(text_lengths) / len(text_lengths)
    max_len = max(text_lengths)
    print(f"\nText length stats: avg={avg_len:.0f} chars, max={max_len} chars")

    print(f"\n→ Next, we apply LoRA to the SFT model and train on medical data.")
    return train_dataset, val_dataset


if __name__ == "__main__":
    load_and_format_medalpaca()
