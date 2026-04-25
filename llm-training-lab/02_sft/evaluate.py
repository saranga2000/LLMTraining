"""
Evaluate SFT model vs base model.

Compares:
  1. Perplexity on validation data (quantitative)
  2. Side-by-side generation on identical prompts (qualitative)

# 📚 LEARN: Perplexity
# Perplexity = e^(cross-entropy loss)
# Intuition: if perplexity = 10, the model is as confused as if it
# had to guess from 10 equally likely options at every step.
#   - Perplexity 1.0 = perfect prediction (impossible in practice)
#   - Perplexity 10 = reasonably confident
#   - Perplexity 100 = quite confused
#   - Perplexity 1000+ = basically guessing
# Lower is better. We expect the SFT model to have lower perplexity
# on instruction-formatted text because it's been trained on that format.
"""

import os
import math

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SFT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sft_model")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def compute_perplexity(model, tokenizer, texts, max_length=512, device="mps"):
    """
    Compute perplexity over a list of texts.

    Perplexity is just exp(average cross-entropy loss) over all tokens.
    It's the same loss we minimize during training, but exponentiated
    to make it more interpretable.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)

            input_ids = encodings["input_ids"]
            # For language modeling, targets = inputs shifted by 1
            # The model handles this internally
            outputs = model(**encodings, labels=input_ids)
            loss = outputs.loss

            num_tokens = input_ids.size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def generate_response(model, tokenizer, prompt, max_new_tokens=150, device="mps"):
    """Generate a response from a model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate():
    print("\n" + "=" * 60)
    print("Phase 2 Evaluation: Base Model vs SFT Model")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load val data
    val_dataset = load_from_disk(os.path.join(DATA_DIR, "val"))
    val_texts = [ex["text"] for ex in val_dataset][:100]
    print(f"Evaluating on {len(val_texts)} validation examples")

    # ─── Load base model ─────────────────────────────────────
    print(f"\nLoading base model: {BASE_MODEL_NAME}...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)

    # ─── Load SFT model ─────────────────────────────────────
    print(f"Loading SFT model: {SFT_MODEL_DIR}...")
    sft_tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR, torch_dtype=torch.bfloat16
    ).to(device)

    # ─── Perplexity comparison ───────────────────────────────
    print("\nComputing perplexity (this may take a minute)...")

    base_ppl, base_loss = compute_perplexity(
        base_model, base_tokenizer, val_texts, device=device
    )
    sft_ppl, sft_loss = compute_perplexity(
        sft_model, sft_tokenizer, val_texts, device=device
    )

    improvement = ((base_ppl - sft_ppl) / base_ppl) * 100

    print(f"\n┌────────────────────────────────────────┐")
    print(f"│  Perplexity Comparison                 │")
    print(f"├──────────────┬───────────┬─────────────┤")
    print(f"│ Model        │ Perplexity│ Avg Loss    │")
    print(f"├──────────────┼───────────┼─────────────┤")
    print(f"│ Base         │ {base_ppl:9.2f} │ {base_loss:11.4f} │")
    print(f"│ SFT          │ {sft_ppl:9.2f} │ {sft_loss:11.4f} │")
    print(f"├──────────────┼───────────┼─────────────┤")
    print(f"│ Improvement  │ {improvement:+8.1f}% │             │")
    print(f"└──────────────┴───────────┴─────────────┘")

    # ─── Side-by-side generation ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Side-by-Side Generation Comparison")
    print(f"{'=' * 60}")

    test_prompts = [
        "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "### Instruction:\nExplain gravity in one sentence.\n\n### Response:\n",
        "### Instruction:\nList three benefits of exercise.\n\n### Response:\n",
    ]

    comparison_text = []
    for prompt in test_prompts:
        base_response = generate_response(base_model, base_tokenizer, prompt, device=device)
        sft_response = generate_response(sft_model, sft_tokenizer, prompt, device=device)

        prompt_short = prompt.split("\n")[1]  # Just the instruction line

        output = f"\nPrompt: {prompt_short}\n"
        output += f"{'─' * 60}\n"
        output += f"BASE: {base_response[len(prompt):][:200]}\n"
        output += f"{'─' * 60}\n"
        output += f"SFT:  {sft_response[len(prompt):][:200]}\n"
        output += f"{'═' * 60}"

        comparison_text.append(output)
        print(output)

    # ─── Save results ────────────────────────────────────────
    eval_path = os.path.join(OUTPUT_DIR, "sft_eval.txt")
    with open(eval_path, "w") as f:
        f.write("Phase 2 Evaluation: Base Model vs SFT Model\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Base Model: {BASE_MODEL_NAME}\n")
        f.write(f"SFT Model: {SFT_MODEL_DIR}\n\n")
        f.write(f"Base Perplexity: {base_ppl:.2f} (loss: {base_loss:.4f})\n")
        f.write(f"SFT Perplexity: {sft_ppl:.2f} (loss: {sft_loss:.4f})\n")
        f.write(f"Improvement: {improvement:+.1f}%\n\n")
        f.write("\n".join(comparison_text))

    print(f"\nResults saved to {eval_path}")

    # Free memory
    del base_model
    del sft_model
    torch.mps.empty_cache() if device == "mps" else None

    print(f"\n✅ Phase 2 Evaluation Complete")
    print(f"   Base PPL: {base_ppl:.2f} | SFT PPL: {sft_ppl:.2f} (↓ {improvement:.1f}% improvement)")


if __name__ == "__main__":
    evaluate()
