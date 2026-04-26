"""
Evaluate LoRA adapter vs SFT base on medical questions.

Compares:
  1. Perplexity on medical validation data
  2. Side-by-side generation on 20 medical questions
"""

import os
import math

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SFT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sft_model")
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "lora_adapter")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def compute_perplexity(model, tokenizer, texts, max_length=256, device="mps"):
    """Compute perplexity over a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            input_ids = encodings["input_ids"]
            outputs = model(**encodings, labels=input_ids)
            num_tokens = input_ids.size(1)
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


def generate_response(model, tokenizer, prompt, max_new_tokens=150, device="mps"):
    """Generate a response from a model."""
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
    print("Phase 3 Evaluation: SFT Base vs SFT + LoRA (Medical)")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load val data
    val_dataset = load_from_disk(os.path.join(DATA_DIR, "val"))
    val_texts = [ex["text"] for ex in val_dataset]
    print(f"Evaluating on {len(val_texts)} medical validation examples")

    # ─── Load SFT base model ────────────────────────────────
    print(f"\nLoading SFT base model...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR, dtype=torch.bfloat16
    ).to(device)

    # ─── Load SFT + LoRA model ──────────────────────────────
    print(f"Loading SFT + LoRA adapter...")
    lora_base = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR, dtype=torch.bfloat16
    ).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, ADAPTER_DIR).to(device)

    # ─── Perplexity comparison ───────────────────────────────
    print("\nComputing perplexity on medical data...")

    sft_ppl, sft_loss = compute_perplexity(sft_model, tokenizer, val_texts, device=device)
    lora_ppl, lora_loss = compute_perplexity(lora_model, tokenizer, val_texts, device=device)

    improvement = ((sft_ppl - lora_ppl) / sft_ppl) * 100

    print(f"\n┌──────────────────────────────────────────────┐")
    print(f"│  Medical Perplexity Comparison               │")
    print(f"├──────────────┬───────────┬───────────────────┤")
    print(f"│ Model        │ Perplexity│ Avg Loss          │")
    print(f"├──────────────┼───────────┼───────────────────┤")
    print(f"│ SFT (base)   │ {sft_ppl:9.2f} │ {sft_loss:17.4f} │")
    print(f"│ SFT + LoRA   │ {lora_ppl:9.2f} │ {lora_loss:17.4f} │")
    print(f"├──────────────┼───────────┼───────────────────┤")
    print(f"│ Improvement  │ {improvement:+8.1f}% │                   │")
    print(f"└──────────────┴───────────┴───────────────────┘")

    # ─── Side-by-side generation ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Side-by-Side Medical Generation (20 questions)")
    print(f"{'=' * 60}")

    medical_questions = [
        "What is hypertension?",
        "What are the symptoms of diabetes?",
        "What is the function of the liver?",
        "How does insulin work?",
        "What is anemia?",
        "What causes a heart attack?",
        "What is the difference between a virus and a bacteria?",
        "What are antibiotics used for?",
        "What is inflammation?",
        "How does the immune system work?",
        "What is cholesterol?",
        "What are the symptoms of pneumonia?",
        "What is a stroke?",
        "What causes allergies?",
        "What is the function of the kidneys?",
        "What is an MRI scan?",
        "What is osteoporosis?",
        "How does blood pressure work?",
        "What is an autoimmune disease?",
        "What causes headaches?",
    ]

    comparison_text = []
    for i, question in enumerate(medical_questions):
        prompt = f"### Medical Question:\n{question}\n\n### Answer:\n"

        sft_response = generate_response(sft_model, tokenizer, prompt, device=device)
        lora_response = generate_response(lora_model, tokenizer, prompt, device=device)

        # Extract just the answer part
        sft_answer = sft_response[len(prompt):][:200]
        lora_answer = lora_response[len(prompt):][:200]

        output = f"\n[{i+1}] {question}\n"
        output += f"{'─' * 60}\n"
        output += f"SFT:  {sft_answer}\n"
        output += f"{'─' * 60}\n"
        output += f"LoRA: {lora_answer}\n"
        output += f"{'═' * 60}"

        comparison_text.append(output)
        print(output)

    # ─── Save results ────────────────────────────────────────
    eval_path = os.path.join(OUTPUT_DIR, "lora_eval.txt")
    with open(eval_path, "w") as f:
        f.write("Phase 3 Evaluation: SFT Base vs SFT + LoRA (Medical)\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"SFT Perplexity: {sft_ppl:.2f} (loss: {sft_loss:.4f})\n")
        f.write(f"LoRA Perplexity: {lora_ppl:.2f} (loss: {lora_loss:.4f})\n")
        f.write(f"Improvement: {improvement:+.1f}%\n\n")
        f.write("\n".join(comparison_text))

    print(f"\nResults saved to {eval_path}")

    # Free memory
    del sft_model, lora_model, lora_base
    if device == "mps":
        torch.mps.empty_cache()

    print(f"\n✅ Phase 3 Evaluation Complete")
    print(f"   SFT PPL on medical: {sft_ppl:.2f} | LoRA PPL: {lora_ppl:.2f} (↓ {improvement:.1f}% improvement)")


if __name__ == "__main__":
    evaluate()
