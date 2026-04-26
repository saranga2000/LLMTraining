"""
Cross-phase comparison: run all 3 models against the same prompts.

Compares:
  1. NanoGPT (Phase 1) — character-level pretrained model
  2. SmolLM2-135M SFT (Phase 2) — instruction-following model
  3. SmolLM2-135M SFT + LoRA (Phase 3) — medical domain model
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PRETRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "01_pretrain")
SFT_MODEL_DIR = os.path.join(BASE_DIR, "outputs", "sft_model")
ADAPTER_DIR = os.path.join(BASE_DIR, "outputs", "lora_adapter")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Add pretrain dir to path so we can import model
sys.path.insert(0, PRETRAIN_DIR)


def load_nanogpt():
    """Load the Phase 1 NanoGPT model."""
    from model import NanoGPT

    checkpoint_path = os.path.join(BASE_DIR, "outputs", "pretrain_best.pt")
    if not os.path.exists(checkpoint_path):
        print("  ⚠ NanoGPT checkpoint not found. Skipping Phase 1 model.")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = NanoGPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load vocabulary from meta.pkl
    import pickle
    data_dir = os.path.join(PRETRAIN_DIR, "data")
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]

    return model, {"stoi": stoi, "itos": itos, "device": device}


def generate_nanogpt(model, meta, prompt, max_new_tokens=150):
    """Generate text from NanoGPT."""
    if model is None:
        return "[NanoGPT checkpoint not found]"

    stoi, itos, device = meta["stoi"], meta["itos"], meta["device"]

    # Encode prompt (character-level)
    try:
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    except KeyError:
        # If prompt has characters not in Shakespeare vocab, use a simple prefix
        idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.7, top_k=50)

    return "".join([itos[i] for i in generated[0].tolist()])


def generate_hf(model, tokenizer, prompt, max_new_tokens=150, device="mps"):
    """Generate text from a HuggingFace model."""
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


def compare_all():
    print("\n" + "=" * 70)
    print("Cross-Phase Comparison: All 3 Models")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # ─── Load Phase 1: NanoGPT ──────────────────────────────
    print("\nLoading Phase 1: NanoGPT (character-level)...")
    nanogpt_model, nanogpt_meta = load_nanogpt()

    # ─── Load Phase 2: SFT model ───────────────────────────
    print("Loading Phase 2: SmolLM2-135M SFT...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR, dtype=torch.bfloat16
    ).to(device)

    # ─── Load Phase 3: SFT + LoRA ──────────────────────────
    print("Loading Phase 3: SmolLM2-135M SFT + LoRA (Medical)...")
    lora_base = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR, dtype=torch.bfloat16
    ).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, ADAPTER_DIR).to(device)

    # ─── Test prompts ───────────────────────────────────────
    # Mix of: general knowledge, instruction-following, and medical
    test_cases = [
        {
            "name": "General Knowledge",
            "nanogpt_prompt": "The king",
            "hf_prompt": "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        },
        {
            "name": "Explanation",
            "nanogpt_prompt": "To be or not",
            "hf_prompt": "### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n",
        },
        {
            "name": "Medical Question",
            "nanogpt_prompt": "The doctor",
            "hf_prompt": "### Medical Question:\nWhat is hypertension?\n\n### Answer:\n",
        },
        {
            "name": "Medical Question (Specific)",
            "nanogpt_prompt": "What is the",
            "hf_prompt": "### Medical Question:\nHow does insulin work in the body?\n\n### Answer:\n",
        },
        {
            "name": "Creative / Open-Ended",
            "nanogpt_prompt": "ROMEO:",
            "hf_prompt": "### Instruction:\nWrite a short poem about the ocean.\n\n### Response:\n",
        },
    ]

    comparison_text = []

    for i, test in enumerate(test_cases):
        print(f"\n{'━' * 70}")
        print(f"  [{i+1}] {test['name']}")
        print(f"{'━' * 70}")

        # Phase 1: NanoGPT (character-level, uses its own prompt)
        nano_response = generate_nanogpt(
            nanogpt_model, nanogpt_meta, test["nanogpt_prompt"], max_new_tokens=150
        )
        nano_short = nano_response[:200]

        # Phase 2: SFT
        sft_response = generate_hf(sft_model, tokenizer, test["hf_prompt"], device=device)
        sft_answer = sft_response[len(test["hf_prompt"]):][:200]

        # Phase 3: LoRA
        lora_response = generate_hf(lora_model, tokenizer, test["hf_prompt"], device=device)
        lora_answer = lora_response[len(test["hf_prompt"]):][:200]

        output = f"\n[{i+1}] {test['name']}\n"
        output += f"    Prompt (NanoGPT): \"{test['nanogpt_prompt']}\"\n"
        output += f"    Prompt (SFT/LoRA): \"{test['hf_prompt'].strip()[:60]}...\"\n"
        output += f"\n┌──────────────────┬{'─' * 50}┐\n"
        output += f"│ NanoGPT (Phase 1)│ {nano_short[:48]:48s} │\n"
        output += f"│                  │ {nano_short[48:96]:48s} │\n"
        output += f"├──────────────────┼{'─' * 50}┤\n"
        output += f"│ SFT     (Phase 2)│ {sft_answer[:48]:48s} │\n"
        output += f"│                  │ {sft_answer[48:96]:48s} │\n"
        output += f"├──────────────────┼{'─' * 50}┤\n"
        output += f"│ LoRA    (Phase 3)│ {lora_answer[:48]:48s} │\n"
        output += f"│                  │ {lora_answer[48:96]:48s} │\n"
        output += f"└──────────────────┴{'─' * 50}┘"

        comparison_text.append(output)
        print(output)

    # ─── Save results ────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "cross_phase_comparison.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Cross-Phase Comparison: All 3 Models\n")
        f.write(f"{'=' * 70}\n\n")
        f.write("Phase 1: NanoGPT (10.8M params, character-level, trained on Shakespeare)\n")
        f.write("Phase 2: SmolLM2-135M SFT (135M params, instruction-tuned on Dolly-15k)\n")
        f.write("Phase 3: SmolLM2-135M SFT + LoRA (135M + 0.9M adapter, medical domain)\n\n")
        f.write("\n".join(comparison_text))

    print(f"\n\nResults saved to {output_path}")

    # Free memory
    del sft_model, lora_model, lora_base, nanogpt_model
    if device == "mps":
        torch.mps.empty_cache()

    print(f"\n✅ Cross-Phase Comparison Complete")


if __name__ == "__main__":
    compare_all()
