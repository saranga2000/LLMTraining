"""
Generate text from a trained NanoGPT checkpoint.

Usage:
    python 01_pretrain/generate.py --prompt "To be or not to be"
    python 01_pretrain/generate.py --prompt "ROMEO:" --temperature 0.5
"""

import os
import sys
import argparse
import pickle

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import NanoGPT

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def main():
    parser = argparse.ArgumentParser(description="Generate text from NanoGPT")
    parser.add_argument("--prompt", type=str, default="", help="Starting text prompt")
    parser.add_argument("--max_tokens", type=int, default=500, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.1=conservative, 1.0=creative)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Only sample from top-k most likely tokens")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(OUTPUT_DIR, "pretrain_best.pt"),
                        help="Path to checkpoint file")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Load vocab
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]

    # Rebuild model from config
    model = NanoGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded (val_loss={checkpoint['best_val_loss']:.4f} at iter {checkpoint['iter_num']})")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print("─" * 60)

    # Encode prompt
    if args.prompt:
        context = torch.tensor(
            [stoi.get(c, 0) for c in args.prompt],
            dtype=torch.long, device=device
        ).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = "".join([itos[i] for i in generated[0].tolist()])
    print(text)
    print("─" * 60)


if __name__ == "__main__":
    main()
