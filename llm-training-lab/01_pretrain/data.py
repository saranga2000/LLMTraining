"""
TinyShakespeare download + character-level tokenization.

This is the simplest possible data pipeline for language modeling:
  1. Download raw text
  2. Build a vocabulary from unique characters
  3. Encode text as integer sequences
  4. Split into train/val
  5. Save as binary tensors
"""

import os
import ssl
import urllib.request
import pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare():
    """Download TinyShakespeare (~1MB of text)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "input.txt")
    if os.path.exists(filepath):
        print(f"Already downloaded: {filepath}")
    else:
        print(f"Downloading TinyShakespeare...")
        # macOS system Python often lacks SSL root certs; use certifi bundle
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        response = urllib.request.urlopen(URL, context=ctx)
        with open(filepath, "wb") as f:
            f.write(response.read())
        print(f"Saved to {filepath}")
    return filepath


def build_vocab_and_encode(filepath):
    """
    Build a character-level vocabulary and encode the text.

    # 📚 LEARN: Character-level tokenization
    # Every unique character in the text becomes a token.
    # 'a' -> 0, 'b' -> 1, ... etc.
    # This is dead simple but means the model must learn spelling,
    # word boundaries, and grammar all from scratch.
    # Real LLMs use subword tokenizers (BPE) with ~50k-100k tokens,
    # which give the model a head start on common words.
    """
    with open(filepath, "r") as f:
        text = f.read()

    # Build vocabulary from unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Create mappings: character <-> integer
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
    itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

    # Encode the entire text as a numpy array of integers
    data = np.array([stoi[ch] for ch in text], dtype=np.uint16)

    return data, chars, stoi, itos, vocab_size


def split_and_save(data, chars, stoi, itos):
    """Split 90/10 into train/val and save as binary files."""
    n = len(data)
    split_idx = int(0.9 * n)

    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Save as binary files (fast loading during training)
    train_data.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_data.tofile(os.path.join(DATA_DIR, "val.bin"))

    # Save vocabulary metadata
    meta = {
        "vocab_size": len(chars),
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
    }
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return train_data, val_data


def main():
    filepath = download_shakespeare()
    data, chars, stoi, itos, vocab_size = build_vocab_and_encode(filepath)
    train_data, val_data = split_and_save(data, chars, stoi, itos)

    print(f"\nVocab size: {vocab_size} chars")
    print(f"Characters: {''.join(chars[:20])}... (showing first 20)")
    print(f"Total tokens: {len(data):,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Saved: {DATA_DIR}/train.bin, val.bin, meta.pkl")
    print(f"\n→ Next, we build the model that will learn to predict these tokens.")


if __name__ == "__main__":
    main()
