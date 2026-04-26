# LLM Training Lab

**Build, train, fine-tune, and evaluate a language model pipeline from scratch — entirely on a MacBook.**

This repo is a hands-on, educational implementation of the complete LLM training pipeline: pre-training, supervised fine-tuning (SFT), and LoRA domain adaptation. Every step is designed to be understood, not just executed.

Built on Apple Silicon (M4, 16GB) using PyTorch's MPS backend. No cloud GPU required.

---

## Who This Is For

- **Product Managers** who want to deeply understand what "training an LLM" actually means — not from slides, but from running code
- **Engineers** new to ML who want to see the full pipeline end-to-end at a tractable scale
- **Anyone** curious about how ChatGPT, Claude, and other LLMs are built

You don't need ML experience. You do need comfort with the command line and Python.

---

## Two Ways to Learn

This repo supports two learning paths:

### Path A: With Claude Code (Recommended)
If you have a Claude subscription and [Claude Code](https://claude.ai/claude-code) installed:

1. Clone this repo and open it in VS Code
2. Open the Claude Code terminal
3. Say: **"Read CLAUDE.md and execute it phase by phase"**

Claude will walk you through every step interactively — explaining what's about to happen before each command, interpreting the output after, answering your questions in real-time, and adapting to your pace. The `CLAUDE.md` file contains a complete teaching plan that tells Claude how to guide you.

This is the richest learning experience. You can ask "why?" at any point, request deeper explanations, or skip ahead.

### Path B: Self-Guided (No AI Assistant Required)
If you don't have Claude Code, or prefer to work independently:

1. Clone this repo
2. Follow **[WORKBOOK.md](llm-training-lab/WORKBOOK.md)** — a step-by-step, self-paced workbook
3. Use **[LEARNING_GUIDE.md](llm-training-lab/LEARNING_GUIDE.md)** as a reference for deep-dive explanations
4. The code is heavily commented with `# LEARN:` markers at every important concept

The workbook tells you what to read, what to run, what the output means, and what to understand at each checkpoint. It's the same journey, just written down instead of interactive.

### Supporting Materials (Both Paths)

| Document | What It Contains |
|----------|-----------------|
| [WORKBOOK.md](llm-training-lab/WORKBOOK.md) | Step-by-step self-guided walkthrough of all phases |
| [LEARNING_GUIDE.md](llm-training-lab/LEARNING_GUIDE.md) | Deep-dive explanations of every concept (attention, LoRA, training dynamics) |
| [MLFLOW_GUIDE.md](llm-training-lab/MLFLOW_GUIDE.md) | How to launch and interpret the experiment tracking UI |
| [TRANSFORMER_WALKTHROUGH.md](llm-training-lab/01_pretrain/TRANSFORMER_WALKTHROUGH.md) | Line-by-line explanation of the transformer code |
| [CLAUDE.md](CLAUDE.md) | Teaching plan for Claude Code (Path A) |

---

## What You'll Build

| Phase | What You Train | What You Learn |
|-------|---------------|----------------|
| **Phase 1** | A transformer (NanoGPT) from scratch on Shakespeare | How attention, embeddings, and the training loop actually work |
| **Phase 2** | SmolLM2-135M fine-tuned on instruction data | How raw language models become instruction-following assistants |
| **Phase 3** | LoRA adapter for medical domain | How to add domain expertise by training only 0.68% of parameters |
| **Phase 4** | Cross-phase evaluation | How to measure and compare model quality |

By the end, you'll have trained three models, compared them side-by-side, and understood every step that took them from random weights to a domain-specialized assistant.

---

## The Pipeline at a Glance

```
Phase 1: PRE-TRAINING                    Phase 2: SFT                          Phase 3: LoRA
━━━━━━━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━                          ━━━━━━━━━━━━━━
Random weights                           Pretrained SmolLM2-135M               SFT model from Phase 2
     |                                        |                                     |
     v                                        v                                     v
Train on Shakespeare                     Train on Dolly-15k instructions       Train on medical flashcards
(character-level, 1M tokens)             (3,000 instruction/response pairs)    (3,500 medical Q&A pairs)
     |                                        |                                     |
     v                                        v                                     v
Model learns ENGLISH                     Model learns to FOLLOW                Model learns MEDICINE
(loss: 4.2 → 1.5)                       INSTRUCTIONS (loss: 2.7 → 2.3)       (loss: 1.8 → 1.3)

10.8M params trained                     135M params trained                   921K params trained (0.68%!)
131 MB checkpoint                        270 MB model                          3.5 MB adapter
```

This is the same pipeline used by OpenAI, Anthropic, Google, and Meta — just at a scale that fits on a laptop.

---

## Results

### Phase 1: Pre-Training (NanoGPT)

Trained a 10.8M-parameter transformer from random weights on Shakespeare. After 3,000 iterations:

```
Input:  "To be or not to be"
Output: "To be or not to be the duke of my lord,
         That shall be so much of the world with thee
         As the world is the name of the contrary..."
```

Not Shakespeare, but grammatically valid English learned purely from statistics — no grammar rules, no dictionaries.

### Phase 2: SFT (SmolLM2-135M)

| Prompt | Base Model (no SFT) | After SFT |
|--------|-------------------|-----------|
| "Explain gravity" | Repeats the prompt in variations | "Gravity is the force that keeps everything in place on Earth" |
| "Capital of France?" | Rambles, eventually gets there | "The capital of France is Paris." |

**Perplexity:** 11.38 → 9.18 (19.4% improvement)

### Phase 3: LoRA (Medical Domain)

| Question | SFT Model | SFT + LoRA |
|----------|-----------|------------|
| Function of the liver? | Generic anatomy description | "Synthesizing bile for fat breakdown" |
| How does insulin work? | "Regulates blood sugar" | "Binds to insulin receptor on cells" |
| What is a stroke? | "A sudden brain attack" | "Blood flow to part of the brain is disrupted, leading to death of brain cells" |

**Perplexity on medical data:** 6.72 → 3.80 (43.5% improvement)

Only **3.5 MB** of adapter weights added this medical expertise to a 270 MB model.

---

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~5 GB free disk space (for models, data, and checkpoints)
- ~2 hours for all training phases

### Setup

```bash
git clone https://github.com/saranga2000/LLMTraining.git
cd LLMTraining

# Create virtual environment and install dependencies
bash setup.sh

# Activate the environment
source llm-training-lab/venv/bin/activate
```

### Run the Full Pipeline

```bash
cd llm-training-lab

# Prevent Mac from sleeping during training
caffeinate -i -s &

# Set environment variables for MPS stability
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# ── Phase 1: Pre-train NanoGPT on Shakespeare ──────────────
python 01_pretrain/data.py          # Download + tokenize Shakespeare
python 01_pretrain/train.py         # Train ~3000 iterations (~10 min)
python 01_pretrain/generate.py --prompt "To be or not to be"

# ── Phase 2: SFT on Dolly-15k ──────────────────────────────
python 02_sft/data.py               # Download + format Dolly-15k
python 02_sft/train.py              # Fine-tune SmolLM2-135M (~40 min)
python 02_sft/evaluate.py           # Compare base vs SFT

# ── Phase 3: LoRA on Medical Data ──────────────────────────
python 03_lora/data.py              # Download + format MedAlpaca
python 03_lora/train.py             # LoRA fine-tuning (~80 min)
python 03_lora/evaluate.py          # Compare SFT vs SFT+LoRA

# ── Phase 4: Cross-Phase Comparison ────────────────────────
python eval/compare_all.py          # Side-by-side all 3 models
python eval/plot_curves.py          # Loss curve visualization

# ── MLflow UI (experiment tracking) ────────────────────────
mlflow server --backend-store-uri ./mlruns \
  --host 127.0.0.1 --port 5000 \
  --disable-security-middleware &
# Open http://127.0.0.1:5000 in your browser
```

---

## Project Structure

```
LLM101-TrainFineTuneEvals-Learn/
├── README.md                          ← You are here
├── CLAUDE.md                          ← Project plan and build instructions
├── setup.sh                           ← One-shot environment setup
│
└── llm-training-lab/
    ├── LEARNING_GUIDE.md              ← Deep-dive explanations of every concept
    ├── MLFLOW_GUIDE.md                ← How to use the MLflow experiment tracker
    │
    ├── 01_pretrain/                   ← Phase 1: Pre-training from scratch
    │   ├── data.py                    │  Download TinyShakespeare, char-level tokenize
    │   ├── model.py                   │  NanoGPT transformer (~150 lines of PyTorch)
    │   ├── train.py                   │  Explicit training loop with MLflow logging
    │   ├── generate.py                │  Interactive text generation from checkpoint
    │   └── TRANSFORMER_WALKTHROUGH.md │  Line-by-line explanation of the transformer
    │
    ├── 02_sft/                        ← Phase 2: Supervised Fine-Tuning
    │   ├── data.py                    │  Dolly-15k download, prompt template formatting
    │   ├── train.py                   │  SFT with HuggingFace SFTTrainer + MLflow
    │   └── evaluate.py                │  Base vs SFT perplexity and generation comparison
    │
    ├── 03_lora/                       ← Phase 3: LoRA / PEFT
    │   ├── data.py                    │  MedAlpaca flashcards, medical Q&A formatting
    │   ├── train.py                   │  LoRA config + PEFT training + MLflow
    │   └── evaluate.py                │  SFT vs SFT+LoRA medical evaluation
    │
    ├── eval/                          ← Phase 4: Cross-phase evaluation
    │   ├── compare_all.py             │  Side-by-side generation across all 3 models
    │   └── plot_curves.py             │  Loss curve plots from MLflow data
    │
    ├── mlruns/                        ← MLflow experiment data (auto-generated)
    └── outputs/                       ← Checkpoints, models, adapters, eval results
        ├── pretrain_best.pt           │  Phase 1: Best NanoGPT checkpoint (131 MB)
        ├── sft_model/                 │  Phase 2: Fine-tuned SmolLM2-135M (270 MB)
        ├── lora_adapter/              │  Phase 3: LoRA adapter only (3.5 MB!)
        ├── sft_eval.txt               │  Phase 2 evaluation results
        ├── lora_eval.txt              │  Phase 3 evaluation results
        └── loss_curves.png            │  Visualization of all training curves
```

---

## Key Concepts Explained

### The Training Loop (The Most Important Thing to Understand)

Every LLM ever trained does exactly this:

```
1. Sample a batch of text sequences
2. Forward pass: model predicts probability of each next token
3. Loss: cross-entropy between predictions and actual next tokens
4. Backward pass: compute gradients (how to adjust each weight)
5. Optimizer step: adjust weights to reduce loss
6. Repeat thousands of times.
```

That's it. GPT-4, Claude, Llama — they all do this. The only differences are the scale of data, model size, and some training tricks.

### What is a Transformer?

A transformer is a stack of repeating blocks. Each block has two parts:

1. **Self-Attention:** Each token looks at all previous tokens and decides which are relevant. Implemented as a "fuzzy dictionary lookup" with Query, Key, and Value matrices.

2. **Feed-Forward Network (MLP):** Processes the attention output through a neural network. This is where the model "thinks" about what it gathered.

Between these, **residual connections** (shortcuts around each block) and **layer normalization** (recalibrating values) keep training stable.

See [TRANSFORMER_WALKTHROUGH.md](llm-training-lab/01_pretrain/TRANSFORMER_WALKTHROUGH.md) for a line-by-line explanation with code.

### Pre-Training vs Fine-Tuning vs LoRA

| Aspect | Pre-Training | SFT | LoRA |
|--------|-------------|-----|------|
| **Starting point** | Random weights | Pretrained model | SFT model |
| **Data** | Massive raw text (books, web) | Instruction/response pairs | Domain-specific Q&A |
| **Parameters trained** | All | All | 0.5-2% (adapter only) |
| **Learning rate** | High (3e-4) | Low (2e-5) | Medium (3e-4) |
| **Purpose** | Learn language | Learn behavior | Learn domain |
| **Analogy** | Learning to read | Learning to be a teacher | Specializing in medicine |
| **Output size** | Full model | Full model | Tiny adapter |

### What is Perplexity?

Perplexity = e^(loss). If perplexity = 10, the model is as confused as if it had to randomly guess from 10 equally likely options at every token.

- Perplexity 1.0 = perfect (impossible)
- Perplexity 3-5 = very confident, specialized
- Perplexity 10 = reasonable
- Perplexity 100+ = struggling

Lower is better.

### What is LoRA?

**Low-Rank Adaptation.** Instead of updating all 135M parameters:

1. Freeze the entire model
2. Add tiny adapter matrices (rank 16) next to attention layers
3. Train only the adapters (~921K params = 0.68%)

The adapter is 3.5 MB vs 270 MB for the full model. You can have dozens of domain adapters (medical, legal, coding, finance) and swap them at inference time without reloading the base model.

---

## Hardware Requirements and Performance

| Phase | Training Time (M4) | Peak Memory | Output Size |
|-------|-------------------|-------------|-------------|
| Phase 1: Pre-train | ~10 minutes | ~2 GB | 131 MB checkpoint |
| Phase 2: SFT | ~40 minutes | ~12 GB | 270 MB model |
| Phase 3: LoRA | ~80 minutes | ~10 GB | 3.5 MB adapter |
| Phase 2 Eval | ~5 minutes | ~8 GB | Text file |
| Phase 3 Eval | ~10 minutes | ~10 GB | Text file |

**Total:** ~2.5 hours of training, ~16 GB disk space

**Memory tips:**
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for flexible MPS memory allocation
- Use `caffeinate -i -s` to prevent Mac sleep during training
- Set `PYTHONUNBUFFERED=1` for real-time training output

---

## MLflow Experiment Tracking

All training runs are logged to MLflow with hyperparameters, metrics, and artifacts.

```bash
# Launch (use 127.0.0.1, not localhost, to avoid 403 errors)
mlflow server --backend-store-uri ./mlruns \
  --host 127.0.0.1 --port 5000 \
  --disable-security-middleware &

open http://127.0.0.1:5000
```

Three experiments are logged:
- `pretrain_shakespeare` — Phase 1 loss curves and hyperparameters
- `sft_dolly` — Phase 2 metrics and generated samples
- `lora_medical` — Phase 3 adapter efficiency stats and samples

See [MLFLOW_GUIDE.md](llm-training-lab/MLFLOW_GUIDE.md) for a detailed walkthrough of what to look for in the UI.

---

## Stack

| Library | Purpose |
|---------|---------|
| `torch` | Deep learning framework (MPS backend for Apple Silicon) |
| `transformers` | HuggingFace model loading and tokenizers |
| `datasets` | HuggingFace dataset loading |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA implementation) |
| `trl` | Training utilities (SFTTrainer) |
| `mlflow` | Experiment tracking and visualization |
| `matplotlib` | Loss curve plotting |
| `accelerate` | Hardware abstraction for training |
| `tokenizers` | Fast tokenization |
| `sentencepiece` | Subword tokenization support |

---

## Further Reading

### Documentation in This Repo
- [LEARNING_GUIDE.md](llm-training-lab/LEARNING_GUIDE.md) — Deep-dive into every concept (attention, LoRA math, training dynamics)
- [MLFLOW_GUIDE.md](llm-training-lab/MLFLOW_GUIDE.md) — How to use the experiment tracking UI
- [TRANSFORMER_WALKTHROUGH.md](llm-training-lab/01_pretrain/TRANSFORMER_WALKTHROUGH.md) — Line-by-line explanation of the NanoGPT transformer code

### External Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original transformer paper (2017)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — The LoRA paper (2021)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — Inspiration for Phase 1
- [HuggingFace PEFT documentation](https://huggingface.co/docs/peft) — LoRA implementation details
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual guide to transformer architecture

---

## How This Maps to Production LLMs

| This Repo | Production (GPT-4, Claude, etc.) |
|-----------|--------------------------------|
| 10.8M params (NanoGPT) | 175B-1T+ parameters |
| 1M tokens (Shakespeare) | 1-15T tokens (internet, books, code) |
| 3,000 SFT examples | 100K+ human-annotated examples |
| 3,500 medical examples | Millions of domain-specific examples |
| Character-level tokenizer | BPE with 100K+ vocabulary |
| Single MacBook | Thousands of GPUs for months |
| Cross-entropy only | + RLHF/DPO for alignment |

The concepts are identical. The scale is different.

---

## License

Educational use. Built with Claude Code on Apple Silicon.
