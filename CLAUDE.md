# LLM Training Lab — CLAUDE.md

> **This is a Claude Code seed file.** Read this fully before doing anything. Execute each phase in order. Ask no clarifying questions — make sensible decisions and proceed.

---

## Teaching Mode — Read This First

**The person you are working with is a senior PM with a strong developer background. They want to deeply understand what is happening at every step — not just have the code run. You are both a builder and a teacher throughout this project.**

### Your teaching obligations

**Before writing any file**, print a short "what we're about to build" explanation:
```
📖 ABOUT TO BUILD: 01_pretrain/model.py
   What: A transformer language model written from scratch in ~150 lines of PyTorch.
   Why: Instead of using a HuggingFace model, writing it ourselves means we see
        exactly what attention, residual connections, and layer norm are doing.
   Key idea: Every token prediction is just a big matrix multiply followed by
             a softmax over the vocabulary. That's it.
```

**Before running any command**, explain what it will do:
```
▶ RUNNING: python 01_pretrain/data.py
   This downloads ~1MB of Shakespeare text, builds a character-level vocabulary
   (65 unique chars), and splits it 90/10 into train/val tensors saved to disk.
   Character-level means no external tokenizer — each character is one token.
   This is the simplest possible tokenization: easy to understand, hard to scale.
```

**After every command completes**, interpret the output:
```
✅ RESULT: data.py finished
   Vocab size: 65 chars  ← This is our entire "language" (letters, punctuation, newlines)
   Train tokens: 1,003,854
   Val tokens: 111,540
   Saved: 01_pretrain/data/train.bin, val.bin
   → Next, we build the model that will learn to predict these tokens.
```

**During training**, explain what the loss numbers mean as they appear:
```
📉 TRAINING PROGRESS (what these numbers mean):
   iter 0:    train loss 4.22  ← ~random. log2(65) ≈ 6 bits/char max entropy.
                                  4.22 nats ≈ 6.09 bits. We know nothing yet.
   iter 300:  train loss 2.15  ← Starting to learn letter frequencies and common words.
   iter 1500: train loss 1.68  ← Learning word patterns, some grammar.
   iter 3000: train loss 1.48  ← Reasonable Shakespeare-like structure.
   Rule of thumb: loss < 1.5 = model has learned something real.
                  loss < 1.2 = starting to sound fluent.
```

### Concept explanation triggers

Whenever you write or encounter any of the following, stop and explain it in plain language before proceeding:

| Trigger | What to explain |
|---|---|
| `nn.Embedding` | What an embedding table is. Analogy: a lookup table mapping token IDs to vectors |
| `scaled_dot_product_attention` | What Q, K, V represent. Analogy: a fuzzy dictionary lookup |
| causal mask / `tril` | Why we mask future tokens — the model can't cheat by looking ahead |
| `cross_entropy` loss | What the model is actually minimizing. Why we use log-probabilities |
| `AdamW` | What momentum and weight decay are doing intuitively |
| `cosine lr schedule` | Why we warm up and then decay learning rate |
| `gradient_accumulation_steps` | Why we need this on constrained hardware. Virtual batch size concept |
| `SFTTrainer` | What "supervised" means here — why we only compute loss on response tokens, not the prompt |
| `LoraConfig(r=16)` | What rank means. Analogy: compressing a photo — r is like JPEG quality level |
| `lora_alpha` | Explain the scaling factor and its relationship to effective learning rate |
| `get_peft_model` | What "freezing" weights means and why it's efficient |
| perplexity | Define it: e^loss. Intuition: how many equally-likely next tokens the model sees |
| `model.generate()` | Explain temperature and top_k sampling intuitively |
| MLflow `log_metric` | What experiment tracking is for — reproducibility and comparison |

### Checkpoint conversations

At the end of each phase, before moving to the next, print a "Checkpoint" summary:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 CHECKPOINT: What you just learned in Phase 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE BIG PICTURE:
  You built a transformer from scratch and trained it to predict the next
  character in Shakespeare text. The model has never seen a grammar rule —
  it learned language structure purely from statistics.

THE TRAINING LOOP (the most important thing to internalize):
  1. Sample a batch of (input, target) token sequences
  2. Forward pass: model predicts probability of each next token
  3. Loss: cross-entropy between predictions and actual next tokens
  4. Backward pass: compute gradients (how to nudge each weight)
  5. Optimizer step: nudge weights in the direction that reduces loss
  6. Repeat 3,000 times.
  That's all training is. Every LLM ever trained does this.

WHAT CHANGES IN PHASE 2:
  Instead of training from random weights, we start from a model that
  already knows English (SmolLM2-135M, trained on 1T tokens). We just
  teach it to follow instructions. The loop is identical — only the
  data format and starting weights change.

THINGS TO NOTICE IN MLFLOW (open http://localhost:5000):
  - The loss curve should show a sharp drop early, then flatten
  - Val loss should track train loss closely (no overfitting at 3k iters)
  - If val loss rises while train loss falls → overfitting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Each phase gets its own checkpoint with the same format.

### Analogies to use

Use these specific analogies when explaining concepts — they work well for someone with a technical PM background:

- **Attention mechanism** → A search engine inside each layer. Q=query, K=keys, V=values. You search your own context for relevant information.
- **Embeddings** → Word2Vec intuition: words that appear in similar contexts get similar vectors. "King - Man + Woman ≈ Queen" is the classic example.
- **Loss / cross-entropy** → Compression. The model is learning to compress Shakespeare as efficiently as possible. Lower loss = better compression = more understanding.
- **LoRA rank** → Like principal components in PCA. High rank = more expressive adaptation, more parameters. Low rank = cheap but less flexible.
- **Fine-tuning vs LoRA** → Fine-tuning is repainting the whole house. LoRA is applying a thin coat of specialist paint to the doors and windows only.
- **Perplexity** → If perplexity = 10, the model is as confused as if it had to guess from 10 equally-likely options at every step. Lower = more confident and correct.
- **SFT instruction masking** → When learning to bake, you don't study the recipe card — only the steps. We mask the prompt tokens so the model only learns from the response.
- **Gradient accumulation** → You can't carry 64 bags of groceries at once, so you make 4 trips of 16. Same math, smaller memory footprint.

### What NOT to do

- Do not print wall-of-text explanations mid-training. Batch your explanations before and after.
- Do not explain the same concept twice unless the person asks.
- Do not over-explain things that are obvious from the code (e.g., "here we import numpy").
- Do not skip the checkpoint summaries — they are the most important teaching moments.

---

## Project Goal

Build a complete, educational LLM training pipeline on this Mac (Apple Silicon, MPS backend) covering:
1. **Pre-training from scratch** — nano-GPT on TinyShakespeare
2. **Supervised Fine-Tuning (SFT)** — SmolLM2-135M on Databricks Dolly-15k
3. **PEFT / LoRA Fine-tuning** — domain adaptation on MedAlpaca Medical Flashcards
4. **Evaluation & comparison** — loss curves, perplexity, generation quality across all 3 phases
5. **MLflow experiment tracking** — local server, all runs logged

The code must be educational: prefer explicit training loops over magic abstractions where instructive.

---

## Machine Context

- **Hardware:** Apple Silicon M4, 16GB unified memory
- **Backend:** PyTorch MPS (Metal Performance Shaders) — use `device = torch.device("mps")` everywhere
- **Python:** Use system Python 3 or create a venv at `./venv`
- **Working directory:** `./llm-training-lab/` — create this and work inside it

---

## Stack

```
torch (MPS)
transformers
datasets
peft
trl
tokenizers
accelerate
mlflow
matplotlib
numpy
sentencepiece
```

Install all with: `pip install torch transformers datasets peft trl tokenizers accelerate mlflow matplotlib numpy sentencepiece`

---

## Project Structure

Create this layout before writing any code:

```
llm-training-lab/
├── CLAUDE.md                  # this file (copy here too)
├── requirements.txt
├── setup.sh                   # one-shot env setup script
│
├── 01_pretrain/
│   ├── data.py                # TinyShakespeare download + tokenize
│   ├── model.py               # NanoGPT from scratch (~10M params)
│   ├── train.py               # training loop with MPS + MLflow logging
│   └── generate.py            # text generation / sampling
│
├── 02_sft/
│   ├── data.py                # Dolly-15k loader, format to prompt template
│   ├── train.py               # SFT with trl.SFTTrainer + MLflow logging
│   └── evaluate.py            # perplexity + generation samples
│
├── 03_lora/
│   ├── data.py                # MedAlpaca flashcards loader
│   ├── train.py               # LoRA config + peft + MLflow logging
│   └── evaluate.py            # domain eval: before vs after LoRA
│
├── eval/
│   ├── compare_all.py         # side-by-side generation comparison across all 3
│   └── plot_curves.py         # loss curves from MLflow, saved as PNG
│
└── outputs/                   # checkpoints, plots, generation samples (gitignored)
```

---

## Phase 1: Pre-Training from Scratch

### Goal
Train a tiny transformer (~10M params) on TinyShakespeare from zero. This teaches the raw pre-training loop: forward pass, loss, backprop, optimizer step.

### Data
- **Dataset:** TinyShakespeare
- **Download URL:** `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Save to `01_pretrain/data/input.txt`
- Tokenizer: character-level (no external tokenizer needed — educational)
- Train/val split: 90/10

### Model (`01_pretrain/model.py`)
Implement a clean NanoGPT. Do NOT use HuggingFace here — write it from scratch:

```python
# Architecture targets (approximate):
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256      # context length
vocab_size = 65       # char-level
dropout = 0.2
# Total params: ~10M
```

Classes to implement:
- `CausalSelfAttention` — multi-head causal self-attention with mask
- `MLP` — 2-layer feedforward with GELU
- `Block` — pre-norm transformer block (LayerNorm → Attn → LayerNorm → MLP)
- `NanoGPT` — stack of blocks + embedding + LM head
- Include a `generate(idx, max_new_tokens, temperature, top_k)` method

### Training Loop (`01_pretrain/train.py`)
Write an explicit training loop (no Trainer abstraction):

```python
# Hyperparameters:
batch_size = 32
learning_rate = 3e-4
max_iters = 3000          # ~5-10 min on M4
eval_interval = 300
eval_iters = 50
optimizer = AdamW with weight decay 0.1
lr_schedule = cosine decay with warmup (100 steps)
```

- Use `device = torch.device("mps")` 
- Log to MLflow: train_loss, val_loss every eval_interval
- Save checkpoint at best val_loss to `outputs/pretrain_best.pt`
- Print progress every 100 steps
- At end, generate 3 samples (200 chars each) and log them as MLflow artifacts

### Generate (`01_pretrain/generate.py`)
- Load checkpoint, generate interactive samples
- Accept `--prompt` CLI arg
- Print generated text to stdout

---

## Phase 2: Supervised Fine-Tuning (SFT)

### Goal
Take a pretrained model (SmolLM2-135M from HuggingFace) and fine-tune it on instruction-following data. This teaches how SFT shapes raw language model behavior into assistant behavior.

### Model
- **Base:** `HuggingFaceTB/SmolLM2-135M`  
- Load with `AutoModelForCausalLM` and `AutoTokenizer`
- Use MPS: `model.to("mps")`

### Data (`02_sft/data.py`)
- **Dataset:** `databricks/databricks-dolly-15k` via HuggingFace datasets
- Format each example into this prompt template:

```
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
```

- If context is empty, omit that section
- Use a random 3,000-example subset for speed (set seed=42)
- Train/val split: 90/10
- Max token length: 512 (truncate)

### Training (`02_sft/train.py`)
Use `trl.SFTTrainer`:

```python
# Config:
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4   # effective batch = 16
learning_rate = 2e-5
warmup_ratio = 0.03
lr_scheduler_type = "cosine"
max_seq_length = 512
fp16 = False   # MPS doesn't support fp16 reliably; use bf16 if available else fp32
```

- Log all metrics to MLflow under experiment `"sft_dolly"`
- Save final model to `outputs/sft_model/`
- After training, run 5 inference samples from val set and log to MLflow

### Evaluate (`02_sft/evaluate.py`)
- Load base model and fine-tuned model
- Compute perplexity on 100 val examples for both
- Print comparison table: `Base PPL vs SFT PPL`
- Run 3 identical prompts through both, print side-by-side
- Save comparison to `outputs/sft_eval.txt`

---

## Phase 3: LoRA / PEFT Fine-Tuning

### Goal
Take the SFT model from Phase 2 and apply LoRA for domain adaptation on medical Q&A. This teaches parameter-efficient fine-tuning: why we freeze most weights, what rank/alpha mean, and how adapters work.

### Model
- **Base:** Load the SFT model saved in `outputs/sft_model/`
- Apply LoRA using `peft`

### LoRA Config
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rank
    lora_alpha=32,           # scaling = alpha/r = 2
    target_modules=["q_proj", "v_proj"],   # attention projections only
    lora_dropout=0.05,
    bias="none",
)
# Print trainable params vs total params — educational!
```

### Data (`03_lora/data.py`)
- **Dataset:** `medalpaca/medical_meadow_medical_flashcards`
- Each example has `input` (question) and `output` (answer)
- Format as:

```
### Medical Question:
{input}

### Answer:
{output}
```

- Use full dataset (~3,500 examples)
- Train/val split: 90/10
- Max token length: 256

### Training (`03_lora/train.py`)
```python
# Config:
num_train_epochs = 5
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
learning_rate = 3e-4        # LoRA can use higher LR
warmup_steps = 50
```

- Log to MLflow under experiment `"lora_medical"`
- Log: trainable_params, total_params, param_efficiency_pct
- Save LoRA adapter (not full model) to `outputs/lora_adapter/`
- LoRA adapters are tiny — log adapter size in MB to MLflow

### Evaluate (`03_lora/evaluate.py`)
- Load SFT base + LoRA adapter
- Compare against SFT base (no adapter) on 20 medical questions
- Print: base answer vs LoRA answer side-by-side
- Compute perplexity on medical val set for both
- Save to `outputs/lora_eval.txt`

---

## Phase 4: Cross-Phase Evaluation & Visualization

### `eval/compare_all.py`
Run all 3 models against the same 5 prompts and print a formatted comparison table:

```
Prompt: "What is the capital of France?"
┌──────────────────┬──────────────────────────────────┐
│ Model            │ Output                           │
├──────────────────┼──────────────────────────────────┤
│ NanoGPT (phase1) │ ...                              │
│ SmolLM SFT       │ ...                              │
│ SmolLM + LoRA    │ ...                              │
└──────────────────┴──────────────────────────────────┘
```

Use a mix of: general knowledge, instruction-following, and medical prompts.

### `eval/plot_curves.py`
- Pull all runs from MLflow local store
- Plot train/val loss curves for all 3 experiments on one figure
- Save to `outputs/loss_curves.png`
- Print a summary table: final train loss, val loss, perplexity, run duration

---

## MLflow Setup

Start MLflow UI as a background process:

```bash
mlflow ui --host 127.0.0.1 --port 5000 &
echo "MLflow UI: http://localhost:5000"
```

All training scripts must call:
```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("<experiment_name>")
```

Structure runs as:
- Experiment: `pretrain_shakespeare` → run: `nanogpt_v1`
- Experiment: `sft_dolly` → run: `smollm2_sft_v1`
- Experiment: `lora_medical` → run: `smollm2_lora_v1`

---

## Execution Order

Run these in sequence. Each step must complete successfully before the next:

```bash
# 0. Setup
bash setup.sh

# 1. Start MLflow
mlflow ui --host 127.0.0.1 --port 5000 &

# 2. Pre-train
cd llm-training-lab
python 01_pretrain/data.py        # download + tokenize shakespeare
python 01_pretrain/train.py       # train ~3000 iters, logs to mlflow
python 01_pretrain/generate.py --prompt "To be or not to be"

# 3. SFT
python 02_sft/data.py             # download + format dolly-15k
python 02_sft/train.py            # fine-tune SmolLM2-135M
python 02_sft/evaluate.py         # compare base vs SFT

# 4. LoRA
python 03_lora/data.py            # download + format medalpaca
python 03_lora/train.py           # apply LoRA, train on medical data
python 03_lora/evaluate.py        # compare SFT vs SFT+LoRA

# 5. Cross-phase eval
python eval/compare_all.py        # side-by-side across all 3
python eval/plot_curves.py        # loss curve plots
```

---

## Error Handling Guidance

- If MPS runs out of memory: reduce `batch_size` by half, increase `gradient_accumulation_steps`
- If `mps` device errors: fallback to `cpu` with a warning (but note it will be slow)
- If HuggingFace download fails: retry once; if still failing, print the manual download URL
- Do not silently swallow exceptions — always print what failed and why

---

## What to Print at the End of Each Phase

Phase 1 (pretrain):
```
✅ Phase 1 Complete
   Model params: 10.7M
   Best val loss: X.XX (iter NNNN)
   Sample: "[generated text]"
   MLflow run: http://localhost:5000/#/experiments/...
```

Phase 2 (SFT):
```
✅ Phase 2 Complete
   Base perplexity: XX.X | SFT perplexity: XX.X (↓ improvement)
   Model saved: outputs/sft_model/
```

Phase 3 (LoRA):
```
✅ Phase 3 Complete
   Trainable params: X.XM / XXX.XM total (X.X% of model)
   Adapter size: X.X MB
   SFT perplexity on medical: XX.X | LoRA perplexity: XX.X
   Adapter saved: outputs/lora_adapter/
```

---

## Key Learning Moments to Surface in Code Comments

Add `# 📚 LEARN:` comments at these points:
- In `model.py`: why causal mask, what attention scores represent
- In `train.py` (phase 1): what the loss number means (bits per character)
- In `02_sft/train.py`: why instruction masking matters (only compute loss on response tokens)
- In `03_lora/train.py`: print and explain the trainable param count; comment on why rank=16 is common
- In `03_lora/train.py`: comment explaining lora_alpha scaling and its effect on learning rate

---

## Modular / MAX Note

After Phase 3 is complete, add a `04_modular/README.md` that explains:
1. What Modular MAX is (inference engine, not training framework)
2. How the trained LoRA adapter from Phase 3 could be exported and served via MAX Container
3. Benchmark comparison format: HuggingFace `model.generate()` throughput vs MAX serving throughput
4. Why Mojo's type system and MLIR backend gives MAX its performance edge vs Python/PyTorch

This is a conceptual README, not runnable code (MAX requires a separate install and is primarily Linux/cloud).

---

*Generated for: LLM Training Lab | Apple M4 MacBook Air 16GB | Educational use*
