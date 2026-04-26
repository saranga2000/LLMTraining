# LLM Training Lab — Comprehensive Learning Guide

> Everything you need to understand about how LLMs are trained, fine-tuned, and evaluated — explained through the code in this repo. Written for technical product managers and engineers who want deep understanding, not just working code.

---

## Table of Contents

1. [The Big Picture: How LLMs Are Built](#the-big-picture)
2. [Phase 1: Pre-Training from Scratch](#phase-1-pre-training-from-scratch)
   - [The Transformer Architecture](#the-transformer-architecture)
   - [The Training Loop](#the-training-loop)
   - [What the Loss Numbers Mean](#what-the-loss-numbers-mean)
3. [Phase 2: Supervised Fine-Tuning (SFT)](#phase-2-supervised-fine-tuning)
   - [Why SFT Exists](#why-sft-exists)
   - [Data Formatting](#sft-data-formatting)
   - [Training Differences from Phase 1](#training-differences-from-phase-1)
   - [Evaluation: Base vs SFT](#evaluation-base-vs-sft)
4. [Phase 3: LoRA / PEFT Fine-Tuning](#phase-3-lora--peft-fine-tuning)
   - [The Problem LoRA Solves](#the-problem-lora-solves)
   - [How LoRA Works](#how-lora-works)
   - [LoRA Configuration Explained](#lora-configuration-explained)
   - [Evaluation: SFT vs LoRA](#evaluation-sft-vs-lora)
5. [Key Concepts Reference](#key-concepts-reference)
6. [Analogies That Work](#analogies-that-work)
7. [Common Pitfalls and What We Learned](#common-pitfalls)

---

## The Big Picture

Every modern LLM (ChatGPT, Claude, Llama, Gemini) is built in the same three stages:

```
Stage 1: Pre-Training          Stage 2: Fine-Tuning (SFT)       Stage 3: Specialization (LoRA)
━━━━━━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Random weights                 Pretrained weights                SFT weights
    ↓                              ↓                                ↓
Train on MASSIVE text          Train on (instruction, response)  Train on domain-specific data
(books, web, code)             pairs from human demonstrations   (medical, legal, coding, etc.)
    ↓                              ↓                                ↓
Model learns LANGUAGE          Model learns to FOLLOW            Model learns DOMAIN EXPERTISE
(grammar, facts, patterns)     INSTRUCTIONS (helpful, structured) (terminology, reasoning patterns)
    ↓                              ↓                                ↓
"Knows English"                "Knows how to be an assistant"    "Knows medicine"
```

This repo implements all three stages on a MacBook, at a scale you can run and understand.

---

## Phase 1: Pre-Training from Scratch

**Code:** `01_pretrain/` | **Model:** NanoGPT (~10.8M params) | **Data:** TinyShakespeare

### The Transformer Architecture

The transformer is the architecture behind every modern LLM. Our NanoGPT implementation in `01_pretrain/model.py` builds it from scratch in ~150 lines. Here's what each piece does:

#### Token Embeddings (`nn.Embedding`)

```python
self.token_embedding = nn.Embedding(vocab_size, n_embd)  # 65 × 384
```

**What it is:** A lookup table. Each of the 65 characters gets a 384-dimensional vector.

**Analogy:** Think of it like a phonebook — given a character ID (0-64), look up its vector. Before training, these vectors are random. After training, similar characters (like 'a' and 'e', both vowels) end up with similar vectors.

**The Word2Vec intuition:** Words that appear in similar contexts get similar vectors. The classic example: "King - Man + Woman ≈ Queen." Our character embeddings learn a simpler version of this.

#### Position Embeddings

```python
self.position_embedding = nn.Embedding(block_size, n_embd)  # 256 × 384
```

**Why needed:** Attention is permutation-invariant — it doesn't know word order. "The cat sat on the mat" and "mat the on sat cat the" would look identical without position information. Position embeddings add "I am token #0, #1, #2..." to each vector.

#### Causal Self-Attention (The Core of Transformers)

This is the mechanism that makes transformers work. It's in `CausalSelfAttention` class.

**Analogy:** A search engine inside each layer. For each token:
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I provide?"

The attention score between tokens i and j is: `score = Q_i · K_j / sqrt(d)`

High score = "token j is relevant to token i." The model then takes a weighted average of all V vectors, weighted by these scores.

```python
# In code:
Q = self.q_proj(x)    # What am I looking for?
K = self.k_proj(x)    # What do I contain?
V = self.v_proj(x)    # What information do I provide?

# Attention weights = softmax(Q @ K^T / sqrt(d))
# Output = weights @ V
```

**Multi-head attention:** Instead of one set of Q/K/V, we have 6 parallel "heads" (n_head=6), each with dimension 64 (384/6). Each head can attend to different things — one might track grammar, another might track meaning, another might track character relationships.

#### The Causal Mask — Why We Mask Future Tokens

```python
# Equivalent to: torch.tril(torch.ones(T, T))
# This creates a lower-triangular matrix:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

**Why:** During training, the model predicts each next token. Token at position 3 should only see tokens 0, 1, 2 — not tokens 4, 5, 6 (that would be cheating). The causal mask sets future attention scores to -infinity, so after softmax they become 0.

**This is what makes it "autoregressive"** — each token can only look backward, never forward.

#### The MLP (Feed-Forward Network)

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)   # 384 → 1536 (expand)
        self.gelu = nn.GELU()                         # Non-linearity
        self.fc2 = nn.Linear(4 * n_embd, n_embd)     # 1536 → 384 (compress back)
```

**What it does:** After attention gathers information from other tokens, the MLP processes that information. It expands to 4x the dimension (giving the model room to compute), applies a non-linear activation (GELU), and compresses back.

**Analogy:** Attention is like reading a book and highlighting relevant passages. The MLP is like thinking about what you highlighted and forming new understanding.

**Note:** MLP = Multilayer Perceptron = Feed-Forward Network (FFN). Same concept, different names used in different papers.

#### Residual Connections and Layer Norm

```python
# In each Block:
x = x + self.attn(self.ln1(x))    # Residual around attention
x = x + self.mlp(self.ln2(x))     # Residual around MLP
```

**Residual connections (the `+`):** Instead of `x = f(x)`, we do `x = x + f(x)`. This means information can flow directly through the network without being forced through every transformation. It prevents the "vanishing gradient" problem in deep networks.

**Layer Norm:** Normalizes the values at each layer to have mean 0 and variance 1. Without it, values can explode or vanish as they pass through 6 layers. Think of it as recalibrating a sensor between each measurement.

#### The LM Head — Making Predictions

```python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # 384 → 65
```

**What it does:** Takes the final 384-dimensional representation and produces 65 scores (logits) — one per character in the vocabulary. After softmax, these become probabilities: "70% chance the next character is 'e', 15% chance it's 't', ..."

**Weight tying:** We share weights between the token embedding and LM head: `self.lm_head.weight = self.token_embedding.weight`. Intuition: if "e" maps to vector [0.3, -0.1, ...] in the embedding, then when the model wants to output "e", it should produce something close to [0.3, -0.1, ...] before the final projection.

### The Training Loop

The training loop in `01_pretrain/train.py` is the most fundamental concept in deep learning. Every LLM ever trained does exactly this:

```
for each training step:
    1. SAMPLE a batch of (input, target) sequences
    2. FORWARD PASS: model predicts probability of each next token
    3. LOSS: cross-entropy between predictions and actual next tokens
    4. BACKWARD PASS: compute gradients (how to nudge each weight)
    5. OPTIMIZER STEP: nudge weights in the direction that reduces loss
```

That's all training is. ChatGPT, Claude, Llama — they all do this, just with more data and bigger models.

#### Cross-Entropy Loss — What the Model Minimizes

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

**What it measures:** How surprised the model is by the correct answer. If the model predicts "e" with 90% confidence and the answer is "e", the loss is low (-log(0.9) = 0.1). If it predicted "e" with 1% confidence, the loss is high (-log(0.01) = 4.6).

**Analogy:** The model is learning to compress Shakespeare as efficiently as possible. Lower loss = better compression = more understanding of the text's patterns.

#### Backpropagation — How Gradients Flow

```python
loss.backward()   # Compute gradients
optimizer.step()  # Update weights
optimizer.zero_grad()  # Reset gradients for next step
```

**`loss.backward()`** — This is backpropagation. PyTorch records every operation in the forward pass as a computational graph. `backward()` walks this graph in reverse, computing ∂loss/∂weight for every parameter using the chain rule. This tells us: "if I increase this weight by a tiny amount, how much does the loss change?"

**Note:** Backpropagation is NOT in model.py — it's handled by PyTorch's autograd system. The forward pass defines the computation; PyTorch automatically derives the backward pass.

#### AdamW Optimizer

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
```

**What it does:** Uses the gradients to update weights. AdamW is smarter than basic gradient descent:
- **Momentum:** Instead of following the current gradient exactly, it maintains a running average of past gradients. Like a heavy ball rolling downhill — it doesn't change direction on every bump.
- **Adaptive learning rates:** Each parameter gets its own effective learning rate based on how much its gradient varies. Frequently-updated params get smaller steps; rare ones get larger steps.
- **Weight decay (0.1):** Slowly shrinks weights toward zero, preventing them from growing too large. Acts as regularization.

#### Cosine Learning Rate Schedule with Warmup

```python
# Warmup: linearly increase LR from 0 to max over 100 steps
# Then: cosine decay from max to ~0 over remaining steps
```

**Why warmup:** At the start, the model's gradients are noisy (random weights → wild predictions). Large learning rate + noisy gradients = instability. We start with tiny steps, ramp up as gradients stabilize.

**Why cosine decay:** After the model has learned the big patterns, we reduce the step size to fine-tune details. Cosine gives a smooth, gradual decay (not a sharp cliff).

### What the Loss Numbers Mean

```
iter    0: train loss 4.22  ← Random. log2(65) ≈ 6 bits. The model knows nothing.
iter  300: train loss 2.15  ← Learning letter frequencies and common words.
iter 1500: train loss 1.68  ← Learning word patterns, some grammar.
iter 3000: train loss 1.48  ← Reasonable Shakespeare-like structure.

Rule of thumb:
  loss < 1.5 = model has learned something real
  loss < 1.2 = starting to sound fluent
```

**Why 4.2 at start?** With 65 characters, random guessing gives loss = ln(65) ≈ 4.17. The model starts slightly worse than random because it hasn't even learned uniform probabilities yet.

---

## Phase 2: Supervised Fine-Tuning

**Code:** `02_sft/` | **Model:** SmolLM2-135M (HuggingFace) | **Data:** Dolly-15k

### Why SFT Exists

A pretrained model (like SmolLM2-135M) knows English — it was trained on ~1 trillion tokens of internet text. But it doesn't know how to follow instructions. It just predicts the next token.

If you give it "What is the capital of France?", it might continue with "What is the capital of Germany? What is the capital of Spain?..." — because on the internet, questions are often followed by more questions.

SFT teaches it: "When you see an instruction, generate a helpful response." We do this by showing it thousands of (instruction, response) pairs.

**This is exactly how ChatGPT was created:** GPT-3 (pretrained) → InstructGPT (SFT on human demonstrations).

### SFT Data Formatting

Each training example follows a structured template:

```
### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris. It is the largest city in France
and serves as the country's political, economic, and cultural center.
```

**Why the template matters:** The model learns to associate the `### Instruction:` / `### Response:` structure with Q&A behavior. This is a lightweight form of "prompt engineering baked into the weights."

If context is provided (background information for the question), it's included:

```
### Instruction:
Summarize the following text.

### Context:
[text to summarize]

### Response:
[summary]
```

### Training Differences from Phase 1

| Aspect | Phase 1 (Pretrain) | Phase 2 (SFT) |
|--------|-------------------|---------------|
| Starting weights | Random | Pretrained (SmolLM2-135M) |
| Data format | Raw text | Structured (instruction/response) |
| Vocabulary | 65 characters | 49,152 subword tokens |
| Learning rate | 3e-4 | 2e-5 (15x smaller!) |
| Why smaller LR? | Learning from scratch | Preserving pretrained knowledge |
| Trainer | Manual loop | HuggingFace SFTTrainer |

**Why a smaller learning rate?** The pretrained model already has good weights from seeing 1 trillion tokens. Large steps would destroy that knowledge. We take small, careful steps — like making minor adjustments to a painting rather than starting over.

#### Gradient Accumulation — Virtual Batch Size

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
# Effective batch size = 1 × 16 = 16
```

**The problem:** We want a batch size of 16 for stable gradients, but 16 examples don't fit in 16GB unified memory (each example needs memory for forward pass, activations, gradients).

**The solution:** Process 1 example at a time, but accumulate gradients for 16 steps before updating weights. Mathematically identical to processing 16 at once.

**Analogy:** You can't carry 64 bags of groceries at once, so you make 4 trips of 16. Same total groceries, smaller load per trip.

#### SFTTrainer — What It Does Under the Hood

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)
```

SFTTrainer handles:
1. Tokenizing the "text" field from the dataset
2. Creating input/target pairs (shifted by one token, same as Phase 1)
3. Running the training loop (forward → loss → backward → optimizer step)
4. Evaluating on the validation set periodically
5. Saving checkpoints

Under the hood, it's doing the exact same thing as our Phase 1 training loop — just wrapped in a convenient API.

### Evaluation: Base vs SFT

**Perplexity comparison on instruction-following data:**

| Model | Perplexity | Interpretation |
|-------|-----------|---------------|
| Base SmolLM2-135M | 11.38 | "Choosing from ~11 options per token" — uncertain |
| SFT SmolLM2-135M | 9.18 | "Choosing from ~9 options" — 19.4% more confident |

**Side-by-side generation — this tells the real story:**

| Prompt | Base Model | SFT Model |
|--------|-----------|-----------|
| "Explain gravity in one sentence." | Repeats the prompt: "Explain gravity in two sentences. Explain gravity in three sentences..." | "Gravity is the force that keeps everything in place on Earth." |
| "What is the capital of France?" | Eventually gets there but rambles | "The capital of France is Paris." — Direct. |
| "List three benefits of exercise." | Starts generating lesson plans | Lists actual benefits |

**Key insight:** The base model is a *completion* model — it just predicts what text comes next. The SFT model is an *instruction-following* model — it understands that questions deserve answers.

---

## Phase 3: LoRA / PEFT Fine-Tuning

**Code:** `03_lora/` | **Model:** SFT model + LoRA adapter | **Data:** MedAlpaca Medical Flashcards

### The Problem LoRA Solves

After Phase 2, the model can follow instructions. But it has no specialized medical knowledge. We could do another round of full fine-tuning, but:

1. **Storage:** Saving a separate 270 MB model for every domain (medical, legal, coding...) adds up fast.
2. **Catastrophic forgetting:** Full fine-tuning on medical data might destroy the general instruction-following ability.
3. **Speed:** Fine-tuning all 135M parameters is slow.

LoRA solves all three problems.

### How LoRA Works

**Full fine-tuning** updates ALL weights:
```
W_new = W_old + ΔW
where ΔW is 384 × 384 = 147,456 parameters per layer
```

**LoRA's insight:** ΔW doesn't need to be full-rank. The weight updates during fine-tuning are LOW RANK — they can be decomposed:
```
ΔW = A × B
where A is (384 × 16) and B is (16 × 384)
A has 6,144 params, B has 6,144 params
Total: 12,288 params instead of 147,456 (12x fewer!)
```

**During the forward pass:**
```
output = W_frozen(x) + (alpha/r) × B(A(x))
         ↑                 ↑
    Original model      LoRA adapter
    (frozen, untouched)  (tiny, trainable)
```

**Analogy:** Full fine-tuning is repainting the whole house (270 MB). LoRA is applying a thin coat of specialist paint to just the doors and windows (3.5 MB). Same house, new specialty.

**Another analogy:** Like PCA (principal component analysis). The rank `r` is how many principal components you keep. High rank = more expressive, more parameters. Low rank = cheaper but less flexible. r=16 is the sweet spot for most tasks.

### LoRA Configuration Explained

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                              # Rank
    lora_alpha=32,                     # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Where to add adapters
    lora_dropout=0.05,                 # Regularization
    bias="none",                       # Don't train bias terms
)
```

#### `r=16` (Rank) — The Compression Level

Each adapter matrix is decomposed into two small matrices of rank 16. Think of it like JPEG quality:
- r=1: Very compressed, loses information
- r=4: Light model, less expressive
- r=16: Good balance (what we use)
- r=64: Barely compressed, almost full fine-tuning
- r=384: Full rank = identical to full fine-tuning

**Common values:** 4, 8, 16, 32, 64. r=16 is standard for domain adaptation tasks.

#### `lora_alpha=32` (Scaling Factor)

The adapter output is scaled by `alpha/r = 32/16 = 2`. This means the adapter's contribution is amplified 2x.

**Think of it as a volume knob** for the adapter. Higher alpha = adapter has more influence on the output. Rule of thumb: `alpha = 2 × r` is a good starting point.

**Relationship to learning rate:** Effective learning rate for LoRA ≈ `learning_rate × (alpha/r)`. With our settings: `3e-4 × 2 = 6e-4` effective LR. This is why LoRA can use higher base learning rates than full fine-tuning.

#### `target_modules=["q_proj", "v_proj"]` — Where to Add Adapters

We only add LoRA adapters to the Query and Value projections in attention. Why not all layers?

- **Q and V** are the most impactful for changing model behavior (empirical finding from the original LoRA paper)
- **K (key)** and **O (output)** projections add parameters with less benefit
- **MLP layers** are less important for behavioral adaptation

This targets exactly the components that matter most with the fewest parameters.

#### `get_peft_model` — Freezing Weights

```python
model = get_peft_model(model, lora_config)
```

This function:
1. Freezes ALL original model parameters (`requires_grad=False`)
2. Injects small LoRA adapter matrices next to q_proj and v_proj
3. Only the adapter matrices are trainable

**The result for our model:**
```
Total params:     135,436,608 (135M)
Trainable params:     921,600 (0.9M)
Frozen params:    134,515,008 (134.5M)
Efficiency:       0.68% of model is being trained
```

Only 0.68% of the model was trained — yet it gained medical expertise.

### Evaluation: SFT vs LoRA

**Perplexity on medical data:**

| Model | Perplexity | Interpretation |
|-------|-----------|---------------|
| SFT (no medical training) | 6.72 | "Choosing from ~7 options per medical token" |
| SFT + LoRA (medical) | 3.80 | "Choosing from ~4 options" — 43.5% improvement! |

**Side-by-side generation on medical questions:**

| Question | SFT Response | LoRA Response |
|----------|-------------|--------------|
| What is hypertension? | Circular: "Hypertension is high blood pressure. Hypertension is..." | Adds causes: "can occur as a result of kidney disease..." |
| Function of the liver? | Generic anatomy description | Specific: "synthesizing bile for fat breakdown" |
| How does insulin work? | Surface-level: "regulates blood sugar" | Mechanism: "binds to insulin receptor on cells" |
| Function of kidneys? | Just names anatomy parts | Explains filtering function and urine production |
| What is an MRI? | Correct-ish, then drifts to new questions | Clean, accurate, stops appropriately |

**Key patterns:**
1. **SFT often "leaks"** — generates new `### Medical Question:` prompts instead of stopping. LoRA learned the flashcard format.
2. **LoRA is more specific** — mentions mechanisms (insulin receptor binding, bile for fat breakdown) vs SFT's surface descriptions.
3. **LoRA is more concise** — trained on flashcard-style Q&A, it gives focused answers.

---

## Key Concepts Reference

### Perplexity
**Definition:** e^(cross-entropy loss)

**Intuition:** If perplexity = 10, the model is as confused as if it had to guess from 10 equally-likely options at every token. Lower = more confident and correct.

| Perplexity | What It Means |
|-----------|---------------|
| 1.0 | Perfect prediction (impossible in practice) |
| 3-5 | Very confident, domain-specialized |
| 10 | Reasonably confident |
| 100 | Quite confused |
| 1000+ | Basically random guessing |

### Temperature and Top-K Sampling

When generating text with `model.generate()`:

**Temperature (0.7):** Controls randomness.
- Temperature = 0: Always pick the most likely token (deterministic, but boring and repetitive)
- Temperature = 0.7: Mostly likely tokens, with some variety (good balance)
- Temperature = 1.0: Sample directly from the probability distribution
- Temperature = 2.0: Very random (creative but potentially nonsensical)

**Top-K (50):** Only consider the top 50 most likely tokens at each step. Everything else gets probability 0. Prevents the model from picking extremely unlikely tokens.

### Token vs Character vs Subword

| Tokenization | Example: "unhappiness" | Vocab Size | Used In |
|-------------|----------------------|-----------|---------|
| Character | u, n, h, a, p, p, i, n, e, s, s | ~65 | Phase 1 (NanoGPT) |
| Subword (BPE) | un, happi, ness | ~49,000 | Phase 2 & 3 (SmolLM2) |
| Word | unhappiness | ~100,000+ | Older models |

Subword tokenization (BPE = Byte Pair Encoding) is the standard for modern LLMs. It balances vocabulary size with sequence length.

---

## Analogies That Work

| Concept | Analogy |
|---------|---------|
| **Attention mechanism** | A search engine inside each layer. Q=query, K=keys, V=values. You search your own context for relevant information. |
| **Embeddings** | A phonebook — given a word ID, look up its vector. After training, similar words have similar vectors. |
| **Loss / cross-entropy** | Compression. The model learns to compress text as efficiently as possible. Lower loss = better compression = more understanding. |
| **LoRA rank** | Like JPEG quality level. High rank = high quality (more params). Low rank = compressed (fewer params). r=16 is the sweet spot. |
| **Fine-tuning vs LoRA** | Fine-tuning is repainting the whole house. LoRA is applying a thin coat of specialist paint to the doors and windows only. |
| **Perplexity** | If perplexity = 10, the model is choosing from 10 equally-likely options at each step. Lower = more confident. |
| **SFT instruction masking** | When learning to bake, you study the steps, not the recipe card. We only compute loss on response tokens. |
| **Gradient accumulation** | Can't carry 64 bags of groceries at once, so make 4 trips of 16. Same total, smaller load. |
| **Residual connections** | An expressway bypass around each layer. Information can flow directly without being forced through every transformation. |
| **Learning rate warmup** | Easing into a workout. Start gentle (small LR), ramp up as the model stabilizes. |
| **Cosine decay** | Landing a plane. Gradual descent as you approach the destination (optimal weights). |
| **Weight tying** | If "cat" maps to vector [0.3, -0.1] going in, the model should produce something near [0.3, -0.1] when it wants to output "cat." |

---

## Common Pitfalls

### MPS (Apple Silicon) Memory Issues

Apple's Metal Performance Shaders (MPS) backend is powerful but has quirks:

- **OOM during backward pass:** The backward pass stores all intermediate activations, requiring 2-3x the memory of the forward pass. Solution: reduce batch size, increase gradient accumulation.
- **OOM during evaluation:** Eval also uses memory. Use `per_device_eval_batch_size=1` and set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to allow flexible memory allocation.
- **Reduce `max_length`:** We went from 512 to 256 tokens to halve memory usage. Trade-off: longer examples get truncated.

**Environment variable to always set:**
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Output Buffering

Python buffers stdout when output is redirected (to a file or pipe). Training progress won't appear in real-time. Fix:
```bash
PYTHONUNBUFFERED=1 python train.py
```

### Mac Sleep Interruptions

If your Mac sleeps during training, the process pauses but doesn't crash. MPS computations resume when the Mac wakes. Wall-clock time will be inflated but results are identical.

To prevent sleep during long training runs:
```bash
caffeinate -i -s  # Prevents sleep until Ctrl+C
```

### TRL API Changes (v1.2+)

The `trl` library evolves rapidly. Key changes we hit:
- `SFTTrainer` no longer accepts `max_seq_length` → use `SFTConfig` with `max_length`
- `warmup_ratio` deprecated → use `warmup_steps` with manual calculation
- `torch_dtype` deprecated in `from_pretrained()` → use `dtype`

### Dataset Size Matters

MedAlpaca has 33,955 examples but CLAUDE.md targets 3,500 for tractable training time. Always subsample for educational/experimental runs:
```python
indices = random.sample(range(len(dataset)), min(3500, len(dataset)))
dataset = dataset.select(indices)
```

---

## Summary: The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    THE LLM TRAINING PIPELINE                │
├──────────┬──────────────┬───────────┬───────────────────────┤
│ Phase    │ What Changes │ Loss      │ What It Learned       │
├──────────┼──────────────┼───────────┼───────────────────────┤
│ 1. Pre-  │ All weights  │ 4.2 → 1.5│ English language       │
│   train  │ from random  │           │ structure, vocabulary  │
├──────────┼──────────────┼───────────┼───────────────────────┤
│ 2. SFT   │ All weights  │ 2.7 → 2.3│ How to follow          │
│          │ from pre-    │           │ instructions           │
│          │ trained      │           │                        │
├──────────┼──────────────┼───────────┼───────────────────────┤
│ 3. LoRA  │ 0.68% of     │ 1.8 → 1.3│ Medical domain         │
│          │ weights via  │           │ expertise              │
│          │ adapters     │           │                        │
├──────────┼──────────────┼───────────┼───────────────────────┤
│ Result: A model that speaks English, follows instructions,  │
│ and has medical expertise — trained entirely on a MacBook.  │
└─────────────────────────────────────────────────────────────┘
```

Each phase builds on the last. The model never forgets — it adds capabilities. This is the same pipeline used by OpenAI, Anthropic, Google, and Meta to build their flagship models. The only differences are scale (billions of parameters, trillions of tokens) and an additional RLHF/DPO step for alignment.
