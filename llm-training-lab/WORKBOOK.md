# LLM Training Lab — Self-Guided Workbook

> A step-by-step, hands-on learning exercise that takes you from zero to training and evaluating LLMs. Follow this sequentially — each step builds on the last.
>
> **Time commitment:** ~3-4 hours (including training time where you can read ahead)

---

## Which path are you on?

### Without Claude Code (this workbook is for you)
Follow this workbook from top to bottom. It explains what to read, what to run, what the output means, and what to take away at each step. Use [LEARNING_GUIDE.md](LEARNING_GUIDE.md) as a companion reference when you want deeper explanations of any concept.

### With Claude Code (interactive path)
If you have a [Claude subscription](https://claude.ai) and Claude Code installed in VS Code:
1. Open this repo folder in VS Code
2. Open the Claude Code panel (Cmd+Shift+P → "Claude Code")
3. Say: **"Read CLAUDE.md and execute it phase by phase"**

Claude will guide you interactively — explaining concepts before running code, interpreting output in real-time, and answering questions on the fly. The `CLAUDE.md` file is a complete teaching plan that tells Claude exactly how to guide you. You can still reference this workbook and the learning guide alongside Claude's explanations.

---

---

## Before You Start

### What You'll Need
- A Mac with Apple Silicon (M1/M2/M3/M4) and 16GB RAM
- Python 3.10 or later
- A terminal and a text editor (VS Code recommended)
- ~5 GB free disk space
- Basic comfort with command line and Python (you don't need ML experience)

### Setup

```bash
# Clone the repo
git clone https://github.com/saranga2000/LLMTraining.git
cd LLMTraining

# Run the setup script (creates venv, installs all dependencies)
bash setup.sh

# Activate the environment (you'll need this in every new terminal)
source llm-training-lab/venv/bin/activate
cd llm-training-lab

# Set environment variables (do this once per terminal session)
export PYTHONUNBUFFERED=1                    # Real-time training output
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Flexible GPU memory

# Prevent your Mac from sleeping during training
caffeinate -i -s &
```

### How to Use This Workbook

Each step follows this pattern:
1. **READ** — Understand what you're about to do and why
2. **LOOK AT THE CODE** — Open the file, read the comments (especially `# LEARN:` comments)
3. **RUN** — Execute the command
4. **INTERPRET** — Understand what the output means
5. **CHECKPOINT** — Review what you learned before moving on

Take your time. The training steps take 10-80 minutes each — use that time to read ahead or review the code.

---

## Phase 1: Pre-Training a Language Model from Scratch

### The Goal

You're going to build a transformer neural network from zero and teach it English by showing it Shakespeare. The model starts knowing absolutely nothing — random weights — and learns to generate coherent text purely from statistical patterns.

This is the foundation of every LLM: GPT-4, Claude, Llama, Gemini. They all start here.

### What is a Transformer?

Before you touch any code, understand the architecture at a high level:

```
Input: "To be or not"
  ↓
[Token Embedding] — Look up a vector for each character
  ↓
[Position Embedding] — Add "I'm token #1, #2, #3..." to each vector
  ↓
[Transformer Block 1]
  ├── [Self-Attention] — Each token looks at all previous tokens
  │                      and decides which are relevant
  └── [Feed-Forward]   — Process the gathered information
  ↓
[Transformer Block 2] — Same thing, deeper understanding
  ↓
... (6 blocks total)
  ↓
[LM Head] — Predict: "What character comes next?"
  ↓
Output: probability distribution over 65 characters
        → most likely: " " (space), "t", "to"...
```

The model predicts one token at a time. To generate text, you predict the next token, append it, and predict again.

---

### Step 1.1: Prepare the Data

**Read first:** Open `01_pretrain/data.py` in your editor. Read through it — it's short.

**What it does:** Downloads Shakespeare's complete works (~1MB of text), splits each character into a "token" (a=0, b=1, c=2, ..., 65 unique characters), and creates train/validation splits.

**Why character-level?** It's the simplest possible tokenization. No external libraries, no complicated algorithms. Each character is one token. Modern LLMs use "subword" tokenization (splitting words into pieces like "un" + "happi" + "ness"), but character-level lets us focus on the model, not the tokenizer.

**Run it:**
```bash
python 01_pretrain/data.py
```

**What you should see:**
```
Vocab size: 65 chars
Train tokens: 1,003,854
Val tokens: 111,540
Saved: data/train.bin, data/val.bin
```

**Interpret:** 65 characters is our entire "language" — letters, punctuation, spaces, newlines. The model will learn to predict which of these 65 comes next, given the ones before it. 1M tokens is tiny by modern standards (GPT-4 trained on trillions), but enough to learn Shakespeare patterns.

---

### Step 1.2: Understand the Model

**Read carefully:** Open `01_pretrain/model.py`. This is the most important file in the entire repo. Read every `# LEARN:` comment.

For a detailed line-by-line explanation, see `01_pretrain/TRANSFORMER_WALKTHROUGH.md`.

**Key things to understand:**

1. **`CausalSelfAttention`** (the core mechanism):
   - Each token creates a Query ("what am I looking for?"), Key ("what do I contain?"), and Value ("what info do I provide?")
   - Attention score = how relevant token j is to token i
   - The causal mask prevents looking at future tokens (no cheating!)
   - Multiple heads (6) let the model attend to different things simultaneously

2. **`MLP`** (the "thinking" layer):
   - Expands to 4x the dimension (384 → 1536), applies non-linearity, compresses back
   - Attention gathers information; MLP processes it

3. **`Block`** (one transformer layer):
   - LayerNorm → Attention → LayerNorm → MLP
   - Residual connections (`x = x + attention(x)`) let information flow directly through

4. **`NanoGPT`** (the full model):
   - Token embeddings + Position embeddings → 6 Blocks → LM Head
   - LM Head outputs 65 scores → softmax → probabilities

5. **`generate()`** method:
   - Temperature controls randomness (0 = deterministic, 1 = creative)
   - Top-K limits choices to the K most likely tokens

**Check your understanding:** Can you trace what happens to the input "Hello" through the model? Each character gets embedded, positions are added, attention looks at previous characters, MLP processes, and the LM head predicts the next character.

---

### Step 1.3: Train the Model

**Read first:** Open `01_pretrain/train.py`. Focus on the training loop — it's the same loop used by every LLM ever trained:

```python
for iter in range(max_iters):          # 3000 iterations
    xb, yb = get_batch('train')         # 1. Sample a batch
    logits, loss = model(xb, yb)        # 2. Forward pass → predictions + loss
    optimizer.zero_grad()               # 3. Reset gradients
    loss.backward()                     # 4. Backward pass → compute gradients
    optimizer.step()                    # 5. Update weights to reduce loss
```

That's it. Five lines. Every LLM training run is this loop.

**Run it:**
```bash
python 01_pretrain/train.py
```

**This takes ~10 minutes.** While it runs, watch the loss numbers:

```
What the loss numbers mean:

iter    0: loss 4.22  ← Random guessing. ln(65) ≈ 4.17 = theoretical maximum entropy.
                         The model knows nothing — it's as confused as if randomly
                         picking from 65 characters.

iter  300: loss 2.15  ← Learning letter frequencies. It now knows 'e' is common,
                         'z' is rare, spaces come after words.

iter 1000: loss 1.68  ← Learning word patterns. "th" is common, "xq" is not.
                         Starting to learn common words like "the", "and", "you".

iter 2000: loss 1.55  ← Learning grammar and structure. Knows that periods end
                         sentences, capital letters start them.

iter 3000: loss 1.48  ← Reasonable Shakespeare-like structure. Names, dialogue
                         patterns, iambic-ish rhythm.

Rule of thumb:
  loss > 3.0 = barely started learning
  loss 2.0   = knows letter and word patterns
  loss 1.5   = has learned real language structure
  loss < 1.2 = quite fluent (we won't reach this with 10M params)
```

**Also watch:** The validation loss (val_loss). If val_loss stays close to train_loss, the model is generalizing well. If val_loss rises while train_loss falls, it's memorizing (overfitting).

---

### Step 1.4: Generate Text

**Run it:**
```bash
python 01_pretrain/generate.py --prompt "To be or not to be"
```

**What you should see:** Shakespeare-ish text. It won't be perfect — this is a tiny model trained on limited data. But it should have:
- Proper word boundaries (spaces in the right places)
- Character names followed by colons (dialogue format)
- Roughly English-looking words
- Some grammatical structure

**Try different prompts:**
```bash
python 01_pretrain/generate.py --prompt "ROMEO:"
python 01_pretrain/generate.py --prompt "The king"
```

---

### Phase 1 Checkpoint

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKPOINT: What you learned in Phase 1
━━━━━━━━���━━━━━━━━━━━━━━��━━━━���━━━━━━━��━━━━━━━━━━━━━

YOU BUILT:
  A 10.8M parameter transformer from scratch that generates
  Shakespeare-like text. It learned language purely from statistics —
  no grammar rules, no dictionaries, no human guidance.

THE KEY INSIGHT:
  The training loop is just: predict next token → measure error →
  adjust weights → repeat. That's ALL training is. Every LLM does this.

WHAT THE MODEL LEARNED:
  - Character frequencies (e > z)
  - Common letter pairs (th, he, in)
  - Word boundaries (spaces after words)
  - Dialogue format (NAME: text)
  - Basic grammar and sentence structure

WHAT IT DID NOT LEARN:
  - How to follow instructions
  - How to answer questions
  - Any factual knowledge
  - When to stop generating

  → Phase 2 fixes the "following instructions" problem.
━━━━━━━���━━━━━━━━━━━━���━━━━━━━��━━━━━━���━━━━━━━━━━━━━━
```

---

## Phase 2: Supervised Fine-Tuning (SFT)

### The Goal

Take a pretrained model that "knows English" and teach it to follow instructions. This is the step that turns a text completion engine into an assistant.

**Analogy:** Phase 1 was like teaching someone to read. Phase 2 is like teaching them to be a helpful teacher — when someone asks a question, give a clear answer.

### Why Not Use NanoGPT?

Our Phase 1 model is too small (10.8M params) for instruction-following. Instead, we use **SmolLM2-135M** — a HuggingFace model pretrained on ~1 trillion tokens. It already knows English, facts, and world knowledge. We just need to teach it the instruction → response pattern.

---

### Step 2.1: Prepare the Data

**Read first:** Open `02_sft/data.py`. 

**What it does:** Downloads Databricks Dolly-15k (15,000 human-written instruction/response pairs), samples 3,000 for tractable training time, and formats each into a template:

```
### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

**Why the template?** The model learns: "When I see `### Instruction:`, a question follows. When I see `### Response:`, I should generate a helpful answer." This is a lightweight way to bake "prompt engineering" into the model's weights.

**Run it:**
```bash
python 02_sft/data.py
```

**What you should see:**
```
Train: 2,700 examples
Val: 300 examples
```

---

### Step 2.2: Understand the Training Setup

**Read:** Open `02_sft/train.py`. Key differences from Phase 1:

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Starting weights | Random | Pretrained (knows English) |
| Learning rate | 3e-4 (aggressive) | 2e-5 (gentle — 15x smaller!) |
| Why? | Learning from scratch | Preserving pretrained knowledge |
| Batch size | 32 real | 1 real × 16 gradient accumulation |
| Why? | Fits in memory | 16GB can't fit larger batches |
| Trainer | Manual loop | HuggingFace SFTTrainer |

**Gradient accumulation explained:**
We want an effective batch size of 16, but can't fit 16 examples in memory at once. So we process 1 example, accumulate its gradient, repeat 16 times, then update weights. Mathematically identical to batch size 16.

*Analogy: You can't carry 64 bags of groceries at once, so you make 4 trips of 16.*

---

### Step 2.3: Train

**Run it:**
```bash
python 02_sft/train.py
```

**This takes ~40 minutes.** Watch for:
- Loss starting around 2.7 and decreasing
- Eval loss (every 100 steps) tracking close to train loss
- Generated samples at the end — these should be coherent instruction-following responses

**After training, you'll see 5 generated samples.** Compare them mentally to what NanoGPT would produce — the difference is dramatic.

---

### Step 2.4: Evaluate

**Run it:**
```bash
python 02_sft/evaluate.py
```

**What it does:** Loads both the original SmolLM2-135M (base) and your fine-tuned version, runs the same prompts through both, and compares.

**What to look for in the output:**

1. **Perplexity comparison:** SFT model should have lower perplexity on instruction-formatted text (it's been trained on this format).

2. **Side-by-side generation:** This is the most revealing part:
   - **Base model:** When asked "Explain gravity," it might repeat the prompt or generate more questions. It's a completion model — it continues text, it doesn't answer questions.
   - **SFT model:** Gives a direct answer. "Gravity is the force that keeps everything in place on Earth."

**Results are saved to** `outputs/sft_eval.txt`

---

### Phase 2 Checkpoint

```
━━━━━━━━━━━━━━━━━━━━━━━━��━━━━━━━━━━━━━━━━━━━━━���━━━
CHECKPOINT: What you learned in Phase 2
━━━━━━━━━━━━━━━━━━━━��━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOU LEARNED:
  SFT transforms a text completion model into an instruction-following
  assistant. The training loop is IDENTICAL to Phase 1 — only the
  data format and starting weights change.

THE KEY INSIGHT:
  A pretrained model already "knows" how to answer questions — the
  knowledge is in its weights from pretraining on the internet. SFT
  just teaches it the FORMAT: "when you see a question, produce an answer."

THIS IS HOW CHATGPT WAS CREATED:
  GPT-3 (pretrained) → InstructGPT (SFT on human demonstrations)
  That's literally it. (Plus RLHF, which is Phase 4 in the real pipeline.)

WHAT'S STILL MISSING:
  The model can follow general instructions, but it has no specialized
  domain knowledge. Ask it a medical question and it gives a generic,
  surface-level answer.

  → Phase 3 fixes this with LoRA domain adaptation.
━━━━━━━━��━━━━━━━━━━━━━━��━━━━━━��━━━━━━━━━━━━━━━━━━━
```

---

## Phase 3: LoRA / PEFT Fine-Tuning

### The Goal

Add medical domain expertise to the SFT model by training only 0.68% of its parameters. This is **LoRA (Low-Rank Adaptation)** — the most important technique in modern LLM customization.

### Why Not Just Fine-Tune Again?

Three problems with full fine-tuning for every domain:

1. **Storage:** 270 MB per domain × 50 domains = 13.5 GB of duplicate models
2. **Catastrophic forgetting:** Full fine-tuning on medical data might destroy general instruction-following ability
3. **Speed:** Training all 135M parameters is slow and expensive

**LoRA's solution:** Freeze all 135M parameters. Add tiny adapter matrices (~921K params) next to the attention layers. Train only the adapters. Result: a 3.5 MB file that adds medical expertise to any compatible base model.

**Analogy:** Full fine-tuning is repainting the entire house for each new color scheme. LoRA is applying removable wallpaper to specific rooms. Same house, swappable styles.

---

### Step 3.1: Prepare Medical Data

**Read:** Open `03_lora/data.py`.

**What it does:** Downloads MedAlpaca medical flashcards (~33,000 Q&A pairs), subsamples to 3,500, and formats each as:

```
### Medical Question:
What is hypertension?

### Answer:
Hypertension is a condition characterized by persistently elevated blood pressure...
```

**Why a different template than Phase 2?** Using `### Medical Question:` instead of `### Instruction:` creates a domain signal. The model learns: "When I see this template, activate medical knowledge."

**Run it:**
```bash
python 03_lora/data.py
```

---

### Step 3.2: Understand LoRA

**Read carefully:** Open `03_lora/train.py`. The `# LEARN:` comments explain the math.

**The key configuration:**
```python
LoraConfig(
    r=16,                              # Rank (compression level)
    lora_alpha=32,                     # Scaling factor (volume knob)
    target_modules=["q_proj", "v_proj"], # Where to add adapters
    lora_dropout=0.05,                 # Regularization
)
```

**What rank means:**
Normal weight update: ΔW is a 384×384 matrix = 147,456 parameters
LoRA: ΔW = A(384×16) × B(16×384) = 12,288 parameters (12x fewer!)

Think of it like JPEG compression:
- r=1: Very compressed, loses detail
- r=16: Good balance (what we use)
- r=64: Barely compressed
- r=384: No compression = full fine-tuning

**What alpha means:**
The adapter output is scaled by alpha/r = 32/16 = 2x. It's a volume knob for the adapter's influence. Higher alpha = adapter contributes more to the output.

**After `get_peft_model()` runs, look at the printed stats:**
```
Total params:     135,436,608
Trainable params:     921,600
Efficiency:       0.68% of model is being trained
```

Only 0.68%. That's the power of LoRA.

---

### Step 3.3: Train

**Run it:**
```bash
python 03_lora/train.py
```

**This takes ~80 minutes.** Watch the loss:
```
Step   10: loss 1.82  ← SFT model has no medical knowledge
Step  100: loss 1.40  ← Rapidly learning medical vocabulary
Step  500: loss 1.27  ← Core medical patterns acquired
Step 1970: loss 1.30  ← Training complete

Eval loss: 1.33       ← Close to train loss = no overfitting ✓
Adapter size: 3.5 MB  ← vs 270 MB for the full model!
```

**After training,** you'll see 5 medical question/answer samples. The model should give medically-specific answers (mentioning mechanisms, specific conditions, proper terminology).

---

### Step 3.4: Evaluate

**Run it:**
```bash
python 03_lora/evaluate.py
```

**What to look for:**

1. **Perplexity drop:** SFT=6.72 → LoRA=3.80 (43.5% improvement on medical text)

2. **Side-by-side answers (20 medical questions):**
   - SFT gives generic, surface-level answers and sometimes "leaks" into generating new questions
   - LoRA gives focused, medically-specific answers with proper terminology
   - Example: "Function of kidneys?" — SFT just names anatomy; LoRA explains the filtering mechanism

**Results saved to** `outputs/lora_eval.txt`

---

### Phase 3 Checkpoint

```
━���━━━━━━━━━━━━���━━━━��━━━━━���━━━━━━━━━━━━━━━━��━━━━━━━
CHECKPOINT: What you learned in Phase 3
━━━━━���━━━━━━━━━━━━━━���━━━━━━━━━━━━━━���━━━━━━━━━━━━━━

YOU LEARNED:
  LoRA adds domain expertise by training only 0.68% of parameters.
  The adapter is 3.5 MB — you could have 100 domain adapters
  (medical, legal, coding, finance) for the cost of one full model.

THE MATH:
  Full fine-tuning: ΔW = 384×384 = 147,456 params per layer
  LoRA (r=16):      ΔW = A(384×16) × B(16×384) = 12,288 params (12x fewer)

WHY IT WORKS:
  The pretrained model already has a rich representation of language.
  Fine-tuning only needs small, structured adjustments — not a complete
  rewrite. These adjustments naturally live in a low-dimensional
  subspace, which LoRA exploits.

THE PIPELINE IS COMPLETE:
  Random weights → Pretrain (learn language) → SFT (learn instructions)
  → LoRA (learn domain) → Ready for deployment

  This is how production LLMs are built. The only differences at
  scale are: more data, more parameters, and an RLHF/DPO step
  for alignment (teaching the model to be helpful and harmless).
━━━━━━━━━━━━━━━━━━━━━━━━━━━���━━━��━━━━━���━━━━━━━���━━━━
```

---

## Phase 4: Cross-Phase Evaluation

### Step 4.1: Compare All Three Models

```bash
python eval/compare_all.py
```

This runs the same prompts through all three models and prints a side-by-side comparison table. You'll see:
- NanoGPT: Generates Shakespeare-like text regardless of the prompt
- SFT: Follows instructions but gives generic answers
- SFT+LoRA: Gives domain-specific medical answers

### Step 4.2: Visualize Loss Curves

```bash
python eval/plot_curves.py
```

This pulls metrics from MLflow and plots training/validation loss curves for all three phases. The plot is saved to `outputs/loss_curves.png`.

### Step 4.3: Explore in MLflow

```bash
# Launch the experiment tracking UI
mlflow server --backend-store-uri ./mlruns \
  --host 127.0.0.1 --port 5000 \
  --disable-security-middleware &
```

Open **http://127.0.0.1:5000** in your browser (use 127.0.0.1, not localhost).

See [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md) for a detailed walkthrough of what to explore.

---

## What to Do Next

Now that you understand the pipeline, here are ways to go deeper:

### Experiment with Hyperparameters
- Change LoRA rank: r=4 vs r=16 vs r=64. How does it affect quality and adapter size?
- Change learning rate: What happens if you use SFT's learning rate (2e-5) for LoRA?
- Change target modules: Add "k_proj" and "o_proj" — does it help?

### Try Different Domains
- Replace MedAlpaca with a legal dataset, coding dataset, or finance dataset
- The LoRA code is domain-agnostic — just change the data

### Scale Up
- Try SmolLM2-360M or SmolLM2-1.7B instead of 135M
- Warning: larger models need more memory. Reduce max_length or batch_size.

### Read the Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The transformer (2017)
- [LoRA](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation (2021)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT / RLHF (2022)

---

## Glossary

| Term | Definition |
|------|-----------|
| **Attention** | Mechanism where each token looks at all previous tokens to decide what's relevant |
| **Autoregressive** | Generating one token at a time, left to right, each conditioned on all previous tokens |
| **Backpropagation** | Algorithm that computes gradients (how to adjust each weight to reduce loss) |
| **BPE** | Byte Pair Encoding — a subword tokenization algorithm used by most modern LLMs |
| **Causal mask** | Prevents tokens from attending to future positions (no cheating during training) |
| **Cross-entropy** | Loss function that measures how surprised the model is by the correct answer |
| **Embedding** | A learned vector representation of a token (word, character, or subword) |
| **Epoch** | One complete pass through the training data |
| **Fine-tuning** | Continuing training on a new dataset, starting from pretrained weights |
| **Gradient** | The direction and magnitude of change needed for each weight to reduce loss |
| **Gradient accumulation** | Processing small batches and accumulating gradients before updating weights |
| **LM Head** | The final layer that converts hidden states to vocabulary-size predictions |
| **LoRA** | Low-Rank Adaptation — adds tiny trainable matrices to a frozen model |
| **Loss** | A number measuring how wrong the model's predictions are (lower = better) |
| **MLP** | Multilayer Perceptron (also called Feed-Forward Network) — processes information after attention |
| **MPS** | Metal Performance Shaders — Apple's GPU computing framework for Apple Silicon |
| **Overfitting** | When a model memorizes training data instead of learning general patterns |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term for methods like LoRA |
| **Perplexity** | e^loss — "how many equally-likely options the model sees at each token" |
| **Pre-training** | Initial training on massive text data to learn language |
| **Rank (r)** | In LoRA, the dimensionality of the adapter matrices (compression level) |
| **Residual connection** | Shortcut that adds a layer's input to its output (x + f(x)) |
| **SFT** | Supervised Fine-Tuning — teaching a model to follow instructions |
| **Temperature** | Controls randomness during text generation (0=deterministic, 1=varied) |
| **Token** | The basic unit of text the model processes (character, subword, or word) |
| **Top-K** | During generation, only consider the K most likely next tokens |
| **Transformer** | The neural network architecture behind all modern LLMs |
| **Weight decay** | Regularization that slowly shrinks weights toward zero to prevent overfitting |
