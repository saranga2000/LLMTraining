# NanoGPT Transformer — Line-by-Line Walkthrough

> This is a companion guide to `model.py`. Read this alongside the code.

---

## The Big Picture

Think of the whole model as a pipeline. Text goes in as characters, flows through layers that process meaning, and comes out as predictions for the next character.

```
Input: "To be or n"  (10 characters)
  | Token embedding: each char -> 384-dim vector
  | + Position embedding: add position info
  | Block 1: attend to context -> process with MLP
  | Block 2: attend deeper -> process
  | ... (6 blocks total)
  | Block 6: final refined representation
  | LM Head: 384-dim -> 65 logits
  | Softmax -> probabilities over 65 chars
Output: 'o' with probability 0.73  (predicting "not")
```

The model will do this ~1 million times during training, each time nudging its 10.77M parameters slightly to get better predictions. After 3000 iterations, it should produce recognizable Shakespeare-like text.

---

## 1. Embeddings — The Input Layer

```python
wte = nn.Embedding(vocab_size, n_embd)   # (65, 384)
wpe = nn.Embedding(block_size, n_embd)   # (256, 384)
```

**Token Embedding (`wte`)**: A lookup table with 65 rows (one per character) and 384 columns. When the model sees character `'a'` (ID 39), it looks up row 39 and gets a 384-dimensional vector. At initialization these vectors are random — during training, characters that behave similarly (like `'a'` and `'e'`, both vowels) end up with similar vectors.

**Position Embedding (`wpe`)**: Same idea, but for *position* in the sequence. Position 0 gets one vector, position 1 gets another, etc., up to position 255 (our max context length). This is how the model knows *where* a character is — without this, `"cat"` and `"tac"` would look identical.

```python
x = tok_emb + pos_emb   # (B, T, 384)
```

We **add** them together. Now each vector encodes both *what* the character is and *where* it sits. This 384-dim vector is the input to the transformer blocks.

---

## 2. CausalSelfAttention — The Core Innovation

This is what makes transformers powerful. The analogy: **a search engine inside each layer**.

```python
self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # one big matrix for Q, K, V
```

One linear layer projects each token's 384-dim vector into **three** vectors of size 384:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information should I pass along?"

```python
q, k, v = self.c_attn(x).split(C, dim=2)
```

Split the output into Q, K, V.

```python
q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
# shape: (B, 6, T, 64)
```

**Multi-head**: Instead of one big 384-dim attention, we split into **6 heads** of 64 dims each. Each head can attend to different things — one might track grammar, another might track rhyme patterns, another might track who is speaking. The model learns what each head specializes in.

```python
att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
```

**Attention scores**: For each pair of tokens, compute Q*K^T — this is a dot product that measures "how relevant is token j to token i?" The `* (64 ** -0.5)` scaling prevents the dot products from getting huge, which would make softmax produce near-one-hot distributions (killing gradients).

```python
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
```

**The causal mask** — this is the line that makes it a *language model*. `self.bias` is a lower-triangular matrix:
```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```
Where there's a 0, we fill with `-inf`. After softmax, `-inf` becomes 0 probability. So token 2 can attend to tokens 0, 1, 2 but NOT to token 3. **The model can never see the future.** This is what forces it to actually *learn* to predict rather than just copy.

```python
att = F.softmax(att, dim=-1)
y = att @ v
```

Softmax normalizes scores to probabilities (sum to 1.0). Then we take a weighted average of the **V** (value) vectors. If token 5 attends strongly to token 2 (high attention score), it pulls in token 2's value vector heavily. The output is a new representation that "knows about" relevant context.

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.resid_dropout(self.c_proj(y))
```

Concatenate all 6 heads back into a 384-dim vector, project through one more linear layer. Done.

---

## 3. MLP — The Thinking Layer

```python
self.c_fc = nn.Linear(n_embd, 4 * n_embd)    # 384 -> 1536
self.c_proj = nn.Linear(4 * n_embd, n_embd)   # 1536 -> 384
self.gelu = nn.GELU()
```

After attention gathers relevant context, the MLP *processes* it. It expands to 4x the dimension (giving more capacity), applies a non-linearity (GELU — a smooth version of ReLU), and projects back down. Think of attention as "collecting evidence" and MLP as "drawing conclusions."

---

## 4. Block — One Transformer Layer

```python
x = x + self.attn(self.ln_1(x))   # residual around attention
x = x + self.mlp(self.ln_2(x))    # residual around MLP
```

Two critical ideas here:

**LayerNorm** (`ln_1`, `ln_2`): Normalizes the vector to zero-mean, unit-variance. This stabilizes training — without it, values can drift to huge or tiny numbers across layers.

**Residual connections** (`x + ...`): The original `x` is added back after each sub-layer. Why? Two reasons:
1. **Gradient flow**: During backprop, gradients can flow directly through the `+` without being squished by attention/MLP operations. This prevents vanishing gradients in deep networks.
2. **Incremental refinement**: Each layer *adds* a small adjustment to the representation rather than replacing it entirely.

Our model stacks **6 of these blocks**. Each one refines the representation further.

---

## 5. LM Head — The Output

```python
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # 384 -> 65
self.transformer.wte.weight = self.lm_head.weight  # weight tying!
```

The final linear layer projects from 384 dims to 65 (one score per character in the vocabulary). These scores are called **logits** — they're unnormalized. Softmax converts them to probabilities.

**Weight tying**: The embedding table (`wte`) maps token IDs -> vectors. The LM head maps vectors -> token scores. These are conceptually inverse operations, so we share the same weight matrix. This saves ~25k parameters and acts as a regularizer.

---

## 6. Loss — What the Model Minimizes

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

Cross-entropy measures: "how surprised was the model by the actual next character?" If the model assigns probability 0.9 to the correct character, loss is low (-log(0.9) = 0.1). If it assigns 0.01, loss is high (-log(0.01) = 4.6). **Training = making the model less surprised by real text.**

---

## 7. Generation — Using the Trained Model

```python
logits = logits[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

1. Get predictions for the last position
2. Divide by **temperature** (>1 = more random, <1 = more conservative)
3. Sample one character from the probability distribution
4. Append it, repeat

---

## Key Analogies

- **Attention mechanism** -> A search engine inside each layer. Q=query, K=keys, V=values. You search your own context for relevant information.
- **Embeddings** -> A lookup table. Words/chars that appear in similar contexts get similar vectors. "King - Man + Woman = Queen" intuition.
- **Loss / cross-entropy** -> Compression. The model is learning to compress Shakespeare as efficiently as possible. Lower loss = better compression = more understanding.
- **Residual connections** -> Express lane on a highway. Lets gradients bypass complex operations.
- **LayerNorm** -> A thermostat. Keeps activations in a stable range so training doesn't blow up.
- **Multi-head attention** -> Multiple search engines, each specialized for different patterns (grammar, semantics, position, etc).
- **Temperature** -> A creativity knob. Low = safe/repetitive, high = wild/creative.

---

*Generated as part of LLM Training Lab — Phase 1*
