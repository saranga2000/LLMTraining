"""
NanoGPT — a minimal transformer language model from scratch.

Architecture (~10M params):
  - 6 transformer blocks, each containing:
    - Multi-head causal self-attention (6 heads)
    - Feedforward MLP (GELU activation)
    - Pre-norm (LayerNorm before each sub-layer)
  - Token + positional embeddings
  - Language model head (linear projection to vocab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    # 📚 LEARN: What is attention?
    # Think of it as a search engine inside each layer.
    #   Q (Query)  = "What am I looking for?"
    #   K (Key)    = "What do I contain?"
    #   V (Value)  = "What information do I provide?"
    #
    # For each token, we compute Q·K^T to get attention scores —
    # how relevant is every other token to this one?
    # Then we use those scores to take a weighted average of V.
    #
    # The "causal" part: we mask out future tokens so the model
    # can't cheat by looking ahead. Token 5 can only attend to
    # tokens 1-5, never to token 6+. This is what makes it
    # autoregressive — it generates one token at a time, left to right.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # All Q, K, V projections in one matrix multiply (efficiency trick)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)  # output projection
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 📚 LEARN: Causal mask (lower-triangular matrix)
        # This is a matrix of 1s below the diagonal and 0s above.
        # When we add -inf to the masked positions, softmax converts
        # those to 0 probability — the model literally cannot see the future.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Compute Q, K, V for all heads in one shot
        q, k, v = self.c_attn(x).split(C, dim=2)

        # Reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: (Q · K^T) / sqrt(d_k)
        # The scaling by sqrt(d_k) prevents the dot products from getting
        # too large, which would push softmax into regions with tiny gradients.
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Apply causal mask: set future positions to -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax converts scores to probabilities (0 to 1, summing to 1)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v  # (B, n_head, T, head_dim)

        # Concatenate heads back together
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Two-layer feedforward network with GELU activation.

    This is the "thinking" part of each transformer block.
    Attention gathers information; the MLP processes it.
    The 4x expansion (n_embd -> 4*n_embd -> n_embd) gives the
    network more capacity to learn complex transformations.
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block: LayerNorm → Attention → LayerNorm → MLP

    # 📚 LEARN: Pre-norm vs post-norm
    # Original transformer: Attn → Add → LayerNorm (post-norm)
    # Modern (GPT-2+):      LayerNorm → Attn → Add   (pre-norm)
    # Pre-norm is more stable during training — gradients flow
    # better through the residual connections.

    # 📚 LEARN: Residual connections (the x + ... pattern)
    # Without residuals, deep networks suffer from vanishing gradients.
    # The "skip connection" lets gradients flow directly backward
    # through the addition, bypassing the attention/MLP layers.
    # It's like having an express lane on the highway.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # residual around attention
        x = x + self.mlp(self.ln_2(x))   # residual around MLP
        return x


class NanoGPT(nn.Module):
    """
    The full language model.

    # 📚 LEARN: nn.Embedding — what is an embedding table?
    # It's a lookup table: each token ID (0-64) maps to a learned vector.
    # At init, these vectors are random. During training, tokens that
    # appear in similar contexts develop similar vectors.
    # This is the "King - Man + Woman ≈ Queen" intuition from Word2Vec.
    #
    # We have TWO embedding tables:
    #   1. Token embeddings: what character is this? (vocab_size × n_embd)
    #   2. Position embeddings: where in the sequence? (block_size × n_embd)
    # These are added together so the model knows both WHAT and WHERE.
    """

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),      # token embeddings
            wpe=nn.Embedding(block_size, n_embd),       # position embeddings
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([
                Block(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]),
            ln_f=nn.LayerNorm(n_embd),                  # final layer norm
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head.
        # Intuition: the embedding maps tokens → vectors, and the LM head
        # maps vectors → token probabilities. They're doing inverse operations,
        # so sharing weights makes sense and saves parameters.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass: tokens in, logits (and optionally loss) out.

        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices (for training)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)     # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)     # (T, n_embd) — broadcast over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Project to vocabulary size to get logits (unnormalized scores)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 📚 LEARN: Cross-entropy loss
        # The model outputs a probability distribution over all 65 chars
        # for each position. Cross-entropy measures how far that distribution
        # is from the true answer (which is just 1.0 on the correct char).
        # Minimizing cross-entropy = maximizing the probability of the
        # correct next character. That's literally all training does.
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively, one token at a time.

        # 📚 LEARN: How generation works
        # 1. Feed the prompt through the model → get logits for next token
        # 2. Apply temperature: higher = more random, lower = more deterministic
        #    temperature=0.1: almost always picks the most likely char
        #    temperature=1.0: samples proportionally to learned probabilities
        #    temperature=2.0: very random, creative but often nonsensical
        # 3. Optionally apply top-k: only consider the k most likely chars
        # 4. Sample one token from the distribution
        # 5. Append it to the sequence, repeat
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence is too long
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last token's logits

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
