"""The Transformer attention mechanism, built on the nanograd autodiff engine.

This is where nanograd pays off: because every operation below is expressed
with :class:`~nanograd.tensor.Tensor`, the gradients for multi-head attention,
the residual blocks and the whole model are produced automatically by
``loss.backward()`` — there is no hand-written backward pass anywhere.

Contents
--------
- :class:`MultiHeadSelfAttention` — scaled dot-product attention with multiple
  heads and an optional causal mask.
- :class:`TransformerBlock` — pre-norm residual block (attention + MLP).
- :class:`TinyTransformer` — a minimal decoder-only model for fixed-length
  sequences.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import nn
from ._random import rng
from .tensor import Tensor


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    For each head ``h`` the layer computes

        Attention(Q, K, V) = softmax(Q Kᵀ / √d_head + mask) V

    over separate learned projections of the input, then concatenates the heads
    and projects back to ``d_model``. With ``causal=True`` a triangular mask
    prevents each position from attending to later positions.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool = True):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.last_attn: Optional[np.ndarray] = None  # cached for visualization

    def _split_heads(self, x: Tensor, b: int, t: int) -> Tensor:
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        return x.reshape(b, t, self.n_heads, self.d_head).transpose((0, 2, 1, 3))

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self._split_heads(self.q_proj(x), b, t)
        k = self._split_heads(self.k_proj(x), b, t)
        v = self._split_heads(self.v_proj(x), b, t)

        scale = 1.0 / np.sqrt(self.d_head)
        scores = (q @ k.transpose((0, 1, 3, 2))) * scale  # (B, H, T, T)

        if self.causal:
            mask = np.triu(np.ones((t, t)), k=1) * -1e9  # 0 on/below diagonal
            scores = scores + Tensor(mask.reshape(1, 1, t, t), requires_grad=False)

        attn = scores.softmax(axis=-1)        # (B, H, T, T)
        self.last_attn = attn.data            # detached numpy copy for plots
        context = attn @ v                    # (B, H, T, d_head)
        merged = context.transpose((0, 2, 1, 3)).reshape(b, t, self.d_model)
        return self.out_proj(merged)

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            params += layer.parameters()
        return params

    def _child_modules(self) -> List[nn.Module]:
        return [self.q_proj, self.k_proj, self.v_proj, self.out_proj]


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal: bool = True):
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def parameters(self) -> List[Tensor]:
        return (self.ln1.parameters() + self.attn.parameters()
                + self.ln2.parameters() + self.ff.parameters())

    def _child_modules(self) -> List[nn.Module]:
        return [self.ln1, self.attn, self.ln2, self.ff]


class TinyTransformer(nn.Module):
    """A minimal decoder-only Transformer for fixed-length integer sequences.

    Token embeddings plus learned positional embeddings feed a stack of
    causal Transformer blocks; a final linear head produces per-position logits
    over the vocabulary. The sequence length is fixed to ``block_size`` to keep
    the implementation tiny (no need for variable-length position slicing).
    """

    def __init__(self, vocab_size: int, block_size: int, d_model: int = 48,
                 n_heads: int = 3, n_layers: int = 2, d_ff: int = 96,
                 causal: bool = True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = Tensor(rng().standard_normal((block_size, d_model)) * 0.02)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, causal)
                       for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx) -> Tensor:
        idx = np.asarray(idx).astype(int)
        b, t = idx.shape
        assert t == self.block_size, (
            f"this minimal model expects fixed length {self.block_size}, got {t}")
        x = self.token_emb(idx) + self.pos_emb          # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                             # (B, T, vocab_size)

    def attention_maps(self) -> List[np.ndarray]:
        """Attention weights ``(B, n_heads, T, T)`` cached from the last forward."""
        return [block.attn.last_attn for block in self.blocks]

    def parameters(self) -> List[Tensor]:
        params = self.token_emb.parameters() + [self.pos_emb]
        for block in self.blocks:
            params += block.parameters()
        params += self.ln_f.parameters() + self.head.parameters()
        return params

    def _child_modules(self) -> List[nn.Module]:
        return [self.token_emb, *self.blocks, self.ln_f, self.head]
