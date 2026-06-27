"""Tests for the attention layers and the TinyTransformer.

Gradient checks confirm that the new ops the Transformer relies on (softmax,
batched/4-D matmul, LayerNorm, embedding lookup) differentiate correctly. The
remaining tests check the attention mechanics (shape, causal masking, rows that
sum to one) and that the full model can learn an algorithmic task.
"""

import numpy as np
import pytest

import nanograd as ng
from nanograd import nn, optim, utils
from nanograd.attention import (MultiHeadSelfAttention, TinyTransformer,
                                TransformerBlock)
from nanograd.tensor import Tensor

from _gradcheck import check_grad

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Gradient checks for the ops attention is built from
# --------------------------------------------------------------------------- #
def test_softmax_gradient():
    weights = rng.standard_normal((4, 5))  # makes the summed output non-trivial
    check_grad(lambda x: x.softmax(axis=-1) * Tensor(weights, requires_grad=False),
               rng.standard_normal((4, 5)))


def test_batched_matmul_3d_3d():
    check_grad(lambda a, b: a @ b, rng.standard_normal((2, 3, 4)),
               rng.standard_normal((2, 4, 5)))


def test_batched_matmul_3d_2d():
    check_grad(lambda a, b: a @ b, rng.standard_normal((2, 3, 4)),
               rng.standard_normal((4, 5)))


def test_batched_matmul_4d():
    check_grad(lambda a, b: a @ b, rng.standard_normal((2, 3, 4, 5)),
               rng.standard_normal((2, 3, 5, 6)))


def test_transpose_4d_axes():
    weights = rng.standard_normal((2, 4, 3, 5))
    check_grad(lambda a: a.transpose((0, 2, 1, 3)) * Tensor(weights, requires_grad=False),
               rng.standard_normal((2, 3, 4, 5)))


def test_layernorm_gradient():
    ln = nn.LayerNorm(5)
    weights = rng.standard_normal((4, 5))
    check_grad(lambda x: ln(x) * Tensor(weights, requires_grad=False),
               rng.standard_normal((4, 5)))


def test_embedding_gradient():
    idx = np.array([[0, 2, 1], [1, 1, 0]])

    def build(w):
        onehot = np.eye(3)[idx]
        return Tensor(onehot, requires_grad=False) @ w

    check_grad(build, rng.standard_normal((3, 4)))


# --------------------------------------------------------------------------- #
# Attention mechanics
# --------------------------------------------------------------------------- #
def test_attention_output_shape():
    ng.manual_seed(0)
    mha = MultiHeadSelfAttention(16, 4, causal=True)
    out = mha(Tensor(rng.standard_normal((2, 5, 16))))
    assert out.shape == (2, 5, 16)


def test_attention_is_causal():
    ng.manual_seed(0)
    mha = MultiHeadSelfAttention(16, 4, causal=True)
    t = 5
    mha(Tensor(rng.standard_normal((2, t, 16))))
    attn = mha.last_attn  # (B, H, T, T)
    upper = np.triu(np.ones((t, t)), k=1).astype(bool)
    assert np.allclose(attn[:, :, upper], 0.0)       # no attention to the future
    assert np.allclose(attn.sum(axis=-1), 1.0)        # each row is a distribution


def test_attention_noncausal_can_see_future():
    ng.manual_seed(0)
    mha = MultiHeadSelfAttention(16, 4, causal=False)
    t = 5
    mha(Tensor(rng.standard_normal((1, t, 16))))
    attn = mha.last_attn
    upper = np.triu(np.ones((t, t)), k=1).astype(bool)
    assert attn[:, :, upper].max() > 0.0
    assert np.allclose(attn.sum(axis=-1), 1.0)


def test_transformer_block_residual_shape():
    ng.manual_seed(0)
    block = TransformerBlock(16, 4, 32)
    out = block(Tensor(rng.standard_normal((2, 6, 16))))
    assert out.shape == (2, 6, 16)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
def test_tiny_transformer_forward_and_gradients():
    ng.manual_seed(0)
    vocab, block = 3, 11
    model = TinyTransformer(vocab, block, d_model=32, n_heads=2, n_layers=2, d_ff=64)
    idx = rng.integers(0, vocab, size=(4, block))
    logits = model(idx)
    assert logits.shape == (4, block, vocab)

    logits.sum().backward()
    assert all(np.any(p.grad != 0) for p in model.parameters())

    maps = model.attention_maps()
    assert len(maps) == 2
    assert maps[0].shape == (4, 2, block, block)


def test_cross_entropy_seq_ignores_masked_positions():
    logits = Tensor(np.array([[[2.0, 0.5, 0.1], [0.2, 1.5, 0.3]]]))  # (1, 2, 3)
    targets = np.array([[-1, 1]])  # first position ignored
    loss = nn.cross_entropy_seq(logits, targets).data

    z = np.array([0.2, 1.5, 0.3])
    z = z - z.max()
    log_probs = z - np.log(np.exp(z).sum())
    assert np.isclose(loss, -log_probs[1])


def test_transformer_learns_to_sort():
    ng.manual_seed(0)
    length, vocab = 6, 3
    block = 2 * length - 1
    x_train, y_train = utils.make_sort_dataset(1500, length, vocab)

    model = TinyTransformer(vocab, block, d_model=32, n_heads=2, n_layers=2, d_ff=64)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(15):
        for xb, yb in utils.iterate_minibatches(x_train, y_train, 64):
            opt.zero_grad()
            nn.cross_entropy_seq(model(xb), yb).backward()
            opt.step()

    exact, per_token = utils.sort_accuracy(model, 256, length, vocab)
    assert per_token > 0.9, f"per-token accuracy too low: {per_token:.3f}"
    assert exact > 0.7, f"exact-match accuracy too low: {exact:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
