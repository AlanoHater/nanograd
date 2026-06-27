"""Neural-network building blocks built on top of :class:`~nanograd.tensor.Tensor`.

Everything here is just a thin, stateful wrapper around tensor operations:
modules hold parameters (``Tensor`` objects with ``requires_grad=True``) and
define a ``forward`` pass. Because the autodiff engine records the forward
operations, gradients for every parameter come for free from
``loss.backward()``.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ._random import rng
from .tensor import Tensor


class Module:
    """Base class for all layers and models."""

    training: bool = True

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Tensor]:
        """Return the list of learnable tensors in this module."""
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def _child_modules(self) -> List["Module"]:
        """Sub-modules to recurse into for train()/eval()."""
        return []

    def train(self, mode: bool = True) -> "Module":
        """Set training mode (affects Dropout and BatchNorm)."""
        self.training = mode
        for module in self._child_modules():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode (Dropout off, BatchNorm uses running stats)."""
        return self.train(False)


class Linear(Module):
    """A fully connected layer: ``y = x @ W + b``.

    Weights are initialised with He ("kaiming") or Xavier scaling so that the
    variance of activations is preserved across layers at the start of training.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "kaiming"):
        if init == "kaiming":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"unknown init {init!r}; use 'kaiming' or 'xavier'")

        w = rng().standard_normal((in_features, out_features)) * std
        self.weight = Tensor(w)
        self.bias = Tensor(np.zeros(out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self) -> List[Tensor]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Flatten(Module):
    """Flatten every dimension except the batch dimension into one vector."""

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)


class Sequential(Module):
    """Chain modules so the output of one is the input of the next."""

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def _child_modules(self) -> List[Module]:
        return self.layers


class Dropout(Module):
    """Inverted dropout: randomly zero a fraction ``p`` of activations.

    Active only in training mode. Surviving activations are scaled by
    ``1 / (1 - p)`` so the expected value is unchanged, which means no rescaling
    is needed at evaluation time.
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (rng().random(x.shape) >= self.p) / (1.0 - self.p)
        return x * Tensor(mask, requires_grad=False)


class BatchNorm1d(Module):
    """Batch normalization for 2-D inputs of shape ``(batch, num_features)``.

    Normalizes each feature using statistics computed over the batch during
    training, then applies a learnable scale (``gamma``) and shift (``beta``).
    At evaluation time it uses running estimates accumulated during training.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = Tensor(np.ones((1, num_features)))
        self.beta = Tensor(np.zeros((1, num_features)))
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            centered = x - mean
            var = (centered * centered).mean(axis=0, keepdims=True)
            # Update running statistics (detached from the graph).
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                 + self.momentum * mean.data)
            self.running_var = ((1 - self.momentum) * self.running_var
                                + self.momentum * var.data)
            normalized = centered / ((var + self.eps) ** 0.5)
        else:
            mean = Tensor(self.running_mean, requires_grad=False)
            std = Tensor(np.sqrt(self.running_var + self.eps), requires_grad=False)
            normalized = (x - mean) / std
        return self.gamma * normalized + self.beta

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class LayerNorm(Module):
    """Layer normalization over the last dimension.

    Unlike BatchNorm, statistics are computed per-sample across the features, so
    it behaves identically in training and evaluation — which is why it is the
    normalization of choice inside Transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / ((var + self.eps) ** 0.5)
        return self.gamma * normalized + self.beta

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Embedding(Module):
    """Lookup table mapping integer ids to dense vectors.

    The lookup is expressed as a one-hot matrix multiply so it reuses the
    engine's ``matmul`` (and therefore its gradient): the rows of the weight
    matrix that were selected receive the incoming gradient.
    """

    def __init__(self, num_embeddings: int, dim: int):
        self.num_embeddings = num_embeddings
        self.weight = Tensor(rng().standard_normal((num_embeddings, dim)) * 0.02)

    def forward(self, idx) -> Tensor:
        idx = np.asarray(idx).astype(int)
        onehot = np.eye(self.num_embeddings)[idx]  # (..., num_embeddings)
        return Tensor(onehot, requires_grad=False) @ self.weight  # (..., dim)

    def parameters(self) -> List[Tensor]:
        return [self.weight]


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #
def mse_loss(pred: Tensor, target) -> Tensor:
    """Mean squared error between predictions and targets."""
    target = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    diff = pred - target
    return (diff * diff).mean()


def cross_entropy(logits: Tensor, labels) -> Tensor:
    """Softmax cross-entropy loss for integer class labels.

    Implemented with the log-sum-exp trick for numerical stability. ``logits``
    has shape ``(N, C)`` and ``labels`` is an integer array of shape ``(N,)``.
    """
    n, c = logits.shape
    labels = np.asarray(labels).astype(int)

    # max along classes is treated as a constant (it does not change the
    # softmax, it only keeps exp() from overflowing).
    maxv = Tensor(logits.data.max(axis=1, keepdims=True), requires_grad=False)
    shifted = logits - maxv
    log_sum_exp = shifted.exp().sum(axis=1, keepdims=True).log()
    log_probs = shifted - log_sum_exp  # (N, C)

    onehot = np.zeros((n, c))
    onehot[np.arange(n), labels] = 1.0
    picked = (log_probs * Tensor(onehot, requires_grad=False)).sum(axis=1)  # (N,)
    return -picked.mean()


def cross_entropy_seq(logits: Tensor, targets, ignore_index: int = -1) -> Tensor:
    """Cross-entropy for sequences of shape ``(batch, time, vocab)``.

    ``targets`` is an integer array ``(batch, time)``. Positions equal to
    ``ignore_index`` contribute no loss (used to ignore the prompt tokens when
    training a model to produce only the answer part of a sequence).
    """
    b, t, vocab = logits.shape
    flat_logits = logits.reshape(b * t, vocab)
    flat_targets = np.asarray(targets).astype(int).reshape(b * t)

    valid = flat_targets != ignore_index
    safe = np.where(valid, flat_targets, 0)
    onehot = np.eye(vocab)[safe] * valid[:, None]  # zero rows for ignored tokens

    maxv = Tensor(flat_logits.data.max(axis=1, keepdims=True), requires_grad=False)
    shifted = flat_logits - maxv
    log_probs = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
    picked = (log_probs * Tensor(onehot, requires_grad=False)).sum(axis=1)
    return -(picked.sum()) / float(valid.sum())
