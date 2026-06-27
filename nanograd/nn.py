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

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """Return the list of learnable tensors in this module."""
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


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
