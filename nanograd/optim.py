"""Gradient-descent optimizers.

An optimizer owns a list of parameters and, on each :meth:`step`, updates
``param.data`` in place using the gradient stored in ``param.grad`` (which was
filled in by ``loss.backward()``).
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Tensor


class Optimizer:
    """Base class: holds parameters and can reset their gradients."""

    def __init__(self, params: Iterable[Tensor]):
        self.params: List[Tensor] = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            if self.momentum:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]
            p.data -= self.lr * grad


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2015) with bias-corrected moments."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# --------------------------------------------------------------------------- #
# Learning-rate schedulers
# --------------------------------------------------------------------------- #
class _Scheduler:
    """Base scheduler. Call :meth:`step` once per epoch to update ``opt.lr``."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def get_lr(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()


class StepLR(_Scheduler):
    """Multiply the learning rate by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class CosineAnnealingLR(_Scheduler):
    """Anneal the learning rate from ``base_lr`` to ``eta_min`` over ``T_max``."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        cosine = (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * cosine
