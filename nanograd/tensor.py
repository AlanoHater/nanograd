"""Reverse-mode automatic differentiation over NumPy arrays.

This module implements a small :class:`Tensor` type that records every
operation performed on it in a dynamic computation graph and can compute the
gradient of a scalar output with respect to every tensor in that graph using
backpropagation (the chain rule).

The design is intentionally tiny and readable. Each operation:

1. computes the forward result as a plain NumPy array;
2. wraps it in a new ``Tensor`` that remembers its parents; and
3. registers a local ``_backward`` closure that knows how to push the incoming
   gradient (``out.grad``) to each parent.

Calling :meth:`Tensor.backward` on a scalar walks the graph once in reverse
topological order and invokes those closures, accumulating gradients in
``Tensor.grad``.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int, list]


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` by summing over broadcasted axes.

    NumPy broadcasting lets operands with different shapes combine, so the
    gradient flowing back through such an operation has the *broadcasted* shape
    and must be summed back down to the shape of the original operand.
    """
    # Sum away any leading axes that broadcasting added.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum (keeping the dim) over axes that were stretched from length 1.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional array that tracks gradients.

    Parameters
    ----------
    data:
        Array-like values held by the tensor. Stored internally as ``float64``.
    _children:
        Internal. The tensors this one was derived from (its parents in the
        computation graph).
    _op:
        Internal. A short label describing the producing operation (handy when
        printing/debugging the graph).
    requires_grad:
        If ``False``, gradients are not accumulated for this tensor. Useful for
        constants so backprop can skip them.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data: ArrayLike, _children: tuple = (), _op: str = "",
                 requires_grad: bool = True):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(x: ArrayLike) -> "Tensor":
        """Wrap a scalar/array constant as a non-differentiable Tensor."""
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    # ------------------------------------------------------------------ #
    # Binary elementwise operations (broadcasting-aware)
    # ------------------------------------------------------------------ #
    def __add__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(
                    -self.data / (other.data ** 2) * out.grad, other.data.shape
                )

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    out.grad @ other.data.swapaxes(-1, -2), self.data.shape
                )
            if other.requires_grad:
                other.grad += _unbroadcast(
                    self.data.swapaxes(-1, -2) @ out.grad, other.data.shape
                )

        out._backward = _backward
        return out

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        assert isinstance(power, (int, float)), "only constant powers are supported"
        out = Tensor(self.data ** power, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")
        count = self.data.size / out.data.size

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += (np.ones_like(self.data) * grad) / count

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Unary math
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Activations
    # ------------------------------------------------------------------ #
    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Shape manipulation
    # ------------------------------------------------------------------ #
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes=None) -> "Tensor":
        out = Tensor(np.transpose(self.data, axes), (self,), "transpose")

        def _backward():
            if not self.requires_grad:
                return
            if axes is None:
                self.grad += np.transpose(out.grad)
            else:
                inverse = np.argsort(axes)
                self.grad += np.transpose(out.grad, inverse)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically-stable softmax along ``axis``.

        Built entirely from primitive ops, so its gradient comes for free from
        the engine. The max is subtracted as a constant for stability.
        """
        shift = Tensor(self.data.max(axis=axis, keepdims=True), requires_grad=False)
        exp = (self - shift).exp()
        return exp / exp.sum(axis=axis, keepdims=True)

    # ------------------------------------------------------------------ #
    # Reflected / unary operators
    # ------------------------------------------------------------------ #
    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-self._coerce(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) + (-self)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) / self

    # ------------------------------------------------------------------ #
    # Backpropagation
    # ------------------------------------------------------------------ #
    def backward(self) -> None:
        """Backpropagate from this (scalar) tensor through the whole graph.

        Builds a reverse topological ordering of the graph rooted at ``self``
        and invokes each node's local ``_backward`` once, accumulating the
        result into every ``Tensor.grad``.
        """
        topo = []
        visited = set()

        def build(node: "Tensor"):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        # Seed: d(self)/d(self) = 1 for every element of the root.
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
