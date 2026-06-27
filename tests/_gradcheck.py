"""Shared numerical gradient-checking helpers for the test suite."""

import numpy as np

from nanograd import Tensor


def numerical_grad(f, x, eps=1e-6):
    """Estimate d f / d x with central differences (x is modified in place)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f(x)
        x[idx] = original - eps
        minus = f(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def check_grad(build, *inputs, atol=1e-5, rtol=1e-4):
    """Check analytical vs numerical grads of ``build(*tensors).sum()``.

    ``build`` maps one Tensor per input array to a single output Tensor.
    """
    tensors = [Tensor(x.copy()) for x in inputs]
    out = build(*tensors)
    out.sum().backward()

    for k in range(len(inputs)):
        def f(xk, k=k):
            args = [inp.copy() for inp in inputs]
            args[k] = xk
            return float(build(*[Tensor(a) for a in args]).data.sum())

        ng = numerical_grad(f, inputs[k].copy())
        assert np.allclose(tensors[k].grad, ng, atol=atol, rtol=rtol), (
            f"gradient mismatch for input {k}\n"
            f"analytical=\n{tensors[k].grad}\nnumerical=\n{ng}"
        )
