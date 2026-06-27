"""Correctness tests for the autodiff engine.

The key tests use *gradient checking*: for every operation we compare the
analytical gradient produced by ``Tensor.backward`` against a numerical
gradient estimated with central finite differences. If the engine's chain-rule
bookkeeping is wrong, these tests fail.
"""

import numpy as np
import pytest

from nanograd import Tensor

rng = np.random.default_rng(0)


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
    """Check analytical vs numerical grads of ``build(*tensors).sum()``."""
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


# --------------------------------------------------------------------------- #
# Elementwise ops
# --------------------------------------------------------------------------- #
def test_add():
    check_grad(lambda a, b: a + b, rng.standard_normal((3, 4)), rng.standard_normal((3, 4)))


def test_sub():
    check_grad(lambda a, b: a - b, rng.standard_normal((3, 4)), rng.standard_normal((3, 4)))


def test_mul():
    check_grad(lambda a, b: a * b, rng.standard_normal((3, 4)), rng.standard_normal((3, 4)))


def test_div():
    a = rng.standard_normal((3, 4))
    b = rng.standard_normal((3, 4)) + 2.0  # keep denominator away from zero
    check_grad(lambda x, y: x / y, a, b)


def test_pow():
    check_grad(lambda a: a ** 3, rng.standard_normal((3, 4)))


# --------------------------------------------------------------------------- #
# Broadcasting
# --------------------------------------------------------------------------- #
def test_broadcast_add_row():
    check_grad(lambda a, b: a + b, rng.standard_normal((4, 5)), rng.standard_normal((5,)))


def test_broadcast_mul_col():
    check_grad(lambda a, b: a * b, rng.standard_normal((4, 5)), rng.standard_normal((4, 1)))


# --------------------------------------------------------------------------- #
# Matmul + reductions
# --------------------------------------------------------------------------- #
def test_matmul():
    check_grad(lambda a, b: a @ b, rng.standard_normal((3, 4)), rng.standard_normal((4, 5)))


def test_sum_axis():
    check_grad(lambda a: a.sum(axis=1), rng.standard_normal((3, 4)))


def test_mean_all():
    check_grad(lambda a: a.mean(), rng.standard_normal((3, 4)))


def test_mean_axis_keepdims():
    check_grad(lambda a: a.mean(axis=0, keepdims=True), rng.standard_normal((3, 4)))


# --------------------------------------------------------------------------- #
# Unary math + activations
# --------------------------------------------------------------------------- #
def test_exp():
    check_grad(lambda a: a.exp(), rng.standard_normal((3, 4)))


def test_log():
    check_grad(lambda a: a.log(), np.abs(rng.standard_normal((3, 4))) + 0.5)


def test_relu():
    # Offset away from 0 so finite differences don't straddle the kink.
    x = rng.standard_normal((3, 4))
    x[np.abs(x) < 1e-2] = 0.5
    check_grad(lambda a: a.relu(), x)


def test_tanh():
    check_grad(lambda a: a.tanh(), rng.standard_normal((3, 4)))


def test_sigmoid():
    check_grad(lambda a: a.sigmoid(), rng.standard_normal((3, 4)))


# --------------------------------------------------------------------------- #
# Shape manipulation
# --------------------------------------------------------------------------- #
def test_reshape():
    check_grad(lambda a: a.reshape(6, 2), rng.standard_normal((3, 4)))


def test_transpose():
    check_grad(lambda a: a.transpose(), rng.standard_normal((3, 4)))


# --------------------------------------------------------------------------- #
# Composite graph: a numerically-stable softmax cross-entropy by hand.
# Exercises broadcasting, exp/log, sum over an axis and reuse of a node.
# --------------------------------------------------------------------------- #
def test_softmax_cross_entropy_composite():
    logits_np = rng.standard_normal((5, 3))
    labels = rng.integers(0, 3, size=5)
    onehot = np.zeros((5, 3))
    onehot[np.arange(5), labels] = 1.0

    def build(logits):
        maxv = Tensor(logits.data.max(axis=1, keepdims=True), requires_grad=False)
        shifted = logits - maxv
        logsumexp = shifted.exp().sum(axis=1, keepdims=True).log()
        log_probs = shifted - logsumexp
        picked = (log_probs * Tensor(onehot, requires_grad=False)).sum(axis=1)
        return -picked.mean()

    check_grad(build, logits_np)


# --------------------------------------------------------------------------- #
# Engine plumbing
# --------------------------------------------------------------------------- #
def test_accumulates_gradient_for_reused_node():
    # If x feeds two paths, grads must add up: y = x*x + x  =>  dy/dx = 2x + 1
    x = Tensor(np.array([3.0]))
    y = x * x + x
    y.backward()
    assert np.allclose(x.grad, 2 * 3.0 + 1.0)


def test_constants_get_no_gradient():
    c = Tensor(np.array([2.0, 3.0]), requires_grad=False)
    x = Tensor(np.array([4.0, 5.0]))
    (c * x).sum().backward()
    assert np.allclose(c.grad, 0.0)
    assert np.allclose(x.grad, c.data)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
