# nanograd

[![CI](https://github.com/AlanoHater/nanograd/actions/workflows/ci.yml/badge.svg)](https://github.com/AlanoHater/nanograd/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A tiny reverse-mode automatic differentiation engine and neural-network library, written from scratch in NumPy.**

`nanograd` implements the core machinery behind modern deep-learning frameworks
(PyTorch, TensorFlow, JAX) in a few hundred lines of readable Python: a dynamic
computation graph, backpropagation, common layers, loss functions and
optimizers. It depends only on NumPy. The goal is **understanding how deep
learning actually works under the hood** — not raw performance.

Every gradient in the engine is verified against numerical finite differences
(*gradient checking*), so the math is provably correct, not just plausible.

---

## Results

A multi-layer perceptron built with `nanograd`, trained purely with gradients
produced by the engine:

| Spiral classification (3 classes) | Sine regression |
|:---:|:---:|
| ![Spiral decision boundary](assets/spiral_decision_boundary.png) | ![Sine regression](assets/regression_sine.png) |
| **99.3%** train accuracy on a non-linearly-separable dataset | Fits a noisy sine wave down to the noise floor |

| Two-moons classification | Training loss |
|:---:|:---:|
| ![Moons decision boundary](assets/moons_decision_boundary.png) | ![Loss curve](assets/spiral_loss_curve.png) |

**Real data — hand-written digits** (8×8 images, 10 classes): a deeper network
with BatchNorm, Dropout and a cosine learning-rate schedule, trained with
mini-batches, reaches **~97% validation accuracy**.

| Predictions — `pred (true)` | Train vs validation accuracy |
|:---:|:---:|
| ![Digit predictions](assets/digits_predictions.png) | ![Digits accuracy](assets/digits_accuracy.png) |

All figures above are produced by the scripts in [`examples/`](examples/).

---

## Features

- **Reverse-mode autodiff** over n-dimensional NumPy arrays with full
  broadcasting support ([`tensor.py`](nanograd/tensor.py)).
- **Operations**: `+ - * / @ **`, `sum`, `mean`, `exp`, `log`, `reshape`,
  `transpose`, and activations `relu`, `tanh`, `sigmoid`.
- **Layers**: `Linear` (He/Xavier init), `ReLU`, `Tanh`, `Sigmoid`,
  `BatchNorm1d`, `Dropout`, and `Sequential`, with `train()` / `eval()` modes
  ([`nn.py`](nanograd/nn.py)).
- **Losses**: mean squared error and numerically-stable softmax cross-entropy.
- **Optimizers**: `SGD` (momentum + weight decay) and `Adam`, plus `StepLR`
  and `CosineAnnealingLR` schedulers ([`optim.py`](nanograd/optim.py)).
- **Training utilities**: synthetic datasets, mini-batch iteration,
  standardization and train/test split ([`utils.py`](nanograd/utils.py)).
- **Tested**: 51 tests including gradient checking for every operation and an
  end-to-end training test.

---

## Quick start

```bash
git clone https://github.com/AlanoHater/nanograd.git
cd nanograd
pip install -r requirements-dev.txt   # numpy, matplotlib, pytest

python -m pytest                      # run the 51-test suite
python examples/spiral_classification.py
python examples/digits_classification.py   # real data (needs scikit-learn)
```

### Train a neural network in a few lines

```python
import nanograd as ng
from nanograd import nn, optim, utils

ng.manual_seed(0)
x, y = utils.make_spiral(n_points=100, n_classes=3)   # non-linear toy data

model = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 3),
)
opt = optim.Adam(model.parameters(), lr=1e-2)

inputs = ng.Tensor(x)
for epoch in range(300):
    opt.zero_grad()
    loss = nn.cross_entropy(model(inputs), y)   # forward
    loss.backward()                             # backprop (autodiff)
    opt.step()                                  # gradient descent

print("accuracy:", utils.accuracy(model(inputs), y))
```

---

## How it works

`nanograd` builds a **computation graph** as you operate on tensors. Each
`Tensor` remembers the tensors it was produced from and a small `_backward`
closure implementing the chain rule for that single operation.

Take `z = x * y`. The forward value is `x * y`; the local derivatives are
`∂z/∂x = y` and `∂z/∂y = x`. When a gradient `∂L/∂z` arrives from downstream,
the closure routes it to the inputs:

```
x.grad += y * z.grad      # ∂L/∂x = ∂L/∂z · ∂z/∂x
y.grad += x * z.grad      # ∂L/∂y = ∂L/∂z · ∂z/∂y
```

Calling `loss.backward()` performs a **reverse topological traversal** of the
graph from the scalar loss, running each node's closure exactly once so every
parameter ends up with the correct accumulated gradient. The optimizer then
nudges each parameter against its gradient. That single loop —
*forward → backward → step* — is all of supervised deep learning.

Broadcasting is handled by summing gradients back down to each operand's
original shape (see `_unbroadcast` in [`tensor.py`](nanograd/tensor.py)).

---

## Why gradient checking matters

A from-scratch autodiff engine is only useful if its derivatives are correct.
For every operation, [`tests/test_tensor.py`](tests/test_tensor.py) compares the
analytical gradient from `backward()` against a numerical estimate using central
differences:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

If the chain-rule bookkeeping were wrong, these tests would fail. This is the
same technique used to debug real deep-learning frameworks.

---

## Project structure

```
nanograd/
├── nanograd/            # the library (NumPy only)
│   ├── tensor.py        # autodiff engine: Tensor + backward()
│   ├── nn.py            # layers (Linear, BatchNorm, Dropout, ...), losses
│   ├── optim.py         # SGD, Adam + LR schedulers
│   ├── utils.py         # datasets, mini-batches, metrics
│   └── _random.py       # reproducible RNG (manual_seed)
├── examples/            # runnable demos that produce the figures above
├── tests/               # 51 tests, incl. gradient checking
├── assets/              # generated figures
└── .github/workflows/   # CI: tests on Python 3.9 / 3.11 / 3.12
```

---

## Possible extensions

- Convolutional layers (`Conv2d`) via im2col
- Recurrent layers (RNN / LSTM)
- A scalar-valued "engine mode" to visualize the computation graph
- Loading larger datasets (e.g. the full MNIST)

---

## License

[MIT](LICENSE) © 2026 AlanoHater
