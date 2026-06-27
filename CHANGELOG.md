# Changelog

All notable changes to this project are documented here.

## [0.2.0] — 2026-06-27

### Added
- `nn.BatchNorm1d` and `nn.Dropout` layers.
- `Module.train()` / `Module.eval()` modes, propagated through `Sequential`.
- Learning-rate schedulers: `optim.StepLR` and `optim.CosineAnnealingLR`.
- `utils.iterate_minibatches` for mini-batch training.
- New example `examples/digits_classification.py`: trains a deeper network on
  scikit-learn's `digits` dataset (~97% validation accuracy) with accuracy
  curves, a confusion matrix and sample-prediction visualizations.
- Additional tests for the new components — **51 tests** total.

## [0.1.0] — 2026-06-27

### Added
- Reverse-mode automatic differentiation `Tensor` engine in NumPy, with full
  broadcasting support.
- `nn`: `Linear`, `ReLU`, `Tanh`, `Sigmoid`, `Sequential`, plus MSE and
  numerically-stable softmax cross-entropy losses.
- `optim`: `SGD` (with momentum and weight decay) and `Adam`.
- Synthetic datasets and runnable examples (spiral, moons, sine regression)
  with saved visualizations.
- Gradient-checking test suite and GitHub Actions CI on Python 3.10–3.12.
