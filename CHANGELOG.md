# Changelog

All notable changes to this project are documented here.

## [0.3.0] — 2026-06-27

### Added
- `Tensor.softmax` (numerically stable) plus verified gradients for batched and
  4-D `matmul`.
- `nn.LayerNorm` and `nn.Embedding`.
- `nanograd.attention`: `MultiHeadSelfAttention` (with causal masking),
  `TransformerBlock` and `TinyTransformer` — a decoder-only Transformer built
  entirely on the autodiff engine.
- `nn.cross_entropy_seq` (sequence loss with `ignore_index`).
- `utils.make_sort_dataset` and `utils.sort_accuracy` for the sorting task.
- New example `examples/sort_transformer.py`: trains the Transformer to sort
  digit sequences (100% exact-match) and saves attention-map heatmaps.
- Attention/Transformer tests — **65 tests** total.

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
