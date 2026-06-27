"""Train a tiny Transformer (built on nanograd) to sort sequences of digits.

The model is a decoder-only Transformer with causal self-attention. Given an
unsorted sequence followed by a separator position, it learns to emit the sorted
sequence one token at a time — a clean algorithmic task whose attention patterns
are interpretable.

Run from the repo root:

    python examples/sort_transformer.py

Saves attention-map heatmaps and a training-accuracy curve into ``assets/``.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import nanograd as ng  # noqa: E402
from nanograd import nn, optim, utils  # noqa: E402
from nanograd.attention import TinyTransformer  # noqa: E402
from nanograd.tensor import Tensor  # noqa: E402

import _plot  # noqa: E402

LENGTH = 6        # sequence length to sort
NUM_DIGITS = 3    # vocabulary: digits 0..NUM_DIGITS-1
EPOCHS = 30
BATCH_SIZE = 64
N_TRAIN = 4000


def main():
    ng.manual_seed(0)
    block_size = 2 * LENGTH - 1

    x_train, y_train = utils.make_sort_dataset(N_TRAIN, LENGTH, NUM_DIGITS)

    model = TinyTransformer(
        vocab_size=NUM_DIGITS, block_size=block_size,
        d_model=64, n_heads=4, n_layers=2, d_ff=128, causal=True,
    )
    opt = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-4)

    acc_hist = []
    for epoch in range(EPOCHS):
        for xb, yb in utils.iterate_minibatches(x_train, y_train, BATCH_SIZE):
            opt.zero_grad()
            loss = nn.cross_entropy_seq(model(xb), yb)
            loss.backward()
            opt.step()
        scheduler.step()

        exact, per_token = utils.sort_accuracy(model, 256, LENGTH, NUM_DIGITS)
        acc_hist.append(exact)
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:3d}  loss {loss.data:.4f}  "
                  f"exact {exact:.3f}  per-token {per_token:.3f}")

    exact, per_token = utils.sort_accuracy(model, 2000, LENGTH, NUM_DIGITS)
    print(f"final  exact-match {exact:.3f}  per-token {per_token:.3f}")

    # ---- visualize attention for one example ----
    sample_x, _ = utils.make_sort_dataset(1, LENGTH, NUM_DIGITS)
    model(sample_x)  # populates each block's cached attention
    maps = [m[0] for m in model.attention_maps()]  # batch index 0 -> (heads, T, T)
    tokens = sample_x[0].tolist()

    assets = os.path.join(ROOT, "assets")
    os.makedirs(assets, exist_ok=True)
    _plot.save_attention_grid(
        maps, tokens, os.path.join(assets, "sort_attention.png"),
        "Causal self-attention while sorting  (input: "
        f"{tokens[:LENGTH]} -> sorted)",
    )
    _plot.save_loss_curve(
        acc_hist, os.path.join(assets, "sort_accuracy.png"),
        "Sort task — exact-match accuracy per epoch",
    )
    print("saved figures to assets/")


if __name__ == "__main__":
    main()
