"""Classify hand-written digits (sklearn's 8x8 ``digits``) with a deep MLP.

This is the most realistic example: a real dataset, a train/validation split,
mini-batch training, BatchNorm + Dropout for regularization, and a cosine
learning-rate schedule -- all powered by nanograd's own autodiff.

Run from the repo root:

    python examples/digits_classification.py

Requires scikit-learn (only for loading the bundled dataset):

    pip install scikit-learn
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

import nanograd as ng  # noqa: E402
from nanograd import nn, optim, utils  # noqa: E402
from nanograd.tensor import Tensor  # noqa: E402

import _plot  # noqa: E402

EPOCHS = 40
BATCH_SIZE = 64


def main():
    ng.manual_seed(0)
    digits = load_digits()
    x_all = digits.data.astype(np.float64)   # (1797, 64)
    y_all = digits.target.astype(int)        # (1797,)

    # Split by index so we can also recover the original 8x8 images later.
    perm = np.random.default_rng(123).permutation(x_all.shape[0])
    n_test = int(0.2 * x_all.shape[0])
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    # Standardize using TRAIN statistics only (no leakage from the test set).
    x_train, mean, std = utils.standardize(x_all[train_idx])
    x_test, _, _ = utils.standardize(x_all[test_idx], mean, std)
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    images_test = digits.images[test_idx]

    model = nn.Sequential(
        nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 10),
    )
    opt = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-4)

    train_acc_hist, val_acc_hist = [], []
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in utils.iterate_minibatches(x_train, y_train, BATCH_SIZE):
            opt.zero_grad()
            loss = nn.cross_entropy(model(Tensor(xb)), yb)
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        train_acc = utils.accuracy(model(Tensor(x_train)), y_train)
        val_acc = utils.accuracy(model(Tensor(x_test)), y_test)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:3d}  lr {opt.lr:.4f}  "
                  f"train {train_acc:.3f}  val {val_acc:.3f}")

    print(f"final  train {train_acc_hist[-1]:.3f}  val {val_acc_hist[-1]:.3f}")

    model.eval()
    val_pred = np.argmax(model(Tensor(x_test)).data, axis=1)

    assets = os.path.join(ROOT, "assets")
    os.makedirs(assets, exist_ok=True)
    _plot.save_accuracy_curves(
        train_acc_hist, val_acc_hist,
        os.path.join(assets, "digits_accuracy.png"),
        "Digits — train vs validation accuracy",
    )
    _plot.save_confusion_matrix(
        y_test, val_pred, 10,
        os.path.join(assets, "digits_confusion_matrix.png"),
        f"Digits — confusion matrix (val acc {val_acc_hist[-1]:.3f})",
    )
    _plot.save_digit_predictions(
        images_test, y_test, val_pred,
        os.path.join(assets, "digits_predictions.png"),
        "Digits — predictions  pred (true)",
    )
    print("saved figures to assets/")


if __name__ == "__main__":
    main()
