"""Train a small convolutional network (built on nanograd) on 8x8 digits.

Demonstrates the Conv2d / MaxPool2d / Flatten layers working together as a
LeNet-style CNN, trained with mini-batches and a cosine learning-rate schedule.

Run from the repo root:

    python examples/cnn_digits.py

Requires scikit-learn (only to load the bundled dataset).
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
from nanograd.conv import Conv2d, MaxPool2d  # noqa: E402
from nanograd.tensor import Tensor  # noqa: E402

import _plot  # noqa: E402

EPOCHS = 20
BATCH_SIZE = 64


def main():
    ng.manual_seed(0)
    digits = load_digits()
    images = digits.images.astype(np.float64)   # (1797, 8, 8)
    labels = digits.target.astype(int)

    perm = np.random.default_rng(123).permutation(images.shape[0])
    n_test = int(0.2 * images.shape[0])
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    mean, std = images[train_idx].mean(), images[train_idx].std()
    x_train = ((images[train_idx] - mean) / std)[:, None, :, :]  # (N, 1, 8, 8)
    x_test = ((images[test_idx] - mean) / std)[:, None, :, :]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = nn.Sequential(
        Conv2d(1, 8, 3, padding=1), nn.ReLU(), MaxPool2d(2),    # 8x8 -> 4x4
        Conv2d(8, 16, 3, padding=1), nn.ReLU(), MaxPool2d(2),   # 4x4 -> 2x2
        nn.Flatten(), nn.Linear(16 * 2 * 2, 10),
    )
    opt = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-4)

    train_acc_hist, val_acc_hist = [], []
    for epoch in range(EPOCHS):
        for xb, yb in utils.iterate_minibatches(x_train, y_train, BATCH_SIZE):
            opt.zero_grad()
            loss = nn.cross_entropy(model(Tensor(xb)), yb)
            loss.backward()
            opt.step()
        scheduler.step()

        train_acc = utils.accuracy(model(Tensor(x_train)), y_train)
        val_acc = utils.accuracy(model(Tensor(x_test)), y_test)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        if (epoch + 1) % 4 == 0:
            print(f"epoch {epoch + 1:3d}  loss {loss.data:.4f}  "
                  f"train {train_acc:.3f}  val {val_acc:.3f}")

    print(f"final  train {train_acc_hist[-1]:.3f}  val {val_acc_hist[-1]:.3f}")

    # ---- visualize the first conv layer's feature maps on one digit ----
    sample = x_test[0:1]                      # (1, 1, 8, 8)
    conv1, relu1 = model.layers[0], model.layers[1]
    feature_maps = relu1(conv1(Tensor(sample))).data[0]  # (8, 8, 8)

    assets = os.path.join(ROOT, "assets")
    os.makedirs(assets, exist_ok=True)
    _plot.save_accuracy_curves(
        train_acc_hist, val_acc_hist,
        os.path.join(assets, "cnn_accuracy.png"),
        "CNN on digits — train vs validation accuracy",
    )
    _plot.save_feature_maps(
        images[test_idx][0], feature_maps,
        os.path.join(assets, "cnn_feature_maps.png"),
        "First conv layer — feature maps for one digit",
    )
    print("saved figures to assets/")


if __name__ == "__main__":
    main()
