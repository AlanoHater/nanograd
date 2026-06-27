"""Train recurrent networks (built on nanograd) on a sequence-memory task.

The task: read a binary sequence and, at the final step, output its *first*
bit. Solving it requires carrying information across every time step, so it is a
direct test of recurrent memory. We train both a vanilla RNN and an LSTM and
plot their learning curves.

Run from the repo root:

    python examples/rnn_recall.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import nanograd as ng  # noqa: E402
from nanograd import nn, optim, utils  # noqa: E402
from nanograd.rnn import RNN  # noqa: E402
from nanograd.tensor import Tensor  # noqa: E402

import _plot  # noqa: E402

LENGTH = 12
EPOCHS = 12
N_TRAIN = 2500
HIDDEN = 32


def train(cell):
    ng.manual_seed(0)
    x_train, y_train = utils.make_recall_dataset(N_TRAIN, LENGTH)
    model = nn.Sequential(RNN(1, HIDDEN, cell=cell), nn.Linear(HIDDEN, 2))
    opt = optim.Adam(model.parameters(), lr=3e-3)

    acc_hist = []
    for epoch in range(EPOCHS):
        for xb, yb in utils.iterate_minibatches(x_train, y_train, 64):
            opt.zero_grad()
            nn.cross_entropy(model(Tensor(xb)), yb).backward()
            opt.step()
        x_val, y_val = utils.make_recall_dataset(1000, LENGTH)
        acc_hist.append(utils.accuracy(model(Tensor(x_val)), y_val))
    return acc_hist


def main():
    print(f"Task: recall the first bit of a length-{LENGTH} binary sequence\n")
    curves = {}
    for cell in ("rnn", "lstm"):
        acc = train(cell)
        curves[cell.upper()] = acc
        print(f"{cell.upper():4s}  final accuracy: {acc[-1]:.3f}")

    assets = os.path.join(ROOT, "assets")
    os.makedirs(assets, exist_ok=True)
    _plot.save_curves(
        curves, os.path.join(assets, "rnn_recall.png"),
        title=f"Recurrent memory — recall first bit (sequence length {LENGTH})",
        ylim=(0.4, 1.02),
    )
    print("saved figure to assets/")


if __name__ == "__main__":
    main()
