"""Tests for the recurrent layers (RNN / LSTM)."""

import numpy as np
import pytest

import nanograd as ng
from nanograd import nn, optim, utils
from nanograd.rnn import RNN, RNNCell, LSTMCell
from nanograd.tensor import Tensor

from _gradcheck import check_grad

rng = np.random.default_rng(0)


def test_rnn_cell_shape():
    cell = RNNCell(3, 5)
    h = cell(Tensor(np.zeros((2, 3))), Tensor(np.zeros((2, 5))))
    assert h.shape == (2, 5)


def test_lstm_cell_shapes():
    cell = LSTMCell(3, 5)
    h, c = cell(Tensor(np.zeros((2, 3))),
                (Tensor(np.zeros((2, 5))), Tensor(np.zeros((2, 5)))))
    assert h.shape == (2, 5)
    assert c.shape == (2, 5)


def test_rnn_returns_final_hidden_shape():
    ng.manual_seed(0)
    net = RNN(4, 8, cell="rnn")
    out = net(Tensor(rng.standard_normal((3, 6, 4))))
    assert out.shape == (3, 8)


def test_rnn_backprop_through_time_gradient():
    # Unrolling the RNN must differentiate correctly across all time steps.
    ng.manual_seed(0)
    net = RNN(3, 5, cell="rnn")
    check_grad(lambda x: net(x), rng.standard_normal((2, 4, 3)))


def test_lstm_backprop_through_time_gradient():
    ng.manual_seed(0)
    net = RNN(3, 5, cell="lstm")
    check_grad(lambda x: net(x), rng.standard_normal((2, 4, 3)))


def test_lstm_learns_to_recall_first_bit():
    ng.manual_seed(0)
    length = 10
    x_train, y_train = utils.make_recall_dataset(2000, length)
    model = nn.Sequential(RNN(1, 32, cell="lstm"), nn.Linear(32, 2))
    opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(10):
        for xb, yb in utils.iterate_minibatches(x_train, y_train, 64):
            opt.zero_grad()
            nn.cross_entropy(model(Tensor(xb)), yb).backward()
            opt.step()

    x_test, y_test = utils.make_recall_dataset(500, length)
    acc = utils.accuracy(model(Tensor(x_test)), y_test)
    assert acc > 0.9, f"LSTM failed the recall task: {acc:.3f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
