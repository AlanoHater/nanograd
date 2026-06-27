"""Recurrent layers (vanilla RNN and LSTM) on the nanograd autodiff engine.

A recurrent network is just the same cell applied at every time step, threading
a hidden state through the sequence. Because each step is built from ordinary
``Tensor`` ops, *backpropagation through time* is handled automatically by
``loss.backward()`` — unrolling the loop simply produces a deeper graph.

This minimal implementation returns the **final hidden state**, which is what a
sequence-classification head consumes.
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import nn
from ._random import rng
from .tensor import Tensor


def _uniform(shape):
    # PyTorch-style init: U(-1/sqrt(hidden), 1/sqrt(hidden)).
    k = 1.0 / np.sqrt(shape[-1])
    return Tensor(rng().uniform(-k, k, size=shape))


class RNNCell(nn.Module):
    """A single Elman RNN step: h' = tanh(x W_ih + h W_hh + b)."""

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.weight_ih = _uniform((input_size, hidden_size))
        self.weight_hh = _uniform((hidden_size, hidden_size))
        self.bias = Tensor(np.zeros(hidden_size))

    def forward(self, x_t: Tensor, h: Tensor) -> Tensor:
        return (x_t @ self.weight_ih + h @ self.weight_hh + self.bias).tanh()

    def parameters(self) -> List[Tensor]:
        return [self.weight_ih, self.weight_hh, self.bias]


class LSTMCell(nn.Module):
    """A single LSTM step with input/forget/cell/output gates."""

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        # One fused weight produces all four gates at once.
        self.weight_ih = _uniform((input_size, 4 * hidden_size))
        self.weight_hh = _uniform((hidden_size, 4 * hidden_size))
        # Initialize the forget-gate bias to 1 so the cell remembers by default
        # (a standard trick that greatly helps long-range memory).
        bias = np.zeros(4 * hidden_size)
        bias[hidden_size:2 * hidden_size] = 1.0
        self.bias = Tensor(bias)

    def forward(self, x_t: Tensor, state):
        h, c = state
        gates = x_t @ self.weight_ih + h @ self.weight_hh + self.bias  # (B, 4H)
        hs = self.hidden_size
        i = gates[:, 0:hs].sigmoid()          # input gate
        f = gates[:, hs:2 * hs].sigmoid()     # forget gate
        g = gates[:, 2 * hs:3 * hs].tanh()    # candidate cell
        o = gates[:, 3 * hs:4 * hs].sigmoid()  # output gate
        c = f * c + i * g
        h = o * c.tanh()
        return h, c

    def parameters(self) -> List[Tensor]:
        return [self.weight_ih, self.weight_hh, self.bias]


class RNN(nn.Module):
    """Run an RNN or LSTM cell over a sequence and return the final hidden state.

    Input shape is ``(batch, time, input_size)``; output is ``(batch, hidden)``.
    """

    def __init__(self, input_size: int, hidden_size: int, cell: str = "lstm"):
        self.hidden_size = hidden_size
        self.is_lstm = cell == "lstm"
        if cell == "lstm":
            self.cell = LSTMCell(input_size, hidden_size)
        elif cell == "rnn":
            self.cell = RNNCell(input_size, hidden_size)
        else:
            raise ValueError("cell must be 'rnn' or 'lstm'")

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h = Tensor(np.zeros((b, self.hidden_size)), requires_grad=False)
        c = Tensor(np.zeros((b, self.hidden_size)), requires_grad=False)
        for step in range(t):
            x_t = x[:, step, :]
            if self.is_lstm:
                h, c = self.cell(x_t, (h, c))
            else:
                h = self.cell(x_t, h)
        return h

    def parameters(self) -> List[Tensor]:
        return self.cell.parameters()

    def _child_modules(self) -> List[nn.Module]:
        return [self.cell]
