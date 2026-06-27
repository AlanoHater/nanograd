"""Convolutional layers for nanograd, implemented with the im2col trick.

A 2-D convolution is turned into a single matrix multiply: ``im2col`` lays out
every sliding window as a column, after which the convolution is just
``weight @ columns``. The matmul (and bias add, and final reshape) are ordinary
engine ops, so their gradients are automatic — only ``im2col`` itself needs a
custom backward (the classic ``col2im`` that scatters gradients back onto the
overlapping input patches). ``MaxPool2d`` is the other custom op here, routing
each gradient to the input position that produced the maximum.
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import nn
from ._random import rng
from .tensor import Tensor


def _im2col(x: Tensor, kh: int, kw: int, stride: int, padding: int,
            out_h: int, out_w: int) -> Tensor:
    """Custom op: ``(B, C, H, W)`` -> ``(B, C*kh*kw, out_h*out_w)``."""
    data = x.data
    b, c, _, _ = data.shape
    if padding > 0:
        xp = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        xp = data

    col = np.zeros((b, c, kh, kw, out_h, out_w), dtype=data.dtype)
    for i in range(kh):
        i_end = i + stride * out_h
        for j in range(kw):
            j_end = j + stride * out_w
            col[:, :, i, j, :, :] = xp[:, :, i:i_end:stride, j:j_end:stride]
    cols = col.reshape(b, c * kh * kw, out_h * out_w)
    out = Tensor(cols, (x,), "im2col")

    def _backward():
        if not x.requires_grad:
            return
        dcol = out.grad.reshape(b, c, kh, kw, out_h, out_w)
        dxp = np.zeros_like(xp)
        for i in range(kh):
            i_end = i + stride * out_h
            for j in range(kw):
                j_end = j + stride * out_w
                dxp[:, :, i:i_end:stride, j:j_end:stride] += dcol[:, :, i, j, :, :]
        if padding > 0:
            x.grad += dxp[:, :, padding:-padding, padding:-padding]
        else:
            x.grad += dxp

    out._backward = _backward
    return out


class Conv2d(nn.Module):
    """2-D convolution over ``(batch, in_channels, height, width)`` inputs."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = self.kw = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * self.kh * self.kw
        std = np.sqrt(2.0 / fan_in)  # He initialization
        weight = rng().standard_normal((out_channels, fan_in)) * std
        self.weight = Tensor(weight)
        self.bias = Tensor(np.zeros((out_channels, 1))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        out_h = (h + 2 * self.padding - self.kh) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kw) // self.stride + 1

        cols = _im2col(x, self.kh, self.kw, self.stride, self.padding, out_h, out_w)
        out = self.weight @ cols                  # (B, out_channels, out_h*out_w)
        if self.bias is not None:
            out = out + self.bias                 # broadcast (out_channels, 1)
        return out.reshape(b, self.out_channels, out_h, out_w)

    def parameters(self) -> List[Tensor]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])


def _max_pool2d(x: Tensor, k: int) -> Tensor:
    """Custom op: non-overlapping max pooling with window/stride ``k``."""
    data = x.data
    b, c, h, w = data.shape
    assert h % k == 0 and w % k == 0, "MaxPool2d expects H, W divisible by kernel_size"
    out_h, out_w = h // k, w // k

    windows = (data.reshape(b, c, out_h, k, out_w, k)
                   .transpose(0, 1, 2, 4, 3, 5)
                   .reshape(b, c, out_h, out_w, k * k))
    argmax = windows.argmax(axis=-1)
    out = Tensor(windows.max(axis=-1), (x,), "maxpool2d")

    def _backward():
        if not x.requires_grad:
            return
        grad_windows = np.zeros((b, c, out_h, out_w, k * k), dtype=data.dtype)
        bi, ci, hi, wi = np.indices((b, c, out_h, out_w))
        grad_windows[bi, ci, hi, wi, argmax] = out.grad
        dx = (grad_windows.reshape(b, c, out_h, out_w, k, k)
                          .transpose(0, 1, 2, 4, 3, 5)
                          .reshape(b, c, h, w))
        x.grad += dx

    out._backward = _backward
    return out


class MaxPool2d(nn.Module):
    """Non-overlapping 2-D max pooling (window = stride = ``kernel_size``)."""

    def __init__(self, kernel_size: int = 2):
        self.k = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        return _max_pool2d(x, self.k)
