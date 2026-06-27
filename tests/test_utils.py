"""Tests for dataset helpers and metrics."""

import numpy as np
import pytest

import nanograd as ng
from nanograd import utils


def test_make_spiral_shapes():
    x, y = utils.make_spiral(n_points=50, n_classes=3)
    assert x.shape == (150, 2)
    assert y.shape == (150,)
    assert set(np.unique(y)) == {0, 1, 2}


def test_make_moons_shapes():
    x, y = utils.make_moons(n_samples=200)
    assert x.shape == (200, 2)
    assert set(np.unique(y)) == {0, 1}


def test_one_hot():
    oh = utils.one_hot(np.array([0, 2, 1]), 3)
    assert np.array_equal(oh, np.eye(3)[[0, 2, 1]])


def test_accuracy():
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert utils.accuracy(scores, [1, 0, 1]) == 1.0
    assert utils.accuracy(scores, [0, 0, 0]) == pytest.approx(1 / 3)


def test_iterate_minibatches_covers_all_samples():
    ng.manual_seed(0)
    x = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    seen = []
    total = 0
    for xb, yb in utils.iterate_minibatches(x, y, batch_size=16):
        assert xb.shape[0] == yb.shape[0]
        assert xb.shape[0] <= 16
        seen.append(yb)
        total += yb.shape[0]
    assert total == 50
    assert sorted(np.concatenate(seen).tolist()) == list(range(50))


def test_iterate_minibatches_no_shuffle_is_ordered():
    x = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    batches = list(utils.iterate_minibatches(x, y, batch_size=4, shuffle=False))
    assert np.array_equal(batches[0][1], [0, 1, 2, 3])
    assert np.array_equal(batches[-1][1], [8, 9])


def test_standardize():
    x = np.random.default_rng(0).standard_normal((100, 3)) * 5 + 2
    xs, mean, std = utils.standardize(x)
    assert np.allclose(xs.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(xs.std(axis=0), 1.0, atol=1e-2)


def test_train_test_split_sizes():
    x = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    xtr, xte, ytr, yte = utils.train_test_split(x, y, test_frac=0.2)
    assert xtr.shape[0] == 40 and xte.shape[0] == 10
    assert ytr.shape[0] == 40 and yte.shape[0] == 10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
