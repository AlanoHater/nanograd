"""Tests for the learning-rate schedulers."""

import numpy as np
import pytest

from nanograd import optim
from nanograd.tensor import Tensor


def _dummy_opt(lr):
    return optim.SGD([Tensor(np.zeros(1))], lr=lr)


def test_step_lr_decays_in_steps():
    opt = _dummy_opt(1.0)
    sched = optim.StepLR(opt, step_size=3, gamma=0.5)
    seen = [opt.lr]
    for _ in range(7):
        sched.step()
        seen.append(opt.lr)
    # epochs 0..2 -> 1.0, 3..5 -> 0.5, 6.. -> 0.25
    assert np.isclose(seen[0], 1.0)
    assert np.isclose(seen[3], 0.5)
    assert np.isclose(seen[6], 0.25)


def test_cosine_annealing_endpoints():
    opt = _dummy_opt(1.0)
    sched = optim.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    lrs = [opt.lr]
    for _ in range(10):
        sched.step()
        lrs.append(opt.lr)
    assert np.isclose(lrs[0], 1.0)        # start at base_lr
    assert np.isclose(lrs[-1], 0.0, atol=1e-12)  # reach eta_min at T_max
    assert np.isclose(lrs[5], 0.5, atol=1e-9)    # halfway -> midpoint
    # monotonically non-increasing
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
