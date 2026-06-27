"""nanograd — a tiny autodiff engine and neural-network library in NumPy.

Quick start
-----------
>>> import nanograd as ng
>>> from nanograd import nn, optim, utils
>>> ng.manual_seed(0)
>>> x, y = utils.make_spiral(n_points=100, n_classes=3)
>>> model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 3))
>>> opt = optim.Adam(model.parameters(), lr=1e-2)
>>> logits = model(ng.Tensor(x))
>>> loss = nn.cross_entropy(logits, y)
>>> loss.backward(); opt.step()
"""

from . import nn, optim, utils, attention
from ._random import manual_seed
from .tensor import Tensor

__all__ = ["Tensor", "manual_seed", "nn", "optim", "utils", "attention"]
__version__ = "0.3.0"
