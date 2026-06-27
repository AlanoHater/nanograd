"""A single, swappable random generator shared across the library.

Keeping one generator behind :func:`manual_seed` makes weight initialisation
and the synthetic datasets reproducible without threading a ``seed`` argument
through every function.
"""

import numpy as np

_rng = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Reseed the library-wide generator for reproducible runs."""
    global _rng
    _rng = np.random.default_rng(seed)


def rng() -> np.random.Generator:
    """Return the current library-wide random generator."""
    return _rng
