"""rliable IQM, with a shim for arch 8.

rliable 1.2 still passes ``random_state`` into the bootstrap. arch 8 renamed
that argument to ``seed`` and rejects unknown keywords, so the stock
``get_interval_estimates`` call fails on the Python 3.14 environment this
project uses. The shim accepts the old name and forwards it.
"""

from __future__ import annotations

import inspect

import numpy as np


def _patch_arch_random_state() -> None:
    from arch.bootstrap.base import IIDBootstrap

    if getattr(IIDBootstrap.__init__, "_p30_patched", False):
        return
    params = inspect.signature(IIDBootstrap.__init__).parameters
    if "random_state" in params or "seed" not in params:
        return
    original = IIDBootstrap.__init__

    def _init(self, *args, random_state=None, seed=None, **kwargs):
        original(self, *args, seed=random_state if seed is None else seed, **kwargs)

    _init._p30_patched = True
    IIDBootstrap.__init__ = _init


def iqm_cost_interval(costs, reps: int = 5_000, seed: int = 0) -> tuple[float, float, float]:
    """IQM of average cost and a 95% bootstrap interval.

    ``costs`` has shape ``(n_runs, n_tasks)`` or ``(n_runs,)``. Lower cost is
    better, but rliable aggregates scores where higher is better, so the
    matrix is negated internally and the result is flipped back to cost.
    Returns ``(iqm, ci_low, ci_high)`` on the cost scale.
    """
    _patch_arch_random_state()
    from rliable import library as rly
    from rliable import metrics

    matrix = np.asarray(costs, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    def _iqm(scores):
        return np.asarray([metrics.aggregate_iqm(scores)], dtype=float)

    point, interval = rly.get_interval_estimates(
        {"score": -matrix},
        _iqm,
        reps=reps,
        random_state=np.random.RandomState(seed),
    )
    score = float(np.asarray(point["score"]).reshape(-1)[0])
    bounds = np.asarray(interval["score"], dtype=float).reshape(2, -1)[:, 0]
    return -score, float(-bounds[1]), float(-bounds[0])
