"""Fast vectorized simulation and summary statistics.

Common random numbers: every method is evaluated on the same demand sequence
for a given seed, so differences between methods are not due to luck of the
draw, and per-seed differences can be compared directly (paired).
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .config import InventoryConfig
from .dynamics import transition

TUNING_SEEDS = tuple(range(1000, 1005))
EVAL_SEEDS = tuple(range(10))
EVAL_DAYS = 20_000
BURN_IN = 1_000


def make_demands(cfg: InventoryConfig, seeds, days: int) -> np.ndarray:
    """(n_seeds, days) Poisson demands, one reproducible stream per seed."""
    return np.stack([np.random.default_rng(s).poisson(cfg.lam, size=days) for s in seeds])


@dataclass
class SimResult:
    avg_cost: np.ndarray        # (n,) mean daily cost after burn-in
    avg_holding: np.ndarray
    avg_lost_penalty: np.ndarray
    avg_ordering: np.ndarray
    fill_rate: np.ndarray       # fraction of demand served
    order_freq: np.ndarray      # fraction of days with an order
    overflow_rate: np.ndarray   # discarded units per day (capacity hits)
    seconds: float
    daily_cost: np.ndarray | None = None  # (n, days) if requested


def simulate(
    policy,
    cfg: InventoryConfig,
    demands: np.ndarray,
    burn_in: int = BURN_IN,
    n_runs: int | None = None,
    record_daily: bool = False,
) -> SimResult:
    """Run `policy` on `demands` of shape (n, T), or (T,) broadcast to `n_runs` copies.

    Every run starts with an empty shelf and pipeline; the first `burn_in` days
    are excluded from the averages.
    """
    demands = np.asarray(demands)
    if demands.ndim == 1:
        demands = np.broadcast_to(demands, (n_runs or 1, demands.size))
    n, T = demands.shape
    if burn_in >= T:
        raise ValueError("burn_in must be shorter than the simulation")

    t0 = time.perf_counter()
    on_hand = np.zeros(n, dtype=np.int64)
    pipeline = np.zeros((n, cfg.pipeline_len), dtype=np.int64)
    totals = {k: np.zeros(n) for k in ("cost", "holding", "lost", "ordering", "sold", "demand", "orders", "overflow")}
    daily = np.empty((n, T)) if record_daily else None

    for t in range(T):
        action = np.broadcast_to(policy(on_hand, pipeline), (n,))
        out = transition(on_hand, pipeline, action, demands[:, t], cfg)
        on_hand, pipeline = out.next_on_hand, out.next_pipeline
        if record_daily:
            daily[:, t] = out.cost
        if t >= burn_in:
            totals["cost"] += out.cost
            totals["holding"] += out.holding_cost
            totals["lost"] += out.lost_cost
            totals["ordering"] += out.order_cost
            totals["sold"] += out.sold
            totals["demand"] += demands[:, t]
            totals["orders"] += action > 0
            totals["overflow"] += out.overflow

    days = T - burn_in
    return SimResult(
        avg_cost=totals["cost"] / days,
        avg_holding=totals["holding"] / days,
        avg_lost_penalty=totals["lost"] / days,
        avg_ordering=totals["ordering"] / days,
        fill_rate=totals["sold"] / np.maximum(totals["demand"], 1),
        order_freq=totals["orders"] / days,
        overflow_rate=totals["overflow"] / days,
        seconds=time.perf_counter() - t0,
        daily_cost=daily,
    )


def evaluate_policy(policy, cfg: InventoryConfig, seeds=EVAL_SEEDS, days=EVAL_DAYS, burn_in=BURN_IN) -> SimResult:
    """Greedy evaluation of one policy, one run per seed (all seeds in parallel)."""
    return simulate(policy, cfg, make_demands(cfg, seeds, days + burn_in), burn_in)


def mean_ci(x, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a t-based confidence interval across seeds."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf(0.5 + confidence / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def paired_gap_pct(costs, reference_costs) -> tuple[float, float]:
    """Per-seed % gap (cost - ref) / ref, summarized as mean and 95% CI half-width."""
    costs, ref = np.asarray(costs), np.asarray(reference_costs)
    return mean_ci(100 * (costs - ref) / ref)
