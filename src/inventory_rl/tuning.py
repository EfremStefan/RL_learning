"""Grid-search tuning of the classic ordering rules on dedicated tuning seeds.

All parameter combinations are simulated at once (one vectorized run per seed),
and the evaluation seeds are never used here, so the tuned rules are not
fitted to the demand they are later scored on.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import InventoryConfig
from .evaluate import BURN_IN, TUNING_SEEDS, make_demands, simulate
from .policies import BaseStockPolicy, SSPolicy

TUNING_DAYS = 10_000


@dataclass
class TuningResult:
    table: pd.DataFrame   # one row per parameter combination, mean cost over tuning seeds
    best: dict
    seconds: float


def _grid_costs(policy, n, cfg, seeds, days, burn_in):
    demands = make_demands(cfg, seeds, days + burn_in)
    return np.mean([simulate(policy, cfg, d, burn_in, n_runs=n).avg_cost for d in demands], axis=0)


def tune_base_stock(cfg: InventoryConfig, S_values=None, seeds=TUNING_SEEDS, days=TUNING_DAYS, burn_in=BURN_IN):
    t0 = time.perf_counter()
    S_values = np.arange(0, cfg.max_position + 2) if S_values is None else np.asarray(S_values)
    costs = _grid_costs(BaseStockPolicy(S_values, cfg), len(S_values), cfg, seeds, days, burn_in)
    table = pd.DataFrame({"S": S_values, "cost": costs})
    best = table.loc[table.cost.idxmin()]
    return TuningResult(table, {"S": int(best.S), "cost": float(best.cost)}, time.perf_counter() - t0)


def tune_ss(cfg: InventoryConfig, S_values=None, seeds=TUNING_SEEDS, days=TUNING_DAYS, burn_in=BURN_IN):
    t0 = time.perf_counter()
    S_values = np.arange(1, cfg.max_position + 2) if S_values is None else np.asarray(S_values)
    pairs = np.array([(s, S) for S in S_values for s in range(0, S)])
    costs = _grid_costs(SSPolicy(pairs[:, 0], pairs[:, 1], cfg), len(pairs), cfg, seeds, days, burn_in)
    table = pd.DataFrame({"s": pairs[:, 0], "S": pairs[:, 1], "cost": costs})
    best = table.loc[table.cost.idxmin()]
    return TuningResult(table, {"s": int(best.s), "S": int(best.S), "cost": float(best.cost)}, time.perf_counter() - t0)
