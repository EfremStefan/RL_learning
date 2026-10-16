"""Exact transition model of the inventory MDP, for dynamic programming.

Given (state, action), the only randomness is the day's demand, so the model
stores for every (state, action, demand) the next state and the day's cost,
together with the demand pmf. Expectations over demand are dot products with
`pmf`.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from .config import InventoryConfig
from .dynamics import transition


def demand_pmf(lam: float, tail: float = 1e-8) -> np.ndarray:
    """Poisson pmf on 0..D_max with P(D > D_max) < tail folded into D_max."""
    d_max = int(poisson.ppf(1.0 - tail, lam))
    while poisson.sf(d_max, lam) >= tail:
        d_max += 1
    pmf = poisson.pmf(np.arange(d_max + 1), lam)
    pmf[-1] += poisson.sf(d_max, lam)
    return pmf


def encode(on_hand, pipeline, cfg: InventoryConfig) -> np.ndarray:
    """Map (on_hand, pipeline) arrays to integer state indices."""
    on_hand = np.asarray(on_hand)
    pipeline = np.asarray(pipeline)
    coords = (on_hand,) + tuple(pipeline[..., i] for i in range(cfg.pipeline_len))
    return np.ravel_multi_index(coords, cfg.state_dims)


def decode(index, cfg: InventoryConfig):
    """Inverse of `encode`: returns (on_hand, pipeline)."""
    coords = np.unravel_index(np.asarray(index), cfg.state_dims)
    on_hand = coords[0]
    pipeline = np.stack(coords[1:], axis=-1) if cfg.pipeline_len else np.zeros(on_hand.shape + (0,), dtype=int)
    return on_hand, pipeline


@dataclass
class InventoryModel:
    cfg: InventoryConfig
    pmf: np.ndarray            # (D,)
    next_state: np.ndarray     # (S, A, D) int32
    cost: np.ndarray           # (S, A, D) cost of the day
    expected_cost: np.ndarray  # (S, A)
    build_seconds: float

    @property
    def reward(self) -> np.ndarray:
        return -self.expected_cost

    @property
    def n_states(self) -> int:
        return self.next_state.shape[0]

    def transition_probs(self, s: int, a: int) -> np.ndarray:
        """Dense P(s' | s, a), mainly for testing."""
        return np.bincount(self.next_state[s, a], weights=self.pmf, minlength=self.n_states)


def build_model(cfg: InventoryConfig, tail: float = 1e-8) -> InventoryModel:
    t0 = time.perf_counter()
    pmf = demand_pmf(cfg.lam, tail)
    S, A, D = cfg.n_states, cfg.n_actions, len(pmf)

    on_hand, pipeline = decode(np.arange(S), cfg)
    on_hand = on_hand[:, None, None]
    pipeline = pipeline[:, None, None, :]
    action = np.arange(A)[None, :, None]
    demand = np.arange(D)[None, None, :]

    shape = (S, A, D)
    out = transition(
        np.broadcast_to(on_hand, shape),
        np.broadcast_to(pipeline, shape + (cfg.pipeline_len,)),
        np.broadcast_to(action, shape),
        np.broadcast_to(demand, shape),
        cfg,
    )
    next_state = encode(out.next_on_hand, out.next_pipeline, cfg).astype(np.int32)
    cost = np.broadcast_to(out.cost, shape).astype(np.float64)
    expected_cost = cost @ pmf

    return InventoryModel(cfg, pmf, next_state, cost, expected_cost, time.perf_counter() - t0)


def model_memory_mb(cfg: InventoryConfig, tail: float = 1e-8) -> float:
    """Memory of `next_state` + `cost` arrays (int32 + float64), without building them."""
    D = len(demand_pmf(cfg.lam, tail))
    return cfg.n_states * cfg.n_actions * D * (4 + 8) / 1e6
