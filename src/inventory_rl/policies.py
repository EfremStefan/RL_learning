"""Ordering policies as vectorized functions (on_hand, pipeline) -> order quantity.

`on_hand` has shape (n,) and `pipeline` shape (n, L-1), so one policy object can
drive many parallel simulations. Rule parameters may themselves be arrays of
shape (n,), which evaluates a whole parameter grid in a single simulation.
"""

import numpy as np

from .config import InventoryConfig
from .model import decode, encode


def inventory_position(on_hand, pipeline) -> np.ndarray:
    return np.asarray(on_hand) + np.asarray(pipeline).sum(axis=-1)


class SSPolicy:
    """(s, S): if the inventory position is at most s, order up to S (capped at A_max)."""

    def __init__(self, s, S, cfg: InventoryConfig):
        self.s = np.asarray(s)
        self.S = np.asarray(S)
        self.cfg = cfg

    def __call__(self, on_hand, pipeline):
        position = inventory_position(on_hand, pipeline)
        order = np.clip(self.S - position, 0, self.cfg.A_max)
        return np.where(position <= self.s, order, 0)

    def __repr__(self):
        return f"SSPolicy(s={self.s}, S={self.S})"


class BaseStockPolicy(SSPolicy):
    """Base-stock: every day, order up to S. Equivalent to (s, S) with s = S - 1."""

    def __init__(self, S, cfg: InventoryConfig):
        super().__init__(np.asarray(S) - 1, S, cfg)

    def __repr__(self):
        return f"BaseStockPolicy(S={self.S})"


class TabularPolicy:
    """Looks up one action per state, e.g. the greedy policy from value iteration."""

    def __init__(self, actions: np.ndarray, cfg: InventoryConfig):
        self.actions = np.asarray(actions)
        self.cfg = cfg

    def __call__(self, on_hand, pipeline):
        return self.actions[encode(on_hand, pipeline, self.cfg)]


def rule_actions(policy, cfg: InventoryConfig) -> np.ndarray:
    """The action a rule takes in every state, for exact evaluation with the DP model."""
    on_hand, pipeline = decode(np.arange(cfg.n_states), cfg)
    return np.asarray(policy(on_hand, pipeline))
