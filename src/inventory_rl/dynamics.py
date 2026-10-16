"""One day of the inventory system, shared by the env, the DP model and the simulator.

All arguments may be numpy arrays of matching leading shape, so the same code
steps a single Gymnasium env, thousands of parallel simulations, or every
(state, action, demand) triple when building the DP model.
"""

from dataclasses import dataclass

import numpy as np

from .config import InventoryConfig


@dataclass
class DayOutcome:
    next_on_hand: np.ndarray
    next_pipeline: np.ndarray
    sold: np.ndarray
    lost: np.ndarray
    left: np.ndarray
    overflow: np.ndarray
    holding_cost: np.ndarray
    lost_cost: np.ndarray
    order_cost: np.ndarray

    @property
    def cost(self) -> np.ndarray:
        return self.holding_cost + self.lost_cost + self.order_cost


def transition(on_hand, pipeline, action, demand, cfg: InventoryConfig) -> DayOutcome:
    """Order `action`, serve `demand`, then receive tomorrow morning's delivery.

    `pipeline[..., 0]` is the order that arrives next; it has `cfg.L - 1` columns.
    """
    on_hand = np.asarray(on_hand)
    pipeline = np.asarray(pipeline)
    action = np.asarray(action)
    demand = np.asarray(demand)

    sold = np.minimum(on_hand, demand)
    lost = demand - sold
    left = on_hand - sold

    if cfg.L > 1:
        arrival = pipeline[..., 0]
        next_pipeline = np.concatenate([pipeline[..., 1:], action[..., None]], axis=-1)
    else:
        arrival = action
        next_pipeline = pipeline

    stocked = left + arrival
    next_on_hand = np.minimum(stocked, cfg.X_max)

    return DayOutcome(
        next_on_hand=next_on_hand,
        next_pipeline=next_pipeline,
        sold=sold,
        lost=lost,
        left=left,
        overflow=stocked - next_on_hand,
        holding_cost=cfg.h * left,
        lost_cost=cfg.p * lost,
        order_cost=cfg.K * (action > 0) + cfg.c * action,
    )
