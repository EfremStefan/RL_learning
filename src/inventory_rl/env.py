import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import InventoryConfig
from .dynamics import transition


class InventoryEnv(gym.Env):
    """Single-item lost-sales inventory with lead time L and Poisson demand.

    Observation: [on_hand, q_1, ..., q_{L-1}], where q_1 arrives tomorrow morning.
    Action: order quantity in {0, ..., A_max}.
    Reward: -(holding + lost-sales penalty + ordering cost) for the day.

    The task is continuing, so `terminated` is always False. `max_steps`
    optionally truncates a run. Passing `options={"demands": seq}` to `reset`
    replays a fixed demand sequence (used for common random numbers).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: InventoryConfig | None = None, max_steps: int | None = None, **overrides):
        base = config or InventoryConfig()
        self.cfg = base.with_(**overrides) if overrides else base
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(self.cfg.n_actions)
        self.observation_space = spaces.MultiDiscrete(np.array(self.cfg.state_dims))
        self._demands = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.on_hand = int(options.get("on_hand", 0))
        self.pipeline = np.array(options.get("pipeline", [0] * self.cfg.pipeline_len), dtype=np.int64)
        if self.pipeline.shape != (self.cfg.pipeline_len,):
            raise ValueError(f"pipeline must have {self.cfg.pipeline_len} entries")
        demands = options.get("demands")
        self._demands = None if demands is None else np.asarray(demands, dtype=np.int64)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid order quantity {action}")

        if self._demands is not None:
            demand = int(self._demands[self.t])
        else:
            demand = int(self.np_random.poisson(self.cfg.lam))

        out = transition(self.on_hand, self.pipeline, action, demand, self.cfg)
        self.on_hand = int(out.next_on_hand)
        self.pipeline = np.asarray(out.next_pipeline, dtype=np.int64)
        self.t += 1

        cost = float(out.cost)
        info = {
            "demand": demand,
            "sold": int(out.sold),
            "lost": int(out.lost),
            "ordered": action,
            "overflow": int(out.overflow),
            "holding": float(out.holding_cost),
            "lost_penalty": float(out.lost_cost),
            "ordering": float(out.order_cost),
            "cost": cost,
        }
        truncated = self.max_steps is not None and self.t >= self.max_steps
        return self._obs(), -cost, False, truncated, info

    def _obs(self):
        return np.concatenate([[self.on_hand], self.pipeline]).astype(np.int64)
