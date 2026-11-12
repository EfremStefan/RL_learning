"""DQN for the continuing inventory MDP (Part 3).

Same action-value idea as tabular Q-learning, with a small network in place of
the table. The learned greedy policy is a vectorized callable, so the fast
simulator in ``evaluate.simulate`` can score it without stepping the Gym env.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from ..config import InventoryConfig
from ..env import InventoryEnv
from ..evaluate import simulate


def observation_scale(cfg: InventoryConfig) -> np.ndarray:
    """Per-coordinate divisor that maps each discrete state feature onto [0, 1]."""
    return np.maximum(np.asarray(cfg.state_dims, dtype=np.float32) - 1.0, 1.0)


def scale_observation(obs, cfg: InventoryConfig) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32) / observation_scale(cfg)


class ScaledInventoryEnv(InventoryEnv):
    """InventoryEnv with observations scaled to [0, 1] for the Q-network.

    Subclasses ``InventoryEnv`` so Stable-Baselines3 keeps the real dynamics
    (a Gym wrapper would otherwise hide ``cfg`` and the demand hook).
    """

    def __init__(self, config: InventoryConfig | None = None, max_steps: int | None = None, **overrides):
        super().__init__(config, max_steps=max_steps, **overrides)
        self._scale = observation_scale(self.cfg)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._scale.size,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return scale_observation(obs, self.cfg), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return scale_observation(obs, self.cfg), reward, terminated, truncated, info


class DQNPolicy:
    """Greedy DQN policy. ``(on_hand, pipeline) -> order`` for the vectorized simulator."""

    def __init__(self, model: DQN, cfg: InventoryConfig):
        self.model = model
        self.cfg = cfg
        self._scale = observation_scale(cfg)

    def __call__(self, on_hand, pipeline) -> np.ndarray:
        on_hand = np.asarray(on_hand)
        pipeline = np.asarray(pipeline)
        n = int(on_hand.shape[0])
        if self.cfg.pipeline_len == 0:
            raw = on_hand.reshape(n, 1)
        else:
            raw = np.concatenate([on_hand.reshape(n, 1), np.reshape(pipeline, (n, -1))], axis=1)
        obs = raw.astype(np.float32) / self._scale
        actions, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(actions, dtype=np.int64).reshape(-1)


def _checkpoint_cost(policy: DQNPolicy, cfg: InventoryConfig, seed: int, days: int, burn_in: int) -> float:
    demands = np.random.default_rng(10_000 + seed).poisson(cfg.lam, size=days + burn_in)
    result = simulate(policy, cfg, demands[None, :], burn_in)
    return float(result.avg_cost[0])


def first_within(costs, steps, target: float | None, tolerance: float = 0.05) -> float:
    """First checkpoint step whose cost is within ``tolerance`` of ``target``, else nan."""
    costs = np.asarray(costs, dtype=float)
    steps = np.asarray(steps)
    if target is None or not np.isfinite(target):
        return float("nan")
    hits = np.flatnonzero(costs <= (1.0 + tolerance) * target)
    return float(steps[hits[0]]) if hits.size else float("nan")


@dataclass
class DQNTrainResult:
    model: DQN
    policy: DQNPolicy
    checkpoints: np.ndarray
    eval_costs: np.ndarray
    seconds: float
    steps_to_target: float


class _CheckpointCallback(BaseCallback):
    def __init__(self, cfg, seed, eval_every, days, burn_in, target_cost):
        super().__init__(verbose=0)
        self.cfg = cfg
        self.seed = seed
        self.eval_every = eval_every
        self.days = days
        self.burn_in = burn_in
        self.target_cost = target_cost
        self.checkpoints: list[int] = []
        self.eval_costs: list[float] = []
        self.steps_to_target = float("nan")

    def _record(self, step: int) -> None:
        if self.checkpoints and self.checkpoints[-1] == step:
            return
        policy = DQNPolicy(self.model, self.cfg)
        cost = _checkpoint_cost(policy, self.cfg, self.seed, self.days, self.burn_in)
        self.checkpoints.append(int(step))
        self.eval_costs.append(cost)
        if (
            self.target_cost is not None
            and np.isnan(self.steps_to_target)
            and cost <= 1.05 * self.target_cost
        ):
            self.steps_to_target = float(step)

    def _on_training_start(self) -> None:
        self._record(0)

    def _on_step(self) -> bool:
        if self.eval_every > 0 and self.num_timesteps % self.eval_every == 0:
            self._record(int(self.num_timesteps))
        return True


def train_dqn(
    cfg: InventoryConfig,
    seed: int,
    total_steps: int = 100_000,
    learning_rate: float = 1e-3,
    gamma: float | None = None,
    eval_every: int = 10_000,
    checkpoint_days: int = 2_000,
    checkpoint_burn_in: int = 200,
    target_cost: float | None = None,
    net_arch: tuple[int, ...] = (64, 64),
    device: str = "cpu",
) -> DQNTrainResult:
    """Train one DQN seed on a continuing inventory trajectory.

    ``target_cost`` is optional. When set, ``steps_to_target`` is the first
    checkpoint whose greedy average cost is within 5% of that reference.
    """
    gamma = cfg.gamma if gamma is None else gamma
    env = ScaledInventoryEnv(cfg)
    callback = _CheckpointCallback(
        cfg, seed, eval_every, checkpoint_days, checkpoint_burn_in, target_cost
    )
    batch_size = min(32, max(8, total_steps // 4))
    learning_starts = min(1_000, max(batch_size, total_steps // 2))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=max(total_steps, batch_size),
        learning_starts=learning_starts,
        batch_size=batch_size,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": list(net_arch)},
        seed=seed,
        device=device,
        verbose=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    callback._record(int(total_steps))
    policy = DQNPolicy(model, cfg)
    return DQNTrainResult(
        model=model,
        policy=policy,
        checkpoints=np.asarray(callback.checkpoints, dtype=np.int64),
        eval_costs=np.asarray(callback.eval_costs, dtype=np.float64),
        seconds=time.perf_counter() - t0,
        steps_to_target=callback.steps_to_target,
    )
