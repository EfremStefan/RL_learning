"""Tabular Q-learning for the continuing inventory MDP."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import InventoryConfig
from ..env import InventoryEnv
from ..evaluate import simulate
from ..model import encode
from ..policies import TabularPolicy


@dataclass
class TrainResult:
    Q: np.ndarray
    greedy_actions: np.ndarray
    checkpoints: np.ndarray       # env steps at each eval
    eval_costs: np.ndarray        # mean daily cost at each checkpoint
    seconds: float
    steps_to_target: float        # first checkpoint within target, or nan


class QLearningAgent:
    """ε-greedy tabular Q-learning."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def act(self, s: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int) -> float:
        td_target = r + self.gamma * self.Q[s_next].max()
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return float(td_error)

    @property
    def greedy_actions(self) -> np.ndarray:
        return self.Q.argmax(axis=1).astype(np.int64)


def _checkpoint_cost(actions: np.ndarray, cfg: InventoryConfig, seed: int, days: int, burn_in: int) -> float:
    """Fast single-seed greedy cost estimate for learning curves."""
    demands = np.random.default_rng(10_000 + seed).poisson(cfg.lam, size=days + burn_in)
    result = simulate(TabularPolicy(actions, cfg), cfg, demands[None, :], burn_in)
    return float(result.avg_cost[0])


def train_q_learning(
    cfg: InventoryConfig,
    seed: int,
    total_steps: int = 200_000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float | None = None,
    eval_every: int = 10_000,
    checkpoint_days: int = 2_000,
    checkpoint_burn_in: int = 200,
    target_cost: float | None = None,
) -> TrainResult:
    """Train on one continuing trajectory; evaluate greedily at checkpoints.

    `target_cost` is the DP (or other) reference average cost. `steps_to_target`
    is the first checkpoint whose greedy cost is <= 1.05 * target_cost.
    """
    gamma = cfg.gamma if gamma is None else gamma
    rng = np.random.default_rng(seed)
    agent = QLearningAgent(cfg.n_states, cfg.n_actions, alpha, gamma, epsilon, rng)

    env = InventoryEnv(cfg)
    obs, _ = env.reset(seed=seed)
    s = int(encode(obs[0], obs[1:], cfg))

    checkpoints: list[int] = []
    eval_costs: list[float] = []
    steps_to_target = float("nan")
    t0 = time.perf_counter()

    def record(step: int):
        cost = _checkpoint_cost(agent.greedy_actions, cfg, seed, checkpoint_days, checkpoint_burn_in)
        checkpoints.append(step)
        eval_costs.append(cost)
        nonlocal steps_to_target
        if target_cost is not None and np.isnan(steps_to_target) and cost <= 1.05 * target_cost:
            steps_to_target = float(step)

    record(0)
    for step in range(1, total_steps + 1):
        a = agent.act(s, explore=True)
        obs, reward, _, _, _ = env.step(a)
        s_next = int(encode(obs[0], obs[1:], cfg))
        agent.update(s, a, reward, s_next)
        s = s_next
        if step % eval_every == 0:
            record(step)

    return TrainResult(
        Q=agent.Q,
        greedy_actions=agent.greedy_actions,
        checkpoints=np.asarray(checkpoints, dtype=np.int64),
        eval_costs=np.asarray(eval_costs, dtype=np.float64),
        seconds=time.perf_counter() - t0,
        steps_to_target=steps_to_target,
    )
