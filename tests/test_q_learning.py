import numpy as np
import pytest

from inventory_rl.agents.q_learning import QLearningAgent, train_q_learning
from inventory_rl.config import InventoryConfig
from inventory_rl.evaluate import evaluate_policy
from inventory_rl.policies import TabularPolicy


def test_q_update_matches_formula():
    agent = QLearningAgent(n_states=3, n_actions=2, alpha=0.5, gamma=0.9, epsilon=0.0,
                           rng=np.random.default_rng(0))
    agent.Q[1, 0] = 1.0
    agent.Q[2, 0] = 4.0
    agent.Q[2, 1] = 2.0
    err = agent.update(1, 0, r=-1.0, s_next=2)
    # TD target = -1 + 0.9 * max(4, 2) = 2.6; error = 2.6 - 1 = 1.6; Q <- 1 + 0.5*1.6 = 1.8
    assert abs(err - 1.6) < 1e-9
    assert abs(agent.Q[1, 0] - 1.8) < 1e-9


def test_greedy_act_without_exploration():
    agent = QLearningAgent(3, 4, epsilon=0.0, rng=np.random.default_rng(0))
    agent.Q[0] = np.array([0.1, 5.0, 0.2, 0.0])
    assert agent.act(0, explore=True) == 1
    assert agent.act(0, explore=False) == 1


def test_short_train_beats_random_on_L1():
    cfg = InventoryConfig(L=1)
    result = train_q_learning(cfg, seed=0, total_steps=20_000, eval_every=20_000,
                              checkpoint_days=1_000, checkpoint_burn_in=100)
    q_pol = TabularPolicy(result.greedy_actions, cfg)
    random_actions = np.random.default_rng(1).integers(0, cfg.n_actions, size=cfg.n_states)
    rnd_pol = TabularPolicy(random_actions, cfg)
    q_cost = evaluate_policy(q_pol, cfg, seeds=(0, 1, 2), days=5_000, burn_in=500).avg_cost.mean()
    r_cost = evaluate_policy(rnd_pol, cfg, seeds=(0, 1, 2), days=5_000, burn_in=500).avg_cost.mean()
    assert q_cost < r_cost


def test_tabular_policy_from_greedy_actions_works():
    cfg = InventoryConfig(L=1)
    result = train_q_learning(cfg, seed=1, total_steps=5_000, eval_every=5_000,
                              checkpoint_days=500, checkpoint_burn_in=50)
    policy = TabularPolicy(result.greedy_actions, cfg)
    sim = evaluate_policy(policy, cfg, seeds=(0,), days=1_000, burn_in=100)
    assert sim.avg_cost.shape == (1,)
    assert np.isfinite(sim.avg_cost[0])


def test_steps_to_target_when_already_good():
    cfg = InventoryConfig(L=1)
    # Very loose target so step 0 counts
    result = train_q_learning(cfg, seed=0, total_steps=1_000, eval_every=1_000,
                              checkpoint_days=200, checkpoint_burn_in=20, target_cost=1e9)
    assert result.steps_to_target == 0.0
