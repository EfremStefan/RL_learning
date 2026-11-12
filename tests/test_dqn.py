import numpy as np

from inventory_rl.agents.dqn import ScaledInventoryEnv, first_within, train_dqn
from inventory_rl.config import InventoryConfig
from inventory_rl.evaluate import evaluate_policy


def test_scaled_observation_is_unit_interval():
    env = ScaledInventoryEnv(InventoryConfig(L=2))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    obs2, reward, terminated, truncated, info = env.step(3)
    assert obs2.shape == (2,)
    assert obs2.min() >= 0.0 and obs2.max() <= 1.0
    assert np.isfinite(reward)
    assert terminated is False


def test_short_dqn_policy_runs_in_simulator():
    cfg = InventoryConfig(L=1)
    result = train_dqn(
        cfg, seed=0, total_steps=400, eval_every=200,
        checkpoint_days=150, checkpoint_burn_in=20,
    )
    assert result.checkpoints[0] == 0
    assert result.checkpoints[-1] == 400
    assert result.eval_costs.shape == result.checkpoints.shape
    assert np.isfinite(result.eval_costs).all()
    sim = evaluate_policy(result.policy, cfg, seeds=(0, 1), days=400, burn_in=40)
    assert sim.avg_cost.shape == (2,)
    assert np.isfinite(sim.avg_cost).all()
    assert ((sim.avg_cost >= 0).all())


def test_dqn_policy_lead_time_two_matches_simulator():
    cfg = InventoryConfig(L=2)
    result = train_dqn(
        cfg, seed=0, total_steps=200, eval_every=200,
        checkpoint_days=80, checkpoint_burn_in=10,
    )
    sim = evaluate_policy(result.policy, cfg, seeds=(0, 1), days=200, burn_in=20)
    assert sim.avg_cost.shape == (2,)
    assert np.isfinite(sim.avg_cost).all()


def test_steps_to_target_when_already_good():
    cfg = InventoryConfig(L=1)
    result = train_dqn(
        cfg, seed=0, total_steps=200, eval_every=200,
        checkpoint_days=80, checkpoint_burn_in=10, target_cost=1e9,
    )
    assert result.steps_to_target == 0.0


def test_first_within_miss_is_nan():
    assert np.isnan(first_within([10.0, 9.0], [0, 100], target=1.0))
    assert first_within([10.0, 1.01], [0, 100], target=1.0) == 100.0


def test_iqm_cost_interval_is_ordered():
    from inventory_rl.rliable_compat import iqm_cost_interval

    costs = np.random.default_rng(0).normal(14.0, 0.4, size=(8, 3))
    iqm, lo, hi = iqm_cost_interval(costs, reps=200, seed=0)
    assert lo <= iqm <= hi
    assert np.isfinite([iqm, lo, hi]).all()
