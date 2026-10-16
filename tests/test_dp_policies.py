import numpy as np
import pytest

from inventory_rl.config import InventoryConfig
from inventory_rl.dp import policy_evaluation, value_iteration
from inventory_rl.env import InventoryEnv
from inventory_rl.evaluate import make_demands, simulate
from inventory_rl.model import build_model
from inventory_rl.policies import BaseStockPolicy, SSPolicy, TabularPolicy, rule_actions


@pytest.fixture(scope="module", params=[1, 2])
def solved(request):
    cfg = InventoryConfig(L=request.param)
    model = build_model(cfg)
    return cfg, model, value_iteration(model, epsilon=1e-4)


def test_value_iteration_satisfies_bellman_optimality(solved):
    cfg, model, vi = solved
    V_pi = policy_evaluation(model, vi.policy)
    assert np.abs(V_pi - vi.V).max() < 1e-2


@pytest.mark.parametrize("rule", ["base_stock", "sS"])
def test_dp_policy_dominates_rules_in_every_state(solved, rule):
    cfg, model, vi = solved
    policy = BaseStockPolicy(4 * cfg.L + 4, cfg) if rule == "base_stock" else SSPolicy(4 * cfg.L, 4 * cfg.L + 6, cfg)
    V_rule = policy_evaluation(model, rule_actions(policy, cfg))
    V_dp = policy_evaluation(model, vi.policy)
    assert (V_dp >= V_rule - 1e-6).all()


def test_policies_respect_order_limits():
    cfg = InventoryConfig(L=3)
    on_hand, pipeline = np.array([0, 5, 20]), np.array([[0, 0], [3, 2], [10, 10]])
    for policy in (BaseStockPolicy(30, cfg), SSPolicy(10, 30, cfg)):
        a = policy(on_hand, pipeline)
        assert ((a >= 0) & (a <= cfg.A_max)).all()
    assert list(SSPolicy(9, 14, cfg)(on_hand, pipeline)) == [10, 0, 0]
    assert list(BaseStockPolicy(14, cfg)(on_hand, pipeline)) == [10, 4, 0]


@pytest.mark.parametrize("L", [1, 3])
def test_vectorized_simulator_matches_env(L):
    cfg = InventoryConfig(L=L)
    demands = make_demands(cfg, [7], 300)[0]
    policy = SSPolicy(3 * L, 3 * L + 7, cfg)

    env = InventoryEnv(cfg)
    obs, _ = env.reset(options={"demands": demands})
    env_costs = []
    for _ in range(len(demands)):
        action = int(policy(obs[:1], obs[None, 1:])[0])
        obs, reward, *_ = env.step(action)
        env_costs.append(-reward)

    sim = simulate(policy, cfg, demands[None, :], burn_in=0, record_daily=True)
    assert np.allclose(sim.daily_cost[0], env_costs)


def test_tabular_policy_follows_dp_actions(solved):
    cfg, model, vi = solved
    policy = TabularPolicy(vi.policy, cfg)
    assert (rule_actions(policy, cfg) == vi.policy).all()
