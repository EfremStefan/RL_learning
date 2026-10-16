import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from inventory_rl.config import InventoryConfig
from inventory_rl.dynamics import transition
from inventory_rl.env import InventoryEnv
from inventory_rl.model import build_model, decode, demand_pmf, encode


@pytest.mark.parametrize("L", [1, 2, 3])
def test_gymnasium_api(L):
    check_env(InventoryEnv(L=L), skip_render_check=True)


@pytest.mark.parametrize(
    "L, on_hand, pipeline, action, demand, expected",
    [
        # lost sales: 5 sold, 2 lost (20), order 4 (5 + 4); the in-transit 3 arrives
        (2, 5, [3], 4, 7, dict(cost=29.0, next_on_hand=3, next_pipeline=[4])),
        # leftovers: 6 units held (6), no order
        (1, 8, [], 0, 2, dict(cost=6.0, next_on_hand=6, next_pipeline=[])),
        # capacity: 18 held (18) + order 10 (15); 28 units do not fit, 8 overflow
        (1, 18, [], 10, 0, dict(cost=33.0, next_on_hand=20, next_pipeline=[], overflow=8)),
        # lead time 3: the oldest in-transit order arrives, the new one joins the back
        (3, 0, [2, 6], 1, 3, dict(cost=36.0, next_on_hand=2, next_pipeline=[6, 1])),
    ],
)
def test_hand_worked_days(L, on_hand, pipeline, action, demand, expected):
    cfg = InventoryConfig(L=L)
    out = transition(on_hand, np.array(pipeline, dtype=int), action, demand, cfg)
    assert float(out.cost) == expected["cost"]
    assert int(out.next_on_hand) == expected["next_on_hand"]
    assert list(np.atleast_1d(out.next_pipeline)) == expected["next_pipeline"]
    if "overflow" in expected:
        assert int(out.overflow) == expected["overflow"]


def test_demand_pmf_sums_to_one():
    pmf = demand_pmf(4.0)
    assert np.isclose(pmf.sum(), 1.0)
    assert (pmf >= 0).all()


@pytest.mark.parametrize("L", [1, 2, 3])
def test_encode_decode_roundtrip(L):
    cfg = InventoryConfig(L=L)
    idx = np.arange(cfg.n_states)
    on_hand, pipeline = decode(idx, cfg)
    assert (encode(on_hand, pipeline, cfg) == idx).all()


@pytest.mark.parametrize("L", [1, 2])
def test_model_rows_are_distributions(L):
    model = build_model(InventoryConfig(L=L))
    for s in range(0, model.n_states, 17):
        for a in (0, 4, 10):
            assert np.isclose(model.transition_probs(s, a).sum(), 1.0)


@pytest.mark.parametrize(
    "L, on_hand, pipeline, action",
    [(1, 3, [], 5), (2, 6, [4], 0), (3, 1, [5, 2], 7)],
)
def test_model_matches_env_monte_carlo(L, on_hand, pipeline, action):
    """Next-state frequencies and mean cost sampled from the env agree with the model."""
    cfg = InventoryConfig(L=L)
    model = build_model(cfg)
    s = int(encode(on_hand, np.array(pipeline, dtype=int), cfg))

    env = InventoryEnv(cfg)
    rng = np.random.default_rng(0)
    n = 20_000
    counts = np.zeros(model.n_states)
    costs = np.empty(n)
    for i in range(n):
        env.reset(seed=int(rng.integers(2**31)), options={"on_hand": on_hand, "pipeline": pipeline})
        obs, reward, *_ = env.step(action)
        counts[encode(obs[0], obs[1:], cfg)] += 1
        costs[i] = -reward

    assert np.abs(counts / n - model.transition_probs(s, action)).max() < 0.01
    assert abs(costs.mean() - model.expected_cost[s, action]) < 4 * costs.std() / np.sqrt(n)
