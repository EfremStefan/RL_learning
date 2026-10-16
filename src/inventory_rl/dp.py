import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

from .model import InventoryModel


@dataclass
class VIResult:
    V: np.ndarray            # optimal discounted value per state (= minus discounted cost)
    Q: np.ndarray            # (S, A)
    policy: np.ndarray       # greedy order quantity per state
    residuals: np.ndarray    # max |V_{k+1} - V_k| per sweep
    seconds: float

    @property
    def sweeps(self) -> int:
        return len(self.residuals)


def bellman_q(model: InventoryModel, V: np.ndarray, gamma: float) -> np.ndarray:
    """Q(s, a) = r(s, a) + gamma * E[V(s')] for all states and actions."""
    return model.reward + gamma * (V[model.next_state] @ model.pmf)


def value_iteration(
    model: InventoryModel,
    gamma: float | None = None,
    epsilon: float = 1e-2,
    max_sweeps: int = 100_000,
) -> VIResult:
    """Synchronous value iteration from V = 0.

    Stops when the sweep-to-sweep change drops below epsilon * (1 - gamma) / (2 * gamma),
    which guarantees the greedy policy is epsilon-optimal.
    """
    gamma = model.cfg.gamma if gamma is None else gamma
    threshold = epsilon * (1 - gamma) / (2 * gamma)
    t0 = time.perf_counter()

    V = np.zeros(model.n_states)
    residuals = []
    for _ in range(max_sweeps):
        V_new = bellman_q(model, V, gamma).max(axis=1)
        residuals.append(np.abs(V_new - V).max())
        V = V_new
        if residuals[-1] < threshold:
            break

    Q = bellman_q(model, V, gamma)
    return VIResult(V, Q, Q.argmax(axis=1), np.array(residuals), time.perf_counter() - t0)


def policy_matrix(model: InventoryModel, actions: np.ndarray) -> csr_matrix:
    """Sparse P_pi(s, s') for a deterministic policy given as one action per state."""
    S = model.n_states
    rows = np.repeat(np.arange(S), len(model.pmf))
    cols = model.next_state[np.arange(S), actions].ravel()
    probs = np.tile(model.pmf, S)
    return csr_matrix((probs, (rows, cols)), shape=(S, S))


def policy_evaluation(model: InventoryModel, actions: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Exact V_pi from the Bellman expectation equation (I - gamma P_pi) V = r_pi."""
    gamma = model.cfg.gamma if gamma is None else gamma
    S = model.n_states
    r_pi = model.reward[np.arange(S), actions]
    A = identity(S, format="csr") - gamma * policy_matrix(model, actions)
    return spsolve(A.tocsc(), r_pi)
