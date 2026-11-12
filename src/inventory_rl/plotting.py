from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import InventoryConfig

METHOD_COLORS = {
    "DP optimal": "tab:blue",
    "(s,S) tuned": "tab:orange",
    "base-stock tuned": "tab:green",
    "base-stock naive": "tab:red",
    "Q-learning": "tab:purple",
    "DQN": "tab:brown",
}


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_residuals(results: dict, ax=None):
    """Bellman residual per sweep of value iteration, one line per lead time."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    for L, vi in results.items():
        ax.semilogy(np.arange(1, vi.sweeps + 1), vi.residuals, label=f"L = {L}")
    ax.set_xlabel("sweep")
    ax.set_ylabel(r"$\max_s |V_{k+1}(s) - V_k(s)|$")
    ax.set_title("Value iteration convergence")
    ax.legend()
    return ax


def plot_policy_L1(actions: np.ndarray, cfg: InventoryConfig, rules: dict | None = None, ax=None):
    """Order quantity against on-hand stock for L = 1, with rule policies overlaid."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    x = np.arange(cfg.X_max + 1)
    ax.step(x, actions, where="mid", lw=2.5, label="DP optimal")
    empty = np.zeros((len(x), 0), dtype=int)
    for name, policy in (rules or {}).items():
        ax.step(x, policy(x, empty), where="mid", ls="--", label=name)
    ax.set_xlabel("stock on hand")
    ax.set_ylabel("order quantity")
    ax.set_title("Ordering policies, lead time L = 1")
    ax.legend()
    return ax


def plot_policy_L2(actions: np.ndarray, cfg: InventoryConfig, ax=None):
    """Heatmap of the optimal order for L = 2: on-hand stock vs the order in transit."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    grid = actions.reshape(cfg.state_dims)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("order in transit (arrives tomorrow)")
    ax.set_ylabel("stock on hand")
    ax.set_title("DP optimal order quantity, L = 2")
    plt.colorbar(im, ax=ax, label="order quantity")
    return ax


def plot_cost_vs_S(tables: dict, ax=None):
    """Base-stock tuning curves: mean cost on the tuning seeds against S."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    for L, table in tables.items():
        line, = ax.plot(table.S, table.cost, label=f"L = {L}")
        best = table.loc[table.cost.idxmin()]
        ax.plot(best.S, best.cost, "o", color=line.get_color())
    ax.set_xlabel("base-stock level S")
    ax.set_ylabel("average cost per day")
    ax.set_title("Base-stock sensitivity to S (tuning seeds)")
    ax.legend()
    return ax


def plot_ss_heatmap(table: pd.DataFrame, ax=None, clip_pct: float = 30.0):
    """(s, S) tuning grid; costs more than `clip_pct`% above the best are shown saturated."""
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    grid = table.pivot(index="s", columns="S", values="cost")
    best = table.cost.min()
    im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="magma_r",
                   vmin=best, vmax=best * (1 + clip_pct / 100),
                   extent=[grid.columns.min() - 0.5, grid.columns.max() + 0.5,
                           grid.index.min() - 0.5, grid.index.max() + 0.5])
    b = table.loc[table.cost.idxmin()]
    ax.plot(b.S, b.s, "c*", ms=14, label=f"best (s={int(b.s)}, S={int(b.S)})")
    ax.set_xlabel("order-up-to level S")
    ax.set_ylabel("reorder point s")
    ax.set_title("(s, S) tuning grid")
    ax.legend(loc="upper left")
    plt.colorbar(im, ax=ax, label="average cost per day")
    return ax


def plot_methods_by_L(summary: pd.DataFrame, ax=None):
    """Average cost per day with 95% CI, grouped by lead time."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    methods = list(dict.fromkeys(summary.method))
    Ls = sorted(summary.L.unique())
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        rows = summary[summary.method == m].set_index("L").loc[Ls]
        ax.bar(np.arange(len(Ls)) + i * width, rows.cost, width, yerr=rows.cost_ci, capsize=3, label=m,
               color=METHOD_COLORS.get(m))
    ax.set_xticks(np.arange(len(Ls)) + width * (len(methods) - 1) / 2, [f"L = {L}" for L in Ls])
    ax.set_ylabel("average cost per day")
    ax.set_title("Final evaluation (10 seeds, 95% CI)")
    ax.set_ylim(bottom=summary.cost.min() * 0.8)
    ax.legend(fontsize=8)
    return ax


def plot_sensitivity(table: pd.DataFrame, x: str, ax=None):
    """Gap to the DP optimum (%, 95% CI) against a cost parameter, one line per rule."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    for m, rows in table[table.method != "DP optimal"].groupby("method", sort=False):
        ax.errorbar(rows[x], rows.gap_dp_pct, yerr=rows.gap_dp_ci, marker="o", capsize=3, label=m,
                    color=METHOD_COLORS.get(m))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel("gap to DP optimum (%)")
    ax.legend()
    return ax


def plot_learning_curves(
    curves_by_L: dict,
    dp_costs: dict | None = None,
    rule_costs: dict | None = None,
    ax=None,
    method: str = "Q-learning",
):
    """Mean checkpoint cost vs training steps with 95% CI across seeds.

    `curves_by_L[L]` is `(steps, costs)` with `costs` shaped `(n_seeds, n_checkpoints)`.
    Dotted lines are DP costs (only for lead times present in `dp_costs`). Dashed
    lines are the tuned (s, S) costs.
    """
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    from scipy import stats

    dp_costs = dp_costs or {}
    for L, payload in curves_by_L.items():
        if isinstance(payload, tuple):
            steps, costs = payload
        else:
            raise TypeError("curves_by_L values must be (steps, costs) tuples")
        steps = np.asarray(steps)
        costs = np.asarray(costs, dtype=float)
        mean = costs.mean(axis=0)
        if costs.shape[0] > 1:
            half = stats.t.ppf(0.975, costs.shape[0] - 1) * costs.std(axis=0, ddof=1) / np.sqrt(costs.shape[0])
        else:
            half = np.zeros_like(mean)
        line, = ax.plot(steps, mean, label=f"{method} L={L}")
        ax.fill_between(steps, mean - half, mean + half, alpha=0.2, color=line.get_color())
        if L in dp_costs and np.isfinite(dp_costs[L]):
            ax.axhline(dp_costs[L], color=line.get_color(), ls=":", lw=1.2, alpha=0.9)
        if rule_costs and L in rule_costs and np.isfinite(rule_costs[L]):
            ax.axhline(rule_costs[L], color=line.get_color(), ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("average cost per day (greedy eval)")
    ax.set_title(f"{method} curves (mean ± 95% CI); dotted = DP, dashed = tuned (s,S)")
    ax.legend(fontsize=8)
    return ax


def plot_hyperparam_sensitivity(table: pd.DataFrame, x: str, ax=None, ylabel: str = "gap to DP optimum (%)"):
    """Final gap (or cost) against a training hyperparameter."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    y = "gap_dp_pct" if "gap_dp_pct" in table.columns else "cost"
    yerr = "gap_dp_ci" if "gap_dp_ci" in table.columns else None
    for L, rows in table.groupby("L", sort=True):
        rows = rows.sort_values(x)
        if yerr and yerr in rows.columns:
            ax.errorbar(rows[x], rows[y], yerr=rows[yerr], marker="o", capsize=3, label=f"L = {L}")
        else:
            ax.plot(rows[x], rows[y], marker="o", label=f"L = {L}")
    if y == "gap_dp_pct":
        ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel if y == "gap_dp_pct" else "average cost per day")
    ax.legend()
    return ax
