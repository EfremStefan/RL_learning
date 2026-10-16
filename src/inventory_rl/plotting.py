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
