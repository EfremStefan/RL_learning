# P30: Inventory control. When does RL beat the classic ordering rules?

A shop decides every day how much to order from its supplier. Demand is random (Poisson), orders arrive after a delay of L days, unsold stock costs money to hold, and empty shelves lose sales. This project compares the exact dynamic-programming optimum, the base-stock and (s, S) rules used in industry, tabular Q-learning, and deep RL as the delay grows from 1 to 6 days.

| Part | Content | Status |
|---|---|---|
| 1 | Gymnasium environment, value iteration (exact optimum, L = 1 to 3), tuned base-stock and (s, S) | done: §§0–8 in [`notebooks/01_part1_env_dp_rules.ipynb`](notebooks/01_part1_env_dp_rules.ipynb) |
| 2 | Tabular Q-learning vs DP vs rules, learning curves, α/ε sensitivity, compute table | done: §§9–15 in the **same** notebook; outputs in [`results/part2/`](results/part2/) |
| 3 | DQN or PPO, delay sweep L = 1 to 6, rliable IQM statistics | planned |

## The MDP

- **State:** `(on_hand, q_1, ..., q_{L-1})`, the stock on the shelf and the orders in transit (`q_1` arrives tomorrow).
- **Action:** order quantity `0 ... A_max`.
- **Reward:** `-(h * leftover + p * lost_sales + K * [order > 0] + c * order)`.
- **Continuing task** (no terminal state). DP uses discount `gamma = 0.99`; all methods are compared on long-run **average cost per day**.

Day timeline: order, then demand is served (the unmet part is lost), then leftovers are charged holding cost, then the oldest order in transit is delivered the next morning.

Default parameters (`src/inventory_rl/config.py`): `lam=4, X_max=20, A_max=10, h=1, p=10, K=5, c=1, gamma=0.99`.

## Repository layout

```
src/inventory_rl/
  config.py, dynamics.py, env.py, model.py, dp.py
  policies.py, tuning.py, evaluate.py, plotting.py
  agents/q_learning.py     Part 2 tabular Q-learning
notebooks/01_part1_env_dp_rules.ipynb   Parts 1–2 narrative (run Part 1 cells before Part 2)
tests/                     env/model/DP/Q-learning tests
results/part1/             Part 1 tables and figures
results/part2/             Part 2 curves, comparisons, sensitivity, compute_time.csv
```

## Running it

### Locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                            # 32 tests
jupyter nbconvert --to notebook --execute --inplace notebooks/01_part1_env_dp_rules.ipynb
```

Full Parts 1–2 execution takes on the order of **5–10 minutes** on a laptop CPU (Part 1 ~1 min; Part 2 training + sensitivity dominate).

### Google Colab

1. Push this folder to a GitHub repository.
2. Open `notebooks/01_part1_env_dp_rules.ipynb` in Colab.
3. Set `REPO_URL` in the first code cell, then **Run all**.
4. Part 2 cells require Part 1 cells to have run in the same session (`vi_results`, `summary`, …).

## Experimental protocol

### Part 1
- **Seeds:** rules tuned on seeds 1000–1004; final eval on seeds 0–9 (20,000 days, 1,000 burn-in).
- **Equal budget / CRN:** same demand sequence per eval seed for every method.
- **Sensitivity:** base-stock S; lost-sales penalty p ∈ {5, 10, 20}.

### Part 2
- **Train seeds:** 0–9; **200,000** steps each; α=0.1, ε=0.1, γ=0.99.
- **Learning curves:** greedy checkpoint every 10,000 steps (2,000-day probe); mean ± 95% CI.
- **Final eval:** average Q-tables across train seeds → greedy `TabularPolicy` → same `EVAL_SEEDS` as Part 1.
- **Sample efficiency:** first checkpoint within 5% of DP cost (not reached under the default budget).
- **Sensitivity:** α ∈ {0.05, 0.1, 0.3} and ε ∈ {0.05, 0.1, 0.2} at L=1 and L=2 (one factor at a time, 5 seeds).
- **Compute (§15 only):** `results/part2/compute_time.csv` — does **not** modify Part 1 §7.

## Figures and tables

### Part 1 → `results/part1/`
| File | Section |
|---|---|
| `state_space_growth.csv` | §2 |
| `dp_timing.csv`, `dp_convergence_and_policies.png` | §3 |
| `tuned_rules.csv`, `rule_tuning.png`, … | §4 |
| `final_evaluation.csv`, `final_evaluation.png` | §5 |
| `sensitivity_p.csv`, `sensitivity_p.png` | §6 |
| `compute_time.csv` | §7 |

### Part 2 → `results/part2/`
| File | Section |
|---|---|
| `learning_curves.npz`, `learning_curves.png` | §10–11 |
| `final_evaluation_qlearning.csv`, `comparison_with_baselines.csv`, `final_comparison.png` | §12 |
| `sensitivity_alpha_epsilon.csv`, `sensitivity_alpha_epsilon.png` | §13 |
| `compute_time.csv` | §15 |

## Part 1 results

| L | DP optimal | (s, S) tuned | base-stock tuned | base-stock naive |
|---|---|---|---|---|
| 1 | 12.77 +/- 0.03 | 12.78 (+0.12% +/- 0.14) | 13.88 (+8.7%) | 16.39 (+28.4%) |
| 2 | 13.35 +/- 0.04 | 13.52 (+1.31% +/- 0.14) | 14.58 (+9.2%) | 16.19 (+21.3%) |
| 3 | 13.70 +/- 0.04 | 13.96 (+1.87% +/- 0.15) | 15.06 (+10.0%) | 16.12 (+17.7%) |

## Part 2 results (default α=0.1, ε=0.1, 200k steps)

| L | Q-learning cost/day | gap vs DP | gap vs tuned (s,S) | seeds within 5% of DP |
|---|---|---|---|---|
| 1 | 13.70 +/- 0.03 | +7.3% | +7.2% | 0/10 |
| 2 | 18.74 +/- 0.06 | +40.4% | +38.6% | 0/10 |
| 3 | 20.97 +/- 0.09 | +53.1% | +50.3% | 0/10 |

- Under this budget, tabular Q-learning does **not** beat tuned (s,S); gaps grow with L.
- Sensitivity: at L=1, **α=0.3** reduced the gap to about **+2.8%**; some L=2 settings were unstable — hyperparameters matter.
- Train time ~3.6–5.0 s per seed; final eval ~0.35 s (see `results/part2/compute_time.csv`).

**Hardware for the reported timings:** AMD64 Family 26 Model 68 (AMD), 16 logical cores, Windows 11, Python 3.14.3, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, gymnasium 1.4.0.

## References

- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., ch. 3 to 6.
- J. Gijsbrechts, R. N. Boute, J. A. Van Mieghem, D. J. Zhang (2022). Can Deep Reinforcement Learning Improve Inventory Management? *Manufacturing & Service Operations Management* 24(3).
- C. D. Hubbs et al. (2020). OR-Gym: A Reinforcement Learning Library for Operations Research Problems. arXiv:2008.06319.
- P. Henderson et al. (2018). Deep Reinforcement Learning that Matters. AAAI.
- R. Agarwal et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS.
