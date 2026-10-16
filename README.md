# P30: Inventory control. When does RL beat the classic ordering rules?

A shop decides every day how much to order from its supplier. Demand is random (Poisson), orders arrive after a delay of L days, unsold stock costs money to hold, and empty shelves lose sales. This project compares the exact dynamic-programming optimum, the base-stock and (s, S) rules used in industry, tabular Q-learning, and deep RL as the delay grows from 1 to 6 days.

| Part | Content | Status |
|---|---|---|
| 1 | Gymnasium environment, value iteration (exact optimum, L = 1 to 3), tuned base-stock and (s, S) | done: [`notebooks/01_part1_env_dp_rules.ipynb`](notebooks/01_part1_env_dp_rules.ipynb) |
| 2 | Tabular Q-learning vs DP vs rules, learning curves, alpha/epsilon sensitivity | planned |
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
  config.py      parameters (InventoryConfig) and the L-sweep presets
  dynamics.py    one day of the system; the single source of truth used by everything below
  env.py         InventoryEnv, a Gymnasium environment (also registered as "Inventory-v0")
  model.py       exact transition model for DP: state encoding, truncated Poisson, (S, A, D) arrays
  dp.py          value iteration and exact policy evaluation
  policies.py    base-stock, (s, S) and tabular (DP) policies, vectorized
  tuning.py      grid search of rule parameters on tuning seeds
  evaluate.py    vectorized simulator with common random numbers, mean / 95% CI, paired gaps
  plotting.py    figures
notebooks/01_part1_env_dp_rules.ipynb   Part 1 narrative, results and figures
tests/                                  env vs model vs simulator consistency, DP optimality
results/part1/                          CSV tables and PNG figures written by the notebook
```

## Running it

### Locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                            # 27 tests, a few seconds
jupyter nbconvert --to notebook --execute --inplace notebooks/01_part1_env_dp_rules.ipynb
```

The last command regenerates every table and figure in `results/part1/` (about one minute). You can also open the notebook in Jupyter or VS Code and run all cells.

### Google Colab

1. Push this folder to a GitHub repository.
2. Open the notebook in Colab (File > Open notebook > GitHub).
3. In the first code cell, set `REPO_URL` to your repository, then run all cells.

The setup cell clones the repository, installs `gymnasium==1.4.0` and adds `src/` to the import path. Colab's preinstalled numpy, scipy, pandas and matplotlib are used as they are, and the hardware/version cell prints exactly which versions ran. Results do not depend on library versions beyond floating-point noise, because all randomness comes from `numpy.random.default_rng(seed)`.

## Experimental protocol (Part 1)

- **Seeds:** rules are tuned on seeds 1000 to 1004 (10,000 days each). Every method is evaluated on seeds 0 to 9 (20,000 days each, after a 1,000-day burn-in).
- **Equal budget:** every method sees the same demand sequence per seed (common random numbers). Gaps are computed per seed and averaged (paired), with 95% t-intervals.
- **Greedy evaluation:** policies are evaluated without exploration on seeds never used for tuning.
- **Sensitivity:** cost against the base-stock level S, and gaps against the lost-sales penalty p in {5, 10, 20} for L = 1 and 2.
- **Learning curves and rliable IQM** start in Part 2 and Part 3, when methods learn.

## Figures and tables

All are produced by `notebooks/01_part1_env_dp_rules.ipynb` (run all cells, or the `nbconvert` command above).

| File in `results/part1/` | Notebook section |
|---|---|
| `state_space_growth.csv` | 2. Environment, state-space growth |
| `dp_timing.csv`, `dp_convergence_and_policies.png` | 3. Value iteration |
| `tuned_rules.csv`, `base_stock_tuning_grid.csv`, `ss_tuning_grid.csv`, `rule_tuning.png` | 4. Classic rules |
| `final_evaluation.csv`, `per_seed_costs.csv`, `final_evaluation.png` | 5. Final evaluation |
| `sensitivity_p.csv`, `sensitivity_p.png` | 6. Sensitivity to p |
| `compute_time.csv` | 7. Compute cost |

## Part 1 results

Average cost per day (10 seeds, 95% CI) and gap to the DP optimum (paired, per seed):

| L | DP optimal | (s, S) tuned | base-stock tuned | base-stock naive |
|---|---|---|---|---|
| 1 | 12.77 +/- 0.03 | 12.78 (+0.12% +/- 0.14) | 13.88 (+8.7%) | 16.39 (+28.4%) |
| 2 | 13.35 +/- 0.04 | 13.52 (+1.31% +/- 0.14) | 14.58 (+9.2%) | 16.19 (+21.3%) |
| 3 | 13.70 +/- 0.04 | 13.96 (+1.87% +/- 0.15) | 15.06 (+10.0%) | 16.12 (+17.7%) |

- With a 1-day delay the tuned (s, S) rule is optimal within noise; the DP policy itself has an (s, S) shape.
- The (s, S) gap grows with the delay, which is where a learned policy may pay off.
- Base-stock pays the fixed order cost almost every day, so it is about 9 to 10% worse.
- Exact DP takes up to 2 s for L = 3. The model needs about 80 MB at L = 4, 0.85 GB at L = 5 and 9 GB at L = 6.

**Hardware for the reported timings:** AMD64 Family 26 Model 68 (AMD), 16 logical cores, Windows 11, Python 3.14.3, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, gymnasium 1.4.0.

## References

- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., ch. 3 to 4.
- J. Gijsbrechts, R. N. Boute, J. A. Van Mieghem, D. J. Zhang (2022). Can Deep Reinforcement Learning Improve Inventory Management? *Manufacturing & Service Operations Management* 24(3).
- C. D. Hubbs et al. (2020). OR-Gym: A Reinforcement Learning Library for Operations Research Problems. arXiv:2008.06319.
- P. Henderson et al. (2018). Deep Reinforcement Learning that Matters. AAAI.
- R. Agarwal et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS.
