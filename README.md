# aiwan_scaling

SB3/DQN-based capability-scaling experiment for **AIWAN: AI With an
Artificial Need** (Rezaei, 2026c). This is the empirical test called
for in AIWAN §7.3: does the dual-loop architecture's crash-avoidance
advantage over a single-loop baseline erode as the goal process's (g's)
optimization capacity increases, as §4.2 predicts? The tabular
Q-learning experiments in the companion repo (`AIWAN-Code`) establish
the architectural result at fixed, modest capability; this repo scales
that test up and adds a third architecture the tabular experiments
don't have.

## Three architectures

| | Arbitration | Learned? |
|---|---|---|
| **A** — single-loop | none (reward shaping: `R = r_goal + λ·r_battery`) | — |
| **B** — dual-loop, hardcoded gate | `λ(b) = sigmoid(k·(b−b_mid))`, fixed formula | No — g has no access to it, cannot influence it |
| **C** — dual-loop, learned gate | a small DQN policy choosing between g's and ρ's action each step | **Yes** — trained via RL on the combined objective `r_goal + r_battery` |

A and B mirror the tabular experiments' SingleLoopAgent/DualLoopAgent
distinction at a larger scale. **C is new to this repo**: B's gate is
a fixed formula g can never influence regardless of capability, which
makes "no erosion" somewhat foregone by construction — not a real
stress-test of §4.2's claim, which is specifically about architectures
where the combination of g and ρ is itself learned. C tests that
directly.

## Environment

Gymnasium reimplementation of the tabular experiments' battery-grid
task: 6×6 grid, single charger, 7-dim normalized observation
(agent/goal/charger position + battery), 5 discrete actions
(UP/DOWN/LEFT/RIGHT/CHARGE). See `env.py`.

- `RewardWrapperA` — combined reward for Architecture A
- `RewardWrapperGoalOnly` / `RewardWrapperBatteryOnly` — used to train
  g and ρ independently for B and C
- `LearnedGateEnv` — 2-action wrapper (defer to g / defer to ρ) used to
  train C's gate on top of already-trained, fixed g and ρ models

## Independent variable: "power" of g

`NETWORK_SIZES` (16 – 2048 hidden units) × `TRAINING_BUDGETS`
(10k – 500k timesteps), 10 seeds. ρ is deliberately kept fixed and
outside this sweep (trained once per seed) — the question is whether a
fixed ρ, or a gate built on top of it, still protects against an
increasingly powerful g. C's gate scales together with g (same
net_size/budget per cell) rather than staying fixed, since a
fixed-size gate would reintroduce the same "foregone by construction"
problem the hardcoded gate has.

Every (network_size, training_budget) cell (24 total) is run under all
conditions:
- **standard** — no adversarial pressure
- **goal_nulling** — g untrained (reward ≡ 0 during its training)
- **indifference** — the battery term is removed from the relevant
  training signal (for A: `λ_battery=0`; for C: the gate's own reward
  drops `r_battery`; for B this condition is a structural no-op, since
  g never sees `r_battery` regardless)
- **override / conflict** — a task-incentive weight `w ∈ {0, 1, 1.5, 2,
  4, 8}` rewards visiting battery-draining tiles (extra resolution at
  1.5, inside the cliff observed in the tabular experiments' Figure 1)

## Files

| File | Purpose |
|---|---|
| `config.py` | All sweep constants — grid sizes, budgets, seeds, override weights, paths |
| `env.py` | Environment + reward wrappers + `LearnedGateEnv` |
| `train.py` | `train_architecture_a`, `train_g`, `train_rho`, `train_gate`, untrained-baseline helpers |
| `arbitration.py` | `DualLoopPolicy` (B) and `LearnedGatePolicy` (C) — combine trained sub-models into a final policy |
| `evaluate.py` | `rollout()` and adapters from each policy type to a common `(obs, battery) -> action` interface |
| `tune_hyperparams.py` | Chooses shared hyperparameters (lr, γ, `λ_battery_a`, `gate_k`, `gate_b_mid`) on a validation seed set disjoint from the main sweep, before any real run |
| `sanity_check.py` | Real-budget (50k step) learning-curve check for A, g, ρ — run before committing to the full sweep |
| `sanity_check_gate.py` | Same, for C's gate specifically — also runs a routing diagnostic (does the gate's g-vs-ρ choice actually vary with battery, or has it collapsed onto a fixed rule?) |
| `run_experiment.py` | Full sweep orchestration; `--quick` for a pipeline smoke test, `--seeds` to split across parallel processes |
| `launch_parallel.sh` | Splits the 10 configured seeds across 4 processes/cores |
| `merge_results.py` | Combines the per-process `results_seeds_*.csv` files into `results/results.csv` |
| `plot_results.py` | Produces the gap-vs-power and crash-rate-vs-w figures from `results.csv` |

## Usage

```bash
pip install gymnasium stable-baselines3 torch pandas matplotlib

# 1. Choose hyperparameters (once, before the real sweep)
python3 tune_hyperparams.py

# 2. Sanity-check training is real, on a real budget, before committing
#    to the multi-day sweep
python3 sanity_check.py
python3 sanity_check_gate.py

# 3. Validate the full pipeline runs end-to-end on a tiny grid
python3 run_experiment.py --quick

# 4a. Full sweep, single process (slow)
python3 run_experiment.py

# 4b. Full sweep, parallelized across seeds (recommended)
bash launch_parallel.sh
# ... wait for all processes to finish, then:
python3 merge_results.py

# 5. Plots
python3 plot_results.py
```

**Runtime warning:** the full sweep (24 cells × 10 seeds × 3
architectures × up to 9 conditions each, network sizes up to 2048,
budgets up to 500k) is on the order of several days even parallelized
across 4 cores — cost scales more steeply with network size than with
training budget, so do not extrapolate total runtime from the smallest
cells alone. Run `--quick` first; it validates the pipeline in seconds
and won't touch `results/results.csv`.

## Results (10 seeds, full 24-cell grid, complete)

At the highest conflict weight tested (w=8), the crash-rate gap between
the single-loop baseline (A) and each dual-loop architecture:

| Training budget | gap (A − B, hardcoded gate) | gap (A − C, learned gate) |
|---|---|---|
| 10,000 | ~0.51 (flat across all budgets/sizes) | **0.088** |
| 50,000 | ~0.51 | 0.005 |
| 200,000 | ~0.51 | −0.003 |
| 500,000 | ~0.51 | **0.0002** |

**B (hardcoded gate) shows no narrowing anywhere in the grid** — its
formula never observes `r_goal` or the conflict weight, so there is
nothing for increased optimization to exploit. Gap stays 0.49–0.56
across all 24 (network size, budget) cells and 10 seeds (mean 0.51).

**C (learned gate) shows exactly the erosion AIWAN §4.2 predicts.** The
gap collapses to statistical indistinguishability from zero as training
budget increases (paired t-test across seeds, t(9)=7.53, **p<0.0001**).
Held at the largest budget, network size alone has almost no further
effect (gap −0.003 to 0.005 across every size tested). A weaker version
of the same pattern appears along the network-size axis alone (0.029 →
0.012, t(9)=1.90, p=0.09) — optimization budget, more than raw
representational capacity, is what drives the fold-back.

This erosion is specific to direct conflict pressure: under
goal-nulling and engineered indifference (no override incentive), C
still outperforms both A and B.

Full numbers, the merged dataset, and the figures generated from it
(Figures 7–8 in the AIWAN paper) are in `results/results.csv` and
`results/*.png` in this repo.

## Status

**Complete.** See §7.3 of the AIWAN paper for the full discussion and
implications of this result.

## Relationship to the AIWAN-Code repo

`AIWAN-Code` (tabular Q-learning, 6×6 and 10×10 grids) and this repo
are independent codebases testing different pieces of the same paper.
`AIWAN-Code`'s `DualLoopAgent` corresponds to this repo's Architecture
B (hardcoded gate) — there is no tabular equivalent of Architecture C,
since a learned gate is only really meaningful once there's a real
policy (not a small enough action/state space that a formula suffices)
doing the arbitration.

## Citation

```
Rezaei, H. (2026b). AIWBN: AI With a Biological Need.
Rezaei, H. (2026c). AIWAN: AI With an Artificial Need.
```

## License

TBD.
