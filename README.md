# aiwan_scaling

SB3/DQN-based capability-scaling experiment for **AIWAN: AI With an Artificial Need** (Rezaei, 2026c). This is the empirical test called for in AIWAN §7.3: does the dual-loop architecture's crash-avoidance advantage over a single-loop baseline erode as the goal process's (g's) optimization capacity increases, as §4.2 predicts? The tabular Q-learning experiments in the companion repo ([AIWAN-Code](https://github.com/hamedrezaeirz/AIWAN-Code)) establish the architectural result at fixed, modest capability; this repo scales that test up and adds a third architecture the tabular experiments don't have.

## Three architectures

| | Arbitration | Learned? |
|---|---|---|
| **A** — single-loop | none (reward shaping: `R = r_goal + λ·r_battery`) | — |
| **B** — dual-loop, hardcoded gate | `λ(b)` = fixed logistic formula (two versions, see below) | No — g has no access to it, cannot influence it |
| **C** — dual-loop, learned gate | a small DQN policy choosing between g's and ρ's action each step | Yes — trained via RL on the combined objective `r_goal + r_battery` |

In every experiment here, ρ is a DQN trained on the battery term alone (64 hidden units, 50,000 steps), held fixed and shared by B and C; only the gate differs between them. (In the tabular AIWAN-Code repo, ρ is a hardcoded navigation policy.)

A and B mirror the tabular experiments' `SingleLoopAgent`/`DualLoopAgent` distinction at a larger scale, but B here is **not** the same object as AIWAN-Code's `DualLoopAgent`: the setting differs (DQN agents with a trained ρ in a Gymnasium environment, rather than tabular Q-learning with a hardcoded ρ), and, as originally run, B's gate direction differed too (next section). C is new to this repo: B's gate is a fixed formula g can never influence regardless of capability, which makes "no erosion" somewhat foregone by construction — not a real stress-test of §4.2's claim, which is specifically about architectures where the combination of g and ρ is itself learned. C tests that directly.

### Two versions of B (important)

The gate in AIWAN §3.2 and in AIWAN-Code is an *urgency* gate: ρ takes over when the battery level is **low**. The original sweep in this repo implemented the formula with the opposite sign:

| Label in the data | Gate | Parameters | Status |
|---|---|---|---|
| `B` (as originally run) | `λ(b) = σ(k·(b − b_mid))` on the battery **level** — ρ is chosen when the battery is **high** | `k=0.1`, `b_mid=40` (from `results/hyperparams.json`) | reverse of §3.2; kept for transparency |
| `B_deficit` (corrected) | `λ(b) = σ(k·(b_mid − b))` — ρ is chosen when the battery is **low** | `k=0.1`, `b_mid=61.97`, fixed in advance as `4 (battery per step) × 10 (worst-case steps to the charger on 6×6) + ln 9 / k` (the §5 derivation), **not tuned on these results** | matches §3.2 |

`B_deficit` is a B-only re-run: g and ρ were retrained on the same seeds and the same 24-cell grid, with every condition. The original rows for A, B and C in `results/results.csv` were **not** re-run, so A and C are unchanged.

## Environment

Gymnasium reimplementation of the tabular experiments' battery-grid task: 6×6 grid, single charger, 7-dim normalized observation (agent/goal/charger position + battery), 5 discrete actions (UP/DOWN/LEFT/RIGHT/CHARGE). See `env.py`.

- `RewardWrapperA` — combined reward for Architecture A
- `RewardWrapperGoalOnly` / `RewardWrapperBatteryOnly` — used to train g and ρ independently for B and C
- `LearnedGateEnv` — 2-action wrapper (defer to g / defer to ρ) used to train C's gate on top of already-trained, fixed g and ρ models

## Independent variable: "power" of g

`NETWORK_SIZES` (16 – 2048 hidden units) × `TRAINING_BUDGETS` (10k – 500k timesteps), 10 seeds. ρ is deliberately kept fixed and outside this sweep (trained once per seed) — the question is whether a fixed ρ, or a gate built on top of it, still protects against an increasingly powerful g. C's gate scales together with g (same `net_size`/budget per cell) rather than staying fixed, since a fixed-size gate would reintroduce the same "foregone by construction" problem the hardcoded gate has.

Every (network_size, training_budget) cell (24 total) is run under all conditions:

- `standard` — no adversarial pressure
- `goal_nulling` — g (and A's policy) **left untrained** (randomly initialized networks, evaluated as-is)
- `indifference` — the battery term is removed from the relevant training signal (for A: `λ_battery=0`; for C: the gate's own reward drops `r_battery`; for B this condition is a structural no-op, since g never sees `r_battery` regardless). **The override weight is 0 here**, unlike §5.3 of the paper (w=5.0): `INDIFFERENCE_TASK_WEIGHT` in `config.py` is defined but not used by the sweep. For this reason A crashes only 0.18 under indifference here (0.00 in the largest cell), unlike its 100% in §5.3.
- `override_w{w}` — a task-incentive weight `w ∈ {0, 1, 1.5, 2, 4, 8}` rewards visiting battery-draining tiles (extra resolution at 1.5, inside the cliff observed in the tabular experiments' Figure 1)

## Files

| File | Purpose |
|---|---|
| `config.py` | All sweep constants — grid sizes, budgets, seeds, override weights, paths |
| `env.py` | Environment + reward wrappers + `LearnedGateEnv` |
| `train.py` | `train_architecture_a`, `train_g`, `train_rho`, `train_gate`, untrained-baseline helpers |
| `arbitration.py` | `DualLoopPolicy` (B, original direction) and `LearnedGatePolicy` (C) — combine trained sub-models into a final policy |
| `evaluate.py` | `rollout()` and adapters from each policy type to a common `(obs, battery) -> action` interface |
| `tune_hyperparams.py` | Chooses shared hyperparameters (lr, γ, `λ_battery_a`, `gate_k`, `gate_b_mid`) on a validation seed set disjoint from the main sweep, before any real run |
| `sanity_check.py` | Real-budget (50k step) learning-curve check for A, g, ρ — run before committing to the full sweep |
| `sanity_check_gate.py` | Same, for C's gate specifically — also runs a routing diagnostic (does the gate's g-vs-ρ choice actually vary with battery, or has it collapsed onto a fixed rule?) |
| `run_experiment.py` | Full sweep orchestration for A, B, C; `--quick` for a pipeline smoke test, `--seeds` to split across parallel processes |
| `launch_parallel.sh` | Splits the 10 configured seeds across 4 processes/cores |
| `merge_results.py` | Combines the per-process `results_seeds_*.csv` files into `results/results.csv` |
| `plot_results.py` | Original gap-vs-power and crash-rate-vs-w figures from `results.csv` (B as originally run, ±SD) |
| **`run_experiment_B_deficit.py`** | B-only re-run with the corrected gate direction (`--quick`, `--pilot`, `--reduced`, default = full grid) |
| **`launch_B_deficit.sh`** | Splits the 10 seeds across 4 processes for the re-run |
| **`merge_B_deficit.py`** | Merges the per-process files into `results/results_B_deficit.csv` (never touches `results.csv`) |
| **`analyze_B_deficit.py`** | Compares `B_deficit` with A, B and C (conditions, gaps, budget trend) |
| **`plot_results_B_deficit.py`** | Figures with both versions of B and ±SE: `gap_vs_power_B_deficit.png`, `crash_rate_vs_w_B_deficit.png` |

## Usage

```bash
pip install gymnasium stable-baselines3 torch pandas matplotlib scipy

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

### Re-running B with the corrected gate direction

```bash
python3 run_experiment_B_deficit.py --quick     # pipeline check, minutes
python3 run_experiment_B_deficit.py --pilot     # one cell (net 256, 100k steps); prints a runtime projection
bash launch_B_deficit.sh                        # full 24-cell grid, 10 seeds, 4 processes
python3 merge_B_deficit.py && python3 analyze_B_deficit.py
python3 plot_results_B_deficit.py
```

`--reduced` runs 3 network sizes × 2 budgets × 3 weights instead of the full grid. The gate's `b_mid` is derived from the environment geometry; `--b-mid` can override it, but do not tune it after looking at results (report both if you do).

**Runtime.** The original full sweep (A, B, C, 24 cells × 10 seeds × up to 9 conditions, network sizes up to 2048, budgets up to 500k) is on the order of several days even parallelized across 4 cores — cost scales more steeply with network size than with training budget, so do not extrapolate total runtime from the smallest cells alone. The B-only re-run (one g per override weight per cell) took about 24 hours on 4 cores. Run `--quick` first; it validates the pipeline in seconds to minutes and does not touch `results/results.csv`.

## Results (10 seeds, full 24-cell grid, complete)

At the highest conflict weight tested (w=8), the crash-rate gap between the single-loop baseline (A) and each dual-loop architecture, by training budget (mean over network sizes and seeds):

| Training budget | gap A − B (as originally run) | gap A − B_deficit (corrected gate) | gap A − C (learned gate) |
|---|---|---|---|
| 10,000 | ~0.51 (flat) | 0.659 | 0.088 |
| 50,000 | ~0.51 | 0.635 | 0.005 |
| 200,000 | ~0.51 | 0.622 | −0.003 |
| 500,000 | ~0.51 | 0.644 | 0.0002 |

**B (hardcoded gate) shows no narrowing anywhere in the grid, with either gate direction** — its formula never observes `r_goal` or the conflict weight, so there is nothing for increased optimization to exploit. As originally run, the gap stays within 0.49–0.56 across all 24 (network size, budget) cells (mean 0.51). With the corrected gate it stays within 0.59–0.72 (mean 0.64) and does not narrow with training budget (0.659 → 0.644; paired t(9)=1.19, p=0.27).

**C (learned gate)** shows exactly the erosion AIWAN §4.2 predicts. The gap collapses to statistical indistinguishability from zero as training budget increases (paired t-test across seeds, t(9)=7.53, p<0.0001). Held at the largest budget, network size alone has almost no further effect (gap −0.003 to 0.005 across every size tested). A weaker version of the same pattern appears along the network-size axis alone (0.029 → 0.012, t(9)=1.90, p=0.09, not significant) — optimization budget, more than raw representational capacity, is what drives the fold-back.

**B's absolute crash rate is high in both versions**, and far above the 0.6–2.7% of the tabular experiments: at w=8, 0.38–0.47 (mean 0.44) as originally run, 0.21–0.37 (mean 0.32) with the corrected gate. The direction accounts for part of the distance to the tabular result, not all of it; the remainder presumably reflects other differences between the settings (DQN with a trained ρ, Gymnasium environment), which were not isolated here.

Mean crash rate over the 24 cells, by condition:

| Condition | A | B (original) | B_deficit (corrected) | C |
|---|---|---|---|---|
| standard | 0.207 | 0.149 | 0.053 | 0.089 |
| goal_nulling | 0.916 | 0.371 | 0.195 | 0.218 |
| indifference (w=0, see above) | 0.180 | 0.152 | 0.053 | 0.114 |
| override w=1 | 0.786 | 0.430 | 0.315 | 0.760 |
| override w=8 | 0.956 | 0.443 | 0.316 | 0.932 |

The erosion is specific to direct conflict pressure. Under goal-nulling and engineered indifference (no override incentive), C outperforms A and the originally-run B, **but not the corrected B**: goal-nulling 0.218 vs. 0.195 (paired t(9)=−0.78, p=0.46, not significant); indifference and standard are better for the corrected B than for C (0.053 vs. 0.114, t(9)=−5.37, p<0.001; 0.053 vs. 0.089, p=0.002). The comparison depends on capability: C is worse than the corrected B in the smallest cells and matches or beats it in the largest (for example 0.000 vs. 0.009 at w=0 in the largest cell).

The corrected B also still crashes where the single-loop agent is safe: in the largest cell (2048 units, 500k steps), `B_deficit` crashes 0.009 and 0.338 at w=0 and w=1, against 0.000 for A at both; at w≥1.5 it crashes less than A.

Full numbers, the merged datasets, and the figures (Figures 7–8 in the AIWAN paper) are in `results/`:

- `results.csv` — original sweep (A, B as originally run, C), 10 seeds
- `results_B_deficit.csv` — corrected-gate re-run, label `B_deficit` (2,160 rows = 24 cells × 10 seeds × 9 conditions); `analysis_B_deficit.txt` is the output of `analyze_B_deficit.py`
- `gap_vs_power_B_deficit.png`, `crash_rate_vs_w_B_deficit.png` — figures with both versions of B (±1 SE over seeds)
- `gap_vs_power.png`, `crash_rate_vs_w.png`, … — original figures (B as originally run, ±1 SD)

## Status

Complete. B re-run with the corrected gate direction (see above). See §7.3 of the AIWAN paper for the full discussion and implications of this result.

## Relationship to the AIWAN-Code repo

AIWAN-Code (tabular Q-learning, 6×6 and 10×10 grids) and this repo are independent codebases testing different pieces of the same paper. AIWAN-Code's `DualLoopAgent` has the same *structure* as this repo's Architecture B with the corrected gate (a fixed urgency gate), but not the same implementation or setting (hardcoded ρ and tabular agents there). There is no tabular equivalent of Architecture C, since a learned gate is only really meaningful once there's a real policy (not a small enough action/state space that a formula suffices) doing the arbitration.

## Citation

Rezaei, H. (2026b). AIWBN: AI With a Biological Need.
Rezaei, H. (2026c). AIWAN: AI With an Artificial Need.

## License

TBD.
