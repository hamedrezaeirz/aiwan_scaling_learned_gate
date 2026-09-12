"""
Central experiment configuration. All constants live here so the other
files don't hardcode anything, and the sweep can be changed without
touching the core code.
"""

import itertools

# --- Environment ---
GRID_SIZE = 6                     # matches the AIWAN paper (tabular Q-learning also used 6x6)
BATTERY_MAX = 100.0
BATTERY_DRAIN_PER_STEP = 4.0
BATTERY_RECHARGE_PER_STEP = 20.0  # matches paper §5
MAX_EPISODE_STEPS = 100

# --- Independent variable: "power" of g ---
# 1024/2048 added on top of the original 16-512 range: the standard-condition
# data showed net=16 was still climbing at 500k steps while net>=64 had
# already fully plateaued by 200k (see the convergence check that motivated
# this change) -- i.e. capacity, not training duration, is the binding
# constraint at the low end, and this project's compute budget is no longer
# a limiting factor, so it's worth checking whether capacity keeps mattering
# at the high end too, rather than assuming 512 was already enough.
NETWORK_SIZES = [[16], [64], [256], [512], [1024], [2048]]

# 1,000,000 was tried as a control point past the original 500k ceiling, but
# with a hard 4-day/4-core deadline now in play, it's the first thing cut:
# the convergence check already showed success_rate flat from 200k to 500k
# for every net_size >= 64 (e.g. net=512: 0.992 -> 1.000), so it was the
# least load-bearing addition, and dropping it back to the original 4 points
# is what makes the full adversarial grid + 10 seeds affordable in time.
TRAINING_BUDGETS = [10_000, 50_000, 200_000, 500_000]

# Original 5 seeds (0-4) are untouched -- every number already reported for
# A/B in the paper stays reproducible from this same list. 5-9 are additional
# seeds, chosen now, before any of them have been run, for the same reason
# the original 5 were fixed in advance per §7: pre-registering seeds (rather
# than adding "just one more" after looking at results) is what keeps a
# seed count increase from being disguised cherry-picking.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

ALGO = "DQN"   # small discrete action/obs space -> usually faster than PPO on weak CPUs

# --- rho: deliberately kept OUTSIDE the g-power sweep (confirmed decision) ---
RHO_NETWORK_SIZE = [64]
RHO_TIMESTEPS = 50_000

# --- Arbitration, Architecture B: probabilistic gate lambda(b) = sigmoid(k*(b - b_mid)) (confirmed decision) ---
GATE_K = 0.2         # sigmoid slope — exact value set by tune_hyperparams.py
GATE_B_MID = 30.0    # urgency midpoint (% battery) — same

# --- Arbitration, Architecture C (learned gate, Design 1) ---
# No separate net_size/timesteps constants here on purpose: the gate is
# trained with the SAME (net_size, budget) cell as whatever g it's paired
# with, scaling together across NETWORK_SIZES x TRAINING_BUDGETS above.
# See NEXT_STEP_learned_gate.md and train.train_gate's docstring for why
# a fixed-size gate would defeat the point of this architecture's test.

# --- Adversarial scenarios (confirmed decision) ---
# goal_nulling:       no training at all (reward=0 means gradient=0)
# indifference:       one fixed training run with battery-weight=0, not a sweep
# override/conflict:  the only condition that's actually swept
#
# 1.5 added inside the original {0,1,2,4,8}: the existing results show
# Architecture A's crash rate collapsing sharply specifically between w=1
# and w=2 (the "cliff" noted in the paper update) while B/C stay gradual --
# this one point sits inside that transition. (3 and 6 were tried too, but
# cut along with the other deadline-driven trims below -- 1.5 is the one
# that actually sits inside the interesting interval, so it's the one kept.)
OVERRIDE_WEIGHTS = [0, 1, 1.5, 2, 4, 8]
INDIFFERENCE_TASK_WEIGHT = 5.0           # matches paper §5.3

# --- Hyperparameter tuning validation set (disjoint from SEEDS on purpose,
# so no run in the final sweep ever contributes to choosing its own
# hyperparameters) ---
VALIDATION_NET_SIZE = [64]
VALIDATION_TIMESTEPS = 50_000
VALIDATION_SEEDS = [100, 101, 102]

# Every (network_size, training_budget) cell now gets goal_nulling,
# indifference, and the override sweep -- not just 3 representative points.
# The 3-point subset was a compute-driven compromise, explicitly flagged as
# a limitation in the results write-up ("only 3 of 16 cells tested... not a
# dense sweep"). Compute IS a real constraint again now (4-day VPS window),
# but the full grid + 10 seeds still fits comfortably inside it once the
# 1M budget tier and extra override points above were cut -- see the time
# estimate this config was sized against before committing to it.
ADVERSARIAL_CAPABILITY_SUBSET = list(itertools.product(NETWORK_SIZES, TRAINING_BUDGETS))

# --- Evaluation ---
# Raised from 100: eval rollouts are cheap relative to training (at most
# MAX_EPISODE_STEPS=100 env steps each, no gradient updates), so this buys
# tighter crash_rate/success_rate estimates for negligible extra cost --
# unlike the training-side changes above, this one is nearly free.
N_EVAL_EPISODES = 200

# --- Paths ---
RESULTS_DIR = "results"
HYPERPARAMS_PATH = f"{RESULTS_DIR}/hyperparams.json"   # filled in by tune_hyperparams.py, not by hand
RESULTS_CSV_PATH = f"{RESULTS_DIR}/results.csv"
