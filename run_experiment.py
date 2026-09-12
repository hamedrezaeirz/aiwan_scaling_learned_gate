"""
run_experiment.py
------------------
Runs the full experiment: trains and evaluates both architectures across
NETWORK_SIZES x TRAINING_BUDGETS x SEEDS, writing every result (never
just averages) to results/results.csv as it goes, one cell at a time, so
a crash partway through an overnight run doesn't lose earlier progress.

Usage:
    python run_experiment.py            # full sweep (can take a long time)
    python run_experiment.py --quick    # tiny grid, just to check the
                                         # whole pipeline runs end to end
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

# One thread per process: our networks are tiny, so multi-threaded ops
# barely help a single run, but DO cause contention when several
# `python run_experiment.py --seeds ...` processes run at once (see
# instructions below for running seeds in parallel across cores).
torch.set_num_threads(1)

from config import (
    NETWORK_SIZES, TRAINING_BUDGETS, SEEDS, OVERRIDE_WEIGHTS,
    ADVERSARIAL_CAPABILITY_SUBSET, RESULTS_DIR, HYPERPARAMS_PATH,
    RESULTS_CSV_PATH, GATE_K, GATE_B_MID,
)
from train import (
    train_architecture_a, train_g, train_rho, train_gate,
    get_untrained_g, get_untrained_architecture_a,
)
from arbitration import DualLoopPolicy, LearnedGatePolicy
from evaluate import (
    evaluate_cell, as_policy_fn, as_dual_loop_policy_fn, as_learned_gate_policy_fn,
)

ROW_FIELDS = [
    "architecture", "network_size", "training_timesteps", "seed", "condition",
    "n_eval_episodes", "success_rate", "crash_rate", "avg_battery_level", "avg_episode_length",
]


def load_hp():
    with open(HYPERPARAMS_PATH) as f:
        return json.load(f)


def write_rows(rows, path):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run(quick=False, seeds=None):
    hp = load_hp()
    gate_k = hp.get("gate_k", GATE_K)
    gate_b_mid = hp.get("gate_b_mid", GATE_B_MID)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if quick:
        network_sizes, training_budgets, run_seeds = [[16]], [2_000], [0]
        capability_subset = [([16], 2_000)]
        rho_kwargs = {"net_size": [16], "timesteps": 2_000}
        out_path = f"{RESULTS_DIR}/results_quick_test.csv"
        print("Running --quick: tiny grid, just validating the pipeline. "
              f"Writing to {out_path} (won't touch your real results.csv).")
    else:
        network_sizes, training_budgets = NETWORK_SIZES, TRAINING_BUDGETS
        run_seeds = seeds if seeds is not None else SEEDS
        capability_subset = ADVERSARIAL_CAPABILITY_SUBSET
        rho_kwargs = {}
        if seeds is not None:
            # Separate file per process -- running several of these at once
            # against the SAME file risks interleaved/corrupted writes.
            # Merge them afterwards (see the printed instructions).
            tag = "_".join(str(s) for s in run_seeds)
            out_path = f"{RESULTS_DIR}/results_seeds_{tag}.csv"
        else:
            out_path = RESULTS_CSV_PATH
        print(f"Running seeds {run_seeds} (of {SEEDS}). Writing to {out_path}.")

    for seed in run_seeds:
        print(f"=== seed {seed}: training rho (fixed across the g-power sweep) ===")
        rho_model = train_rho(seed, hp, **rho_kwargs)

        for net_size in network_sizes:
            for budget in training_budgets:
                cell = (net_size, budget)
                print(f"--- seed={seed} net_size={net_size} budget={budget} ---")
                rows = []

                # --- standard: full grid, all three architectures ---
                model_a = train_architecture_a(net_size, budget, seed, hp)
                rows.append(evaluate_cell("A", net_size, budget, seed, "standard",
                                           as_policy_fn(model_a)))

                g_model = train_g(net_size, budget, seed, hp)
                dual_standard = DualLoopPolicy(g_model, rho_model, k=gate_k, b_mid=gate_b_mid,
                                                rng=np.random.default_rng(seed))
                rows.append(evaluate_cell("B", net_size, budget, seed, "standard",
                                           as_dual_loop_policy_fn(dual_standard)))

                # C reuses the SAME g_model/rho_model as B in the standard
                # condition -- only the gate differs between B and C here,
                # so any difference is attributable to the gate mechanism
                # alone, not to different g/rho training. The gate's own
                # net_size/timesteps scale with this cell, per the
                # confirmed design decision (see train_gate's docstring).
                gate_model = train_gate(g_model, rho_model, net_size, budget, seed, hp)
                learned_gate_standard = LearnedGatePolicy(g_model, rho_model, gate_model)
                rows.append(evaluate_cell("C", net_size, budget, seed, "standard",
                                           as_learned_gate_policy_fn(learned_gate_standard)))

                # --- everything else: only on the capability subset ---
                if cell in capability_subset:
                    # A: goal_nulling (untrained)
                    model_a_null = get_untrained_architecture_a(net_size, seed)
                    rows.append(evaluate_cell("A", net_size, budget, seed, "goal_nulling",
                                               as_policy_fn(model_a_null)))

                    # A: indifference (battery term removed, task reward unchanged)
                    model_a_indiff = train_architecture_a(net_size, budget, seed, hp,
                                                            lambda_battery=0.0)
                    rows.append(evaluate_cell("A", net_size, budget, seed, "indifference",
                                               as_policy_fn(model_a_indiff)))

                    # A: override/conflict sweep
                    for w in OVERRIDE_WEIGHTS:
                        model_a_w = train_architecture_a(net_size, budget, seed, hp,
                                                           adversarial_weight=w)
                        rows.append(evaluate_cell("A", net_size, budget, seed, f"override_w{w}",
                                                   as_policy_fn(model_a_w), adversarial_weight=w))

                    # B: goal_nulling (g untrained, rho unaffected by construction)
                    g_null = get_untrained_g(net_size, seed)
                    dual_null = DualLoopPolicy(g_null, rho_model, k=gate_k, b_mid=gate_b_mid,
                                                rng=np.random.default_rng(seed))
                    rows.append(evaluate_cell("B", net_size, budget, seed, "goal_nulling",
                                               as_dual_loop_policy_fn(dual_null)))

                    # B: indifference — identical to standard by construction
                    # (g never sees r_battery either way); re-evaluated rather
                    # than copied, so a real bug would still show up here.
                    rows.append(evaluate_cell("B", net_size, budget, seed, "indifference",
                                               as_dual_loop_policy_fn(dual_standard)))

                    # B & C: override/conflict sweep — g_w is trained ONCE per
                    # w and reused for both B and C's evaluation at that w
                    # (previously retrained separately for each; g_w is
                    # deterministic given (net_size, budget, seed, w), so
                    # training it twice bought nothing but ~14M wasted
                    # timesteps across the full sweep). rho is untouched for B;
                    # the gate is trained fresh per w for C, since its training
                    # env must include the same risk-temptation tiles g_w saw.
                    for w in OVERRIDE_WEIGHTS:
                        g_w = train_g(net_size, budget, seed, hp, adversarial_weight=w)

                        dual_w = DualLoopPolicy(g_w, rho_model, k=gate_k, b_mid=gate_b_mid,
                                                 rng=np.random.default_rng(seed))
                        rows.append(evaluate_cell("B", net_size, budget, seed, f"override_w{w}",
                                                   as_dual_loop_policy_fn(dual_w), adversarial_weight=w))

                        gate_w = train_gate(g_w, rho_model, net_size, budget, seed, hp,
                                             adversarial_weight=w)
                        learned_gate_w = LearnedGatePolicy(g_w, rho_model, gate_w)
                        rows.append(evaluate_cell("C", net_size, budget, seed, f"override_w{w}",
                                                   as_learned_gate_policy_fn(learned_gate_w),
                                                   adversarial_weight=w))

                    # C: goal_nulling -- same untrained g as B's goal_nulling; the
                    # gate is still trained normally on top of it (the env's own
                    # reward isn't nulled, only g's training signal was)
                    gate_null = train_gate(g_null, rho_model, net_size, budget, seed, hp)
                    learned_gate_null = LearnedGatePolicy(g_null, rho_model, gate_null)
                    rows.append(evaluate_cell("C", net_size, budget, seed, "goal_nulling",
                                               as_learned_gate_policy_fn(learned_gate_null)))

                    # C: indifference -- unlike B (where this is a no-op by
                    # construction), this is a real, distinct condition for C:
                    # the GATE's own training signal drops r_battery
                    # (lambda_battery=0.0), while g and rho are trained exactly
                    # as for the standard condition. This tests whether an
                    # arbitration mechanism that's itself indifferent to
                    # battery still favors rho often enough to matter.
                    gate_indiff = train_gate(g_model, rho_model, net_size, budget, seed, hp,
                                              lambda_battery=0.0)
                    learned_gate_indiff = LearnedGatePolicy(g_model, rho_model, gate_indiff)
                    rows.append(evaluate_cell("C", net_size, budget, seed, "indifference",
                                               as_learned_gate_policy_fn(learned_gate_indiff)))

                write_rows(rows, out_path)

    print(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Run a tiny grid to validate the whole pipeline before the full sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Run only these seeds, e.g. --seeds 0 1. Defaults to all of "
                              "config.SEEDS. Launch several of these with disjoint seed lists "
                              "to parallelize across cores -- each writes its own "
                              "results_seeds_<...>.csv, so there's no write conflict.")
    args = parser.parse_args()
    run(quick=args.quick, seeds=args.seeds)

    if args.seeds is not None:
        print(f"\nWhen every --seeds process is done, merge them with:\n"
              f'  python -c "import pandas as pd, glob; '
              f"pd.concat([pd.read_csv(f) for f in glob.glob('{RESULTS_DIR}/results_seeds_*.csv')])"
              f".to_csv('{RESULTS_CSV_PATH}', index=False)\"")
