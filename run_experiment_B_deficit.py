"""
run_experiment_B_deficit.py
----------------------------
B-only re-run of the §7.3 grid with the gate fed in the direction §3.2/§5
specify (urgency rises as battery LEVEL falls), instead of the inverted
direction the original run_experiment.py used.

Place this file in aiwan_scaling/ next to run_experiment.py, config.py,
train.py, arbitration.py and evaluate.py. It changes none of them and never
touches results/results.csv: output goes to results/results_B_deficit_*.csv
with architecture label "B_deficit", so A, B (old gate) and C stay as they were.

Gate (pre-specified, NOT tuned after seeing results):
    lambda(level) = sigmoid(k * (b_mid - level)),   k = hp gate_k
    b_mid = drain * worst_case_distance + ln(9)/k
(k is read from results/hyperparams.json: 0.1 for the §7.3 run, the same k the
original B used, so only the direction and midpoint change. With k=0.1 this gives
b_mid = 4.0 * 10 + 21.97 = 61.97; the original B used b_mid = 40.)
i.e. the reflex fires with probability 0.9 once the remaining battery only
just covers the worst-case trip home (same rule as the §5 derivation).
Do not change --b-mid after looking at results; if you must, report both.

Modes:
    --quick     tiny pipeline check (a few minutes at most)
    --pilot     ONE cell (net [256], budget 100,000), seed 0, all weights;
                prints wall-clock time and a projection for the other modes
    --reduced   sizes {16,256,2048} x budgets {10k,500k} x w {0,1.5,8}, 10 seeds
    (default)   full §7.3 grid: 6 sizes x 4 budgets x 6 weights, 10 seeds
    --seeds 0 1 2   restrict to some seeds (one output file per process)

Per (size, budget, seed) it trains one g per override weight (w=0 doubles as
the standard / indifference g, as in the original data where standard and
override_w0 coincide) and evaluates: standard, goal_nulling, indifference,
override_w{w}. goal_nulling uses an untrained g, as in the original.
"""

import argparse
import math
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)

from config import NETWORK_SIZES, TRAINING_BUDGETS, SEEDS, OVERRIDE_WEIGHTS, RESULTS_DIR, GATE_K
from run_experiment import ROW_FIELDS, load_hp, write_rows
from train import train_g, train_rho, get_untrained_g
from arbitration import DualLoopPolicy
from evaluate import evaluate_cell, as_dual_loop_policy_fn

ARCH = "B_deficit"
BATTERY_DRAIN = 4.0          # env.py default battery_drain
WORST_CASE_DISTANCE = 10     # 6x6 grid, charger at (0,0): 5 + 5


class UrgencyGatePolicy(DualLoopPolicy):
    """lambda(level) = sigmoid(k * (b_mid - level)): rho is chosen with
    high probability when the battery LEVEL is low (the §3.2/§5 direction)."""

    def _lambda(self, battery_level):
        return 1.0 / (1.0 + np.exp(-self.k * (self.b_mid - battery_level)))


def derived_b_mid(k):
    return BATTERY_DRAIN * WORST_CASE_DISTANCE + math.log(9.0) / k


def n_steps(sizes, budgets, weights, seeds):
    return len(weights) * sum(budgets) * len(sizes) * len(seeds)


def plan(mode, seeds):
    if mode == "quick":
        return [[16]], [2_000], [0, 8], [0]
    if mode == "pilot":
        return [[256]], [100_000], list(OVERRIDE_WEIGHTS), [0]
    if mode == "reduced":
        return [[16], [256], [2048]], [10_000, 500_000], [0, 1.5, 8], (seeds or SEEDS)
    return NETWORK_SIZES, TRAINING_BUDGETS, list(OVERRIDE_WEIGHTS), (seeds or SEEDS)


def run(mode, seeds=None, b_mid=None):
    hp = load_hp()
    k = hp.get("gate_k", GATE_K)
    b_mid = derived_b_mid(k) if b_mid is None else b_mid
    sizes, budgets, weights, run_seeds = plan(mode, seeds)
    assert 0 in weights, "w=0 is the standard/indifference g and must be included"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tag = "_".join(str(s) for s in run_seeds)
    if mode in ("quick", "pilot"):
        out_path = f"{RESULTS_DIR}/results_B_deficit_{mode}_test.csv"
        if os.path.exists(out_path):
            os.remove(out_path)
    else:
        out_path = f"{RESULTS_DIR}/results_B_deficit_seeds_{tag}.csv"
        if os.path.exists(out_path):
            print(f"WARNING: {out_path} exists and will be APPENDED to (duplicate rows if re-run).")

    rho_kwargs = {"net_size": [16], "timesteps": 2_000} if mode == "quick" else {}
    total = n_steps(sizes, budgets, weights, run_seeds)
    print(f"mode={mode} seeds={run_seeds} k={k} b_mid={b_mid:.2f} -> {out_path}")
    print(f"g-training steps to run: {total:,} (+ rho per seed)")

    t0 = time.time()
    done_steps = 0
    for seed in run_seeds:
        print(f"=== seed {seed}: training rho ===", flush=True)
        rho_model = train_rho(seed, hp, **rho_kwargs)

        for net_size in sizes:
            for budget in budgets:
                tc = time.time()
                rows = []
                g_models = {}
                for w in weights:
                    g_models[w] = train_g(net_size, budget, seed, hp, adversarial_weight=w)
                    done_steps += budget

                def policy_for(g):
                    return as_dual_loop_policy_fn(
                        UrgencyGatePolicy(g, rho_model, k=k, b_mid=b_mid,
                                          rng=np.random.default_rng(seed)))

                g_std = g_models[0]
                rows.append(evaluate_cell(ARCH, net_size, budget, seed, "standard", policy_for(g_std)))
                g_null = get_untrained_g(net_size, seed)
                rows.append(evaluate_cell(ARCH, net_size, budget, seed, "goal_nulling", policy_for(g_null)))
                # B never sees r_battery, so indifference == standard by construction
                rows.append(evaluate_cell(ARCH, net_size, budget, seed, "indifference", policy_for(g_std)))
                for w in weights:
                    rows.append(evaluate_cell(ARCH, net_size, budget, seed, f"override_w{w}",
                                              policy_for(g_models[w]), adversarial_weight=w))
                write_rows(rows, out_path)
                el = time.time() - t0
                print(f"seed={seed} net={net_size} budget={budget}: cell {time.time()-tc:.0f}s, "
                      f"total {el/3600:.2f}h, {done_steps/total:.0%} of g-steps", flush=True)

    el = time.time() - t0
    print(f"Done in {el/3600:.2f} h. Results: {out_path}")
    if mode == "pilot":
        sps = done_steps / el
        print(f"\nPilot speed (net [256], includes eval): {sps:,.0f} g-steps/s on this core.")
        for m, lab in (("reduced", "reduced"), ("full", "full")):
            s, b, w, sd = plan(m, None)
            st = n_steps(s, b, w, sd)
            print(f"  {lab}: {st:,} steps -> ~{st/sps/3600:.1f} core-hours "
                  f"(divide by number of parallel cores; nets of 1024/2048 are slower than [256], "
                  f"so treat this as a lower bound)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--reduced", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--b-mid", type=float, default=None,
                    help="override the derived b_mid (do not tune after seeing results)")
    a = ap.parse_args()
    mode = "quick" if a.quick else "pilot" if a.pilot else "reduced" if a.reduced else "full"
    run(mode, seeds=a.seeds, b_mid=a.b_mid)
