"""
tune_hyperparams.py
--------------------
Small validation run to choose hyperparameters BEFORE the real sweep, per
the paper's methodology note (§7): nothing here is chosen after looking at
final results. run_experiment.py reads results/hyperparams.json rather
than accepting hand-tuned values.

Uses config.VALIDATION_SEEDS, which is disjoint from config.SEEDS, so no
run in the final sweep ever contributes to choosing its own hyperparameters.

Design choice: learning_rate and gamma are shared across Architecture A,
g, and rho. Giving one architecture better-tuned optimization settings
than the other would bias the A-vs-B comparison the whole experiment is
about, so both get the same values, chosen on A's task (the harder,
combined one). The same fairness logic applies to lambda_battery_a (A's
combination weight) and GATE_K/GATE_B_MID (B's combination parameters):
both get their own tuning pass so neither architecture is handicapped by
an arbitrary default.
"""

import json
import os

import numpy as np

from config import (
    HYPERPARAMS_PATH, RESULTS_DIR,
    VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, VALIDATION_SEEDS,
)
from train import train_architecture_a, train_g, train_rho
from arbitration import DualLoopPolicy
from evaluate import evaluate_cell, as_policy_fn, as_dual_loop_policy_fn

LR_CANDIDATES = [1e-3, 5e-4]
GAMMA_CANDIDATES = [0.95, 0.99]
LAMBDA_CANDIDATES = [0.5, 1.0, 2.0, 5.0]
GATE_K_CANDIDATES = [0.1, 0.2, 0.4]
GATE_B_MID_CANDIDATES = [20.0, 30.0, 40.0]


def _score(row):
    """success_rate + (1 - crash_rate): favors neither architecture nor
    either hypothesis (collapse vs. resistance) — it just rewards a policy
    that reaches the goal and doesn't crash, whatever combination that is."""
    return row["success_rate"] + (1.0 - row["crash_rate"])


def _mean_score_over_seeds(build_and_eval_fn):
    return float(np.mean([_score(build_and_eval_fn(s)) for s in VALIDATION_SEEDS]))


def tune_lr_gamma():
    print("Tuning learning_rate / gamma on Architecture A...")
    best_score, best_hp = -np.inf, None
    for lr in LR_CANDIDATES:
        for gamma in GAMMA_CANDIDATES:
            hp = {"learning_rate": lr, "gamma": gamma, "lambda_battery_a": 1.0}

            def build_and_eval(seed, hp=hp):
                model = train_architecture_a(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, hp)
                return evaluate_cell("A", VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed,
                                      "tuning", as_policy_fn(model), n_episodes=30)

            score = _mean_score_over_seeds(build_and_eval)
            print(f"  lr={lr} gamma={gamma} -> score={score:.3f}")
            if score > best_score:
                best_score, best_hp = score, {"learning_rate": lr, "gamma": gamma}
    print(f"Chosen: {best_hp} (score={best_score:.3f})")
    return best_hp


def tune_lambda_battery_a(lr_gamma_hp):
    print("Tuning lambda_battery_a on Architecture A...")
    best_score, best_lambda = -np.inf, None
    for lam in LAMBDA_CANDIDATES:
        hp = {**lr_gamma_hp, "lambda_battery_a": lam}

        def build_and_eval(seed, hp=hp, lam=lam):
            model = train_architecture_a(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, hp,
                                          lambda_battery=lam)
            return evaluate_cell("A", VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed,
                                  "tuning", as_policy_fn(model), n_episodes=30)

        score = _mean_score_over_seeds(build_and_eval)
        print(f"  lambda_battery_a={lam} -> score={score:.3f}")
        if score > best_score:
            best_score, best_lambda = score, lam
    print(f"Chosen: lambda_battery_a={best_lambda} (score={best_score:.3f})")
    return best_lambda


def tune_gate_params(hp):
    """
    Trains one validation g and one validation rho per validation seed
    (using the shared hp already chosen above), then searches over
    GATE_K / GATE_B_MID on top of those fixed, already-trained policies.
    This mirrors how lambda_battery_a is tuned for A: it's the
    arbitration/combination parameter that gets searched, not the
    underlying policies.
    """
    print("Training validation g/rho for gate tuning...")
    g_models = {s: train_g(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, s, hp) for s in VALIDATION_SEEDS}
    rho_models = {s: train_rho(s, hp) for s in VALIDATION_SEEDS}

    print("Tuning GATE_K / GATE_B_MID on Architecture B...")
    best_score, best_gate = -np.inf, None
    for k in GATE_K_CANDIDATES:
        for b_mid in GATE_B_MID_CANDIDATES:

            def build_and_eval(seed, k=k, b_mid=b_mid):
                dual = DualLoopPolicy(g_models[seed], rho_models[seed], k=k, b_mid=b_mid)
                return evaluate_cell("B", VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed,
                                      "tuning", as_dual_loop_policy_fn(dual), n_episodes=30)

            score = _mean_score_over_seeds(build_and_eval)
            print(f"  k={k} b_mid={b_mid} -> score={score:.3f}")
            if score > best_score:
                best_score, best_gate = score, {"gate_k": k, "gate_b_mid": b_mid}
    print(f"Chosen: {best_gate} (score={best_score:.3f})")
    return best_gate


if __name__ == "__main__":
    # Full run — this takes a while (roughly 30 training runs at
    # VALIDATION_TIMESTEPS each). Run this once, before run_experiment.py,
    # and don't re-run it after looking at final results:
    #   python tune_hyperparams.py
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lr_gamma = tune_lr_gamma()
    lambda_battery_a = tune_lambda_battery_a(lr_gamma)
    gate = tune_gate_params(lr_gamma)

    final_hp = {**lr_gamma, "lambda_battery_a": lambda_battery_a, **gate}
    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(final_hp, f, indent=2)
    print(f"\nSaved to {HYPERPARAMS_PATH}: {final_hp}")
