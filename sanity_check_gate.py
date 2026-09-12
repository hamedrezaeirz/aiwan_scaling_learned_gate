"""
sanity_check_gate.py
---------------------
Run this BEFORE committing Architecture C to the multi-day sweep --
companion to sanity_check.py, which only covers A/g/rho. Reuses that
file's reference baselines rather than duplicating them.

Checks two things a flat or degenerate result on the real sweep would
otherwise hide:

  1. Learning curve for the gate -- success/crash rate of the FULL
     composed system (g + rho + gate-so-far), evaluated periodically
     during the gate's training, using the same rollout() the real
     experiment trusts. g and rho are trained once, fully, and held
     fixed throughout -- only the gate is what's "learning" here, so a
     flat curve means something is wrong with the gate specifically
     (reward scale, learning rate, an env bug), not with g or rho.
     Plotted against Architecture B's hardcoded gate on the SAME g/rho,
     as the most direct comparison this project has: does a learned
     gate reach parity with a tuned hardcoded one, or does it plateau
     below it?
  2. A routing diagnostic: whether the trained gate's g-vs-rho choice
     actually varies with battery level, or has collapsed onto always
     picking one sub-policy regardless of state. A gate that always
     defers to rho would still score reasonably on crash_rate (since
     rho alone avoids crashing) while quietly failing to test anything
     about g's influence -- exactly the degenerate solution the
     combined objective doesn't rule out by construction, so it's worth
     checking directly rather than inferring it from the aggregate
     metrics above.

Usage:
    python sanity_check_gate.py
"""

import numpy as np
import matplotlib.pyplot as plt

from env import BatteryGridWorldEnv, LearnedGateEnv
from train import _make_dqn, train_g, train_rho
from arbitration import DualLoopPolicy, LearnedGatePolicy
from evaluate import rollout, as_learned_gate_policy_fn
from sanity_check import (
    CHECK_TIMESTEPS, CHECK_INTERVAL, CHECK_NET_SIZE, CHECK_SEED,
    random_baseline, greedy_toward_goal_baseline,
)

N_ROUTING_EPISODES = 50
BATTERY_BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


def gate_learning_curve(g_model, rho_model, hp):
    """
    Mirrors sanity_check.learning_curve()'s chunked train-then-evaluate
    loop, but the object being trained (the gate) isn't what's evaluated
    directly -- what's evaluated is the composed LearnedGatePolicy, since
    that's the system the real experiment actually scores.
    """
    env = LearnedGateEnv(BatteryGridWorldEnv(), g_model, rho_model)
    gate_model = _make_dqn(env, CHECK_NET_SIZE, CHECK_SEED, hp)

    steps_done, success_rates, crash_rates = [], [], []
    n_chunks = CHECK_TIMESTEPS // CHECK_INTERVAL
    for i in range(n_chunks):
        gate_model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
        composed = LearnedGatePolicy(g_model, rho_model, gate_model)
        metrics = rollout(as_learned_gate_policy_fn(composed), n_episodes=30, seed=1000 + i)
        steps_done.append((i + 1) * CHECK_INTERVAL)
        success_rates.append(metrics["success_rate"])
        crash_rates.append(metrics["crash_rate"])
        print(f"  [gate] step={steps_done[-1]:>7} "
              f"success_rate={metrics['success_rate']:.2f} crash_rate={metrics['crash_rate']:.2f}")

    return steps_done, success_rates, crash_rates, gate_model


def routing_diagnostic(g_model, rho_model, gate_model, n_episodes=N_ROUTING_EPISODES):
    """
    Runs real rollouts under the fully-trained composed policy and
    records, at every step, the battery level and which sub-policy the
    gate picked. Buckets by battery level and reports the fraction of
    steps routed to rho per bucket -- if this fraction is ~constant
    across buckets (all 0.0, all 1.0, or flat at some value regardless
    of bucket), the gate has learned a degenerate, state-independent
    rule and isn't doing the kind of conditional arbitration this
    architecture is meant to test, whatever the aggregate crash_rate says.
    """
    env = BatteryGridWorldEnv()
    bucket_counts = {b: [0, 0] for b in BATTERY_BUCKETS}  # [n_rho, n_total]

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=4000 + ep)
        for _ in range(env.max_episode_steps):
            gate_action, _ = gate_model.predict(obs, deterministic=True)
            gate_action = int(gate_action)
            battery_pct = 100.0 * env.battery / env.battery_max
            for lo, hi in BATTERY_BUCKETS:
                if lo <= battery_pct < hi or (hi == 100 and battery_pct == 100):
                    bucket_counts[(lo, hi)][1] += 1
                    bucket_counts[(lo, hi)][0] += gate_action  # action==1 means "defer to rho"
                    break
            sub_model = rho_model if gate_action == 1 else g_model
            real_action, _ = sub_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(real_action)
            if terminated or truncated:
                break

    print("\nRouting diagnostic -- fraction of steps the gate deferred to rho, by battery level:")
    fractions = []
    for (lo, hi) in BATTERY_BUCKETS:
        n_rho, n_total = bucket_counts[(lo, hi)]
        frac = n_rho / n_total if n_total > 0 else float("nan")
        fractions.append(frac)
        print(f"  battery in [{lo:>3}, {hi:>3}): n={n_total:>5}  fraction routed to rho = {frac:.2f}")

    spread = np.nanmax(fractions) - np.nanmin(fractions)
    if spread < 0.05:
        print(f"  WARNING: spread across buckets is only {spread:.2f} -- the gate may have "
              f"collapsed onto a state-independent rule (always g or always rho) rather than "
              f"learning battery-dependent routing. Worth a closer look before the real sweep.")
    else:
        print(f"  Spread across buckets: {spread:.2f} -- routing does vary with battery level.")

    return fractions


if __name__ == "__main__":
    import json
    from config import HYPERPARAMS_PATH
    with open(HYPERPARAMS_PATH) as f:
        hp = json.load(f)
    print(f"Using hyperparameters from {HYPERPARAMS_PATH}: {hp}\n")

    print("Reference baselines (no learning involved):")
    rand = random_baseline()
    print(f"  random policy:      success_rate={rand['success_rate']:.2f} "
          f"crash_rate={rand['crash_rate']:.2f}")
    greedy = greedy_toward_goal_baseline()
    print(f"  greedy-toward-goal: success_rate={greedy['success_rate']:.2f} "
          f"crash_rate={greedy['crash_rate']:.2f}")

    print(f"\nTraining g and rho fully ({CHECK_TIMESTEPS} steps each, held fixed for the gate)...")
    g_model = train_g(CHECK_NET_SIZE, CHECK_TIMESTEPS, CHECK_SEED, hp)
    rho_model = train_rho(CHECK_SEED, hp)

    print("\nEvaluating Architecture B (hardcoded gate) on this SAME g/rho, for comparison:")
    dual = DualLoopPolicy(g_model, rho_model, k=hp["gate_k"], b_mid=hp["gate_b_mid"])
    from evaluate import as_dual_loop_policy_fn
    b_metrics = rollout(as_dual_loop_policy_fn(dual), n_episodes=100, seed=5000)
    print(f"  B: success_rate={b_metrics['success_rate']:.2f} crash_rate={b_metrics['crash_rate']:.2f}")

    print("\nLearning curve, gate (Architecture C):")
    steps_c, succ_c, crash_c, gate_model = gate_learning_curve(g_model, rho_model, hp)

    routing_diagnostic(g_model, rho_model, gate_model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(steps_c, succ_c, marker="o", label="C (learned gate)")
    axes[0].axhline(b_metrics["success_rate"], color="tab:orange", linestyle="-.",
                     label="B (hardcoded gate, same g/rho)")
    axes[0].axhline(rand["success_rate"], color="gray", linestyle="--", label="random baseline")
    axes[0].axhline(greedy["success_rate"], color="black", linestyle=":", label="greedy-toward-goal")
    axes[0].set_xlabel("gate training steps")
    axes[0].set_ylabel("success_rate")
    axes[0].set_title("Does the gate's success_rate climb toward B's?")
    axes[0].legend(fontsize="small")

    axes[1].plot(steps_c, crash_c, marker="o", label="C (learned gate)")
    axes[1].axhline(b_metrics["crash_rate"], color="tab:orange", linestyle="-.",
                     label="B (hardcoded gate, same g/rho)")
    axes[1].axhline(rand["crash_rate"], color="gray", linestyle="--", label="random baseline")
    axes[1].set_xlabel("gate training steps")
    axes[1].set_ylabel("crash_rate")
    axes[1].set_title("Does the gate's crash_rate drop toward B's?")
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig("results/sanity_check_gate_learning_curves.png", dpi=150)
    print("\nSaved results/sanity_check_gate_learning_curves.png")
