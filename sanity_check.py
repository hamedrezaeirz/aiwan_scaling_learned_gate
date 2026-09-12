"""
sanity_check.py
----------------
Run this BEFORE committing to the multi-day sweep. Checks that training
is producing real learning, not an artifact, using a moderate real
budget (not a 2000-step toy run):

  1. Learning curves for Architecture A, g, and rho -- success/crash rate
     evaluated periodically during training, using the exact same
     rollout() code path the real experiment trusts. A flat curve here
     means something is wrong (reward scale, learning rate, a bug) --
     better to find that out in ~10 minutes than after days of compute.
  2. Two reference baselines: a random policy, and a hand-coded
     "always move straight toward the goal" heuristic that ignores
     battery on purpose -- gives a sense of what "good" should look like,
     so a curve that's technically "above random" but far below the
     heuristic is still a red flag, not a pass.

Usage:
    python sanity_check.py
"""

import matplotlib.pyplot as plt

from env import (
    BatteryGridWorldEnv, RewardWrapperA, RewardWrapperGoalOnly,
    RewardWrapperBatteryOnly, UP, DOWN, LEFT, RIGHT,
)
from train import _make_dqn
from evaluate import rollout, as_policy_fn

CHECK_TIMESTEPS = 50_000     # a real budget, not a toy one
CHECK_INTERVAL = 5_000       # evaluate every this many steps
CHECK_NET_SIZE = [64]        # a middle-of-the-road size from NETWORK_SIZES
CHECK_SEED = 0


def learning_curve(build_env_fn, hp, label):
    """
    Trains in CHECK_INTERVAL-sized chunks, evaluating with our own
    rollout() after each chunk -- the same measurement the real
    experiment relies on, applied early and often instead of just once
    at the very end of a multi-day run.
    """
    env = build_env_fn()
    model = _make_dqn(env, CHECK_NET_SIZE, CHECK_SEED, hp)

    steps_done, success_rates, crash_rates = [], [], []
    n_chunks = CHECK_TIMESTEPS // CHECK_INTERVAL
    for i in range(n_chunks):
        model.learn(total_timesteps=CHECK_INTERVAL, reset_num_timesteps=False)
        metrics = rollout(as_policy_fn(model), n_episodes=30, seed=1000 + i)
        steps_done.append((i + 1) * CHECK_INTERVAL)
        success_rates.append(metrics["success_rate"])
        crash_rates.append(metrics["crash_rate"])
        print(f"  [{label}] step={steps_done[-1]:>7} "
              f"success_rate={metrics['success_rate']:.2f} crash_rate={metrics['crash_rate']:.2f}")

    return steps_done, success_rates, crash_rates


def random_baseline(n_episodes=100):
    env = BatteryGridWorldEnv()
    return rollout(lambda obs, battery: env.action_space.sample(), n_episodes, seed=2000)


def greedy_toward_goal_baseline(n_episodes=100):
    """
    Hand-coded, ignoring battery on purpose: always shrink whichever axis
    distance to the goal is larger. This is a rough ceiling for what a
    well-trained g's success_rate should approach -- a trained g that
    can't get close to this is undertrained or broken, regardless of
    what its crash_rate looks like.
    """
    def policy_fn(obs, battery_level):
        ax, ay, gx, gy = obs[0], obs[1], obs[2], obs[3]
        dx, dy = gx - ax, gy - ay
        if abs(dx) >= abs(dy):
            return RIGHT if dx > 0 else LEFT
        return DOWN if dy > 0 else UP

    return rollout(policy_fn, n_episodes, seed=3000)


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
          f"crash_rate={greedy['crash_rate']:.2f} (ignores battery on purpose, "
          f"a high crash_rate here is expected and fine)")

    print("\nLearning curve, Architecture A:")
    steps_a, succ_a, crash_a = learning_curve(
        lambda: RewardWrapperA(BatteryGridWorldEnv(), lambda_battery=hp["lambda_battery_a"]),
        hp, "A")

    print("\nLearning curve, g (goal-only):")
    steps_g, succ_g, _ = learning_curve(
        lambda: RewardWrapperGoalOnly(BatteryGridWorldEnv()), hp, "g")

    print("\nLearning curve, rho (battery-only):")
    # "success" isn't meaningful for rho (it never sees the goal) -- what
    # matters is crash_rate dropping toward 0, so that's what we track.
    steps_r, _, crash_r = learning_curve(
        lambda: RewardWrapperBatteryOnly(BatteryGridWorldEnv()), hp, "rho")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(steps_a, succ_a, marker="o", label="A")
    axes[0].plot(steps_g, succ_g, marker="o", label="g")
    axes[0].axhline(rand["success_rate"], color="gray", linestyle="--", label="random baseline")
    axes[0].axhline(greedy["success_rate"], color="black", linestyle=":", label="greedy-toward-goal")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("success_rate")
    axes[0].set_title("Does success_rate actually climb?")
    axes[0].legend(fontsize="small")

    axes[1].plot(steps_a, crash_a, marker="o", label="A")
    axes[1].plot(steps_r, crash_r, marker="o", label="rho")
    axes[1].axhline(rand["crash_rate"], color="gray", linestyle="--", label="random baseline")
    axes[1].set_xlabel("training steps")
    axes[1].set_ylabel("crash_rate")
    axes[1].set_title("Does crash_rate actually drop?")
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig("results/sanity_check_learning_curves.png", dpi=150)
    print("\nSaved results/sanity_check_learning_curves.png")
