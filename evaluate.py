"""
evaluate.py
-----------
Rollout + metrics. rollout() doesn't care whether it's evaluating
Architecture A or B — it only needs a policy_fn(obs, battery_level) -> action
callable. as_policy_fn() and as_dual_loop_policy_fn() below adapt a plain
SB3 model and a DualLoopPolicy, respectively, to that one interface.
"""

import numpy as np
from env import BatteryGridWorldEnv
from config import N_EVAL_EPISODES


def as_policy_fn(sb3_model):
    """Adapts a plain SB3 model (Architecture A, or a lone untrained g for
    the goal-nulling scenario) to the (obs, battery_level) -> action interface."""
    def fn(obs, battery_level):
        action, _ = sb3_model.predict(obs, deterministic=True)
        return int(action)
    return fn


def as_dual_loop_policy_fn(dual_loop_policy):
    """Adapts a DualLoopPolicy to the same interface, for symmetry with
    as_policy_fn() even though DualLoopPolicy.predict already matches it."""
    def fn(obs, battery_level):
        return int(dual_loop_policy.predict(obs, battery_level, deterministic=True))
    return fn


def as_learned_gate_policy_fn(learned_gate_policy):
    """Adapts a LearnedGatePolicy (Architecture C) to the same
    (obs, battery_level) -> action interface. battery_level is accepted
    for signature compatibility with rollout() but unused: the learned
    gate gets everything it needs through obs, unlike DualLoopPolicy's
    hardcoded lambda(b)."""
    def fn(obs, battery_level):
        return int(learned_gate_policy.predict(obs, deterministic=True))
    return fn


def rollout(policy_fn, n_episodes, seed=None, adversarial_weight=0.0):
    """
    Runs n_episodes on a fresh BatteryGridWorldEnv and aggregates metrics.

    seed, if given, makes the evaluation batch reproducible: episode i is
    reset with seed + i, so the same call always evaluates the same set of
    goal/agent layouts — this matters for the reproducibility requirement
    in the paper's methodology note (§7): re-running an evaluation should
    reproduce the same numbers, not just the same training.
    """
    env = BatteryGridWorldEnv(adversarial_weight=adversarial_weight)
    successes = crashes = 0
    battery_levels, lengths = [], []

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        steps = 0
        info = {}
        for _ in range(env.max_episode_steps):
            action = policy_fn(obs, env.battery)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        successes += int(info.get("reached_goal", False))
        crashes += int(info.get("crashed", False))
        battery_levels.append(info.get("battery", 0.0))
        lengths.append(steps)

    return {
        "n_eval_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "crash_rate": crashes / n_episodes,
        "avg_battery_level": float(np.mean(battery_levels)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def evaluate_cell(architecture, network_size, timesteps, seed, condition,
                   policy_fn, n_episodes=N_EVAL_EPISODES, adversarial_weight=0.0):
    """
    Runs rollout() and packages the result into one row matching
    results.csv's schema (see the design doc). Doesn't train anything —
    training happens in run_experiment.py; this only evaluates an
    already-trained policy_fn under a given condition.
    """
    metrics = rollout(policy_fn, n_episodes, seed=seed, adversarial_weight=adversarial_weight)
    return {
        "architecture": architecture,
        "network_size": str(network_size),
        "training_timesteps": timesteps,
        "seed": seed,
        "condition": condition,
        **metrics,
    }


if __name__ == "__main__":
    # Quick manual smoke test — wires together train.py + arbitration.py +
    # this file's rollout, on the smallest possible combination:
    #   python evaluate.py
    from train import train_architecture_a, train_g, train_rho, train_gate
    from arbitration import DualLoopPolicy, LearnedGatePolicy
    from config import GATE_K, GATE_B_MID

    hp = {"learning_rate": 1e-3, "gamma": 0.99, "lambda_battery_a": 1.0}

    print("Training tiny models (a few seconds)...")
    model_a = train_architecture_a(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    g_model = train_g(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    rho_model = train_rho(seed=0, hp=hp)
    dual = DualLoopPolicy(g_model, rho_model, k=GATE_K, b_mid=GATE_B_MID)
    gate_model = train_gate(g_model, rho_model, net_size=[16], timesteps=2_000, seed=0, hp=hp)
    learned_gate = LearnedGatePolicy(g_model, rho_model, gate_model)

    row_a = evaluate_cell("A", [16], 2_000, 0, "standard",
                           as_policy_fn(model_a), n_episodes=20)
    row_b = evaluate_cell("B", [16], 2_000, 0, "standard",
                           as_dual_loop_policy_fn(dual), n_episodes=20)
    row_c = evaluate_cell("C", [16], 2_000, 0, "standard",
                           as_learned_gate_policy_fn(learned_gate), n_episodes=20)

    print("Architecture A:", row_a)
    print("Architecture B:", row_b)
    print("Architecture C:", row_c)
