"""
train.py
--------
Training functions. This version only has Architecture A; the g/rho
functions for Architecture B are added to this same file in the next stage.
"""

from stable_baselines3 import DQN
from env import (
    BatteryGridWorldEnv,
    RewardWrapperA,
    RewardWrapperGoalOnly,
    RewardWrapperBatteryOnly,
    LearnedGateEnv,
)


def _make_dqn(env, net_arch, seed, hp):
    """Shared DQN settings for every training run in this project — defined
    once here so it isn't duplicated in train_architecture_a and (later)
    train_g/train_rho. Reads learning_rate and gamma from hp rather than
    hardcoding them, since per §7 these must come from tune_hyperparams.py,
    not be tuned by hand. buffer_size is cut from SB3's 1M default to 100k:
    our env's episodes are short and budgets go up to 500k steps, so 100k
    transitions is still generous, and it matters when several training
    processes run in parallel on a memory-limited machine (each process
    gets its own buffer)."""
    return DQN(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": net_arch},
        learning_rate=hp.get("learning_rate", 1e-3),
        gamma=hp.get("gamma", 0.99),
        buffer_size=100_000,
        seed=seed,
        verbose=0,
    )


def train_architecture_a(net_size, timesteps, seed, hp,
                          adversarial_weight=0.0, lambda_battery=None):
    """
    Architecture A: a single policy, R = r_goal + lambda_battery * r_battery.

    lambda_battery=None  -> comes from hp (tune_hyperparams.py output) -> standard condition
    lambda_battery=0.0   -> indifference scenario (coordination point #3, one fixed run)
    adversarial_weight>0 -> override/conflict scenario (sweep over OVERRIDE_WEIGHTS)

    One function covers all three conditions for A — no duplicated code.
    """
    lam = lambda_battery if lambda_battery is not None else hp.get("lambda_battery_a", 1.0)
    env = RewardWrapperA(
        BatteryGridWorldEnv(adversarial_weight=adversarial_weight),
        lambda_battery=lam,
    )
    model = _make_dqn(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_g(net_size, timesteps, seed, hp, adversarial_weight=0.0):
    """
    g in Architecture B — sees r_goal only, never r_battery.

    adversarial_weight > 0 -> override/conflict scenario (sweep over OVERRIDE_WEIGHTS)
    standard and indifference conditions both use this same call with
    adversarial_weight=0 — g never sees the battery term either way, so
    "indifference" changes nothing for g by construction (that's the point
    of the architecture, not an oversight — see coordination point #3).
    """
    env = RewardWrapperGoalOnly(BatteryGridWorldEnv(adversarial_weight=adversarial_weight))
    model = _make_dqn(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_rho(seed, hp, net_size=None, timesteps=None):
    """
    rho in Architecture B — trained once per seed, independent of the
    g-power sweep (coordination point #2, confirmed). The same rho model
    is reused across every (network_size, training_budget) cell for that
    seed. net_size/timesteps default to config.RHO_NETWORK_SIZE /
    config.RHO_TIMESTEPS; the override is only used by run_experiment.py's
    --quick mode.
    """
    from config import RHO_NETWORK_SIZE, RHO_TIMESTEPS
    net_size = net_size or RHO_NETWORK_SIZE
    timesteps = timesteps or RHO_TIMESTEPS
    env = RewardWrapperBatteryOnly(BatteryGridWorldEnv())
    model = _make_dqn(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def train_gate(g_model, rho_model, net_size, timesteps, seed, hp,
                adversarial_weight=0.0, lambda_battery=None):
    """
    Architecture C (learned gate) -- see NEXT_STEP_learned_gate.md, Design 1.

    g_model and rho_model are already-trained, fixed policies (rho exactly
    as in Architecture B; g trained the same way B's g is, including under
    adversarial_weight for the override sweep -- pass the matching g_model
    for whichever condition is being run). Only the gate is optimized here,
    via LearnedGateEnv's 2-action space, on the combined objective.

    net_size/timesteps are meant to be the SAME (net_size, timesteps) cell
    as whatever g this gate is paired with -- the confirmed design decision
    is that the gate must scale with g's power, or this reduces to another
    fixed component one level removed and the test is foregone by
    construction again, just like the hardcoded gate it's meant to improve on.

    lambda_battery=None  -> 1.0, the standard combined objective
    lambda_battery=0.0   -> engineered-indifference analogue for this
                            architecture (drops the battery term from the
                            GATE's training signal only -- g and rho are
                            unaffected, trained exactly as for B)
    adversarial_weight>0 -> override/conflict scenario; must match the
                            weight g_model was itself trained under, since
                            the gate's env needs the same risk-temptation
                            tiles active that g saw.
    """
    lam = lambda_battery if lambda_battery is not None else 1.0
    env = LearnedGateEnv(
        BatteryGridWorldEnv(adversarial_weight=adversarial_weight),
        g_model, rho_model, lambda_battery=lam,
    )
    model = _make_dqn(env, net_size, seed, hp)
    model.learn(total_timesteps=timesteps)
    return model


def get_untrained_g(net_size, seed):
    """
    goal-nulling scenario: reward == 0 means gradient == 0, so training is
    meaningless — we just initialize the network randomly and evaluate
    that, rather than burning compute "training" on a constant-zero signal
    (coordination point #3).
    """
    env = RewardWrapperGoalOnly(BatteryGridWorldEnv())
    model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": net_size}, seed=seed, verbose=0)
    return model  # no model.learn() call


def get_untrained_architecture_a(net_size, seed):
    """
    goal-nulling for Architecture A: for a single loop, U is the *entire*
    training signal, so nulling it means there is no gradient anywhere,
    not just on the task term. We evaluate a randomly-initialized policy,
    mirroring get_untrained_g above.
    """
    env = RewardWrapperA(BatteryGridWorldEnv(), lambda_battery=0.0)
    model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": net_size}, seed=seed, verbose=0)
    return model  # no model.learn() call


if __name__ == "__main__":
    # Quick manual smoke test with the smallest combination — takes a few seconds:
    #   python train.py
    from arbitration import DualLoopPolicy
    from config import GATE_K, GATE_B_MID

    hp = {"learning_rate": 1e-3, "gamma": 0.99, "lambda_battery_a": 1.0}

    print("--- Architecture A ---")
    model_a = train_architecture_a(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    env = RewardWrapperA(BatteryGridWorldEnv(), lambda_battery=1.0)
    obs, _ = env.reset(seed=0)
    total_reward, info = 0.0, {}
    for _ in range(100):
        action, _ = model_a.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"reached_goal={info['reached_goal']} crashed={info['crashed']} "
          f"total_reward={total_reward:.2f}")

    print("--- Architecture B ---")
    g_model = train_g(net_size=[16], timesteps=2_000, seed=0, hp=hp)
    rho_model = train_rho(seed=0, hp=hp)
    dual = DualLoopPolicy(g_model, rho_model, k=GATE_K, b_mid=GATE_B_MID)

    raw_env = BatteryGridWorldEnv()
    obs, _ = raw_env.reset(seed=0)
    total_reward, info = 0.0, {}
    for _ in range(100):
        action = dual.predict(obs, battery_level=raw_env.battery, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"reached_goal={info['reached_goal']} crashed={info['crashed']} "
          f"total_reward={total_reward:.2f}")
