"""
arbitration.py
---------------
Combines g and rho into a single action at rollout/evaluation time, using
the probabilistic urgency gate confirmed for this project (matches the
AIWAN paper's §3.2 exactly):

    lambda(b) = sigmoid(k * (b - b_mid))

With probability lambda(b), the executed action is rho's; otherwise it's g's.
This is a faithful re-implementation of the paper's stochastic
discretization of the blend — see AIWAN §3.2's note that lambda(b) governs
a selection probability here, not a continuous interpolation weight.
"""

import numpy as np


class DualLoopPolicy:
    def __init__(self, g_model, rho_model, k, b_mid, rng=None):
        self.g_model = g_model
        self.rho_model = rho_model
        self.k = k
        self.b_mid = b_mid
        self.rng = rng if rng is not None else np.random.default_rng()

    def _lambda(self, battery_level):
        return 1.0 / (1.0 + np.exp(-self.k * (battery_level - self.b_mid)))

    def predict(self, obs, battery_level, deterministic=True):
        """
        battery_level must be passed in un-normalized units (0..battery_max),
        matching config.GATE_B_MID's units — not the [0,1]-normalized value
        that's inside obs. obs itself is the full 7-dim observation, exactly
        what g_model/rho_model were trained on.
        """
        lam = self._lambda(battery_level)
        if self.rng.random() < lam:
            action, _ = self.rho_model.predict(obs, deterministic=deterministic)
        else:
            action, _ = self.g_model.predict(obs, deterministic=deterministic)
        return action


class LearnedGatePolicy:
    """
    Architecture C: arbitration is a learned 2-action policy (gate_model,
    trained via env.LearnedGateEnv / train.train_gate) choosing between
    g's and rho's action at each step, instead of DualLoopPolicy's
    hardcoded lambda(b). Unlike DualLoopPolicy, the gate here sees the
    full observation (not just battery level) and was optimized on the
    combined objective -- so it can, in principle, learn a decision
    boundary that isn't purely battery-driven, which is exactly the
    design difference this architecture exists to test. No battery_level
    argument is needed: the gate gets everything it was trained on
    through obs alone.
    """

    def __init__(self, g_model, rho_model, gate_model):
        self.g_model = g_model
        self.rho_model = rho_model
        self.gate_model = gate_model

    def predict(self, obs, deterministic=True):
        gate_action, _ = self.gate_model.predict(obs, deterministic=deterministic)
        gate_action = int(gate_action)
        sub_model = self.rho_model if gate_action == 1 else self.g_model
        action, _ = sub_model.predict(obs, deterministic=deterministic)
        return action
