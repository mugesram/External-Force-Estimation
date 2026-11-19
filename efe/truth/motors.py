"""Motor/ESC truth model: open-loop PWM ESC, rotor-speed lag, quadratic thrust.

    omega_cmd,i = (u_scale * u_i + u_offset) * (V / V0)^(gamma / 2)     Pegasus' u -> omega map, scaled by voltage
    d omega_i / dt = (omega_cmd,i - omega_i) / tau                        first-order motor lag (exact discretisation)
    T_i = k_f,i * omega_i^2

gamma = 2 is the physics the filter assumes (thrust ~ V^2); other values are a deliberate truth/model mismatch.
numpy only: shared by the Tier 0 simulator and the Isaac/Pegasus thrust curve.
"""

from __future__ import annotations

import math

import numpy as np


class MotorModel:
    def __init__(self, k_f=8.54858e-6, n=4, tau=0.03, V0=16.8, u_scale=1000.0, u_offset=100.0,
                 voltage_exponent=2.0, k_f_spread=0.0, rng=None):
        self.n = int(n)
        k_f = np.broadcast_to(np.asarray(k_f, dtype=float), (self.n,)).copy()
        if k_f_spread > 0:
            rng = rng or np.random.default_rng(0)
            k_f *= 1.0 + rng.uniform(-k_f_spread, k_f_spread, self.n)
        self.k_f = k_f
        self.tau = float(tau)
        self.V0 = float(V0)
        self.u_scale, self.u_offset = float(u_scale), float(u_offset)
        self.voltage_exponent = float(voltage_exponent)
        self.omega = np.zeros(self.n)

    @classmethod
    def from_config(cls, cfg, rng=None):
        keys = ("k_f", "tau", "V0", "u_scale", "u_offset", "voltage_exponent", "k_f_spread")
        return cls(**{k: cfg[k] for k in keys if k in cfg}, rng=rng)

    # --- truth quantities the filter estimates --------------------------------------------------------------
    @property
    def u0(self):
        return self.u_offset / self.u_scale

    def k_b(self):
        """Per-motor thrust at full command and V = V0 (mean over motors): the filter's k_b."""
        return float(np.mean(self.k_f)) * (self.u_scale + self.u_offset) ** 2

    def k_eff(self, V):
        """Per-motor thrust at full command at voltage V."""
        return self.k_b() * (V / self.V0) ** self.voltage_exponent

    def omega_cmd(self, u, V):
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        scale = max(V, 0.0) / self.V0
        return (self.u_scale * u + self.u_offset) * scale ** (self.voltage_exponent / 2.0)

    def reset(self, u, V):
        self.omega = self.omega_cmd(u, V)

    def step(self, u, V, dt):
        """Advance the rotor speeds by dt; returns per-rotor thrust [N]."""
        ref = self.omega_cmd(u, V)
        if self.tau > 0:
            keep = math.exp(-dt / self.tau)
            self.omega = keep * self.omega + (1.0 - keep) * ref
        else:
            self.omega = ref
        return self.thrusts()

    def thrusts(self):
        return self.k_f * self.omega ** 2
