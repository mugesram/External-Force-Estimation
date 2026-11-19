"""EfeIMU: Pegasus' IMU plus an accelerometer bias that actually reaches the output.

The stock Pegasus IMU in this checkout computes an accelerometer bias but never adds it (the term is commented out),
and drives its random walk with a uniform instead of a Gaussian draw. A real IMU has a bias, and PX4's
sensor_combined is not bias-corrected, so the force estimator has to deal with one. This subclass adds a seeded
Gauss-Markov bias (turn-on sigma, random walk, correlation time), in the FRD body frame of the output, and exposes it
as `acc_bias_frd` for the truth log.
"""

from __future__ import annotations

import math

import numpy as np
from pegasus.simulator.logic.sensors import IMU


class EfeIMU(IMU):
    def __init__(self, config=None, acc_bias_sigma=0.196, acc_bias_rw=0.006, acc_bias_tau=300.0, seed=0):
        super().__init__(config or {})
        self._rng = np.random.default_rng(seed)
        self._tau = float(acc_bias_tau)
        self._rw = float(acc_bias_rw)
        self.acc_bias_frd = self._rng.normal(0.0, acc_bias_sigma, 3) if acc_bias_sigma > 0 else np.zeros(3)

    def update(self, state, dt: float):
        out = super().update(state, dt)
        if out is None:
            return None
        phi = math.exp(-dt / self._tau)
        sd = math.sqrt(self._rw ** 2 * self._tau / 2.0 * (1.0 - math.exp(-2.0 * dt / self._tau)))
        self.acc_bias_frd = phi * self.acc_bias_frd + sd * self._rng.normal(0.0, 1.0, 3)
        out["linear_acceleration"] = np.asarray(out["linear_acceleration"]) + self.acc_bias_frd
        self._state = out
        return out
