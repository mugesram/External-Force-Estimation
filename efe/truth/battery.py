"""Battery truth model: V = OCV(SoC) - R0 * I, with I a monotonic function of thrust.

numpy only, Python 3.10+: the same class runs in the Tier 0 simulator and inside Isaac Sim's Python.
"""

from __future__ import annotations

import numpy as np

# Li-ion/LiPo resting voltage per cell vs state of charge
OCV_SOC = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OCV_CELL = [3.00, 3.30, 3.45, 3.60, 3.68, 3.74, 3.80, 3.86, 3.94, 4.02, 4.10, 4.20]


class BatteryModel:
    """cells: series cells. capacity_ah: deliberately small, so the voltage sags 20-30 % within one ~2 min flight.
    current = current_coeff * sum_i T_i^1.5 (a momentum-theory-shaped thrust -> power map) + idle_current."""

    def __init__(self, cells=4, capacity_ah=0.45, r0=0.03, current_coeff=0.425, idle_current=0.5, soc0=1.0):
        self.cells = int(cells)
        self.capacity_as = float(capacity_ah) * 3600.0
        self.r0 = float(r0)
        self.current_coeff = float(current_coeff)
        self.idle_current = float(idle_current)
        self.soc = float(soc0)
        self.current = 0.0
        self.voltage = self.ocv()

    @classmethod
    def from_config(cls, cfg):
        return cls(**{k: cfg[k] for k in ("cells", "capacity_ah", "r0", "current_coeff", "idle_current", "soc0")
                      if k in cfg})

    def ocv(self):
        return self.cells * float(np.interp(self.soc, OCV_SOC, OCV_CELL))

    def step(self, rotor_thrusts, dt):
        """Advance by dt with the given per-rotor thrusts [N]; returns the loaded terminal voltage."""
        t = np.clip(np.asarray(rotor_thrusts, dtype=float), 0.0, None)
        self.current = self.idle_current + self.current_coeff * float(np.sum(t ** 1.5))
        self.soc = max(self.soc - self.current * dt / self.capacity_as, 0.0)
        self.voltage = self.ocv() - self.r0 * self.current
        return self.voltage
