"""VoltageLaggedThrustCurve: Pegasus' rotors with an open-loop ESC (voltage-scaled), motor lag and a battery.

Pegasus' PX4 backend turns PX4's motor command u into a rotor-speed reference omega_ref = 1000 u + 100 (0 when
disarmed) and its QuadraticThrustCurve applies k_f omega_ref^2 instantly. Here the reference is turned back into u,
and the truth model of efe.truth.motors applies: omega_cmd = (1000 u + 100) (V/V0), first-order lag, T_i = k_f omega_i^2,
with V from efe.truth.battery driven by the rotor thrusts.
"""

from __future__ import annotations

import numpy as np
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve

from efe.truth.battery import BatteryModel
from efe.truth.motors import MotorModel


class VoltageLaggedThrustCurve(QuadraticThrustCurve):
    def __init__(self, vehicle_cfg: dict, battery: BatteryModel, rng=None,
                 rolling_moment_coefficient=1e-6, rot_dir=(-1, -1, 1, 1)):
        self.motors = MotorModel.from_config(vehicle_cfg, rng=rng)
        super().__init__({
            "num_rotors": 4,
            "rotor_constant": list(self.motors.k_f),
            "rolling_moment_coefficient": [rolling_moment_coefficient] * 4,
            "rot_dir": list(rot_dir),
            "max_rotor_velocity": [float(vehicle_cfg.get("omega_max", 1100.0))] * 4,
        })
        self.battery = battery
        self.u = np.zeros(4)
        self.armed = False

    def set_input_reference(self, input_reference):
        ref = np.asarray(input_reference, dtype=float)
        self.armed = bool(np.any(ref > 0.0))
        m = self.motors
        self.u = np.clip((ref - m.u_offset) / m.u_scale, 0.0, 1.0) if self.armed else np.zeros(4)
        self._input_reference = list(ref)

    def update(self, state, dt: float):
        m = self.motors
        V = self.battery.voltage
        if self.armed:
            thrusts = m.step(self.u, V, dt)
        else:
            m.omega = m.omega * np.exp(-dt / max(m.tau, 1e-6))   # spinning down
            thrusts = m.thrusts()
        self.battery.step(thrusts, dt)
        self._velocity = list(m.omega)
        self._force = list(thrusts)
        self._rolling_moment = float(np.sum(np.asarray(self._rolling_moment_coefficient) * m.omega ** 2
                                            * np.asarray(self._rot_dir)))
        return self._force, self._velocity, self._rolling_moment
