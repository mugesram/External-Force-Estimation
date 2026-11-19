"""EfeMultirotor: Pegasus' Multirotor with lockstep ordering, an external-force injector and a truth logger.

Order within one physics step (the lockstep protocol's own):
  1. read this step's state into the sensors (IMU, GPS, ...)
  2. exchange with PX4: send the sensors, receive the motor commands PX4 computed from them
  3. apply the rotor forces of those commands, drag, and the scheduled external force
  4. log the truth for this step, stamped with the time PX4 sees (the backend's time_usec)
Stock Pegasus applies the previous command first and reads sensors after, several steps of extra latency that a real
flight controller doesn't have.
"""

from __future__ import annotations

import numpy as np
from pegasus.simulator.logic.vehicles.multirotor import Multirotor
from pegasus.simulator.logic.vehicles.vehicle import Vehicle
from scipy.spatial.transform import Rotation

from efe.estimation.frames import FRD_FROM_FLU, NED_FROM_ENU, quat_xyzw_enu_flu_to_wxyz_ned_frd

G = 9.80665


class EfeMultirotor(Multirotor):
    def __init__(self, *args, injector=None, logger=None, imu=None, scenario_start=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.injector = injector
        self.logger = logger
        self.imu = imu
        self.scenario_start = float(scenario_start)
        self.sim_time = 0.0
        self.F_ext_ned = np.zeros(3)
        self._v_prev_enu = None

    def update_sensors(self, dt: float):
        pass  # Pegasus' own physics callback; the sensors are read inside update() instead

    def update(self, dt: float):
        self.sim_time += dt
        Vehicle.update_sensors(self, dt)                    # 1. this step's state into the sensors
        for backend in self._backends:                      # 2. PX4 exchange
            backend.update(dt)
        ref = self._backends[0].input_reference() if self._backends else [0.0] * 4
        self._thrusters.set_input_reference(ref)
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)
        articulation = self.get_dc_interface().get_articulation(self._stage_prefix)
        for i in range(4):                                  # 3. forces
            self.apply_force([0.0, 0.0, forces_z[i]], body_part="/rotor" + str(i))
            self.handle_propeller_visual(i, forces_z[i], articulation)
        self.apply_torque([0.0, 0.0, rolling_moment], "/body")
        self.apply_force(self._drag.update(self._state, dt), body_part="/body")
        R_enu_flu = Rotation.from_quat(self._state.attitude).as_matrix()
        if self.injector is not None:
            # scenario time on PX4's clock (the backend's time_usec), the clock the offboard node flies on. It is not
            # the simulator's own time: Pegasus only starts advancing time_usec once PX4 has connected.
            self.F_ext_ned = np.asarray(self.injector(self.px4_time() - self.scenario_start), dtype=float)
            F_body_flu = R_enu_flu.T @ (NED_FROM_ENU.T @ self.F_ext_ned)
            if np.any(F_body_flu):
                self.apply_force(F_body_flu.tolist(), body_part="/body")
        if self.logger is not None:                         # 4. truth
            self._log(dt, forces_z, R_enu_flu)

    def px4_time(self):
        backend = self._backends[0] if self._backends else None
        utime = getattr(backend, "_current_utime", None)
        return utime * 1e-6 if utime is not None else self.sim_time

    def _log(self, dt, forces_z, R_enu_flu):
        st = self._state
        v_enu = np.asarray(st.linear_velocity, dtype=float)
        # specific force as Pegasus' IMU computes it (finite difference of the velocity), in FRD
        a_enu = np.zeros(3) if self._v_prev_enu is None else (v_enu - self._v_prev_enu) / dt
        self._v_prev_enu = v_enu
        f_flu = R_enu_flu.T @ (a_enu - np.array([0.0, 0.0, -G]))
        backend = self._backends[0] if self._backends else None
        thr = self._thrusters
        self.logger.log(
            t_usec=getattr(backend, "_current_utime", int(round(self.sim_time * 1e6))),
            p=NED_FROM_ENU @ np.asarray(st.position, dtype=float),
            v=NED_FROM_ENU @ v_enu,
            q=quat_xyzw_enu_flu_to_wxyz_ned_frd(st.attitude),
            F=self.F_ext_ned,
            T=float(np.sum(forces_z)),
            omega=np.asarray(thr.motors.omega),
            u=np.asarray(thr.u),
            V=thr.battery.voltage,
            soc=thr.battery.soc,
            k=thr.motors.k_b(),
            k_eff=thr.motors.k_eff(thr.battery.voltage),
            lam=1.0 / max(thr.motors.tau, 1e-9),
            b=self.imu.acc_bias_frd if self.imu is not None else np.zeros(3),
            f=FRD_FROM_FLU @ f_flu,
            w=FRD_FROM_FLU @ np.asarray(st.angular_velocity, dtype=float),
        )
