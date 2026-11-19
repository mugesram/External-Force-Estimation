"""Translational momentum observer (Tomić-style), the non-probabilistic baseline.

    r(t) = K_I * ( m v(t) - m v(0) - integral_0^t ( f_model + r ) dt )
    f_model = -T R e3 + m g e3 - R D R^T v

Differentiating gives dr/dt = K_I (F_ext - r): the residual r is F_ext through a first-order low-pass of bandwidth
K_I. The thrust model is the KF's: fixed bench gain and lag, no voltage input. Velocity comes from GNSS (held between
fixes). It has no covariance, so it gets accuracy metrics only, no NEES/NIS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import models as M
from .frames import E3


@dataclass
class MOConfig:
    name: str = "MO"
    k0: float | None = None
    lam0: float | None = None
    gain: float = 2.0          # K_I [1/s]: bandwidth of the force estimate
    attitude_mode: str = "gyro"
    motor_delay: float | None = None


class MomentumObserver:
    def __init__(self, cfg: MOConfig, prm: M.VehicleParams):
        self.cfg, self.prm = cfg, prm
        self.F = np.zeros(3)
        self.T = prm.mass * prm.g
        self.v_hat = np.zeros(3)      # (m v(0) + integral) / m: the model's velocity

    def init(self, v):
        self.v_hat = np.array(v, dtype=float)

    def step(self, dt, u, R, v_meas):
        if dt <= 0:
            return
        prm, c = self.prm, self.cfg
        m = prm.mass
        self.T += (1 - math.exp(-c.lam0 * dt)) * (c.k0 * M.sigma(u, prm.u0) - self.T)
        v_meas = np.asarray(v_meas, dtype=float)
        f_model = -self.T * R @ E3 + m * prm.g * E3 - R @ prm.D @ R.T @ v_meas
        self.v_hat = self.v_hat + (f_model + self.F) / m * dt
        self.F = c.gain * m * (v_meas - self.v_hat)
