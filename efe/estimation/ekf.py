"""The force EKF. One class covers every filter in the comparison through its configuration:

  KF          estimate_k=False, estimate_lambda=False, use_voltage=False, estimate_bias=False
              (k and lam fixed: the model is linear in the state, so the EKF is exactly a linear KF)
  EKF-A       estimate_k, estimate_lambda;                 no voltage, no bias
  EKF-B       estimate_k, estimate_lambda, use_voltage;    no bias
  EKF-B+bias  everything

A state that is not estimated keeps zero covariance and zero process noise, so it stays at its prior exactly and never
takes part in an update.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from . import models as M


@dataclass
class FilterConfig:
    name: str = "EKF-B+bias"
    estimate_k: bool = True
    estimate_lambda: bool = True
    use_voltage: bool = True
    estimate_bias: bool = True
    estimate_tilt: bool = False         # EKF-C: tilt error of the attitude input
    # process noise spectral densities (per second): variance added per step = q * dt
    q_p: float = 1e-6
    q_v: float = 3e-3          # (m/s)^2/s, unmodelled acceleration
    q_F: float = 2.0           # N^2/s, force random walk: sets the force bandwidth
    q_T: float = 1e-2          # N^2/s, thrust model error
    q_k: float = 1e-6          # N^2/s (EKF-A needs more: it has to follow the battery)
    q_lam: float = 1e-2        # (1/s)^2/s
    q_b: float = 2.6e-4        # (m/s^2)^2/s: 2 sigma_b^2 / tau_b of the IMU bias (0.196 m/s^2, 300 s)
    q_eps: float = 1e-7        # rad^2/s: the attitude estimator's tilt error drifts slowly
    # measurement noise (standard deviations)
    r_acc: float = 0.18        # m/s^2, accelerometer noise + vibration + model error
    r_gnss_pos: tuple = (0.02, 0.02, 0.03)
    r_gnss_vel: float = 0.05
    # initial uncertainty (standard deviations)
    p0_p: float = 0.05
    p0_v: float = 0.1
    p0_F: float = 2.0
    p0_T: float = 1.0
    p0_k_rel: float = 0.08
    p0_lam_rel: float = 0.3
    p0_b: float = 0.3
    p0_eps: float = 0.035      # rad (2 deg)
    # priors (bench values); filled from config/priors.yaml if None
    k0: float | None = None
    lam0: float | None = None
    # options
    voltage_lp_hz: float = 2.0
    motor_delay: float | None = None  # s, command -> thrust transport delay; None = the run's meta motor_delay_s
    attitude_mode: str = "gyro"   # gyro | slerp | truth
    gate_gnss: float | None = 30.0  # chi^2 gate for GNSS updates (None = off)
    gate_reset_after: int = 5       # accept GNSS again after this many consecutive rejections (no lock-out)
    k_pseudo_sigma: float | None = None  # pseudo-measurement k = k0 with this sigma, at GNSS times

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown filter config keys: {sorted(unknown)}")
        d = dict(d)
        if "r_gnss_pos" in d:
            d["r_gnss_pos"] = tuple(d["r_gnss_pos"])
        return cls(**d)

    def active(self):
        """Mask of the states this filter estimates."""
        a = np.ones(M.NX, dtype=bool)
        a[M.IK] = self.estimate_k
        a[M.IL] = self.estimate_lambda
        a[M.IB] = self.estimate_bias
        a[M.IE] = self.estimate_tilt
        return a

    def q_diag(self):
        q = np.zeros(M.NX)
        q[M.IP] = self.q_p
        q[M.IV] = self.q_v
        q[M.IF] = self.q_F
        q[M.IT] = self.q_T
        q[M.IK] = self.q_k
        q[M.IL] = self.q_lam
        q[M.IB] = self.q_b
        q[M.IE] = self.q_eps
        return np.where(self.active(), q, 0.0)


class ForceEKF:
    def __init__(self, cfg: FilterConfig, prm: M.VehicleParams):
        if cfg.k0 is None or cfg.lam0 is None:
            raise ValueError("FilterConfig needs priors k0 and lam0")
        self.cfg, self.prm = cfg, prm
        self.active = cfg.active()
        self.qd = cfg.q_diag()
        self.R_acc = np.eye(3) * cfg.r_acc ** 2
        self.R_gnss = np.diag(list(np.square(cfg.r_gnss_pos)) + [cfg.r_gnss_vel ** 2] * 3)
        self.H_gnss = M.H_gnss()
        self.x = np.zeros(M.NX)
        self.P = np.zeros((M.NX, M.NX))
        self.n_gated = 0
        self._gnss_rejected_in_row = 0

    def init(self, p, v, T0=None):
        c = self.cfg
        x = np.zeros(M.NX)
        x[M.IP], x[M.IV] = p, v
        x[M.IT] = self.prm.mass * self.prm.g if T0 is None else T0
        x[M.IK], x[M.IL] = c.k0, c.lam0
        sd = np.zeros(M.NX)
        sd[M.IP], sd[M.IV], sd[M.IF] = c.p0_p, c.p0_v, c.p0_F
        sd[M.IT] = c.p0_T
        sd[M.IK] = c.p0_k_rel * c.k0
        sd[M.IL] = c.p0_lam_rel * c.lam0
        sd[M.IB] = c.p0_b
        sd[M.IE] = c.p0_eps
        sd = np.where(self.active, sd, 0.0)
        self.x, self.P = x, np.diag(sd ** 2)

    # ------------------------------------------------------------------
    def predict(self, dt, u, V, R):
        if dt <= 0:
            return
        c = self.cfg
        J = M.F_jac(self.x, u, V, R, dt, self.prm, c.use_voltage)
        self.x = M.f(self.x, u, V, R, dt, self.prm, c.use_voltage)
        self.P = J @ self.P @ J.T + np.diag(self.qd * dt)
        # keep the parameters physical
        self.x[M.IL] = min(max(self.x[M.IL], 1.0), 500.0)
        self.x[M.IK] = max(self.x[M.IK], 0.1)

    def _update(self, nu, H, Rn, gate=None):
        S = H @ self.P @ H.T + Rn
        Sinv = np.linalg.inv(S)
        nis = float(nu @ Sinv @ nu)
        if gate is not None and nis > gate:
            self.n_gated += 1
            return nis, False
        K = self.P @ H.T @ Sinv
        K[~self.active] = 0.0
        self.x = self.x + K @ nu
        IKH = np.eye(M.NX) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rn @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return nis, True

    def update_acc(self, z, R):
        nu = z - M.h_acc(self.x, R, self.prm)
        return self._update(nu, M.H_acc(self.x, R, self.prm), self.R_acc)

    def update_gnss(self, pos, vel):
        """GNSS update with a chi^2 gate. A gate that keeps rejecting means the filter, not the receiver, has drifted
        (e.g. an unmodelled accelerometer bias): after gate_reset_after rejections in a row the fix is fused anyway,
        as PX4's EKF2 does with its innovation-check timeouts, so the filter cannot lock itself out."""
        nu = np.concatenate([pos, vel]) - M.h_gnss(self.x)
        gate = self.cfg.gate_gnss
        if self._gnss_rejected_in_row >= self.cfg.gate_reset_after:
            gate = None
        nis, used = self._update(nu, self.H_gnss, self.R_gnss, gate)
        self._gnss_rejected_in_row = 0 if used else self._gnss_rejected_in_row + 1
        return nis, used

    def update_k_pseudo(self):
        c = self.cfg
        if c.k_pseudo_sigma is None or not c.estimate_k:
            return
        H = np.zeros((1, M.NX))
        H[0, M.IK] = 1.0
        self._update(np.array([c.k0 - self.x[M.IK]]), H, np.array([[c.k_pseudo_sigma ** 2]]))
