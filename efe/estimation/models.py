"""Process and measurement models of the force EKF, with analytic Jacobians.

State (17):  x = [p(3), v(3), F(3), T, k, lam, b_a(3), eps(2)]
  p, v   position, velocity, world NED
  F      external force, world NED [N]                      <- the target
  T      actual collective thrust [N]
  k      thrust gain: per-motor thrust at full command (and at V0 when the voltage input is used) [N]
  lam    motor bandwidth 1/tau [1/s]
  b_a    accelerometer bias, body FRD [m/s^2]
  eps    tilt error of the attitude input, world x/y [rad]: R_true = (I + [eps]x) R  (EKF-C only; otherwise held at 0)

Inputs:  u_i (motor commands, 0..1), V (battery voltage), R (attitude, body FRD -> world NED).

Thrust command:   T_cmd = k * s(V) * sigma(u)
                  s(V) = (V / V0)^2             (1 when the voltage input is not used)
                  sigma(u) = sum_i ((u_i + u0) / (1 + u0))^2
Process (step dt; the thrust row is the exact discretisation of dT/dt = lam (k s sigma - T) for an input held over
the step, so lam is not biased by the step size; Euler would read lam about 5 % low at dt = 4 ms):
  p+ = p + v dt
  v+ = v + [(-T E R e3 + F - R D R^T v) / m + g e3] dt          E = I + [eps]x
  T+ = T + (1 - exp(-lam dt)) (k s sigma - T)
  F, k, lam, b_a, eps: random walks
Accelerometer (specific force, FRD):  h = [-T e3 + R^T E^T (F - R D R^T v)] / m + b_a
The tilt error only enters through the thrust direction in the dynamics (the accelerometer measures thrust in the body
frame), so it scales with T: throttle changes separate it from F, which does not.
GNSS:  h = [p; v]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .frames import E3

# state layout
IP = slice(0, 3)     # position
IV = slice(3, 6)     # velocity
IF = slice(6, 9)     # external force
IT = 9               # thrust
IK = 10              # thrust gain
IL = 11              # motor bandwidth
IB = slice(12, 15)   # accelerometer bias
IE = slice(15, 17)   # attitude tilt error (world x, y)
NX = 17
STATE_NAMES = ["px", "py", "pz", "vx", "vy", "vz", "Fx", "Fy", "Fz", "T", "k", "lam", "bax", "bay", "baz", "ex", "ey"]


def skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def _eps(x):
    return np.array([x[15], x[16], 0.0])


@dataclass
class VehicleParams:
    mass: float = 1.5                       # kg
    g: float = 9.80665                      # m/s^2
    drag: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.3, 0.0]))  # body-frame linear drag N/(m/s)
    V0: float = 16.8                        # V, voltage of the bench test
    u0: float = 0.1                         # thrust-curve offset: omega ~ (u + u0)
    n_motors: int = 4

    @property
    def D(self):
        return np.diag(self.drag)


def sigma(u, u0):
    """Normalised thrust curve: sum_i ((u_i + u0)/(1 + u0))^2. Equals n_motors at full command."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return float(np.sum(((u + u0) / (1.0 + u0)) ** 2))


def voltage_scale(V, V0):
    return (V / V0) ** 2


def f(x, u, V, R, dt, prm: VehicleParams, use_voltage=True):
    """One step of the process model (noise-free)."""
    s = voltage_scale(V, prm.V0) if use_voltage else 1.0
    sg = sigma(u, prm.u0)
    m = prm.mass
    Dw = R @ prm.D @ R.T
    xn = x.copy()
    xn[IP] = x[IP] + x[IV] * dt
    b3 = R @ E3
    acc = (-x[IT] * (b3 + np.cross(_eps(x), b3)) + x[IF] - Dw @ x[IV]) / m + prm.g * E3
    xn[IV] = x[IV] + acc * dt
    xn[IT] = x[IT] + (1.0 - math.exp(-x[IL] * dt)) * (x[IK] * s * sg - x[IT])
    return xn


def F_jac(x, u, V, R, dt, prm: VehicleParams, use_voltage=True):
    """Jacobian of f with respect to x."""
    s = voltage_scale(V, prm.V0) if use_voltage else 1.0
    sg = sigma(u, prm.u0)
    m = prm.mass
    Dw = R @ prm.D @ R.T
    J = np.eye(NX)
    J[IP, IV] = np.eye(3) * dt
    J[IV, IV] = np.eye(3) - Dw * dt / m
    J[IV, IF] = np.eye(3) * dt / m
    b3 = R @ E3
    J[IV, IT] = -(b3 + np.cross(_eps(x), b3)) * dt / m
    J[IV, IE] = (x[IT] / m * skew(b3) * dt)[:, :2]      # d(-T eps x b3)/d eps = T [b3]x
    a = math.exp(-x[IL] * dt)
    J[IT, IT] = a
    J[IT, IK] = (1.0 - a) * s * sg
    J[IT, IL] = dt * a * (x[IK] * s * sg - x[IT])
    return J


def h_acc(x, R, prm: VehicleParams):
    """Predicted accelerometer reading (specific force, body FRD)."""
    Dw = R @ prm.D @ R.T
    w = x[IF] - Dw @ x[IV]
    return (-x[IT] * E3 + R.T @ (w - np.cross(_eps(x), w))) / prm.mass + x[IB]


def H_acc(x, R, prm: VehicleParams):
    m = prm.mass
    Dw = R @ prm.D @ R.T
    Et = np.eye(3) - skew(_eps(x))      # E^T
    H = np.zeros((3, NX))
    H[:, IV] = -R.T @ Et @ Dw / m
    H[:, IF] = R.T @ Et / m
    H[:, IT] = -E3 / m
    H[:, IB] = np.eye(3)
    H[:, IE] = (R.T @ skew(x[IF] - Dw @ x[IV]) / m)[:, :2]   # d(-eps x w)/d eps = [w]x
    return H


def h_gnss(x):
    return np.concatenate([x[IP], x[IV]])


def H_gnss():
    H = np.zeros((6, NX))
    H[:, 0:6] = np.eye(6)
    return H
