import numpy as np
import pytest

from efe.estimation import models as M
from efe.truth.battery import BatteryModel
from efe.truth.forces import ForceSchedule
from efe.truth.motors import MotorModel


def test_motor_steady_state_matches_filter_thrust_model():
    """With gamma = 2 the truth's steady thrust equals k_b * s(V) * sigma(u): the filter's model is exact there."""
    mot = MotorModel(tau=0.04)
    u, V = np.array([0.4, 0.5, 0.6, 0.7]), 15.0
    mot.reset(u, V)
    T = mot.thrusts().sum()
    assert T == pytest.approx(mot.k_b() * M.voltage_scale(V, mot.V0) * M.sigma(u, mot.u0), rel=1e-12)


def test_motor_lag_time_constant():
    mot = MotorModel(tau=0.04)
    mot.reset(np.full(4, 0.4), 16.8)
    w0 = mot.omega.copy()
    w_ref = mot.omega_cmd(np.full(4, 0.7), 16.8)
    t, dt = 0.0, 1e-4
    while t < 0.04 - 1e-12:
        mot.step(np.full(4, 0.7), 16.8, dt)
        t += dt
    frac = (mot.omega - w0) / (w_ref - w0)
    assert np.allclose(frac, 1 - np.exp(-1), atol=1e-3)


def test_battery_sags_under_load():
    b = BatteryModel()
    v0 = b.voltage
    for _ in range(int(120 / 0.01)):
        b.step(np.full(4, 3.7), 0.01)
    assert b.voltage < 0.85 * v0
    assert 0.0 <= b.soc < 0.5


def test_force_schedule():
    fs = ForceSchedule([
        {"type": "step", "t0": 1, "t1": 2, "F": [0, 0, 5]},
        {"type": "impulse", "t0": 3, "dur": 0.1, "F": [10, 0, 0]},
        {"type": "sine", "t0": 4, "t1": 6, "amp": [2, 0, 0], "freq": 0.5, "bias": [0, 1, 0]},
    ])
    assert np.allclose(fs(0.5), 0)
    assert np.allclose(fs(1.5), [0, 0, 5])
    assert np.allclose(fs(3.05), [10, 0, 0])
    assert np.allclose(fs(3.2), 0)
    assert np.allclose(fs(4.5), [2, 1, 0])
