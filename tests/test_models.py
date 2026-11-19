"""The analytic Jacobians against finite differences: guards every change to the model."""
import numpy as np
import pytest

from efe.estimation import models as M
from efe.estimation.frames import quat_to_rot


def random_case(rng):
    x = np.zeros(M.NX)
    x[M.IP] = rng.normal(0, 3, 3)
    x[M.IV] = rng.normal(0, 2, 3)
    x[M.IF] = rng.normal(0, 3, 3)
    x[M.IT] = rng.uniform(5, 25)
    x[M.IK] = rng.uniform(8, 12)
    x[M.IL] = rng.uniform(10, 40)
    x[M.IB] = rng.normal(0, 0.2, 3)
    x[M.IE] = rng.normal(0, 0.03, 2)
    q = rng.normal(size=4)
    R = quat_to_rot(q / np.linalg.norm(q))
    u = rng.uniform(0.2, 0.9, 4)
    V = rng.uniform(13.0, 16.8)
    return x, u, V, R


def num_jac(fun, x, eps=1e-6):
    y0 = fun(x)
    J = np.zeros((len(y0), len(x)))
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        J[:, i] = (fun(x + d) - fun(x - d)) / (2 * eps)
    return J


@pytest.mark.parametrize("use_voltage", [True, False])
def test_process_jacobian(use_voltage):
    rng = np.random.default_rng(2)
    prm = M.VehicleParams()
    for _ in range(20):
        x, u, V, R = random_case(rng)
        dt = 0.004
        Jn = num_jac(lambda xx: M.f(xx, u, V, R, dt, prm, use_voltage), x)
        Ja = M.F_jac(x, u, V, R, dt, prm, use_voltage)
        assert np.allclose(Ja, Jn, atol=1e-8, rtol=1e-6)


def test_accel_jacobian():
    rng = np.random.default_rng(3)
    prm = M.VehicleParams()
    for _ in range(20):
        x, u, V, R = random_case(rng)
        Jn = num_jac(lambda xx: M.h_acc(xx, R, prm), x)
        assert np.allclose(M.H_acc(x, R, prm), Jn, atol=1e-8)


def test_gnss_jacobian():
    rng = np.random.default_rng(4)
    x, *_ = random_case(rng)
    assert np.allclose(M.H_gnss(), num_jac(M.h_gnss, x))


def test_hover_equilibrium():
    """Level hover with T = m g, no force: zero acceleration, accelerometer reads -g along body z (FRD)."""
    prm = M.VehicleParams()
    x = np.zeros(M.NX)
    x[M.IT] = prm.mass * prm.g
    x[M.IK], x[M.IL] = 10.0, 25.0
    u = np.full(4, 0.5)
    xn = M.f(x, u, prm.V0, np.eye(3), 0.01, prm)
    assert np.allclose(xn[M.IV], 0.0)
    assert np.allclose(M.h_acc(x, np.eye(3), prm), [0, 0, -prm.g])


def test_sigma_full_and_zero():
    assert M.sigma(np.ones(4), 0.1) == pytest.approx(4.0)
    assert M.sigma(np.zeros(4), 0.1) == pytest.approx(4 * (0.1 / 1.1) ** 2)
