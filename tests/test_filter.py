"""Known-answer tests of the filters on short Tier 0 flights."""
import numpy as np
import pytest

from efe.estimation import models as M
from efe.estimation.configs import load_filter
from efe.estimation.runner import run_filter, vehicle_params
from efe.evaluation.metrics import errors
from efe.pysim.sim import simulate
from efe.scenario import load_scenario

SHORT = {
    "duration": 40.0,
    "trajectory": [
        {"t": [0, 5], "type": "hold", "pos": [0, 0, -3]},
        {"t": [5, 20], "type": "zsteps", "pos": [0, 0, -3], "amp": 0.5, "period": 2.5},
        {"t": [20, 40], "type": "hold", "pos": [0, 0, -3]},
    ],
    "forces": [{"type": "step", "t0": 25, "t1": 40, "F": [0, 0, 5.0]}],
    "phases": {}, "events": {},
}


@pytest.fixture(scope="module")
def run():
    sc = load_scenario("S1_main", overrides=SHORT)
    return simulate(sc, seed=7)


def test_ekf_tracks_step(run):
    cfg = load_filter("EKF-B+bias")
    res = run_filter(run, cfg, vehicle_params(run))
    e, _ = errors(res, run)
    w = (res.t > 30) & (res.t < 40)
    assert np.sqrt(np.mean(e[w, M.IF][:, 2] ** 2)) < 0.4
    w0 = (res.t > 15) & (res.t < 25)
    assert np.all(np.sqrt(np.mean(e[w0, M.IF] ** 2, axis=0)) < 0.4)


def test_kf_is_biased_by_battery_sag(run):
    """No voltage input and a fixed gain: as the battery sags the KF must report a fake downward force."""
    res = run_filter(run, load_filter("KF"), vehicle_params(run))
    e, _ = errors(res, run, use_voltage=False)
    w = (res.t > 30) & (res.t < 40)
    assert np.mean(e[w, M.IF][:, 2]) > 0.5


def test_inactive_states_stay_at_prior(run):
    cfg = load_filter("KF")
    res = run_filter(run, cfg, vehicle_params(run))
    assert np.all(res.x[:, M.IK] == cfg.k0)
    assert np.all(res.x[:, M.IL] == cfg.lam0)
    assert np.all(res.x[:, M.IB] == 0.0)


def test_gnss_gate_cannot_lock_out(run):
    """A large unmodelled accelerometer bias must not make the GNSS gate reject every fix and the filter diverge."""
    biased = run.with_streams(imu=run.imu.assign(ax=run.imu["ax"] + 0.5, ay=run.imu["ay"] - 0.5))
    res = run_filter(biased, load_filter("EKF-B"), vehicle_params(run))
    e, _ = errors(res, biased)
    assert np.max(np.abs(e[:, M.IV])) < 1.0
