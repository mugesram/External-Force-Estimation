"""Loading filter configurations (config/filters/*.yaml) with the bench priors (config/priors.yaml)."""

from __future__ import annotations

from pathlib import Path

from ..scenario import CONFIG, load_yaml
from .ekf import FilterConfig
from .momentum_observer import MOConfig

FILTERS = {"KF": "kf", "EKF-A": "ekf_a", "EKF-B": "ekf_b", "EKF-B+bias": "ekf_b_bias", "EKF-C": "ekf_c", "MO": "mo"}


def load_priors(path=None):
    """The bench-identified priors (efe.tools.bench -> config/priors_bench.yaml) if present, else config/priors.yaml."""
    if path is None:
        path = CONFIG / "priors_bench.yaml" if (CONFIG / "priors_bench.yaml").exists() else CONFIG / "priors.yaml"
    return load_yaml(path)


def load_filter(name, priors=None, overrides=None):
    """A FilterConfig (or MOConfig) by display name ('EKF-B+bias') or file stem ('ekf_b_bias')."""
    stem = FILTERS.get(name, name)
    d = load_yaml(CONFIG / "filters" / f"{stem}.yaml")
    d.update(overrides or {})
    priors = priors if priors is not None else load_priors()
    d.setdefault("k0", float(priors["k0"]))
    d.setdefault("lam0", float(priors["lam0"]))
    if d.pop("type", None) == "momentum_observer":
        return MOConfig(**d)
    return FilterConfig.from_dict(d)


def all_filters(priors=None, names=("KF", "EKF-A", "EKF-B", "EKF-B+bias", "MO")):
    return [load_filter(n, priors) for n in names]


def filter_file(name):
    return Path(CONFIG / "filters" / f"{FILTERS.get(name, name)}.yaml")
