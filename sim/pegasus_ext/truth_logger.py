"""Per-physics-step truth, buffered in memory and written once as Parquet (or CSV if pyarrow is missing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLUMNS = ["t_usec", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "Fx", "Fy", "Fz", "T",
           "w0", "w1", "w2", "w3", "u0", "u1", "u2", "u3", "V", "SoC", "k", "k_eff", "lam", "bax", "bay", "baz",
           "fx", "fy", "fz", "wx", "wy", "wz"]


class TruthLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []

    def log(self, t_usec, p, v, q, F, T, omega, u, V, soc, k, k_eff, lam, b, f, w):
        self.rows.append(np.concatenate([[t_usec], p, v, q, F, [T], omega, u, [V, soc, k, k_eff, lam], b, f, w]))

    def save(self):
        import pandas as pd

        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(np.array(self.rows), columns=COLUMNS)
        df["t_usec"] = df["t_usec"].astype(np.int64)
        try:
            df.to_parquet(self.path.with_suffix(".parquet"), index=False)
            return self.path.with_suffix(".parquet")
        except Exception:  # Isaac's Python may lack pyarrow
            df.to_csv(self.path.with_suffix(".csv"), index=False)
            return self.path.with_suffix(".csv")
