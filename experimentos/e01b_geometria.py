"""Electrode-gap sweep: sensing depth and sensitivity to an installation air gap.

    python -X utf8 experimentos/e01b_geometria.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisterna.config import RESULTS                 # noqa: E402
from cisterna.sensor import Probe, solve_field      # noqa: E402

GRID = dict(extent=0.3, hmax=2e-3, r=1.15)


def depth(pr, eps=15.0):
    F, (x, y, V, _) = solve_field(pr, eps, return_field=True, **GRID)
    Ex = np.diff((V[:, :-1] + V[:, 1:]).real / 2, axis=0) / np.diff(x)[:, None]
    Ey = np.diff((V[:-1, :] + V[1:, :]).real / 2, axis=1) / np.diff(y)[None, :]
    xc, yc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    soil = (X >= pr.half_width) | (Y >= pr.tb / 2 + pr.tc)
    dens = (Ex ** 2 + Ey ** 2) * np.outer(np.diff(x), np.diff(y)) * soil
    dist = np.hypot(np.maximum(X - pr.half_width, 0), np.maximum(Y - pr.tb / 2 - pr.tc, 0))
    o = np.argsort(dist.ravel())
    cum = np.cumsum(dens.ravel()[o]); cum /= cum[-1]
    d = dist.ravel()[o]
    return F.real, np.interp(0.5, cum, d) * 1e3, np.interp(0.9, cum, d) * 1e3


rows = []
for s in (1e-3, 2e-3, 5e-3, 10e-3):
    pr = Probe(s=s)
    F15, d50, d90 = depth(pr)
    F5 = solve_field(pr, 5.0, **GRID).real
    Fg15 = solve_field(replace(pr, gap=0.25e-3), 15.0, **GRID).real
    Fg5 = solve_field(replace(pr, gap=0.25e-3), 5.0, **GRID).real
    rows.append(dict(gap_electrodes_mm=s * 1e3, F_eps5=F5, F_eps15=F15, d50_mm=d50, d90_mm=d90,
                     sens_dF_deps=(F15 - F5) / 10, sens_with_airgap=(Fg15 - Fg5) / 10,
                     airgap_loss_pct=100 * (1 - (Fg15 - Fg5) / (F15 - F5))))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "sensor" / "geometria.csv", index=False)
