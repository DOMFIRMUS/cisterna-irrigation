"""Sensitivity of the closed-loop results (water-limited reference areas: tomato 100 m2, cowpea 200 m2):
NASA POWER wind instead of the FAO-56 default, a loamy-sand soil, a lower catchment runoff coefficient,
a terminal value for water left in the cistern, and coarser / time-blocked policy tables.

    python -X utf8 experimentos/e03_sensibilidade.py [workers]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisterna.agro import CROPS, SOILS, Farm                                           # noqa: E402
from cisterna.config import RESULTS, SITE_IDS, TRAIN_YEARS                             # noqa: E402
from cisterna.controle import solve_sdp                                                # noqa: E402
from cisterna.dados.clima import load, planting_doy                                    # noqa: E402
from cisterna.energia import PowerSystem, pv_energy_wh                                 # noqa: E402
from cisterna.simulacao import RUNS, build_climatology, run_years                      # noqa: E402

AREA = {"tomato": 100, "cowpea": 200}
CTRL = [("timer", "S0", "perfect", 1), ("threshold", "S0", "perfect", 1), ("ration", "S0", "perfect", 1),
        ("sdp", "S0", "perfect", 1), ("threshold", "S2", "ultrasonic", 16), ("sdp", "S2", "ultrasonic", 16)]


def block(pol, k):
    """Policy held constant over blocks of k days (the firmware stores one slice per block)."""
    idx = (np.arange(pol.shape[0]) // k) * k
    return pol[idx]


def one(args):
    site, crop_name = args
    crop = CROPS[crop_name]
    rows = []

    def run(variant, df, soil, farm, controllers=CTRL, lam=0.0, nx=21, nv=27, k_block=1, cap=None):
        sow, _ = planting_doy(df, TRAIN_YEARS)
        clim = build_climatology(df, crop, soil, farm, sow)
        pol, _ = solve_sdp(crop, soil, farm, clim, nx=nx, nv=nv, lam=lam)
        if k_block > 1:
            pol = block(pol, k_block)
        for j, (c, s, l, B) in enumerate(controllers):
            seed = 1000 + RUNS.index((c, s, l, B))          # same draws as e02 for the same configuration
            r, _ = run_years(df, crop, soil, farm, sow, c, clim, pol, s, l, B, seed=seed, cap_series=cap)
            r = r[r.year >= 2011]
            rows.append(dict(site=site, crop=crop_name, variant=variant, controller=c, soil_sensor=s, level_sensor=l,
                             yield_ratio=r.yield_ratio.mean(), yield_p10=r.yield_ratio.quantile(0.1),
                             fail=(r.yield_ratio < 0.5).mean(), irr_m3=r.irr_m3.mean(), v_end=r.v_end.mean(),
                             table_bytes=int(np.ceil(pol.shape[0] / k_block)) * nx * nv))

    farm = Farm(area_m2=AREA[crop_name])
    base_df = load(site)
    soil = SOILS["sandy_loam"]
    run("base", base_df, soil, farm)
    run("wind_power", load(site, wind="power"), soil, farm)
    run("loamy_sand", base_df, SOILS["loamy_sand"], farm)
    run("runoff_0.8", base_df, soil, replace(farm, runoff_coef=0.8, initial_loss_mm=1.0))
    for pv in (50, 80):        # pumping limited each day by that day's PV energy (no battery buffer)
        ps = PowerSystem(pv_w=pv)
        e = pv_energy_wh(ps, base_df.rs.values, base_df.tmax.values, (base_df.tmax.values + base_df.tmin.values) / 2)
        e_m3 = ps.pump_w * (1000 / ps.pump_lpm) / 60
        cap = pd.Series(np.clip((e - ps.base_wh_day) / e_m3, 0, None), index=base_df.index)
        run(f"pv{pv}_direct", base_df, soil, farm, cap=cap)
    sdp_only = [c for c in CTRL if c[0] == "sdp"]
    run("lambda_0.005", base_df, soil, farm, sdp_only, lam=0.005)
    run("lambda_0.010", base_df, soil, farm, sdp_only, lam=0.010)
    run("grid_11x14", base_df, soil, farm, sdp_only, nx=11, nv=14)
    run("block_5d", base_df, soil, farm, sdp_only, k_block=5)
    run("block_10d", base_df, soil, farm, sdp_only, k_block=10)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    out = RESULTS / "controle"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, c) for s in SITE_IDS for c in ("tomato", "cowpea")]
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for job, df in zip(jobs, ex.map(one, jobs)):
            parts.append(df)
            print(job, f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(parts).to_csv(out / "sensibilidade.csv", index=False, float_format="%.5g")
    print("done")
