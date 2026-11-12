"""Multi-year closed-loop runs: the cistern is simulated continuously from 1991 to 2025, one crop
season per year sown at the end of the local rainy season; policies and schedules are designed on
the training years (1991-2010) and scored on the test years (2011-2025)."""
from dataclasses import replace

import numpy as np
import pandas as pd

from cisterna.agro import CROPS, SOILS, Farm, SeasonInputs, presowing_state, simulate_season, kcb_adjusted
from cisterna.config import TRAIN_YEARS
from cisterna.controle import (SENSORS, Climatology, ETHargreaves, Ideal, LevelSensor, PolicyController,
                               Ration, SoilSensor, Threshold, Timer, solve_sdp)
from cisterna.dados.clima import load, planting_doy


def season_dates(year, sow_doy, T):
    start = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(sow_doy) - 1)
    return start, start + pd.Timedelta(days=T - 1)


def season_inputs(df, start, T):
    sl = df.loc[start:start + pd.Timedelta(days=T - 1)]
    if len(sl) < T:
        return None
    return SeasonInputs(sl.et0.values, sl.rain.values, sl.u2.values, sl.rhmin.values, sl.tmax.values,
                        sl.tmin.values, sl.et0_hs.values, sl.index.dayofyear.values)


def x0_at(df, start, crop, soil, days=90):
    pre = df.loc[start - pd.Timedelta(days=days):start - pd.Timedelta(days=1)]
    return presowing_state(soil, crop, pre.rain.values, pre.et0.values)


def build_climatology(df, crop, soil, farm, sow_doy, years=TRAIN_YEARS, window=3):
    T = crop.length
    big = replace(farm, v_max_m3=1e9, max_m3_day=1e9)
    rng = np.random.default_rng(0)
    kcb, ke, req, tps, x0s = [], [], [], [], []
    st = crop.stage_of_day()
    for y in years:
        start, _ = season_dates(y, sow_doy, T)
        w = season_inputs(df, start, T)
        if w is None:
            continue
        x0 = x0_at(df, start, crop, soil)
        # reference run: threshold control with a perfect sensor and unlimited water (irrigation
        # requirement and evaporation under the usual, intermittent drip wetting)
        r = simulate_season(w, crop, soil, big, Threshold(crop, soil), SoilSensor(), LevelSensor(), 1e8, x0, rng,
                            B=1, keep_daily=True)
        k = kcb_adjusted(crop, w.u2, w.rhmin)
        kcb.append(k)
        ke.append(r.daily["e"][:, 0] / np.maximum(w.et0, 1e-6))
        req.append(r.daily["irr_mm"][:, 0])
        tps.append(np.bincount(st, weights=k * w.et0, minlength=4))
        x0s.append(x0)
    et0_s, rain_s = [], []
    for t in range(T):
        e, p = [], []
        for y in years:
            start, _ = season_dates(y, sow_doy, T)
            day = start + pd.Timedelta(days=t)
            sl = df.loc[day - pd.Timedelta(days=window):day + pd.Timedelta(days=window)]
            e.extend(sl.et0.values); p.extend(sl.rain.values)
        et0_s.append(e)
        rain_s.append(p)
    K = min(len(e) for e in et0_s)
    return Climatology(np.mean(kcb, 0), np.mean(ke, 0), np.mean(req, 0), np.mean(tps, 0),
                       np.array([e[:K] for e in et0_s]), np.array([p[:K] for p in rain_s]),
                       float(np.mean([1 - 0.5 * x for x in x0s])))


def make_controller(name, crop, soil, farm, clim, policy=None):
    if name == "timer":
        return Timer(clim)
    if name == "threshold":
        return Threshold(crop, soil)
    if name == "et":
        return ETHargreaves(crop, soil)
    if name == "ration":
        return Ration(crop, soil, farm, clim)
    if name in ("sdp", "oracle"):
        return PolicyController(crop, soil, farm, policy)
    raise ValueError(name)


def run_years(df, crop, soil, farm, sow_doy, ctrl_name, clim, policy, sensor="S0", level="perfect",
              B=1, seed=1, years=range(1991, 2026), keep_daily_years=(), cap_series=None):
    """Continuous run; returns one row per (year, replicate)."""
    rng = np.random.default_rng(seed)
    T = crop.length
    v = np.full(B, 0.5 * farm.v_max_m3)
    day = pd.Timestamp(min(years), 1, 1)
    rows, daily = [], {}
    rain = df.rain
    for y in years:
        start, end = season_dates(y, sow_doy, T)
        w = season_inputs(df, start, T)
        if w is None:
            break
        for p in rain.loc[day:start - pd.Timedelta(days=1)].values:      # off-season refill
            v = np.minimum(v + farm.runoff_m3(p), farm.v_max_m3)
        v_sow = v.copy()
        x0 = x0_at(df, start, crop, soil)
        if ctrl_name == "oracle":
            pol, _ = solve_sdp(crop, soil, farm, clim, et0_s=w.et0[:, None], rain_s=w.rain[:, None],
                               fsub=1 - 0.5 * x0)
            ctrl = make_controller("oracle", crop, soil, farm, clim, pol)
        else:
            ctrl = make_controller(ctrl_name, crop, soil, farm, clim, policy)
        f_season = farm
        if cap_series is not None:
            cap = cap_series.loc[start:end].values
            f_season = replace(farm, max_m3_day=np.minimum(cap, farm.max_m3_day))
        r = simulate_season(w, crop, soil, f_season, ctrl, SoilSensor(**SENSORS[sensor]), LevelSensor(level),
                            v, x0, rng, B=B, keep_daily=y in keep_daily_years)
        if y in keep_daily_years:
            daily[y] = r.daily
        v = r.v_end_m3.copy()
        day = end + pd.Timedelta(days=1)
        for b in range(B):
            rows.append(dict(year=y, rep=b, yield_ratio=r.yield_ratio[b], yield_mult=r.yield_ratio_mult[b],
                             irr_m3=r.irrigation_m3[b],
                             pre_m3=r.preplant_m3[b], dp_mm=r.deep_perc_mm[b], evap_mm=r.evap_mm[b],
                             transp_mm=r.transp_mm[b], v_sow=v_sow[b], v_end=r.v_end_m3[b],
                             days_empty=r.days_empty[b], days_severe=r.days_severe[b], x0=x0))
    return pd.DataFrame(rows), daily


# configurations of the main experiment
AREAS = {"tomato": (50, 75, 100, 125, 150), "cowpea": (100, 150, 200, 250, 300)}
RUNS = [  # controller, soil sensor, level sensor, replicates
    ("timer", "S0", "perfect", 1), ("et", "S0", "perfect", 1), ("oracle", "S0", "perfect", 1),
    ("threshold", "S0", "perfect", 1), ("threshold", "S2", "ultrasonic", 16), ("threshold", "S1", "ultrasonic", 16),
    ("ration", "S0", "perfect", 1), ("ration", "S2", "ultrasonic", 16), ("ration", "S1", "ultrasonic", 16),
    ("sdp", "S0", "perfect", 1), ("sdp", "S2", "ultrasonic", 16), ("sdp", "S1", "ultrasonic", 16),
    ("sdp", "S2", "floats", 16),
]


def run_site_crop(args):
    site, crop_name, soil_name, areas = args
    df = load(site)
    crop, soil = CROPS[crop_name], SOILS[soil_name]
    sow, _ = planting_doy(df, TRAIN_YEARS)
    out = []
    for area in areas:
        farm = Farm(area_m2=area)
        clim = build_climatology(df, crop, soil, farm, sow)
        pol, _ = solve_sdp(crop, soil, farm, clim)
        for k, (c, s, l, B) in enumerate(RUNS):
            r, _ = run_years(df, crop, soil, farm, sow, c, clim, pol, s, l, B, seed=1000 + k)
            r["controller"], r["soil_sensor"], r["level_sensor"] = c, s, l
            r["site"], r["crop"], r["soil"], r["area_m2"], r["sow_doy"] = site, crop_name, soil_name, area, sow
            out.append(r)
    return pd.concat(out, ignore_index=True)
