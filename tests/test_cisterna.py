"""Unit and consistency tests: python -X utf8 -m pytest -q tests"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisterna.agro import CROPS, SOILS, Farm, SeasonInputs, simulate_season, yield_ratio    # noqa: E402
from cisterna.controle import ACTIONS, LevelSensor, PolicyController, SoilSensor, Threshold  # noqa: E402
from cisterna.dados.clima import esat, et0_hargreaves, et0_pm, ra                         # noqa: E402
from cisterna.sensor import (Oscillator, Probe, cps_analytic, eps_crim, period, period_ideal,  # noqa: E402
                             solve_field, theta_crim)
from firmware.controle_mpy import Policy, pack_policy                                     # noqa: E402


def synthetic_weather(T, rain_every=0):
    rng = np.random.default_rng(3)
    rain = np.zeros(T)
    if rain_every:
        rain[::rain_every] = 12.0
    return SeasonInputs(et0=4 + rng.random(T), rain=rain, u2=np.full(T, 2.0), rhmin=np.full(T, 40.0),
                        tmax=np.full(T, 33.0), tmin=np.full(T, 21.0), et0_hs=np.full(T, 5.0), doy=np.arange(T) + 120)


def test_fao56_example_values():
    # FAO-56 Annex 2 Table 2.3: es at 25 degC = 3.168 kPa; Example 8: Ra on 3 September (J = 246), 20 S = 32.2 MJ m-2 d-1
    assert abs(esat(25.0) - 3.168) < 2e-3
    assert abs(ra(-20.0, 246) - 32.2) < 0.05


def test_et0_reasonable():
    e = et0_pm(np.array([33.0]), np.array([21.0]), np.array([17.0]), np.array([21.0]), np.array([2.0]),
               np.array([96.0]), -9.4, np.array([200]))
    h = et0_hargreaves(np.array([33.0]), np.array([21.0]), -9.4, np.array([200]))
    assert 4.0 < e[0] < 7.0 and 4.0 < h[0] < 7.0


@pytest.mark.parametrize("crop", ["tomato", "cowpea"])
@pytest.mark.parametrize("rain_every", [0, 9])
def test_season_mass_balance(crop, rain_every):
    c, s = CROPS[crop], SOILS["sandy_loam"]
    farm = Farm(area_m2=100, v_max_m3=1e6, max_m3_day=1e6)
    w = synthetic_weather(c.length, rain_every)
    x0 = 0.4
    r = simulate_season(w, c, s, farm, Threshold(c, s), SoilSensor(), LevelSensor(), 1e5, x0, np.random.default_rng(0),
                        B=1, keep_daily=True)
    taw = r.daily["taw"][:, 0]
    dr_end = r.daily["dr"][-1, 0]
    # water added to the root zone = rain + net irrigation (incl. pre-plant) ; removed = ET + percolation + root growth uptake
    net_irr = r.irrigation_m3[0] * farm.efficiency * 1000 / farm.area_m2
    pre = r.preplant_m3[0] * farm.efficiency * 1000 / farm.area_m2
    taw0 = s.taw(c.zr[0])
    fsub = 1 - 0.5 * x0
    root_dep = 1000 * (s.theta_fc - s.theta_wp) * (c.zr[1] - c.zr[0]) * (1 - fsub)
    lhs = x0 * taw0 + root_dep + r.evap_mm[0] + r.transp_mm[0] + r.deep_perc_mm[0]
    rhs = dr_end + w.rain.sum() + net_irr + pre
    assert abs(lhs - rhs) < 1e-6 * max(lhs, 1)
    assert 0 <= dr_end <= taw[-1] + 1e-9


def test_yield_models():
    c = CROPS["cowpea"]
    tp = np.array([[10.0, 50, 80, 20]])
    assert yield_ratio(c, tp, tp)[0] == pytest.approx(1.0)
    ta = tp.copy(); ta[0, 1] = 25.0
    assert yield_ratio(c, ta, tp)[0] == pytest.approx(1 - 1.10 * 0.5)
    assert yield_ratio(c, ta, tp, "multiplicative")[0] == pytest.approx(1 - 1.10 * 0.5)


def test_field_model_vs_coplanar_strips():
    pr = Probe(w=4e-3, s=2e-3, tb=0.0, tc=0.0)
    F = solve_field(pr, 1.0, extent=0.3, hmin=2e-6, hmax=4e-3, r=1.25, validation=True).real
    assert abs(F / cps_analytic(4e-3, 2e-3) - 1) < 0.006


def test_field_model_is_linear_in_homogeneous_permittivity():
    pr = Probe(w=4e-3, s=2e-3, tb=0.0, tc=0.0)
    F1 = solve_field(pr, 1.0, extent=0.1, validation=True)
    F7 = solve_field(pr, 7.0, extent=0.1, validation=True)
    assert abs(F7 / F1 - 7) < 1e-9


def test_oscillator_matches_closed_form():
    osc = Oscillator(tpd=0.0)
    for C in (5e-12, 40e-12):
        T = period(1e4, C, 1e-6, 1e-15, 0.0, osc)
        c_series = 1e-6 * 1e-15 / (1e-6 + 1e-15)
        assert abs(T / period_ideal(1e4, C + osc.c_par + c_series, osc) - 1) < 1e-8


def test_crim_roundtrip():
    th = np.linspace(0.02, 0.4, 9)
    assert np.allclose(theta_crim(eps_crim(th, 30.0), 30.0), th)


def test_firmware_matches_simulator_policy():
    c, s = CROPS["cowpea"], SOILS["sandy_loam"]
    farm = Farm(area_m2=200)
    rng = np.random.default_rng(5)
    table = rng.integers(0, len(ACTIONS), size=(c.length, 21, 27), dtype=np.uint8)
    taw = s.taw(c.root_curve())
    blob = pack_policy(table.tolist(), ACTIONS.tolist(), taw.tolist(), farm.v_max_m3, farm.area_m2, farm.efficiency,
                       s.theta_fc, s.theta_wp)
    fw = Policy(blob)
    sim = PolicyController(c, s, farm, table)
    th = rng.uniform(0.05, 0.25, 4000)
    v = rng.uniform(0, 52, 4000)
    t = rng.integers(0, c.length, 4000)
    for k in range(4000):
        net_sim = sim.decide(int(t[k]), np.array([th[k]]), np.array([v[k]]), None)[0]
        litres_fw = fw.litres(int(t[k]), float(th[k]), float(v[k]))
        assert litres_fw == pytest.approx(net_sim * farm.area_m2 / farm.efficiency, rel=2e-3, abs=0.2)
    assert len(blob) < 30 + 7 + 2 * c.length + c.length * 21 * 27 + 1
