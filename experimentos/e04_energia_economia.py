"""Energy sizing (35-year daily balance with the pumping schedule of the proposed controller) and cost
analysis of the sensing/control upgrade and of the whole system.

    python -X utf8 experimentos/e04_energia_economia.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisterna.agro import CROPS, SOILS, Farm                                  # noqa: E402
from cisterna.config import RESULTS, SITE_IDS, TRAIN_YEARS                    # noqa: E402
from cisterna.controle import solve_sdp                                       # noqa: E402
from cisterna.dados.clima import load, planting_doy                           # noqa: E402
from cisterna.energia import BATTERIES, BOM, PowerSystem, bom_total, pv_energy_wh, simulate  # noqa: E402
from cisterna.simulacao import build_climatology, run_years, season_dates     # noqa: E402

OUT = RESULTS / "energia_economia"
AREA = {"tomato": 100, "cowpea": 200}
# economic parameters (base, low, high) - sources in the paper
ECON = {
    "tomato": dict(ym_kg_m2=(4.0, 3.0, 6.0), price=(3.14, 2.0, 6.0)),      # IBGE PAM 2024: 59.8 t/ha state mean
    "cowpea": dict(ym_kg_m2=(0.29, 0.20, 0.29), price=(7.0, 5.0, 12.0)),   # green grain, Ramos et al. 2014; Nogueira et al. 2019
}
UPGRADE_BRL = 15.0 + 8.0 + 15.0 + 52.0 - 15.0     # proposed probe + ICs + thermometer + level sensor - generic sensor
RATE = 0.08


def schedules():
    """Daily gross pumping (m3) of the proposed controller (S2 probe, ultrasonic level) at every site."""
    rows = {}
    for site in SITE_IDS:
        df = load(site)
        sow, _ = planting_doy(df, TRAIN_YEARS)
        for crop_name, area in AREA.items():
            crop, soil, farm = CROPS[crop_name], SOILS["sandy_loam"], Farm(area_m2=area)
            clim = build_climatology(df, crop, soil, farm, sow)
            pol, _ = solve_sdp(crop, soil, farm, clim)
            years = list(range(1991, 2026))
            r, daily = run_years(df, crop, soil, farm, sow, "sdp", clim, pol, "S2", "ultrasonic", 1, seed=77,
                                 keep_daily_years=years)
            s = pd.Series(0.0, index=df.loc["1991":"2025"].index)
            for y, d in daily.items():
                start, _ = season_dates(y, sow, crop.length)
                idx = pd.date_range(start, periods=crop.length)
                s.loc[idx] = d["irr_mm"][:, 0] * area / 1000 / farm.efficiency
            rows[(site, crop_name)] = (s, df.loc["1991":"2025"])
            print(site, crop_name, "season mean m3", round(r.irr_m3.mean(), 1), flush=True)
    return rows


def energy(rows):
    out = []
    for (site, crop_name), (s, df) in rows.items():
        for pv in (20, 30, 50, 80):
            for bname, (wh, dod, eff) in BATTERIES.items():
                ps = PowerSystem(pv_w=pv, battery_wh=wh, dod=dod, eff_batt=eff)
                pv_wh = pv_energy_wh(ps, df.rs.values, df.tmax.values, (df.tmax.values + df.tmin.values) / 2)
                dlv, dfr, soc, short, e_m3 = simulate(ps, pv_wh, s.values)
                season = s.values > 0
                pumping_days = season.sum()
                out.append(dict(site=site, crop=crop_name, pv_w=pv, battery=bname,
                                pump_wh_per_m3=e_m3, demand_m3=s.sum(), delivered_m3=dlv.sum(),
                                days_with_deferral=int(((dfr > 1e-6) & season).sum()), irrigation_days=int(pumping_days),
                                frac_days_deferred=float(((dfr > 1e-6) & season).sum() / max(pumping_days, 1)),
                                mean_delay_days=float(dfr.sum() / max(s.sum(), 1e-9)),
                                max_deferred_m3=float(dfr.max()), base_load_short_days=short,
                                pv_kwh_day_dry=float(np.mean(pv_wh[df.index.month.isin([6, 7, 8, 9])]) / 1000)))
    e = pd.DataFrame(out)
    e.to_csv(OUT / "energia_dimensionamento.csv", index=False, float_format="%.5g")
    return e


def economics():
    res = pd.read_csv(RESULTS / "controle" / "resultados_principais.csv.gz")
    te = res[res.year >= 2011]
    key = ["site", "crop", "area_m2"]
    m = te.groupby(key + ["controller", "soil_sensor", "level_sensor"]).yield_ratio.mean().unstack([3, 4, 5])
    rows = []
    for idx, r in m.iterrows():
        site, crop, area = idx
        ym, price = ECON[crop]["ym_kg_m2"][0], ECON[crop]["price"][0]
        val = lambda yr: yr * ym * area * price
        sdp = r[("sdp", "S2", "ultrasonic")]
        for base_name, base_key in (("threshold_S1", ("threshold", "S1", "ultrasonic")),
                                    ("threshold_S2", ("threshold", "S2", "ultrasonic")),
                                    ("timer", ("timer", "S0", "perfect"))):
            gain = val(sdp) - val(r[base_key])
            rows.append(dict(site=site, crop=crop, area_m2=area, baseline=base_name, d_yield_ratio=sdp - r[base_key],
                             gain_brl_season=gain, payback_seasons_upgrade=UPGRADE_BRL / gain if gain > 0 else np.inf,
                             revenue_sdp_brl=val(sdp)))
    ec = pd.DataFrame(rows)
    ec.to_csv(OUT / "economia_ganhos.csv", index=False, float_format="%.5g")
    # sensitivity of the upgrade gain (vs generic-sensor threshold) to yield potential and price
    sens = []
    for crop in ("tomato", "cowpea"):
        sub = ec[(ec.crop == crop) & (ec.baseline == "threshold_S1")]
        dmean = sub.groupby("area_m2").d_yield_ratio.mean()
        for ym in np.linspace(*ECON[crop]["ym_kg_m2"][1:], 5):
            for pr in np.linspace(*ECON[crop]["price"][1:], 5):
                for area, d in dmean.items():
                    g = d * ym * area * pr
                    sens.append(dict(crop=crop, area_m2=area, ym_kg_m2=ym, price=pr, gain_brl=g,
                                     payback_seasons=UPGRADE_BRL / g if g > 0 else np.inf))
    pd.DataFrame(sens).to_csv(OUT / "economia_sensibilidade.csv", index=False, float_format="%.5g")
    # whole system: equivalent annual cost (one season per year)
    life = {"node": 5, "power": 3, "drip": 3}      # power: the lead-acid battery limits the kit
    crf = lambda n: RATE * (1 + RATE) ** n / ((1 + RATE) ** n - 1)
    eac = sum(q * u * crf(life[g]) for (_, g, q, u, _) in BOM)
    bom = pd.DataFrame(BOM, columns=["item", "group", "qty", "unit_brl", "source"])
    bom["total_brl"] = bom.qty * bom.unit_brl
    bom.to_csv(OUT / "lista_materiais.csv", index=False)
    summary = dict(total_brl=bom_total(), node_brl=bom_total({"node"}), power_brl=bom_total({"power"}),
                   drip_brl=bom_total({"drip"}), upgrade_brl=UPGRADE_BRL, equivalent_annual_cost_brl=eac,
                   brl_per_usd=5.51, minimum_wage_2026_brl=1621.0)
    pd.Series(summary).to_csv(OUT / "custos_resumo.csv")
    print(summary)
    return ec


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = schedules()
    e = energy(rows)
    print(e.groupby(["pv_w", "battery"]).frac_days_deferred.agg(["mean", "max"]).round(4).to_string())
    ec = economics()
    print(ec.groupby(["crop", "baseline"])[["d_yield_ratio", "gain_brl_season", "payback_seasons_upgrade"]].median().round(3))
