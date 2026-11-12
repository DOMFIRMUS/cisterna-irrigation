"""Every number quoted in the manuscript, computed from the result files -> 04_resultados/numeros.json and the
tables of the paper (04_resultados/tabelas/*.md).

    python -X utf8 experimentos/e06_numeros.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisterna.agro import CROPS                                        # noqa: E402
from cisterna.config import RESULTS, SITES, TRAIN_YEARS                # noqa: E402
from cisterna.dados.clima import load, planting_doy                    # noqa: E402

REF = {"tomato": 100, "cowpea": 200}
N = {}
TAB = RESULTS / "tabelas"


def r3(x):
    return float(np.round(x, 3))


def boot_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def sites_climate():
    rows = []
    for sid, name, st, lat, lon, reg in SITES:
        df = load(sid)
        d = df.loc["1991":"2025"]
        ann = d.groupby(d.index.year)[["rain", "rain_power", "et0", "et0_power_wind", "et0_hs"]].sum().mean()
        sow, _ = planting_doy(df, TRAIN_YEARS)
        rows.append(dict(site=sid, name=f"{name}-{st}", lat=lat, lon=lon, region=reg, rain=ann.rain,
                         rain_power=ann.rain_power, et0=ann.et0, et0_power=ann.et0_power_wind, et0_hs=ann.et0_hs,
                         sow_month=pd.Timestamp(2001, 1, 1).replace(month=1) + pd.Timedelta(days=sow - 1)))
    c = pd.DataFrame(rows)
    c["sow_month"] = c.sow_month.dt.strftime("%b")
    N["rain_min"], N["rain_max"] = int(round(c.rain.min())), int(round(c.rain.max()))
    N["et0_min"], N["et0_max"] = int(round(c.et0.min())), int(round(c.et0.max()))
    N["et0_power_mean"], N["et0_mean"], N["et0_hs_mean"] = int(round(c.et0_power.mean())), int(round(c.et0.mean())), int(round(c.et0_hs.mean()))
    N["et0_power_excess_pct"] = int(round(100 * (c.et0_power.mean() / c.et0.mean() - 1)))
    pet = c.set_index("site").loc["PET"]
    N["pet_et0_pm_power"], N["pet_et0_pm"], N["pet_et0_hs"] = int(round(pet.et0_power)), int(round(pet.et0)), int(round(pet.et0_hs))
    # in-season rain and ET0 over the test seasons
    from cisterna.agro import CROPS as _CR
    for crop_name in ("tomato", "cowpea"):
        rr, ee = [], []
        for sid in c.site:
            df = load(sid)
            sow, _ = planting_doy(df, TRAIN_YEARS)
            for y in range(2011, 2026):
                st = pd.Timestamp(y, 1, 1) + pd.Timedelta(days=int(sow) - 1)
                sl = df.loc[st:st + pd.Timedelta(days=_CR[crop_name].length - 1)]
                if len(sl) == _CR[crop_name].length:
                    rr.append(sl.rain.sum()); ee.append(sl.et0.sum())
        N[f"season_rain_{crop_name}"] = int(round(np.mean(rr)))
        N[f"season_et0_{crop_name}"] = int(round(np.mean(ee)))
        N[f"season_rain_pct_et0_{crop_name}"] = int(round(100 * np.mean(rr) / np.mean(ee)))
    lines = ["| Site | Lat. (°) | Lon. (°) | Region | Rainfall (mm yr⁻¹) | ET0 (mm yr⁻¹) | Sowing |", "|---|---|---|---|---|---|---|"]
    for _, r in c.sort_values("rain").iterrows():
        lines.append(f"| {r['name']} | {r.lat:.2f} | {r.lon:.2f} | {r.region} | {r.rain:.0f} | {r.et0:.0f} | 1 {r.sow_month} |")
    (TAB / "tab_locais.md").write_text("\n".join(lines), encoding="utf-8")
    return c


def control():
    res = pd.read_csv(RESULTS / "controle" / "resultados_principais.csv.gz")
    te = res[res.year >= 2011]
    N["n_seasons_total"] = int(len(res))
    v = res[(res.controller == "sdp") & (res.soil_sensor == "S0") & (res.crop == "cowpea") & (res.area_m2 == 200)]
    N["frac_sow_full"] = r3((v.v_sow >= 51.5).mean())
    N["frac_sow_below_half"] = r3((v.v_sow < 26).mean())
    N["median_v_sow"] = round(float(v.v_sow.median()), 1)
    N["frac_sow_full_test"] = r3((v[v.year >= 2011].v_sow >= 51.5).mean())
    for site, g in v.groupby("site"):
        N[f"frac_full_{site}"] = r3((g.v_sow >= 51.5).mean())
    out = {}
    for crop, area in REF.items():
        s = te[(te.crop == crop) & (te.area_m2 == area)]
        cfg = s.groupby(["controller", "soil_sensor", "level_sensor"])
        tab = cfg.yield_ratio.agg(mean="mean", p10=lambda x: x.quantile(0.1)).join(
            cfg.yield_ratio.apply(lambda x: (x < 0.5).mean()).rename("fail")).join(
            cfg.irr_m3.mean().rename("irr")).join(cfg.days_empty.mean().rename("empty")).join(
            cfg.v_end.mean().rename("v_end")).join(cfg.yield_mult.mean().rename("mult"))
        out[crop] = tab
        for (c, ss, ls), row in tab.iterrows():
            k = f"{crop}_{c}_{ss}_{ls}"
            N[k + "_mean"], N[k + "_p10"], N[k + "_fail"] = r3(row["mean"]), r3(row.p10), r3(row.fail)
            N[k + "_empty"], N[k + "_mult"] = round(float(row["empty"]), 1), r3(row.mult)
        # paired comparisons by site-year (replicates averaged)
        def series(c, ss, ls):
            q = s[(s.controller == c) & (s.soil_sensor == ss) & (s.level_sensor == ls)]
            return q.groupby(["site", "year"]).yield_ratio.mean()
        for a, b, tag in ((("sdp", "S0", "perfect"), ("threshold", "S0", "perfect"), "S0"),
                          (("sdp", "S2", "ultrasonic"), ("threshold", "S2", "ultrasonic"), "S2"),
                          (("sdp", "S1", "ultrasonic"), ("threshold", "S1", "ultrasonic"), "S1"),
                          (("sdp", "S0", "perfect"), ("ration", "S0", "perfect"), "S0ration"),
                          (("sdp", "S0", "perfect"), ("oracle", "S0", "perfect"), "S0oracle")):
            x, y = series(*a), series(*b)
            d = (x - y).dropna().values
            lo, hi = boot_ci(d)
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            N[f"{crop}_gain_{tag}"], N[f"{crop}_gain_{tag}_lo"], N[f"{crop}_gain_{tag}_hi"] = r3(d.mean()), r3(lo), r3(hi)
            N[f"{crop}_gain_{tag}_rel_pct"] = round(float(100 * d.mean() / y.mean()), 1)
            N[f"{crop}_gain_{tag}_p"] = p
            N[f"{crop}_gain_{tag}_n"] = int(len(d))
            N[f"{crop}_gain_{tag}_frac_pos"] = r3((d > 0.005).mean())
            N[f"{crop}_gain_{tag}_frac_neg"] = r3((d < -0.005).mean())
        # gains in the dry years 2012-2017 and in the other test years, perfect sensing and proposed probe
        for tag, a, b in (("S0", ("sdp", "S0", "perfect"), ("threshold", "S0", "perfect")),
                          ("S2", ("sdp", "S2", "ultrasonic"), ("threshold", "S2", "ultrasonic"))):
            d = (series(*a) - series(*b)).dropna()
            yrs = d.index.get_level_values("year")
            N[f"{crop}_gain_{tag}_dry"] = r3(d[(yrs >= 2012) & (yrs <= 2017)].mean())
            N[f"{crop}_gain_{tag}_other"] = r3(d[(yrs < 2012) | (yrs > 2017)].mean())
        # drought years 2012-2017
        dr = s[s.year.between(2012, 2017)]
        for c in ("sdp", "threshold"):
            q = dr[(dr.controller == c) & (dr.soil_sensor == "S0")]
            N[f"{crop}_{c}_S0_drought_mean"] = r3(q.yield_ratio.mean())
    # all areas, perfect sensing, largest gap
    g = te[te.soil_sensor == "S0"].groupby(["crop", "area_m2", "controller"]).yield_ratio.mean().unstack()
    g["gain"] = g.sdp - g.threshold
    for crop in ("tomato", "cowpea"):
        gg = g.loc[crop]
        N[f"{crop}_max_gain_area"] = int(gg.gain.idxmax()); N[f"{crop}_max_gain"] = r3(gg.gain.max())
        N[f"{crop}_smallest_area_gain"] = r3(gg.gain.iloc[0])
    # results table (reference areas)
    names = {("timer", "S0", "perfect"): "Climatological timer", ("et", "S0", "perfect"): "ET book-keeping (Hargreaves)",
             ("threshold", "S0", "perfect"): "Threshold, perfect sensing", ("ration", "S0", "perfect"): "Rationing, perfect sensing",
             ("sdp", "S0", "perfect"): "**SDP, perfect sensing**", ("oracle", "S0", "perfect"): "Perfect-forecast DP",
             ("threshold", "S2", "ultrasonic"): "Threshold, proposed probe", ("ration", "S2", "ultrasonic"): "Rationing, proposed probe",
             ("sdp", "S2", "ultrasonic"): "**SDP, proposed probe**", ("sdp", "S2", "floats"): "SDP, proposed probe, float switches",
             ("threshold", "S1", "ultrasonic"): "Threshold, generic probe", ("ration", "S1", "ultrasonic"): "Rationing, generic probe",
             ("sdp", "S1", "ultrasonic"): "SDP, generic probe"}
    lines = ["| Controller and sensing | Tomato Ya/Ym | Tomato P10 | Tomato P(Ya/Ym<0.5) | Cowpea Ya/Ym | Cowpea P10 | Cowpea P(Ya/Ym<0.5) |",
             "|---|---|---|---|---|---|---|"]
    for k, nm in names.items():
        t, cw = out["tomato"].loc[k], out["cowpea"].loc[k]
        lines.append(f"| {nm} | {t['mean']:.3f} | {t.p10:.3f} | {t.fail:.3f} | {cw['mean']:.3f} | {cw.p10:.3f} | {cw.fail:.3f} |")
    (TAB / "tab_resultados.md").write_text("\n".join(lines), encoding="utf-8")


def sensitivity():
    s = pd.read_csv(RESULTS / "controle" / "sensibilidade.csv")
    s["cfg"] = s.controller + "-" + s.soil_sensor
    t = s.pivot_table(index=["crop", "variant"], columns="cfg", values="yield_ratio", aggfunc="mean")
    lab = {"base": "Base case", "wind_power": "NASA POWER wind (higher ET0)", "loamy_sand": "Loamy-sand soil",
           "runoff_0.8": "Well-kept catchment, runoff coefficient 0.8", "pv50_direct": "Pumping limited by a 50 W module (no battery)", "pv80_direct": "Pumping limited by an 80 W module (no battery)", "lambda_0.005": "Terminal value 0.005 m⁻³",
           "lambda_0.010": "Terminal value 0.010 m⁻³", "grid_11x14": "Coarser table (11 × 14 states)",
           "block_5d": "Table held over 5-day blocks", "block_10d": "Table held over 10-day blocks"}
    lines = ["| Variant | Tomato: threshold (S2) | Tomato: SDP (S2) | Cowpea: threshold (S2) | Cowpea: SDP (S2) | Table size, cowpea (kB) |",
             "|---|---|---|---|---|---|"]
    by = s[s.controller == "sdp"].groupby(["crop", "variant"]).table_bytes.first()
    for v, nm in lab.items():
        def g(crop, c):
            try:
                x = t.loc[(crop, v), c]
                return "–" if pd.isna(x) else f"{x:.3f}"
            except KeyError:
                return "–"
        lines.append(f"| {nm} | {g('tomato', 'threshold-S2')} | {g('tomato', 'sdp-S2')} | {g('cowpea', 'threshold-S2')} | "
                     f"{g('cowpea', 'sdp-S2')} | {by.loc[('cowpea', v)] / 1000:.1f} |")
        for crop in ("tomato", "cowpea"):
            for c in ("sdp-S2", "threshold-S2", "sdp-S0", "threshold-S0"):
                try:
                    x = t.loc[(crop, v), c]
                    if not pd.isna(x):
                        N[f"sens_{crop}_{v}_{c}"] = r3(x)
                except KeyError:
                    pass
            N[f"bytes_{crop}_{v}"] = int(by.loc[(crop, v)])
    vend = s[s.controller == "sdp"].groupby(["crop", "variant"]).v_end.mean()
    for (crop, v), x in vend.items():
        N[f"vend_{crop}_{v}"] = round(float(x), 1)
    (TAB / "tab_sensibilidade.md").write_text("\n".join(lines), encoding="utf-8")


def sensor():
    S = RESULTS / "sensor"
    val = pd.read_csv(S / "validacao_cps.csv")
    for g in ("coarse", "medium", "fine"):
        N[f"cps_err_{g}_max"] = round(float(val[val.grid == g].rel_err_pct.abs().max()), 2)
    fits = json.loads((S / "circuito_equivalente.json").read_text())
    for k, f in fits.items():
        N[f"fit_{k}_err"] = round(100 * max(f["fit_max_rel_err"], f["check_max_rel_err"]), 1)
        N[f"Cc_{k}_pF"] = round(f["Cc_pF"], 1); N[f"C0_{k}_pF"] = round(f["C0_pF"], 2)
        N[f"Cs_per_eps_{k}_pF"] = round(f["Cs_per_eps_pF"], 3); N[f"Gs_per_sigma_{k}"] = round(f["Gs_per_sigma_S_per_Sm"], 4)
    res = json.loads((S / "resumo.json").read_text())
    for k, v in res["depth"].items():
        N[k] = round(v, 1)
    geo = pd.read_csv(S / "geometria.csv").set_index("gap_electrodes_mm")
    for gap, r in geo.iterrows():
        N[f"geo_{int(gap)}_d90"] = round(r.d90_mm, 1); N[f"geo_{int(gap)}_airgap_loss"] = int(round(r.airgap_loss_pct))
    gp = pd.read_csv(S / "efeito_folga_ar.csv")
    base = gp[gp.gap_mm == 0].set_index("eps_soil").F
    g25 = gp[gp.gap_mm == 0.25].set_index("eps_soil").F
    N["airgap025_drop_eps15_pct"] = int(round(100 * (1 - g25.loc[15.0] / base.loc[15.0])))
    mc = pd.read_csv(S / "monte_carlo_resumo.csv")
    lines = ["| σ_p (S m⁻¹) | Low-cost class, 22 kΩ | Single, 1 kΩ | Dual, 1 + 2.2 kΩ |", "|---|---|---|---|"]
    for sp_ in sorted(mc.sigma_p.unique()):
        row = mc[mc.sigma_p == sp_].set_index("method")
        lines.append(f"| {sp_:g} | {row.loc['lowcost', 'rmse']:.3f} ({row.loc['lowcost', 'bias']:+.3f}) | "
                     f"{row.loc['hf', 'rmse']:.3f} ({row.loc['hf', 'bias']:+.3f}) | {row.loc['dual', 'rmse']:.3f} ({row.loc['dual', 'bias']:+.3f}) |")
        for m in ("lowcost", "hf", "dual"):
            N[f"mc_{m}_{sp_:g}_rmse"] = r3(row.loc[m, "rmse"])
    (TAB / "tab_sensor.md").write_text("\n".join(lines), encoding="utf-8")
    raw = pd.read_csv(S / "monte_carlo_leitura.csv.gz")
    d = raw[raw.method == "dual"]
    ratio = d.sigma_p_hat / d.sigma_p
    N["dual_sigma_within_factor2"] = r3(((ratio > 0.5) & (ratio < 2)).mean())
    flag = d.sigma_p_hat > 0.15
    N["dual_flag_hit_saline"] = r3(flag[d.sigma_p >= 0.2].mean())
    N["dual_flag_false_fresh"] = r3(flag[d.sigma_p <= 0.05].mean())


def energy_costs():
    E = RESULTS / "energia_economia"
    en = pd.read_csv(E / "energia_dimensionamento.csv")
    N["pump_wh_per_m3"] = int(round(en.pump_wh_per_m3.iloc[0]))
    for (pv, b), g in en.groupby(["pv_w", "battery"]):
        key = f"en_{pv}_{b.split()[0]}_{b.split()[-1]}"
        N[key + "_frac"] = round(100 * g.frac_days_deferred.mean(), 2)
        N[key + "_frac_max"] = round(100 * g.frac_days_deferred.max(), 2)
        N[key + "_delay"] = round(g.mean_delay_days.mean(), 3)
    N["pv_kwh_day_dry_50"] = round(en[en.pv_w == 50].pv_kwh_day_dry.mean(), 3)
    c = pd.read_csv(E / "custos_resumo.csv", index_col=0).iloc[:, 0]
    for k in ("total_brl", "node_brl", "power_brl", "drip_brl", "upgrade_brl", "equivalent_annual_cost_brl"):
        N[k] = round(float(c[k]), 0)
    N["total_usd"] = int(round(c.total_brl / c.brl_per_usd))
    N["node_usd"] = int(round(c.node_brl / c.brl_per_usd))
    N["total_minimum_wages"] = round(float(c.total_brl / c.minimum_wage_2026_brl), 2)
    ec = pd.read_csv(E / "economia_ganhos.csv")
    ec = ec[ec.area_m2 == ec.crop.map(REF)]
    for (crop, base), g in ec.groupby(["crop", "baseline"]):
        N[f"econ_{crop}_{base}_gain"] = int(round(g.gain_brl_season.median()))
        N[f"econ_{crop}_{base}_payback"] = round(float(g.payback_seasons_upgrade.replace(np.inf, np.nan).median()), 2)
        N[f"econ_{crop}_{base}_payback_max"] = round(float(g.payback_seasons_upgrade.replace(np.inf, np.nan).max()), 2)
        N[f"econ_{crop}_revenue_sdp"] = int(round(g.revenue_sdp_brl.median()))
    bom = pd.read_csv(E / "lista_materiais.csv")
    lines = ["| Item | Group | Qty | Unit price (BRL) | Total (BRL) | Price basis |", "|---|---|---|---|---|---|"]
    for _, r in bom.iterrows():
        lines.append(f"| {r['item']} | {r['group']} | {r.qty} | {r.unit_brl:.2f} | {r.total_brl:.2f} | {str(r.source).replace('R$', 'BRL')} |")
    lines.append(f"| **Total** | | | | **{bom.total_brl.sum():.2f}** | USD {bom.total_brl.sum() / 5.51:.0f} at 5.51 BRL per USD |")
    (TAB / "tab_materiais.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    sites_climate()
    control()
    sensitivity()
    sensor()
    energy_costs()
    (RESULTS / "numeros.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(N), "numbers")
