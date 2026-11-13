"""Figures of the paper (Elsevier column widths, shared group style).

    python -X utf8 experimentos/e05_figuras.py [fig numbers...]
"""
import sys
from pathlib import Path

import matplotlib.patches as mp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import estilo_figuras as ef                                                       # noqa: E402
from cisterna.agro import CROPS, SOILS, Farm                                      # noqa: E402
from cisterna.config import RESULTS, SITES, SITE_IDS, TRAIN_YEARS                 # noqa: E402
from cisterna.controle import ACTIONS, solve_sdp                                  # noqa: E402
from cisterna.dados.clima import load, planting_doy                               # noqa: E402
from cisterna.simulacao import build_climatology, run_years                       # noqa: E402

FIG = RESULTS / "figuras"
SEN = RESULTS / "sensor"
CTL = RESULTS / "controle"
C = ef.COLORS
CTRL_STYLE = {  # fixed identity of each controller in every figure
    "sdp": dict(color=C[0], marker="o", label="Cistern-aware SDP (proposed)"),
    "threshold": dict(color=C[1], marker="s", label="Soil-moisture threshold"),
    "ration": dict(color=C[2], marker="^", label="Proportional rationing"),
    "et": dict(color=C[3], marker="D", label="ET (Hargreaves) book-keeping"),
    "timer": dict(color=C[4], marker="v", label="Climatological timer"),
    "oracle": dict(color=ef.INK2, marker="x", label="Perfect-forecast DP"),
}
REF_AREA = {"tomato": 100, "cowpea": 200}


def save(fig, name, width="elsevier-2"):
    info = ef.save(fig, FIG / name, formats=("pdf", "png"), journal=width)
    print(name, info["size_mm"], info["issues"] or "")


def box(ax, xy, w, h, text, fc, ec=ef.INK2, fs=8.5, **kw):
    ax.add_patch(mp.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=ef.INK, **kw)


def arrow(ax, a, b, color=ef.INK2, style="-|>", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=0.9, ls=ls, shrinkA=1, shrinkB=1))


def fig01():
    fig, ax = ef.figure("elsevier-2", 95)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    water, elec, energy = ef.tint(C[0], 0.25), ef.tint(C[2], 0.25), ef.tint(C[3], 0.3)
    box(ax, (0.1, 3.95), 1.9, 0.85, "Concrete catchment\n(calçadão, 200 m²)", ef.tint(ef.INK2, 0.15))
    box(ax, (0.1, 1.1), 1.9, 2.25, "", water)
    ax.text(1.05, 1.75, "Cistern\n52 m³ (P1+2)", ha="center", va="center", fontsize=8)
    box(ax, (0.25, 2.55), 1.6, 0.65, "Ultrasonic\nlevel sensor", elec)
    box(ax, (2.55, 1.55), 1.35, 0.9, "12 V pump", water)
    box(ax, (4.35, 1.55), 1.35, 0.9, "Filter +\ndrip laterals", water)
    box(ax, (6.15, 0.35), 3.75, 2.55, "", ef.tint(C[5], 0.18))
    ax.text(8.02, 0.55, "Crop root zone (tomato or cowpea)", ha="center", fontsize=8)
    box(ax, (6.3, 0.95), 1.0, 0.95, "Coated\nPCB probe", elec)
    box(ax, (7.5, 0.95), 1.12, 0.95, "Oscillator\n(two R)", elec)
    box(ax, (8.7, 0.95), 1.1, 0.95, "Soil\nthermometer", elec)
    box(ax, (3.95, 3.55), 2.75, 1.25, "ESP32 node\npolicy table π(t, x, V)\n+ (θ, σ) inversion table", ef.tint(C[0], 0.35))
    box(ax, (7.25, 3.65), 2.6, 1.05, "PV 50 W + battery\n+ PWM controller", energy)
    arrow(ax, (1.05, 3.95), (1.05, 3.35), C[0])
    arrow(ax, (2.0, 2.0), (2.55, 2.0), C[0]); arrow(ax, (3.9, 2.0), (4.35, 2.0), C[0]); arrow(ax, (5.7, 2.0), (6.15, 2.0), C[0])
    arrow(ax, (1.85, 2.95), (3.95, 3.95), C[2], ls="--")
    for xs, xe in ((8.06, 6.5), (9.25, 6.7)):
        arrow(ax, (xs, 1.9), (xe, 3.55), C[2], ls="--")
    arrow(ax, (7.3, 1.425), (7.5, 1.425), ef.INK2, "-")
    arrow(ax, (4.3, 3.55), (3.3, 2.45), C[1])
    ax.text(3.35, 3.05, "MOSFET\non/off", fontsize=8, color=C[1], ha="center")
    arrow(ax, (7.25, 4.2), (6.7, 4.2), C[3])
    ax.text(0.1, 0.05, "Blue: water   Green dashed: measurements   Orange: actuation   Yellow: energy",
            fontsize=8, color=ef.INK2)
    save(fig, "fig01_arquitetura")


def fig02():
    d = np.load(SEN / "campo_eps15.npz")
    x, y, V = d["x"] * 1e3, d["y"] * 1e3, d["V"]
    geo = pd.read_csv(SEN / "geometria.csv")
    prof = pd.read_csv(SEN / "profundidade_sensivel.csv")
    fig, axs = ef.figure("elsevier-2", 62, ncols=3, gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    ax = axs[0]
    m = (x <= 14); n = (y <= 9)
    X, Y = np.meshgrid(x[m], y[n], indexing="ij")
    full = np.concatenate([-X[::-1], X]), np.concatenate([Y[::-1], Y])
    Vq = V[np.ix_(m, n)]
    VV = np.concatenate([1 - Vq[::-1], Vq])
    XX = np.concatenate([-X[::-1], X]); YY = np.concatenate([Y[::-1], Y])
    cs = ax.contourf(XX, YY, VV, levels=np.linspace(0, 1, 21), cmap=ef.cmap("diverging"))
    ax.contourf(XX, -YY, VV, levels=np.linspace(0, 1, 21), cmap=ef.cmap("diverging"))
    ax.add_patch(mp.Rectangle((-8, -0.8), 16, 1.6, fc=ef.tint(ef.INK2, 0.35), ec="none"))
    for sgn in (-1, 1):
        for yy in (0.8, -0.8):
            ax.plot([sgn * 1, sgn * 5], [yy, yy], color=ef.INK, lw=2.2, solid_capstyle="butt")
    ax.set_aspect("equal"); ax.set_xlim(-14, 14); ax.set_ylim(-9, 9)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.text(0, 0, "FR-4", ha="center", va="center", fontsize=8, color=ef.INK)
    fig.colorbar(cs, ax=ax, shrink=0.8, label="Potential (V)", ticks=[0, 0.5, 1])
    ax = axs[1]
    ax.plot(prof.dist_mm, prof.energy_fraction, color=C[0])
    for q, ls in ((0.5, ":"), (0.9, "--")):
        dq = np.interp(q, prof.energy_fraction, prof.dist_mm)
        ax.axhline(q, color=ef.INK2, lw=0.6, ls=ls); ax.axvline(dq, color=ef.INK2, lw=0.6, ls=ls)
        ax.text(dq + 0.6, q - 0.08, f"{dq:.1f} mm", fontsize=8, color=ef.INK2)
    ax.set_xlim(0, 25); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Distance from probe surface (mm)"); ax.set_ylabel("Cumulative field-energy fraction in soil")
    ax = axs[2]
    ax.plot(geo.gap_electrodes_mm, geo.d90_mm, **ef.series(0, line=True), label="Sensing depth d90 (mm)")
    ax2y = geo.airgap_loss_pct / 10
    ax.plot(geo.gap_electrodes_mm, ax2y, **ef.series(1, line=True), label="Sensitivity lost to 0.25 mm air gap (% / 10)")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(["1", "2", "5", "10"])
    ax.set_xlabel("Electrode gap s (mm)"); ax.set_ylim(0, 10.5)
    ax.legend(fontsize=8, loc="lower right")
    ef.panel_labels(axs)
    save(fig, "fig02_sonda_campo")


def fig03():
    mp_ = pd.read_csv(SEN / "mapas_capacitancia.csv")
    mp_ = mp_[mp_.tc_um == 25]
    sel = [(22e3, "R = 22 kΩ (0.5–2.5 MHz)"), (2.2e3, "R = 2.2 kΩ (12–30 MHz)"), (1e3, "R = 1 kΩ (25–60 MHz)")]
    have = sorted(mp_.R_ohm.unique())
    fig, axs = ef.figure("elsevier-2", 62, ncols=3, sharey=True)
    sig = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    cm = ef.cmap("sequential", len(sig) + 2)
    for ax, (R, title) in zip(axs, sel):
        Rn = min(have, key=lambda r: abs(np.log(r / R)))
        for k, s in enumerate(sig):
            dd = mp_[(mp_.R_ohm == Rn) & (np.isclose(mp_.sigma_p, s))]
            ax.plot(dd.theta, dd.C_app_pF, color=cm(k + 2), lw=1.2, label=f"{s:g}")
        ax.set_yscale("log"); ax.set_title(f"R = {Rn/1e3:g} kΩ", fontsize=9)
        ax.set_xlabel("Volumetric water content θ (m³ m⁻³)")
    axs[0].set_ylabel("Apparent capacitance (pF)")
    axs[0].legend(title="σ_p (S m⁻¹)", fontsize=8, title_fontsize=8, loc="upper left", ncol=2)
    ef.panel_labels(axs)
    save(fig, "fig03_capacitancia_aparente")


def fig04():
    s = pd.read_csv(SEN / "monte_carlo_resumo.csv")
    mc = pd.read_csv(SEN / "monte_carlo_leitura.csv.gz")
    fig, axs = ef.figure("elsevier-2", 64, ncols=2)
    lab = {"lowcost": ("Single frequency, R = 22 kΩ (low-cost class)", 1), "hf": ("Single frequency, R = 1 kΩ", 2),
           "dual": ("Dual frequency, 1 kΩ + 2.2 kΩ (proposed)", 0)}
    ax = axs[0]
    for mth, (name, i) in lab.items():
        d = s[s.method == mth]
        ax.plot(d.sigma_p, d.rmse, **ef.series(i, line=True), label=name)
    ax.axhspan(0, 0.03, color=ef.tint(C[2], 0.2), lw=0)
    ax.text(0.021, 0.022, "±0.03 m³ m⁻³ target", fontsize=8, color=ef.INK2)
    ax.set_xscale("log"); ax.set_xlabel("Pore-water conductivity σ_p (S m⁻¹)"); ax.set_ylabel("RMSE of θ (m³ m⁻³)")
    ax.legend(fontsize=8, loc="upper left")
    ax = axs[1]
    d = mc[mc.method == "dual"]
    for k, sp_ in enumerate(sorted(d.sigma_p.unique())):
        dd = d[d.sigma_p == sp_]
        ax.scatter(np.full(len(dd), sp_) * np.exp(np.random.default_rng(k).normal(0, 0.06, len(dd))), dd.sigma_p_hat,
                   s=4, color=C[0], alpha=0.35, lw=0)
    lim = [0.01, 2.0]
    ax.plot(lim, lim, color=ef.INK2, lw=0.7, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True σ_p (S m⁻¹)"); ax.set_ylabel("σ_p estimated by the dual-frequency readout")
    ef.panel_labels(axs)
    save(fig, "fig04_monte_carlo_sensor")


def fig05():
    res = pd.read_csv(CTL / "resultados_principais.csv.gz", usecols=["site", "crop", "area_m2", "controller",
                                                                       "soil_sensor", "year", "v_sow"])
    rows = []
    for sid, name, st, lat, lon, reg in SITES:
        df = load(sid).loc["1991":"2025"]
        ann = df.groupby(df.index.year)[["rain", "et0"]].sum().mean()
        rows.append(dict(site=sid, label=f"{name}-{st}", rain=ann.rain, et0=ann.et0))
    cl = pd.DataFrame(rows).sort_values("rain")
    fig, axs = ef.figure("elsevier-2", 66, ncols=2)
    ax = axs[0]
    yy = np.arange(len(cl))
    ax.barh(yy - 0.2, cl.rain, 0.4, color=C[0], label="Rainfall (CHIRPS)", **ef.GAP)
    ax.barh(yy + 0.2, cl.et0, 0.4, color=C[1], label="ET0 (FAO-56 PM)", **ef.GAP)
    ax.set_yticks(yy); ax.set_yticklabels(cl.label); ax.set_xlabel("Mean annual depth 1991–2025 (mm)")
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, borderaxespad=0, frameon=False)
    ax = axs[1]
    v = res[(res.crop == "cowpea") & (res.area_m2 == 200) & (res.controller == "sdp") & (res.soil_sensor == "S0") & (res.year >= 1991)]
    data = [v[v.site == s].v_sow.values for s in cl.site]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, medianprops=dict(color=ef.INK, lw=1),
                    flierprops=dict(marker="o", ms=2, mfc=ef.INK2, mec="none"))
    for b in bp["boxes"]:
        b.set_facecolor(ef.tint(C[0], 0.35)); b.set_edgecolor(C[0])
    ax.set_yticks(yy + 1); ax.set_yticklabels([])
    ax.axvline(52, color=ef.INK2, lw=0.6, ls="--"); ax.set_xlabel("Water stored at sowing (m³), 1991–2025")
    ef.panel_labels(axs)
    save(fig, "fig05_locais_clima")


def _policy(site="SET", crop_name="cowpea"):
    df = load(site)
    crop, soil, farm = CROPS[crop_name], SOILS["sandy_loam"], Farm(area_m2=REF_AREA[crop_name])
    sow, _ = planting_doy(df, TRAIN_YEARS)
    clim = build_climatology(df, crop, soil, farm, sow)
    pol, _ = solve_sdp(crop, soil, farm, clim)
    return df, crop, soil, farm, sow, clim, pol


def fig06(cache):
    df, crop, soil, farm, sow, clim, pol = cache
    days = [(10, "Day 10, initial stage (Ky 0.20)"), (45, "Day 45, development (Ky 1.10)"), (65, "Day 65, mid-season (Ky 0.75)")]
    fig, axs = ef.figure("elsevier-2", 62, ncols=3, sharey=True)
    cm = ef.cmap("sequential", len(ACTIONS))
    for ax, (t, title) in zip(axs, days):
        img = ACTIONS[pol[t]]
        im = ax.imshow(img.T, origin="lower", aspect="auto", extent=[0, 1, 0, farm.v_max_m3], cmap=ef.cmap("sequential"),
                       vmin=0, vmax=1)
        ax.axvline(crop.p_tab, color=ef.INK, lw=0.7, ls="--")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Root-zone depletion x = Dr/TAW")
        ax.grid(False)
    axs[0].set_ylabel("Water in the cistern V (m³)")
    cb = fig.colorbar(im, ax=axs, shrink=0.85, label="Target depletion after irrigation (1 = do not irrigate)")
    ef.panel_labels(axs)
    save(fig, "fig06_politica_sdp")


def fig07(cache):
    df, crop, soil, farm, sow, clim, pol = cache
    year = 2013
    out = {}
    for c in ("threshold", "sdp"):
        r, daily = run_years(df, crop, soil, farm, sow, c, clim, pol, "S0", "perfect", 1, keep_daily_years=[year])
        out[c] = (r[r.year == year].iloc[0], daily[year])
    fig, axs = ef.figure("elsevier-2", 70, nrows=2, ncols=1, sharex=True)
    t = np.arange(crop.length)
    for c, (row, d) in out.items():
        st = CTRL_STYLE[c]
        axs[0].plot(t, d["v"][:, 0], color=st["color"], label=f"{st['label']} (Ya/Ym = {row.yield_ratio:.2f})")
        axs[1].plot(t, d["dr"][:, 0] / d["taw"][:, 0], color=st["color"])
    for b in np.cumsum(crop.stages)[:-1]:
        for ax in axs:
            ax.axvline(b, color=ef.INK2, lw=0.5, ls=":")
    axs[1].axhline(crop.p_tab, color=ef.INK2, lw=0.6, ls="--")
    axs[0].set_ylabel("Cistern volume (m³)"); axs[1].set_ylabel("Depletion Dr/TAW"); axs[1].set_xlabel("Days after sowing")
    axs[0].legend(fontsize=8, loc="upper right")
    ef.panel_labels(axs)
    save(fig, "fig07_temporada_exemplo")


def fig08():
    res = pd.read_csv(CTL / "resultados_principais.csv.gz")
    te = res[(res.year >= 2011) & (res.soil_sensor == "S0")]
    fig, axs = ef.figure("elsevier-2", 120, nrows=2, ncols=2, sharex="col")
    for col, crop in enumerate(("tomato", "cowpea")):
        for row, (stat, name) in enumerate(((lambda g: g.mean(), "Mean"), (lambda g: g.quantile(0.1), "10th percentile"))):
            ax = axs[row, col]
            for c, st in CTRL_STYLE.items():
                g = te[(te.crop == crop) & (te.controller == c)].groupby("area_m2").yield_ratio
                v = stat(g)
                ax.plot(v.index, v.values, color=st["color"], marker=st["marker"], ms=4,
                        ls="--" if c == "oracle" else "-", label=st["label"])
            ax.set_ylim(0, 1.02)
            if row == 0:
                ax.set_title(crop.capitalize(), fontsize=9)
            else:
                ax.set_xlabel("Irrigated area (m²)")
            if col == 0:
                ax.set_ylabel(f"{name} of Ya/Ym, 2011–2025")
    axs[1, 1].legend(fontsize=8, loc="lower left")
    ef.panel_labels(axs.ravel())
    save(fig, "fig08_produtividade_area")


def fig09():
    res = pd.read_csv(CTL / "resultados_principais.csv.gz")
    te = res[res.year >= 2011]
    fig, axs = ef.figure("elsevier-2", 64, ncols=2, sharey=True)
    combos = [("S0", "perfect", "Perfect sensing"), ("S2", "ultrasonic", "Proposed probe + ultrasonic level"),
              ("S1", "ultrasonic", "Generic low-cost probe + ultrasonic level")]
    ctrls = ["threshold", "ration", "sdp"]
    for ax, crop in zip(axs, ("tomato", "cowpea")):
        sub = te[(te.crop == crop) & (te.area_m2 == REF_AREA[crop])]
        for k, (s, l, name) in enumerate(combos):
            vals = [sub[(sub.controller == c) & (sub.soil_sensor == s) & (sub.level_sensor == l)].yield_ratio.mean() for c in ctrls]
            ax.bar(np.arange(3) + (k - 1) * 0.27, vals, 0.27, color=[C[0], C[2], C[1]][k] if False else ef.tint(C[0], [0.95, 0.6, 0.3][k]),
                   edgecolor=C[0], lw=0.6, label=name)
        tm = sub[(sub.controller == "timer")].yield_ratio.mean()
        ax.axhline(tm, color=CTRL_STYLE["timer"]["color"], lw=0.8, ls="--", label="Climatological timer")
        ax.set_xticks(range(3)); ax.set_xticklabels(["Threshold", "Rationing", "SDP"])
        ax.set_title(f"{crop.capitalize()}, {REF_AREA[crop]} m²", fontsize=9); ax.set_ylim(0, 1)
    axs[0].set_ylabel("Relative yield Ya/Ym (2011–2025)")
    axs[1].legend(fontsize=8, loc="upper left")
    ef.panel_labels(axs)
    save(fig, "fig09_sensores_no_laco")


def fig10():
    en = pd.read_csv(RESULTS / "energia_economia" / "energia_dimensionamento.csv")
    sens = pd.read_csv(RESULTS / "energia_economia" / "economia_sensibilidade.csv")
    fig, axs = ef.figure("elsevier-2", 64, ncols=2)
    ax = axs[0]
    for k, (b, g) in enumerate(en.groupby("battery")):
        m = g.groupby("pv_w").frac_days_deferred.mean() * 100
        mx = g.groupby("pv_w").frac_days_deferred.max() * 100
        ax.plot(m.index, m.values, **ef.series(k, line=True), label=b)
        ax.fill_between(m.index, m.values, mx.values, color=ef.tint(C[k], 0.25), lw=0)
    ax.set_xlabel("PV module (W)"); ax.set_ylabel("Irrigation days with deferred pumping (%)")
    ax.legend(fontsize=8)
    ax = axs[1]
    for k, crop in enumerate(("tomato", "cowpea")):
        d = sens[(sens.crop == crop) & (sens.area_m2 == REF_AREA[crop])]
        mid = d[np.isclose(d.ym_kg_m2, d.ym_kg_m2.max())]
        ax.plot(mid.price, mid.payback_seasons.clip(upper=10), **ef.series(k, line=True),
                label=f"{crop.capitalize()} ({REF_AREA[crop]} m²)")
    ax.axhline(1, color=ef.INK2, lw=0.6, ls="--")
    ax.set_xlabel("Produce price (BRL kg⁻¹)"); ax.set_ylabel("Payback of the sensing upgrade (seasons)")
    ax.legend(fontsize=8)
    ef.panel_labels(axs)
    save(fig, "fig10_energia_economia")


if __name__ == "__main__":
    ef.use()
    FIG.mkdir(parents=True, exist_ok=True)
    want = set(sys.argv[1:]) or {str(i) for i in range(1, 11)}
    simple = {"1": fig01, "2": fig02, "3": fig03, "4": fig04, "5": fig05, "8": fig08, "9": fig09, "10": fig10}
    for k, f in simple.items():
        if k in want:
            f()
    if want & {"6", "7"}:
        cache = _policy()
        if "6" in want:
            fig06(cache)
        if "7" in want:
            fig07(cache)
