"""Sensor experiment: field-model validation, equivalent circuit, sensing depth, air gap, oscillator
validation, apparent-capacitance maps and the Monte Carlo of the single- and dual-frequency readouts.

    python -X utf8 experimentos/e01_sensor.py
Outputs in 04_resultados/sensor/.
"""
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisterna.config import RESULTS                                    # noqa: E402
from cisterna.sensor import (EPS0, Oscillator, Probe, apparent_c, build_grid, cps_analytic,  # noqa: E402
                             eps_crim, eps_water, fit_lumped, period, period_ideal, probe_elements,
                             sigma_bulk, solve_field)

OUT = RESULTS / "sensor"
GRID = dict(extent=0.3, hmax=2e-3, r=1.15)
R_HF, R_LF, R_LC = 1.0e3, 2.2e3, 22e3   # timing resistors: high, second (dual) and low-cost-class setting (ohm)
RS = (R_HF, R_LF, R_LC)
SIGMAS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]    # pore-water conductivity levels of the Monte Carlo (S/m)


def validation():
    rows = []
    for (w, s) in ((4e-3, 2e-3), (4e-3, 0.5e-3)):
        pr = Probe(w=w, s=s, tb=0.0, tc=0.0)
        ana = cps_analytic(w, s)
        for name, (hmin, hmax, r) in (("coarse", (2e-6, 4e-3, 1.25)), ("medium", (2e-6, 1e-3, 1.10)),
                                      ("fine", (1e-6, 0.5e-3, 1.06))):
            x, y = build_grid(pr, 0.3, hmin, hmax, r, validation=True)
            F = solve_field(pr, 1.0, extent=0.3, hmin=hmin, hmax=hmax, r=r, validation=True).real
            rows.append(dict(w_mm=w * 1e3, s_mm=s * 1e3, grid=name, nodes=len(x) * len(y), F_fv=F,
                             F_analytic=ana, rel_err_pct=100 * (F - ana) / ana))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "validacao_cps.csv", index=False)


def lumped_fits():
    pts = [complex(e, -im) for e in (1, 4, 10, 20, 35) for im in (0, 30, 300, 3000)]
    checks = [complex(15, -100), complex(8, -1000), complex(30, -10), complex(25, 0), complex(6, -20)]
    fits = {}
    for tc in (25e-6, 100e-6, 300e-6):
        pr = Probe(tc=tc)
        lm, err, _ = fit_lumped(pr, pts, **GRID)
        chk = [abs(lm.F(e) - solve_field(pr, e, **GRID)) / abs(solve_field(pr, e, **GRID)) for e in checks]
        L = pr.length
        fits[f"{tc * 1e6:.0f}"] = dict(F0=lm.F0, Fc=lm.Fc, Fs=lm.Fs, fit_max_rel_err=float(err),
                                       check_max_rel_err=float(max(chk)), C0_pF=EPS0 * lm.F0 * L * 1e12,
                                       Cc_pF=EPS0 * lm.Fc * L * 1e12, Cs_per_eps_pF=EPS0 * lm.Fs * L * 1e12,
                                       Gs_per_sigma_S_per_Sm=lm.Fs * L)
        print(tc, fits[f"{tc * 1e6:.0f}"], flush=True)
    (OUT / "circuito_equivalente.json").write_text(json.dumps(fits, indent=1))
    return fits


def sensing_depth_and_gap():
    pr = Probe()
    F, (x, y, V, eps) = solve_field(pr, 15.0, return_field=True, **GRID)
    xc, yc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    Vc = (V[:-1, :-1] + V[1:, :-1] + V[:-1, 1:] + V[1:, 1:]).real / 4
    Ex = np.diff((V[:, :-1] + V[:, 1:]).real / 2, axis=0) / np.diff(x)[:, None]
    Ey = np.diff((V[:-1, :] + V[1:, :]).real / 2, axis=1) / np.diff(y)[None, :]
    area = np.outer(np.diff(x), np.diff(y))
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    soil = (Xc >= pr.half_width) | (Yc >= pr.tb / 2 + pr.tc)
    dens = (Ex ** 2 + Ey ** 2) * area * soil
    # distance from the board surface (outside the board footprint: distance to the board edge)
    dx = np.maximum(Xc - pr.half_width, 0); dy = np.maximum(Yc - pr.tb / 2 - pr.tc, 0)
    dist = np.hypot(dx, dy)
    order = np.argsort(dist.ravel())
    cum = np.cumsum(dens.ravel()[order]); cum /= cum[-1]
    d = dist.ravel()[order]
    depth = {f"d{int(q * 100)}_mm": float(np.interp(q, cum, d) * 1e3) for q in (0.5, 0.8, 0.9, 0.95)}
    prof = pd.DataFrame({"dist_mm": np.linspace(0, 40, 81)})
    prof["energy_fraction"] = np.interp(prof.dist_mm / 1e3, d, cum)
    prof.to_csv(OUT / "profundidade_sensivel.csv", index=False)
    np.savez_compressed(OUT / "campo_eps15.npz", x=x, y=y, V=V.real)
    rows = []
    for gap in (0.0, 0.25e-3, 0.5e-3, 1.0e-3):
        for e in (5.0, 15.0, 30.0):
            rows.append(dict(gap_mm=gap * 1e3, eps_soil=e, F=solve_field(replace(pr, gap=gap), e, **GRID).real))
    gp = pd.DataFrame(rows)
    gp.to_csv(OUT / "efeito_folga_ar.csv", index=False)
    print("depth", depth, flush=True)
    return depth


def oscillator_validation():
    rows = []
    osc0 = Oscillator(tpd=0.0)
    for C in (5e-12, 20e-12, 60e-12):
        for R in (1e3, 1e4, 3.3e5):
            T = period(R, C, 1e-6, 1e-15, 0.0, osc0)
            Ti = period_ideal(R, C + osc0.c_par, osc0)
            rows.append(dict(C_pF=C * 1e12, R_ohm=R, T_exact_ns=T * 1e9, T_closed_ns=Ti * 1e9, rel_diff=abs(T - Ti) / Ti))
    pd.DataFrame(rows).to_csv(OUT / "validacao_oscilador.csv", index=False)


def capacitance_maps(fits):
    rows = []
    for tc_key, fit in fits.items():
        lm = type("L", (), {})()
        pr = Probe(tc=float(tc_key) * 1e-6)
        from cisterna.sensor import Lumped
        lm = Lumped(fit["F0"], fit["Fc"], fit["Fs"])
        for R in (1e3, 2.2e3, 4.7e3, 10e3, 22e3, 47e3):
            for sp_ in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
                for th in np.linspace(0.02, 0.42, 21):
                    eb = eps_crim(th); sb = sigma_bulk(eb, sp_)
                    f = 1 / period(R, *probe_elements(lm, pr, eb, sb))
                    rows.append(dict(tc_um=float(tc_key), R_ohm=R, sigma_p=sp_, theta=th, eps_b=eb, sigma_b=sb,
                                     f_MHz=f / 1e6, C_app_pF=apparent_c(f, R) * 1e12))
    pd.DataFrame(rows).to_csv(OUT / "mapas_capacitancia.csv", index=False)


# --------------------------------------------------------------------------------------- Monte Carlo
class Readout:
    """Nominal forward model tabulated for the inversion, as the firmware would store it."""

    def __init__(self, lm, pr, osc=Oscillator()):
        self.lm, self.pr, self.osc = lm, pr, osc
        self.th = np.linspace(0.0, 0.45, 46)
        self.ls = np.linspace(-2.0, 0.2, 23)                # log10 sigma_p (S/m)
        self.tt = np.array([15.0, 25.0, 35.0, 45.0])
        self.tab = {}
        for R in RS:
            a = np.empty((len(self.tt), len(self.th), len(self.ls)))
            for i, t in enumerate(self.tt):
                for j, th in enumerate(self.th):
                    for k, l in enumerate(self.ls):
                        eb = eps_crim(th, t); sb = sigma_bulk(eb, 10 ** l, t)
                        a[i, j, k] = apparent_c(1 / period(R, *probe_elements(lm, pr, eb, sb), osc), R, osc)
            self.tab[R] = RegularGridInterpolator((self.tt, self.th, self.ls), a)

    def c_nominal_air_water(self, R, t=25.0):
        osc, lm, pr = self.osc, self.lm, self.pr
        air = apparent_c(1 / period(R, *probe_elements(lm, pr, 1.0, 1e-6), osc), R, osc)
        wat = apparent_c(1 / period(R, *probe_elements(lm, pr, eps_water(t), 5e-4), osc), R, osc)
        return air, wat

    def invert(self, c, t, method):
        t = float(np.clip(t, 15, 45))
        if method == "dual":
            def res(p):
                pts = np.array([[t, p[0], p[1]]])
                return [(self.tab[R](pts)[0] - c[R]) / c[R] for R in (R_HF, R_LF)]
            best = min(((th, l) for th in np.linspace(0.02, 0.43, 14) for l in np.linspace(-1.9, 0.1, 9)),
                       key=lambda p: np.sum(np.square(res(p))))
            sol = least_squares(res, best, bounds=([0.0, -2.0], [0.45, 0.2]))
            return sol.x[0], 10 ** sol.x[1]
        R = R_HF if method == "hf" else R_LC
        ref = math.log10(0.02)           # single-frequency readouts assume a non-saline soil
        cs = np.array([self.tab[R]([[t, th, ref]])[0] for th in self.th])
        return float(np.interp(c[R], cs, self.th)), 0.02


def monte_carlo(lm_fit, n_per_level=250, seed=7):
    from cisterna.sensor import Lumped
    pr = Probe()
    lm = Lumped(lm_fit["F0"], lm_fit["Fc"], lm_fit["Fs"])
    osc = Oscillator()
    t0 = time.time()
    ro = Readout(lm, pr, osc)
    print(f"forward tables {time.time() - t0:.0f}s", flush=True)
    nom = {R: ro.c_nominal_air_water(R) for R in RS}
    rng = np.random.default_rng(seed)
    rows = []
    for sp_ in SIGMAS:
        for n in range(n_per_level):
            # one manufactured sensor
            o = replace(osc, vtp=osc.vtp * rng.normal(1, 0.03), vtn=osc.vtn * rng.normal(1, 0.03),
                        tpd=max(osc.tpd + rng.normal(0, 0.5e-9), 1e-9), c_par=osc.c_par + rng.normal(0, 0.5e-12))
            tc_f = rng.normal(1, 0.10)
            lm_s = Lumped(lm.F0 * rng.normal(1, 0.03), lm.Fc / tc_f, lm.Fs * rng.normal(1, 0.02))
            rtol = {R: R * rng.normal(1, 0.005) for R in RS}
            tempco_tpd = 0.002                                   # 1/K, not calibrated
            # factory calibration in air and deionised water at 25 degC (gain and offset per setting)
            gain = {}
            for R in RS:
                ca = apparent_c(1 / period(rtol[R], *probe_elements(lm_s, pr, 1.0, 1e-6), o), R, osc)
                cw = apparent_c(1 / period(rtol[R], *probe_elements(lm_s, pr, eps_water(25), 5e-4), o), R, osc)
                b = (nom[R][1] - nom[R][0]) / (cw - ca)
                gain[R] = (b, nom[R][0] - b * ca)
            # field condition: a different soil than the one assumed by CRIM, other temperature
            th = rng.uniform(0.05, 0.35)
            t_soil = rng.uniform(20, 38)
            por, esol = rng.normal(0.42, 0.04), rng.uniform(4.0, 5.5)
            eb = float(eps_crim(th, t_soil, por, esol))
            sb = float(sigma_bulk(eb, sp_ * rng.uniform(0.8, 1.25), t_soil))
            o_t = replace(o, tpd=o.tpd * (1 + tempco_tpd * (t_soil - 25)))
            c = {}
            for R in RS:
                f = 1 / period(rtol[R], *probe_elements(lm_s, pr, eb, sb), o_t)
                f = round(f * 0.1) / 0.1                         # 0.1 s gate: +-10 Hz counting resolution
                b, a = gain[R]
                c[R] = a + b * apparent_c(f, R, osc)
            t_meas = t_soil + rng.normal(0, 0.5)
            for method in ("lowcost", "hf", "dual"):
                th_hat, s_hat = ro.invert(c, t_meas, method)
                rows.append(dict(sigma_p=sp_, theta=th, t_soil=t_soil, method=method, theta_hat=th_hat,
                                 sigma_p_hat=s_hat, err=th_hat - th, c_hf=c[R_HF], c_lf=c[R_LF], c_lc=c[R_LC]))
        print(f"sigma_p={sp_}: {time.time() - t0:.0f}s", flush=True)
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "monte_carlo_leitura.csv.gz", index=False, float_format="%.6g")
    summ = (mc.assign(abs_err=mc.err.abs())
              .groupby(["method", "sigma_p"])
              .agg(bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                   p95_abs=("abs_err", lambda e: float(np.quantile(e, 0.95))))
              .reset_index())
    summ.to_csv(OUT / "monte_carlo_resumo.csv", index=False)
    print(summ.to_string(), flush=True)
    return summ


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    oscillator_validation()
    fits = lumped_fits()
    depth = sensing_depth_and_gap()
    capacitance_maps(fits)
    summ = monte_carlo(fits["25"])
    validation()
    (OUT / "resumo.json").write_text(json.dumps(dict(depth=depth, R_HF=R_HF, R_LF=R_LF, R_LC=R_LC,
                                                     runtime_s=time.time() - t0), indent=1))
    print("done", time.time() - t0)
