"""Irrigation controllers, sensor-error models in the loop, and the cistern-aware stochastic dynamic
programme (SDP) whose policy is stored as a lookup table on the microcontroller.

Controllers return a NET depth (mm over the cropped area) for the day, from what the sensors report.
"""
from dataclasses import dataclass

import numpy as np

from cisterna.agro import kcb_adjusted

ACTIONS = np.array([1.0, 0.80, 0.60, 0.45, 0.30, 0.15, 0.0])   # target relative depletion after irrigation


# ----------------------------------------------------------------------------------- sensors in the loop
class SoilSensor:
    """theta_meas = theta + b + (g - 1)(theta - 0.15) + r_t + e_t, with a per-sensor bias b and gain g,
    an AR(1) representativeness error r (point reading vs root-zone mean) and white noise e."""

    def __init__(self, bias_sd=0.0, gain_sd=0.0, noise_sd=0.0, rep_sd=0.0, rep_rho=0.95, offset=0.0):
        self.bias_sd, self.gain_sd, self.noise_sd = bias_sd, gain_sd, noise_sd
        self.rep_sd, self.rep_rho, self.offset = rep_sd, rep_rho, offset

    def reset(self, B, rng):
        self.b = rng.normal(0, self.bias_sd, B) + self.offset
        self.g = rng.normal(1, self.gain_sd, B)
        self.r = rng.normal(0, self.rep_sd, B)

    def measure(self, theta, t, rng):
        B = theta.shape[0]
        self.r = self.rep_rho * self.r + np.sqrt(1 - self.rep_rho ** 2) * rng.normal(0, self.rep_sd, B)
        return theta + self.b + (self.g - 1) * (theta - 0.15) + self.r + rng.normal(0, self.noise_sd, B)


class LevelSensor:
    """'perfect'; 'ultrasonic' (random reading noise and a fixed per-unit bias, in m3); 'floats' (four
    switches at 25, 50, 75 % of the volume: the controller sees the middle of the band)."""

    def __init__(self, kind="perfect", noise_m3=0.30, bias_m3=0.15):
        self.kind, self.noise_m3, self.bias_sd = kind, noise_m3, bias_m3

    def reset(self, B, rng):
        self.b = rng.normal(0, self.bias_sd, B) if self.kind == "ultrasonic" else np.zeros(B)

    def measure(self, v, vmax, rng):
        if self.kind == "perfect":
            return v.copy()
        if self.kind == "ultrasonic":
            return np.clip(v + self.b + rng.normal(0, self.noise_m3, v.shape[0]), 0, vmax)
        band = np.minimum(np.floor(v / vmax * 4), 3)
        return (band + 0.5) / 4 * vmax


SENSORS = {
    "S0": dict(),                                                                   # perfect
    "S1": dict(bias_sd=0.04, gain_sd=0.15, noise_sd=0.01, rep_sd=0.02),             # generic low-cost
    "S2": dict(bias_sd=0.015, gain_sd=0.05, noise_sd=0.005, rep_sd=0.02),           # proposed probe
}


# ------------------------------------------------------------------------------------------ climatology
@dataclass
class Climatology:
    """What a controller may know in advance: built from the training years only."""
    kcb: np.ndarray        # mean adjusted Kcb per crop day
    ke: np.ndarray         # mean evaporation coefficient under drip, no stress
    req: np.ndarray        # mean net irrigation requirement per day (mm), no stress
    tp_stage: np.ndarray   # mean potential transpiration per stage (mm)
    et0_s: np.ndarray      # (T, K) scenario ET0 samples
    rain_s: np.ndarray     # (T, K) scenario rain samples (same draws)
    fsub: float            # prior available-water fraction of the subsoil


class Ideal:
    """Refills to field capacity every day from a perfect sensor: the no-stress reference."""

    def __init__(self, crop, soil):
        self.zr, self.soil = crop.root_curve(), soil

    def reset(self, B):
        pass

    def decide(self, t, th, v, w):
        return np.maximum(self.soil.theta_fc - th, 0) * self.zr[t] * 1000

    def applied(self, t, inet):
        pass


class Timer:
    def __init__(self, clim):
        self.req = clim.req

    def reset(self, B):
        self.B = B

    def decide(self, t, th, v, w):
        return np.full(self.B, self.req[t])

    def applied(self, t, inet):
        pass


class Threshold:
    """Irrigate back to field capacity when the measured depletion reaches p x TAW."""

    def __init__(self, crop, soil):
        self.zr, self.soil, self.p = crop.root_curve(), soil, crop.p_tab

    def reset(self, B):
        pass

    def decide(self, t, th, v, w):
        s = self.soil
        x = (s.theta_fc - th) / (s.theta_fc - s.theta_wp)
        return np.where(x >= self.p, np.maximum(s.theta_fc - th, 0) * self.zr[t] * 1000, 0.0)

    def applied(self, t, inet):
        pass


class ETHargreaves:
    """Sensorless book-keeping controller: Hargreaves ET0 from the on-board thermometer (yesterday),
    single Kc of FAO-56 Table 12, no rain gauge; refills the book-kept depletion when it reaches RAW."""

    def __init__(self, crop, soil):
        self.kc = crop.kc_single_curve()
        self.taw = soil.taw(crop.root_curve())
        self.p = crop.p_tab

    def reset(self, B):
        self.d = np.zeros(B)

    def decide(self, t, th, v, w):
        e0 = w.et0_hs[t - 1] if t > 0 else w.et0_hs[0]
        self.d = self.d + self.kc[t] * e0
        return np.where(self.d >= self.p * self.taw[t], self.d, 0.0)

    def applied(self, t, inet):
        self.d = np.maximum(self.d - inet, 0)


class Ration:
    """Proportional-rationing heuristic: triggered like the threshold controller (measured depletion
    >= p TAW), but each refill is scaled by rho = (water in the cistern) / (expected remaining
    requirement of the season, training climatology), capped at 1. Identical to the threshold
    controller while water suffices; spreads the deficit over the rest of the season otherwise."""

    def __init__(self, crop, soil, farm, clim):
        self.zr, self.soil, self.farm, self.p = crop.root_curve(), soil, farm, crop.p_tab
        self.rem = np.cumsum(clim.req[::-1])[::-1]

    def reset(self, B):
        pass

    def decide(self, t, th, v, w):
        f, s = self.farm, self.soil
        avail = v * f.efficiency * 1000 / f.area_m2
        rho = np.clip(avail / max(self.rem[t], 1e-6), 0, 1)
        x = (s.theta_fc - th) / (s.theta_fc - s.theta_wp)
        dep = np.maximum(s.theta_fc - th, 0) * self.zr[t] * 1000
        return np.where(x >= self.p, rho * dep, 0.0)

    def applied(self, t, inet):
        pass


class PolicyController:
    """Runs a lookup table pol[t, ix, iv] -> action index, as the firmware does."""

    def __init__(self, crop, soil, farm, table):
        self.taw = soil.taw(crop.root_curve())
        self.soil, self.farm, self.tab = soil, farm, table
        self.nx, self.nv = table.shape[1], table.shape[2]

    def reset(self, B):
        pass

    def decide(self, t, th, v, w):
        s = self.soil
        x = np.clip((s.theta_fc - th) / (s.theta_fc - s.theta_wp), 0, 1)
        ix = np.rint(x * (self.nx - 1)).astype(int)
        iv = np.rint(np.clip(v / self.farm.v_max_m3, 0, 1) * (self.nv - 1)).astype(int)
        target = ACTIONS[self.tab[t, ix, iv]]
        return np.maximum(x - target, 0) * self.taw[t]

    def applied(self, t, inet):
        pass


# ----------------------------------------------------------------------------------------------- SDP
def _interp2(V, x, v, nx, nv, vmax):
    """Bilinear interpolation of V[nx, nv] at arrays x in [0, 1], v in [0, vmax]."""
    fx = np.clip(x, 0, 1) * (nx - 1)
    fv = np.clip(v / vmax, 0, 1) * (nv - 1)
    i0 = np.minimum(fx.astype(np.int32), nx - 2)
    j0 = np.minimum(fv.astype(np.int32), nv - 2)
    ax, av = fx - i0, fv - j0
    f = V.ravel()
    k = i0 * nv + j0
    v00, v10, v01, v11 = f.take(k), f.take(k + nv), f.take(k + 1), f.take(k + nv + 1)
    return v00 + ax * (v10 - v00) + av * (v01 - v00) + ax * av * (v11 - v10 - v01 + v00)


def solve_sdp(crop, soil, farm, clim, nx=21, nv=27, model="auto", lam=0.0, et0_s=None, rain_s=None, fsub=None):
    """Backward induction over (relative depletion x, cistern volume v) for every crop day.

    Reward of day t = c_t * Ta_t, with c_t = Ky_stage / Tp_stage (stage-wise model, linearised) or
    Ky / sum(Tp) (seasonal model); Ta_t = Ks(x after irrigation) * Kcb_t * ET0. Transition with the
    sampled (ET0, rain) pairs of that crop day; rain refills the cistern through the catchment.
    Returns the policy table (T, nx, nv) of action indices (uint8) and the value function at t = 0.
    """
    et0_s = clim.et0_s if et0_s is None else et0_s
    rain_s = clim.rain_s if rain_s is None else rain_s
    fsub = clim.fsub if fsub is None else fsub
    T = crop.length
    st = crop.stage_of_day()
    taw = soil.taw(crop.root_curve())
    taw_next = np.append(taw[1:], taw[-1])
    if model == "auto":
        model = "stage" if crop.ky_stage is not None else "season"
    if model == "stage":
        c = np.asarray(crop.ky_stage)[st] / clim.tp_stage[st]
    else:
        c = np.full(T, crop.ky_season / clim.tp_stage.sum())
    xg = np.linspace(0, 1, nx)[:, None, None]
    vg = np.linspace(0, farm.v_max_m3, nv)[None, :, None]
    act = ACTIONS[None, None, :]
    A, eff, vmax = farm.area_m2, farm.efficiency, farm.v_max_m3
    Vn = lam * np.broadcast_to(np.linspace(0, vmax, nv)[None, :], (nx, nv)).astype(float)
    pol = np.zeros((T, nx, nv), np.uint8)
    for t in range(T - 1, -1, -1):
        want = np.maximum(xg - act, 0) * taw[t]                                     # mm net
        vol = np.minimum(np.minimum(want * A / 1000 / eff, vg), farm.max_m3_day)    # (nx, nv, na)
        xp = xg - vol * eff * 1000 / A / taw[t]
        p = np.clip(crop.p_tab + 0.04 * (5 - clim.kcb[t] * et0_s[t].mean()), 0.1, 0.8)
        ks = np.clip(np.where(xp > p, (1 - xp) / (1 - p), 1.0), 0, 1)               # (nx, nv, na)
        e0, pr = et0_s[t][None, None, None, :], rain_s[t][None, None, None, :]
        ta = ks[..., None] * clim.kcb[t] * e0
        dr = xp[..., None] * taw[t] + ta + clim.ke[t] * e0 - pr + (taw_next[t] - taw[t]) * (1 - fsub)
        x2 = np.clip(dr, 0, taw_next[t]) / taw_next[t]
        v2 = np.minimum((vg - vol)[..., None] + farm.runoff_m3(pr), vmax)
        q = (c[t] * ta + _interp2(Vn, x2, v2, nx, nv, vmax)).mean(-1)                # (nx, nv, na)
        best = np.argmax(q > q.max(-1, keepdims=True) - 1e-12, axis=-1)   # ties -> least water
        pol[t] = best
        Vn = np.take_along_axis(q, best[..., None], -1)[..., 0]
    return pol, Vn
