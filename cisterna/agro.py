"""Crop-soil-cistern daily simulator (FAO-56 dual crop coefficient, drip irrigation).

Allen et al. (1998), chapter 7: basal Kcb curve (eq. 66), climate adjustment of Kcb mid/end (eq. 70),
Kc max (eq. 72), fraction cover (eq. 76), exposed-wetted fraction for trickle irrigation (eq. 75),
evaporation reduction Kr (eq. 74), evaporation-layer balance (eq. 77), root-zone balance (eq. 85) and
water-stress coefficient Ks (eq. 84) with p adjusted for ETc (Table 22 footnote). Yield from the
FAO-33 response factors (Doorenbos & Kassam, 1979), seasonal linear form and the stage-wise
multiplicative form (Rao et al., 1988) with the stage values reproduced in Steduto et al. (2012,
FAO-66, Table 2).

Every array carries a leading batch axis B (independent replicates of the sensor errors), so a whole
year runs as one vectorised loop over days.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Crop:
    name: str
    stages: tuple            # lengths of initial, development, mid and late stages (days)
    kcb: tuple               # Kcb ini, mid, end (FAO-56 Table 17)
    h_max: float             # m (FAO-56 Table 12)
    zr: tuple                # root depth at planting and maximum (m)
    p_tab: float             # depletion fraction for no stress (FAO-56 Table 22)
    ky_season: float         # FAO-56 Table 24
    ky_stage: tuple = None   # per stage (initial, development, mid, late) or None
    kc_single: tuple = (0.6, 1.15, 0.8)   # FAO-56 Table 12, used only by the ET controller
    yield_max_kg_m2: float = 1.0
    price_brl_kg: float = 1.0

    @property
    def length(self):
        return int(sum(self.stages))

    def stage_of_day(self):
        return np.repeat(np.arange(4), self.stages)

    def kcb_curve(self):
        """Daily tabulated Kcb (eq. 66): constant, linear rise, constant, linear decline."""
        ini, mid, end = self.kcb
        l1, l2, l3, l4 = self.stages
        k = np.concatenate([np.full(l1, ini), ini + (mid - ini) * (np.arange(1, l2 + 1) / l2),
                            np.full(l3, mid), mid + (end - mid) * (np.arange(1, l4 + 1) / l4)])
        return k

    def kc_single_curve(self):
        ini, mid, end = self.kc_single
        l1, l2, l3, l4 = self.stages
        return np.concatenate([np.full(l1, ini), ini + (mid - ini) * (np.arange(1, l2 + 1) / l2),
                               np.full(l3, mid), mid + (end - mid) * (np.arange(1, l4 + 1) / l4)])

    def root_curve(self):
        """Linear root growth to the maximum at the end of the development stage."""
        l1, l2 = self.stages[0], self.stages[1]
        f = np.clip(np.arange(1, self.length + 1) / (l1 + l2), 0, 1)
        return self.zr[0] + (self.zr[1] - self.zr[0]) * f


@dataclass(frozen=True)
class Soil:
    name: str
    theta_fc: float
    theta_wp: float
    rew: float               # mm (FAO-56 Table 19)
    ze: float = 0.10         # m

    @property
    def tew(self):
        return 1000 * (self.theta_fc - 0.5 * self.theta_wp) * self.ze      # eq. 73

    def taw(self, zr):
        return 1000 * (self.theta_fc - self.theta_wp) * np.asarray(zr)    # eq. 82


@dataclass(frozen=True)
class Farm:
    area_m2: float = 100.0
    v_max_m3: float = 52.0           # P1+2 cisterna-calcadao
    catch_m2: float = 200.0          # calcadao
    runoff_coef: float = 0.50        # mean measured on four calcadao cisterns in Pernambuco (Santana et al., 2026)
    initial_loss_mm: float = 0.0     # already contained in the measured coefficient
    efficiency: float = 0.90         # drip application efficiency
    fw_drip: float = 0.35            # FAO-56 Table 20, trickle 0.3-0.4
    max_m3_day: float = 0.9          # what a 50 W PV module pumps on a mean dry-season day (energia.py)

    def runoff_m3(self, rain_mm):
        return self.runoff_coef * self.catch_m2 * np.maximum(rain_mm - self.initial_loss_mm, 0) / 1000


CROPS = {
    "tomato": Crop("tomato", (30, 40, 40, 25), (0.15, 1.10, 0.70), 0.6, (0.25, 0.80), 0.40, 1.05, None,
                   (0.6, 1.15, 0.80)),
    "cowpea": Crop("cowpea", (20, 30, 30, 10), (0.15, 1.00, 0.55), 0.4, (0.20, 0.60), 0.45, 1.15,
                   (0.20, 1.10, 0.75, 0.20), (0.4, 1.05, 0.60)),
}
SOILS = {
    "sandy_loam": Soil("sandy_loam", 0.22, 0.10, 8.0),
    "loamy_sand": Soil("loamy_sand", 0.15, 0.06, 6.0),
}


def kcb_adjusted(crop, u2, rhmin):
    """Kcb curve with the climate adjustment of eq. 70 on the mid and end values (stage means of u2
    and RHmin); Kcb end is adjusted only when it is >= 0.45, as FAO-56 prescribes."""
    st = crop.stage_of_day()
    h = crop.h_max

    def adj(s):
        m = st == s
        return (0.04 * (np.mean(u2[m]) - 2) - 0.004 * (np.mean(rhmin[m]) - 45)) * (h / 3) ** 0.3

    ini, mid, end = crop.kcb
    mid2 = mid + adj(2)
    end2 = end + adj(3) if end >= 0.45 else end
    l1, l2, l3, l4 = crop.stages
    return np.concatenate([np.full(l1, ini), ini + (mid2 - ini) * (np.arange(1, l2 + 1) / l2),
                           np.full(l3, mid2), mid2 + (end2 - mid2) * (np.arange(1, l4 + 1) / l4)])


def crop_height(crop, kcb):
    ini, mid = crop.kcb[0], kcb.max()
    h = crop.h_max * np.clip((kcb - ini) / max(mid - ini, 1e-6), 0.05, 1)
    st = crop.stage_of_day()
    h[st >= 2] = crop.h_max
    return h


def yield_ratio(crop, ta_stage, tp_stage, model="additive"):
    """Relative yield Ya/Ym from stage sums of actual and potential transpiration, shape (B, 4).

    additive (Stewart et al., 1977): 1 - Ya/Ym = sum_s Ky_s (1 - Ta_s/Tp_s), the SDP objective;
    multiplicative (Rao et al., 1988): Ya/Ym = prod_s [1 - Ky_s (1 - Ta_s/Tp_s)];
    crops without stage factors use the seasonal form 1 - Ya/Ym = Ky (1 - sum Ta / sum Tp)."""
    tp = np.maximum(tp_stage, 1e-9)
    if crop.ky_stage is None:
        r = ta_stage.sum(-1) / tp.sum(-1)
        return np.clip(1 - crop.ky_season * (1 - r), 0, 1)
    ky = np.asarray(crop.ky_stage)
    d = ky * (1 - ta_stage / tp)
    if model == "multiplicative":
        return np.prod(np.clip(1 - d, 0, 1), axis=-1)
    return np.clip(1 - d.sum(-1), 0, 1)


@dataclass
class SeasonInputs:
    """Weather of one season, arrays of length T (crop days)."""
    et0: np.ndarray
    rain: np.ndarray
    u2: np.ndarray
    rhmin: np.ndarray
    tmax: np.ndarray
    tmin: np.ndarray
    et0_hs: np.ndarray
    doy: np.ndarray


@dataclass
class SeasonResult:
    yield_ratio: np.ndarray
    yield_ratio_mult: np.ndarray
    irrigation_m3: np.ndarray        # gross volume pumped from the cistern (B,)
    preplant_m3: np.ndarray
    deep_perc_mm: np.ndarray
    evap_mm: np.ndarray
    transp_mm: np.ndarray
    v_end_m3: np.ndarray
    days_empty: np.ndarray           # days on which the cistern could not deliver the request
    days_severe: np.ndarray          # days with Ks < 0.2
    daily: dict = field(default_factory=dict)


def presowing_state(soil, crop, rain, et0):
    """Bare-soil balance of the planting-depth layer over the days before sowing -> relative depletion."""
    taw0 = soil.taw(crop.zr[0])
    tew, rew = soil.tew, soil.rew
    dr, de = 0.5 * taw0, 0.5 * tew
    for p, e0 in zip(rain, et0):
        dr = max(dr - p, 0.0)
        de = max(de - p, 0.0)
        kr = 1.0 if de <= rew else max((tew - de) / (tew - rew), 0.0)
        e = kr * 1.15 * e0            # Kc max of a bare wet soil ~ 1.15 under semi-arid wind/RH
        de = min(de + e, tew)
        dr = min(dr + e, taw0)
    return dr / taw0


def simulate_season(w, crop, soil, farm, controller, sensor, level, v0, x0, rng, B=1, keep_daily=False):
    """Run one season. v0: cistern volume at sowing (B,), x0: relative depletion before sowing.

    Order within a day: the controller sees the end-of-previous-day state through the sensors and
    sets the net depth; the pump delivers what the cistern holds (limited by the pump capacity);
    then rain, evaporation and transpiration of the day act (FAO-56 daily balance).
    """
    T = crop.length
    st = crop.stage_of_day()
    kcb = kcb_adjusted(crop, w.u2, w.rhmin)
    h = crop_height(crop, kcb)
    zr = crop.root_curve()
    taw = soil.taw(zr)
    tew, rew = soil.tew, soil.rew
    eff, A = farm.efficiency, farm.area_m2

    v = np.broadcast_to(np.asarray(v0, float), (B,)).copy()
    # pre-plant irrigation: bring the planting layer to field capacity (same for every controller)
    taw0 = soil.taw(crop.zr[0])
    pre_mm = x0 * taw0
    pre_m3 = np.minimum(pre_mm * A / 1000 / eff, v)
    v -= pre_m3
    dr = np.full(B, pre_mm) - pre_m3 * eff * 1000 / A
    fsub = 1 - 0.5 * x0          # available-water fraction of the subsoil reached by growing roots
    de = np.zeros(B)
    ta_st, tp_st = np.zeros((B, 4)), np.zeros((B, 4))
    irr_m3 = np.zeros(B); dp = np.zeros(B); ev = np.zeros(B); tr = np.zeros(B)
    empty = np.zeros(B); severe = np.zeros(B)
    rec = {k: np.zeros((T, B)) for k in ("dr", "taw", "v", "irr_mm", "ks", "theta_meas", "e", "ta")} if keep_daily else None
    controller.reset(B)
    sensor.reset(B, rng)
    level.reset(B, rng)
    zr_prev = crop.zr[0]
    for t in range(T):
        # root growth: new soil enters with its own depletion
        dz = zr[t] - zr_prev
        dr = dr + 1000 * (soil.theta_fc - soil.theta_wp) * dz * (1 - fsub)
        zr_prev = zr[t]
        dr = np.clip(dr, 0, taw[t])
        theta = soil.theta_fc - dr / (1000 * zr[t])
        th_m = sensor.measure(theta, t, rng)
        v_m = level.measure(v, farm.v_max_m3, rng)
        req = controller.decide(t, th_m, v_m, w)
        req = np.maximum(req, 0)
        cap = farm.max_m3_day[t] if np.ndim(farm.max_m3_day) else farm.max_m3_day
        vol = np.minimum(np.minimum(req * A / 1000 / eff, v), cap)
        empty += (req * A / 1000 / eff > v + 1e-9) & (req > 0.05)
        v -= vol
        inet = vol * eff * 1000 / A
        controller.applied(t, inet)
        irr_m3 += vol
        dr = dr - inet
        # dual crop coefficient
        ks_dr = np.maximum(dr, 0)
        p = np.clip(crop.p_tab + 0.04 * (5 - kcb[t] * w.et0[t]), 0.1, 0.8)
        raw = p * taw[t]
        ks = np.where(ks_dr > raw, (taw[t] - ks_dr) / ((1 - p) * taw[t]), 1.0)
        ks = np.clip(ks, 0, 1)
        kcmax = max(1.2 + (0.04 * (w.u2[t] - 2) - 0.004 * (w.rhmin[t] - 45)) * (h[t] / 3) ** 0.3, kcb[t] + 0.05)
        fc = np.clip(((kcb[t] - 0.15) / (kcmax - 0.15)) ** (1 + 0.5 * h[t]), 0, 0.99)
        rain_wet = w.rain[t] >= 3.0
        fw = 1.0 if rain_wet else farm.fw_drip
        few = min(1 - fc, fw) if rain_wet else max((1 - 2 / 3 * fc) * fw, 0.01)
        de = de - w.rain[t] - np.where(inet > 0, inet / fw, 0)
        dpe = np.maximum(-de, 0)
        de = np.maximum(de, 0)
        kr = np.where(de > rew, np.maximum((tew - de) / (tew - rew), 0), 1.0)
        ke = np.minimum(kr * (kcmax - kcb[t]), few * kcmax)
        e = ke * w.et0[t]
        t_act = ks * kcb[t] * w.et0[t]
        de = np.minimum(de + e / few, tew)
        dr = dr - w.rain[t] + e + t_act
        perc = np.maximum(-dr, 0)
        dr = np.maximum(dr, 0)
        dp += perc
        ev += e; tr += t_act
        ta_st[:, st[t]] += t_act
        tp_st[:, st[t]] += kcb[t] * w.et0[t]
        severe += ks < 0.2
        v = np.minimum(v + farm.runoff_m3(w.rain[t]), farm.v_max_m3)
        if keep_daily:
            rec["dr"][t], rec["taw"][t], rec["v"][t], rec["irr_mm"][t] = dr, taw[t], v, inet
            rec["ks"][t], rec["theta_meas"][t], rec["e"][t], rec["ta"][t] = ks, th_m, e, t_act
    yr = yield_ratio(crop, ta_st, tp_st)
    ym = yield_ratio(crop, ta_st, tp_st, "multiplicative")
    return SeasonResult(yr, ym, irr_m3, pre_m3, dp, ev, tr, v, empty, severe, rec or {})
