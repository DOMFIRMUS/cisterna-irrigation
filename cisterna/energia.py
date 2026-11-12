"""Daily energy balance of the autonomous node (PV module + battery + 12 V pump + electronics) and the
bill of materials used in the cost analysis.

PV energy per day: E = P_stc * H * PR * [1 + gamma (T_cell - 25)], H in kWh m-2 day-1 (NASA POWER all-sky
irradiation), cell temperature from the NOCT relation T_cell = T_air + (NOCT - 20)/800 * G (Skoplaki and
Palyvos, 2009) evaluated with the mean daytime irradiance G = H / 12 h and the mean daytime air
temperature (Tmean + Tmax)/2. Pumping that the energy cannot cover is deferred to the next day.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerSystem:
    pv_w: float = 50.0
    battery_wh: float = 84.0        # nominal (12 V x 7 Ah)
    dod: float = 0.5                # usable fraction (sealed lead-acid 0.5; LiFePO4 0.8)
    eff_batt: float = 0.85          # round trip
    pr: float = 0.80                # wiring, PWM controller, soiling, mismatch
    gamma: float = -0.004           # 1/K
    noct: float = 45.0
    pump_w: float = 48.0            # electrical power at the drip operating point
    pump_lpm: float = 3.5           # flow at that point
    base_wh_day: float = 3.2        # controller self-consumption (10 mA at 12 V) + node (ESP32, sensors)


BATTERIES = {"SLA 12V 7Ah": (84.0, 0.5, 0.80), "LiFePO4 12.8V 6Ah": (76.8, 0.8, 0.92),
             "LiFePO4 12.8V 12Ah": (153.6, 0.8, 0.92)}


def pv_energy_wh(ps, rs_mj, tmax, tmean):
    h = np.asarray(rs_mj) / 3.6                              # kWh m-2 day-1
    g = h * 1000 / 12.0                                      # mean daytime irradiance, W m-2
    tcell = (np.asarray(tmax) + np.asarray(tmean)) / 2 + (ps.noct - 20) / 800 * g
    return ps.pv_w * h * ps.pr * (1 + ps.gamma * (tcell - 25))


def simulate(ps, pv_wh, demand_m3):
    """Daily loop; returns delivered volume, deferred volume per day, battery state and unmet base load."""
    e_m3 = ps.pump_w * (1000 / ps.pump_lpm) / 60            # Wh per m3
    cap = ps.battery_wh * ps.dod
    soc = cap
    queue = 0.0
    n = len(pv_wh)
    delivered, deferred, socs = np.zeros(n), np.zeros(n), np.zeros(n)
    base_short = 0
    for d in range(n):
        avail = soc + pv_wh[d]                               # Wh that can be spent today
        need_base = ps.base_wh_day
        if avail < need_base:
            base_short += 1
        avail -= need_base
        want = queue + demand_m3[d]
        vol = min(want, max(avail, 0) / e_m3)
        delivered[d] = vol
        queue = want - vol
        deferred[d] = queue
        spent = need_base + vol * e_m3
        net = pv_wh[d] - spent
        soc = min(cap, soc + (net * ps.eff_batt if net > 0 else net))
        soc = max(soc, 0.0)
        socs[d] = soc
    return delivered, deferred, socs, base_short, e_m3


# Bill of materials (BRL, November 2026). 'src' identifies the price source in the paper's supplement;
# 'estimate' marks items priced from typical listings without a retained source.
BOM = [
    # item, group, qty, unit_brl, src
    ("ESP32 module board (with pulse counter)", "node", 1, 40.0, "ML listing ESP32-C3 R$ 34-36.50; ESP32 with PCNT priced alike (estimate)"),
    ("Probe PCB, FR-4 1.6 mm, solder mask, 100 x 18 mm", "node", 1, 15.0, "estimate (5-board prototype service + shipping)"),
    ("74LVC1G14 + 74LVC1G66 + 74LVC1G80 + resistors", "node", 1, 8.0, "estimate"),
    ("Waterproof DS18B20 temperature probe", "node", 1, 15.0, "estimate"),
    ("Waterproof ultrasonic level sensor JSN-SR04T", "node", 1, 52.0, "USD 8.99-10.00 x 5.51 BRL/USD"),
    ("Logic-level MOSFET switch module", "node", 1, 10.0, "estimate"),
    ("Buck converter 12 V -> 5 V", "node", 1, 10.0, "estimate"),
    ("IP65 enclosure, cables, connectors", "node", 1, 60.0, "estimate"),
    ("PV 50 W + 12 V 7 Ah battery + 10 A PWM controller (kit)", "power", 1, 589.99, "ML listing ZTROON kit"),
    ("12 V diaphragm pump, 80 W, 5.5 L/min", "power", 1, 179.55, "Magalu listing"),
    ("Drip hose 16 mm with emitters, 100 m", "drip", 2, 132.0, "Magalu listings R$ 99.52-164.61"),
    ("Disc filter, PE main line, fittings", "drip", 1, 100.0, "estimate"),
]


def bom_total(groups=None):
    return sum(q * u for (_, g, q, u, _) in BOM if groups is None or g in groups)
