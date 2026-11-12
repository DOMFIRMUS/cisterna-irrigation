"""Daily weather for the simulations: NASA POWER + CHIRPS, FAO-56 reference evapotranspiration.

Equations follow Allen et al. (1998, FAO Irrigation and Drainage Paper 56): Penman-Monteith (eq. 6)
with Tmean = (Tmax + Tmin)/2 (eq. 9), ea from the dew point (eq. 14), extraterrestrial radiation
(eq. 21), clear-sky radiation Rso = (0.75 + 2e-5 z) Ra (eq. 37), net long-wave (eq. 39); and
Hargreaves & Samani (1985) as written in FAO-56 eq. 52 for the temperature-only controller.
"""
import numpy as np
import pandas as pd

from cisterna.config import CLIMA, SITES

SIGMA = 4.903e-9  # MJ K-4 m-2 day-1


def esat(t):
    """Saturation vapour pressure (kPa) at air temperature t (degC), FAO-56 eq. 11."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def ra(lat_deg, doy):
    """Extraterrestrial radiation (MJ m-2 day-1), FAO-56 eqs. 21-25."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


def altitude_from_pressure(p_kpa):
    """Invert FAO-56 eq. 7, P = 101.3 ((293 - 0.0065 z)/293)^5.26."""
    return 293.0 / 0.0065 * (1 - (np.asarray(p_kpa) / 101.3) ** (1 / 5.26))


def et0_pm(tmax, tmin, tdew, rs, u2, p_kpa, lat, doy):
    tmean = (tmax + tmin) / 2
    delta = 4098 * esat(tmean) / (tmean + 237.3) ** 2
    gamma = 0.000665 * p_kpa
    es = (esat(tmax) + esat(tmin)) / 2
    ea = np.minimum(esat(tdew), es)
    z = altitude_from_pressure(p_kpa)
    rso = (0.75 + 2e-5 * z) * ra(lat, doy)
    rns = (1 - 0.23) * rs
    ratio = np.clip(rs / rso, 0.3, 1.0)
    rnl = SIGMA * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    return (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))


def et0_hargreaves(tmax, tmin, lat, doy):
    return 0.0023 * ((tmax + tmin) / 2 + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0)) * 0.408 * ra(lat, doy)


U2_DEFAULT = 2.0   # m/s: FAO-56 (ch. 3) default when measured wind is missing or unreliable


def load(site_id, wind="fao_default"):
    """Daily frame indexed by date with the variables used everywhere else.

    The NASA POWER 2-m wind (MERRA-2) averages ~4 m/s in the Sao Francisco valley, about twice the
    station values, and inflates Penman-Monteith ET0 by ~25 %. By default the FAO-56 global default
    u2 = 2 m/s replaces it (wind="fao_default"); wind="power" keeps it for the sensitivity analysis."""
    site = next(s for s in SITES if s[0] == site_id)
    lat = site[3]
    pw = pd.read_csv(CLIMA / "nasa_power" / f"{site_id}.csv", index_col="date", parse_dates=True)
    ch = pd.read_csv(CLIMA / "chirps" / f"{site_id}.csv", index_col="date", parse_dates=True)
    df = pw.join(ch, how="left")
    df = df.interpolate(limit=5).ffill().bfill()   # NASA POWER and CHIRPS have no or isolated gaps
    doy = df.index.day_of_year.values
    u2 = df.WS2M if wind == "power" else pd.Series(U2_DEFAULT, index=df.index)
    out = pd.DataFrame(index=df.index)
    out["tmax"], out["tmin"], out["tdew"] = df.T2M_MAX, df.T2M_MIN, df.T2MDEW
    out["rh"], out["u2"], out["rs"] = df.RH2M, u2, df.ALLSKY_SFC_SW_DWN
    out["u2_power"] = df.WS2M
    out["p_kpa"] = df.PS
    out["rain"] = df.P_CHIRPS.clip(lower=0)
    out["rain_power"] = df.PRECTOTCORR
    out["rhmin"] = np.clip(100 * esat(df.T2MDEW) / esat(df.T2M_MAX), 5, 100)
    out["et0"] = et0_pm(df.T2M_MAX, df.T2M_MIN, df.T2MDEW, df.ALLSKY_SFC_SW_DWN, u2, df.PS, lat, doy)
    out["et0_power_wind"] = et0_pm(df.T2M_MAX, df.T2M_MIN, df.T2MDEW, df.ALLSKY_SFC_SW_DWN, df.WS2M, df.PS, lat, doy)
    out["et0_hs"] = et0_hargreaves(df.T2M_MAX, df.T2M_MIN, lat, doy)
    out["ra"] = ra(lat, doy)
    out.attrs = {"site": site_id, "lat": lat, "lon": site[4], "name": site[1], "state": site[2],
                 "z_m": float(np.median(altitude_from_pressure(df.PS))), "wind": wind}
    return out


def planting_doy(df, years, window=4):
    """First day of the month that follows the wettest `window`-month block (training years)."""
    sub = df[df.index.year.isin(list(years))]
    monthly = sub.rain.groupby(sub.index.month).sum() / len(list(years))
    m = monthly.reindex(range(1, 13)).values
    tot = [sum(m[(k + j) % 12] for j in range(window)) for k in range(12)]
    first = int(np.argmax(tot))                    # 0-based month starting the wettest block
    month_after = (first + window) % 12 + 1        # 1-based month after the block
    return pd.Timestamp(2001, month_after, 1).dayofyear, monthly
