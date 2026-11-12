"""Download the daily climate series used by the simulations (no account or key needed).

    python -X utf8 -m cisterna.dados.baixar_clima            # every site, skips files already present
    python -X utf8 -m cisterna.dados.baixar_clima PET SET    # some sites

NASA POWER (daily point API, community AG): temperature, dew point, humidity, wind, solar radiation,
surface pressure and precipitation. CHIRPS v2.0 daily rainfall through the ClimateSERV API (SERVIR),
mean over a 0.05 x 0.05 deg box centred on the site. Every request is appended to
02_dados/clima/registro_downloads.jsonl with time, URL, status and MD5 of the stored file, so the
data provenance in the paper is written by the tool.
"""
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from cisterna.config import CLIMA, PERIOD, SITES

POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN",
                "CLRSKY_SFC_SW_DWN", "PRECTOTCORR", "PS"]
CSERV = "https://climateserv.servirglobal.net/api"
UA = {"User-Agent": "cisterna-research/1.0 (UFPE)"}
LOG = CLIMA / "registro_downloads.jsonl"


def _get(url, timeout=300):
    for wait in (0, 10, 30, 90):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccups and 5xx: retry
            err = e
    raise RuntimeError(f"failed after retries: {url}: {err}")


def _log(**rec):
    rec["ts"] = dt.datetime.now().isoformat(timespec="seconds")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def power(site):
    sid, _, _, lat, lon, _ = site
    out = CLIMA / "nasa_power" / f"{sid}.csv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    q = {"parameters": ",".join(POWER_PARAMS), "community": "AG", "longitude": lon, "latitude": lat,
         "start": PERIOD[0].replace("-", ""), "end": PERIOD[1].replace("-", ""), "format": "JSON"}
    url = POWER + "?" + urllib.parse.urlencode(q)
    raw = _get(url)
    js = json.loads(raw)
    raw_path = out.with_suffix(".json.gz")
    raw_path.write_bytes(gzip.compress(raw, mtime=0))
    par = js["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in par.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.replace(-999.0, float("nan"))
    df.to_csv(out, float_format="%.3f")
    elev = js["geometry"]["coordinates"][2]
    _log(source="NASA POWER daily point v" + js["header"]["api"]["version"], site=sid, url=url,
         rows=len(df), grid_elevation_m=elev, md5_raw=hashlib.md5(raw).hexdigest(), file=str(out.name))
    return out


def _box(lat, lon, half=0.025):
    ring = [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half],
            [lon - half, lat + half], [lon - half, lat - half]]
    return json.dumps({"type": "Polygon", "coordinates": [ring]}, separators=(",", ":"))


def chirps(site, chunk_years=7):
    sid, _, _, lat, lon, _ = site
    out = CLIMA / "chirps" / f"{sid}.csv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    y0, y1 = int(PERIOD[0][:4]), int(PERIOD[1][:4])
    parts = []
    for a in range(y0, y1 + 1, chunk_years):
        b = min(a + chunk_years - 1, y1)
        q = {"datatype": 0, "begintime": f"01/01/{a}", "endtime": f"12/31/{b}", "intervaltype": 0,
             "operationtype": 5, "dateType_Category": "default", "isZip_CurrentDataType": "false",
             "geometry": _box(lat, lon)}
        url = CSERV + "/submitDataRequest/?" + urllib.parse.urlencode(q)
        job = json.loads(_get(url))[0]
        for _ in range(240):
            prog = json.loads(_get(f"{CSERV}/getDataRequestProgress/?id={job}"))[0]
            if prog >= 100 or prog < 0:
                break
            time.sleep(5)
        if prog < 0:
            raise RuntimeError(f"ClimateSERV job failed: {sid} {a}-{b}")
        raw = _get(f"{CSERV}/getDataFromRequest/?id={job}")
        recs = json.loads(raw)["data"]
        part = pd.Series({pd.to_datetime(r["date"], format="%m/%d/%Y"): r["value"]["avg"] for r in recs})
        parts.append(part)
        _log(source="CHIRPS v2.0 daily via ClimateSERV", site=sid, url=url, years=f"{a}-{b}",
             rows=len(part), md5_raw=hashlib.md5(raw).hexdigest(), file=str(out.name))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    s[s < -900] = float("nan")
    df = s.rename("P_CHIRPS").to_frame()
    df.index.name = "date"
    df.to_csv(out, float_format="%.3f")
    return out


def main(ids):
    sites = [s for s in SITES if not ids or s[0] in ids]
    pd.DataFrame(SITES, columns=["id", "name", "state", "lat", "lon", "region"]).to_csv(
        CLIMA / "locais.csv", index=False)
    for s in sites:
        p = power(s)
        c = chirps(s)
        print(f"{s[0]}: {p.name} ok, {c.name} ok", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
