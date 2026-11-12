"""Main closed-loop experiment: 10 sites x 2 crops x 5 areas x 13 controller/sensor configurations,
continuous 1991-2025 runs (training 1991-2010, test 2011-2025).

    python -X utf8 experimentos/e02_controle.py [workers]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cisterna.config import RESULTS, SITE_IDS          # noqa: E402
from cisterna.simulacao import AREAS, run_site_crop     # noqa: E402

OUT = RESULTS / "controle"


def main(workers=9):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, c, "sandy_loam", AREAS[c]) for s in SITE_IDS for c in ("tomato", "cowpea")]
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(workers) as ex:
        for (s, c, _, _), df in zip(jobs, ex.map(run_site_crop, jobs)):
            parts.append(df)
            print(f"{s} {c}: {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(OUT / "resultados_principais.csv.gz", index=False, float_format="%.5g")
    print("saved", len(res), "rows")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
