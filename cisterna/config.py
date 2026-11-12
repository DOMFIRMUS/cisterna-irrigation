"""Paths and simulation sites shared by every module."""
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]          # repository root
ROOT = CODE
DATA = ROOT / "02_dados"
CLIMA = DATA / "clima"
RESULTS = ROOT / "04_resultados"

PERIOD = ("1991-01-01", "2025-12-31")
TRAIN_YEARS = range(1991, 2011)       # policy design (SDP climatology)
TEST_YEARS = range(2011, 2026)        # out-of-sample evaluation, includes the 2012-2017 drought

# Municipal seats (IBGE coordinates rounded to 0.01 deg). Gridded data (NASA POWER 0.5 x 0.625 deg,
# CHIRPS 0.05 deg) make sub-kilometre precision irrelevant.
SITES = [
    # id, name, state, lat, lon, region
    ("PET", "Petrolina", "PE", -9.39, -40.50, "Sertao do Sao Francisco"),
    ("CAB", "Cabrobo", "PE", -8.51, -39.31, "Sertao do Sao Francisco"),
    ("IBI", "Ibimirim", "PE", -8.54, -37.69, "Sertao do Moxoto"),
    ("SET", "Serra Talhada", "PE", -7.99, -38.30, "Sertao do Pajeu"),
    ("OUR", "Ouricuri", "PE", -7.88, -40.08, "Sertao do Araripe"),
    ("ARC", "Arcoverde", "PE", -8.42, -37.05, "Sertao do Moxoto (transition)"),
    ("SUR", "Surubim", "PE", -7.83, -35.75, "Agreste Setentrional"),
    ("PAT", "Patos", "PB", -7.02, -37.28, "Sertao Paraibano"),
    ("QXM", "Quixeramobim", "CE", -5.20, -39.29, "Sertao Central"),
    ("IRE", "Irece", "BA", -11.30, -41.86, "Chapada Diamantina (north)"),
]
SITE_IDS = [s[0] for s in SITES]
REFERENCE_SITE = "SET"
