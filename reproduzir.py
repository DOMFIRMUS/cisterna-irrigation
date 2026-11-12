"""Runs the whole study in order (about 1.5 h on 10 cores): python -X utf8 reproduzir.py"""
import subprocess
import sys

STEPS = ["e01_sensor.py", "e01b_geometria.py", "e02_controle.py", "e03_sensibilidade.py",
         "e04_energia_economia.py", "e05_figuras.py", "e06_numeros.py"]
for s in STEPS:
    print("==", s, flush=True)
    subprocess.run([sys.executable, "-X", "utf8", f"experimentos/{s}"], check=True)
