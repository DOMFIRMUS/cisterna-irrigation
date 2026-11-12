# cisterna-irrigation

Code and data of the article *Cistern-aware autonomous drip irrigation for semi-arid family farms: low-cost probe
design and embedded stochastic control tested in simulation at ten sites* (submitted to *Computers and Electronics in
Agriculture*).

The package `cisterna` designs and simulates a low-cost autonomous drip controller for the 52 m3 rainwater
cisterns of the Brazilian semi-arid:

| Module | Content |
|---|---|
| `cisterna/sensor.py` | quasi-static finite-volume field model of a coated PCB capacitive probe (complex permittivity), equivalent circuit, exact Schmitt-trigger oscillator model |
| `cisterna/agro.py` | FAO-56 dual crop coefficient water balance under drip, yield response (FAO-33/66), cistern balance |
| `cisterna/controle.py` | timer, threshold, ET, rationing and cistern-aware stochastic dynamic programming (SDP) controllers; sensor errors in the loop |
| `cisterna/simulacao.py` | continuous 1991-2025 runs, training 1991-2010, test 2011-2025 |
| `cisterna/energia.py` | daily PV/battery/pump energy balance and bill of materials |
| `cisterna/dados/` | download (NASA POWER, CHIRPS via ClimateSERV) and FAO-56 reference evapotranspiration |
| `firmware/controle_mpy.py` | MicroPython-compatible decision layer that reads the policy table (`policy.bin`) |
| `experimentos/` | e01 sensor, e01b geometry, e02 main control experiment, e03 sensitivity, e04 energy and costs, e05 figures, e06 numbers and tables |
| `tests/` | 12 tests: FAO-56 reference values, water-balance closure, field solver vs analytic solution, oscillator vs closed form, firmware vs simulator |

## Install and reproduce

Python 3.12. From this folder:

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e _estilo_figuras
.venv/Scripts/python -X utf8 -m pytest -q tests
.venv/Scripts/python -X utf8 reproduzir.py
```

`reproduzir.py` runs every experiment in order and rewrites `04_resultados/`. The climate series used in the
article are in `02_dados/clima/` (with `registro_downloads.jsonl`, the log of every request); delete them to
download again (`python -X utf8 -m cisterna.dados.baixar_clima`). The full run takes about 1.5 h on 10 cores.

## Data sources

- NASA POWER daily point API v2.10.3, community AG (https://power.larc.nasa.gov/), accessed 11 November 2026.
- CHIRPS v2.0 daily rainfall (Funk et al., 2015, https://doi.org/10.1038/sdata.2015.66) through the SERVIR
  ClimateSERV API (https://climateserv.servirglobal.net/), accessed 11 November 2026.
- Crop and soil parameters: FAO Irrigation and Drainage Papers 56, 33 and 66 (see the article).

## Licences

Code: MIT (`LICENSE`). Data and results produced here: CC BY 4.0 (`LICENSE-data.txt`); NASA POWER and CHIRPS keep
their own open terms. The bundled Source Sans 3 font in `_estilo_figuras` is under the SIL Open Font License 1.1.

## Contact

Jean Firmino Cardoso, Department of Nuclear Energy, Federal University of Pernambuco (jean.firmino@ufpe.br).
