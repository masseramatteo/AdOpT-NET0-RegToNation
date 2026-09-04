"""
Solar anchor library — turns a sampled mean PV capacity factor into a real
hourly irradiance series.

Why this exists
---------------
The LHS samples ``solar_cf_mean``: the annual-mean PV capacity factor, a
dimensionless, geography-free quantity. The model, however, needs an hourly
``ghi``/``dni``/``dhi`` series, because ``adopt_net0`` always runs pvlib from
climate data (``genericTechnologies/res.py:_perform_fitting_pv``); there is no
path to inject a capacity factor directly.

This module bridges the two. A small library of real European TMY series is
downloaded once and cached. Each anchor is run through the same pvlib chain
``res.py`` uses, giving its own mean capacity factor. At run time the anchor
closest to the requested ``solar_cf_mean`` is selected and its irradiance is
scaled by a small residual factor (typically within +-7%) so the realized mean
lands on target.

The hourly *shape* therefore always comes from a real site; only the *level* is
set by the sampled parameter.

Data source
-----------
PVGIS TMY API, same endpoint and same three columns as
``adopt_net0/data_preprocessing/data_loading.py:import_jrc_climate_data``:

    G(h)  -> ghi   global horizontal irradiance
    Gb(n) -> dni   beam (direct) normal irradiance
    Gd(h) -> dhi   diffuse horizontal irradiance

Note these are the *horizontal* components. Do not substitute ``G(i)``, which
is plane-of-array irradiance and already contains the tilt gain.

Usage
-----
Build the cache once (needs network)::

    python four_node_configuration/preprocess/solar_anchors.py

Then at run time::

    from solar_anchors import load_calibration, select_anchor
    calib = load_calibration()
    anchor, scale, predicted = select_anchor(0.125, calib)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Anchor sites, chosen for their capacity factor rather than their identity,
#: spaced so the nearest-anchor residual scale stays small. The place names are
#: provenance only -- the study is geography-free and reports ``solar_cf_mean``.
ANCHORS = [
    ("aalborg_dk", 57.05, 9.92),
    ("krakow_pl", 50.06, 19.94),
    ("paris_fr", 48.86, 2.35),
    ("budapest_hu", 47.50, 19.04),
    ("lyon_fr", 45.76, 4.84),
    ("milano_it", 45.46, 9.19),
    ("roma_it", 41.90, 12.50),
    ("madrid_es", 40.42, -3.70),
]

#: PV system assumed when calibrating. Must match the defaults in
#: ``adopt_net0/components/technologies/genericTechnologies/res.py:93-96``,
#: otherwise the calibrated capacity factors will not match what the model
#: actually computes.
PV_SYSTEM = {
    "tilt": 18,
    "surface_azimuth": 180,
    "module_name": "SunPower_SPR_X20_327",
    "inverter_eff": 0.96,
}

#: Residual scale factors probed when calibrating each anchor.
SCALE_GRID = np.round(np.linspace(0.80, 1.20, 9), 4)

_HERE = Path(__file__).resolve().parent
ANCHOR_DIR = _HERE / "data" / "solar_anchors"
CALIBRATION_FILE = ANCHOR_DIR / "calibration.json"

_PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/tmy?"
_CLIMATE_COLUMNS = ["ghi", "dni", "dhi", "temp_air", "ws10"]


# --------------------------------------------------------------------------
# PVGIS download
# --------------------------------------------------------------------------

def fetch_tmy(lat: float, lon: float, timeout: int = 90) -> pd.DataFrame:
    """
    Download one typical meteorological year from PVGIS.

    :param float lat: latitude in decimal degrees
    :param float lon: longitude in decimal degrees
    :param int timeout: request timeout in seconds
    :return: DataFrame with columns ghi, dni, dhi, temp_air, ws10 (8760 rows)
    :rtype: pd.DataFrame
    """
    import requests

    response = requests.get(
        _PVGIS_TMY_URL,
        params={"lon": lon, "lat": lat, "outputformat": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    hourly = response.json()["outputs"]["tmy_hourly"]

    frame = pd.DataFrame(
        {
            "ghi": [row["G(h)"] for row in hourly],
            "dni": [row["Gb(n)"] for row in hourly],
            "dhi": [row["Gd(h)"] for row in hourly],
            "temp_air": [row["T2m"] for row in hourly],
            "ws10": [row["WS10m"] for row in hourly],
        }
    )
    if len(frame) != 8760:
        raise ValueError(f"PVGIS returned {len(frame)} rows for ({lat}, {lon}), expected 8760")
    return frame


# --------------------------------------------------------------------------
# pvlib chain -- mirrors res.py so the calibration is meaningful
# --------------------------------------------------------------------------

def _build_pv_system():
    import pvlib

    module = pvlib.pvsystem.retrieve_sam("CECMod")[PV_SYSTEM["module_name"]]
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]
    system = pvlib.pvsystem.PVSystem(
        surface_tilt=PV_SYSTEM["tilt"],
        surface_azimuth=PV_SYSTEM["surface_azimuth"],
        module_parameters=module,
        inverter_parameters={"pdc0": 5000, "eta_inv_nom": PV_SYSTEM["inverter_eff"]},
        temperature_model_parameters=temperature_model_parameters,
    )
    return system, module


def capacity_factor(climate: pd.DataFrame, lat: float, lon: float,
                    alt: float = 0.0) -> np.ndarray:
    """
    Hourly PV capacity factor for a climate series at a given location.

    Replicates the chain in ``res.py:_perform_fitting_pv`` so the values are
    what the model itself will compute.

    :param pd.DataFrame climate: columns ghi, dni, dhi, temp_air, ws10
    :param float lat: latitude of the node
    :param float lon: longitude of the node
    :param float alt: altitude of the node
    :return: hourly capacity factor, one value per row of ``climate``
    :rtype: np.ndarray
    """
    import pvlib

    system, module = _build_pv_system()
    location = pvlib.location.Location(lat, lon, tz="UTC", altitude=alt)
    model_chain = pvlib.modelchain.ModelChain(
        system, location, spectral_model="no_loss", aoi_model="physical"
    )

    weather = pd.DataFrame(
        {
            "ghi": climate["ghi"].to_numpy(dtype=float),
            "dni": climate["dni"].to_numpy(dtype=float),
            "dhi": climate["dhi"].to_numpy(dtype=float),
            "temp_air": climate["temp_air"].to_numpy(dtype=float),
            "wind_speed": climate["ws10"].to_numpy(dtype=float),
        },
        index=pd.date_range("2021-01-01 00:00", periods=len(climate), freq="h", tz="UTC"),
    )
    model_chain.run_model(weather)
    return (model_chain.results.ac.p_mp / module.STC).to_numpy()


def scale_climate(climate: pd.DataFrame, scale: float) -> pd.DataFrame:
    """
    Scale the three irradiance components by a common factor.

    Scaling all three together preserves the closure ``ghi ~ dni*cos(z) + dhi``
    and the beam/diffuse ratio, so the series stays internally consistent. Air
    temperature is deliberately left untouched: at the residual scales used
    here (within +-7%) the module-temperature effect is second order.

    :param pd.DataFrame climate: columns ghi, dni, dhi, temp_air, ws10
    :param float scale: multiplicative factor
    :return: scaled copy
    :rtype: pd.DataFrame
    """
    scaled = climate.copy()
    for column in ("ghi", "dni", "dhi"):
        scaled[column] = scaled[column] * scale
    return scaled


# --------------------------------------------------------------------------
# Cache build / load
# --------------------------------------------------------------------------

def build(out_dir: Path = ANCHOR_DIR, scale_grid: np.ndarray = SCALE_GRID) -> dict:
    """
    Download every anchor, calibrate it, and write the cache.

    Writes one ``<name>.csv`` per anchor plus ``calibration.json`` mapping
    residual scale to realized mean capacity factor.

    :param Path out_dir: destination folder
    :param np.ndarray scale_grid: residual scales to probe per anchor
    :return: the calibration dict that was written
    :rtype: dict
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = {"pv_system": PV_SYSTEM, "anchors": {}}

    for name, lat, lon in ANCHORS:
        print(f"[ANCHOR] {name}: downloading TMY ({lat}, {lon}) ...")
        climate = fetch_tmy(lat, lon)
        climate.to_csv(out_dir / f"{name}.csv", sep=";", index=False)

        cf_grid = []
        for scale in scale_grid:
            cf = capacity_factor(scale_climate(climate, float(scale)), lat, lon)
            cf_grid.append(float(cf.mean()))
        base = float(np.interp(1.0, scale_grid, cf_grid))

        calibration["anchors"][name] = {
            "lat": lat,
            "lon": lon,
            "cf_base": base,
            "scale_grid": [float(s) for s in scale_grid],
            "cf_grid": cf_grid,
            "ghi_annual_kwh_m2": float(climate["ghi"].sum() / 1000.0),
        }
        print(f"          base CF {base:.4f} | CF range over scale grid "
              f"{cf_grid[0]:.4f} - {cf_grid[-1]:.4f}")

    covered = sorted(a["cf_base"] for a in calibration["anchors"].values())
    calibration["cf_min"] = covered[0]
    calibration["cf_max"] = covered[-1]
    calibration["max_gap"] = float(max(np.diff(covered))) if len(covered) > 1 else 0.0

    with open(out_dir / "calibration.json", "w") as handle:
        json.dump(calibration, handle, indent=2)

    print(f"\n[ANCHOR] wrote {len(ANCHORS)} anchors to {out_dir}")
    print(f"[ANCHOR] CF coverage {calibration['cf_min']:.4f} - {calibration['cf_max']:.4f}, "
          f"largest gap {calibration['max_gap']:.4f}")
    return calibration


def load_calibration(path: Path = CALIBRATION_FILE) -> dict:
    """
    Read the anchor calibration written by :func:`build`.

    :param Path path: path to calibration.json
    :return: calibration dict
    :rtype: dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Solar anchor calibration not found: {path}\n"
            "Build it once with: python four_node_configuration/preprocess/solar_anchors.py"
        )
    with open(path, "r") as handle:
        return json.load(handle)


def select_anchor(target_cf: float, calibration: dict) -> tuple[str, float, float]:
    """
    Pick the anchor that reaches ``target_cf`` with the smallest residual scale.

    :param float target_cf: requested annual-mean capacity factor
    :param dict calibration: as returned by :func:`load_calibration`
    :return: (anchor name, residual scale, predicted mean capacity factor)
    :rtype: tuple
    """
    best = None
    for name, entry in calibration["anchors"].items():
        cf_grid = np.asarray(entry["cf_grid"], dtype=float)
        scale_grid = np.asarray(entry["scale_grid"], dtype=float)
        # cf is monotone increasing in scale, so a plain interpolation inverts it
        scale = float(np.interp(target_cf, cf_grid, scale_grid))
        predicted = float(np.interp(scale, scale_grid, cf_grid))
        penalty = abs(scale - 1.0)
        if best is None or penalty < best[3]:
            best = (name, scale, predicted, penalty)

    name, scale, predicted, penalty = best
    if penalty > 0.25:
        raise ValueError(
            f"solar_cf_mean={target_cf:.4f} is outside the anchor library "
            f"({calibration['cf_min']:.4f} - {calibration['cf_max']:.4f}); "
            f"nearest anchor '{name}' would need scale {scale:.3f}. "
            "Add an anchor site or narrow the sampled range."
        )
    return name, scale, predicted


def load_anchor_climate(name: str, out_dir: Path = ANCHOR_DIR) -> pd.DataFrame:
    """
    Read one cached anchor series.

    :param str name: anchor name
    :param Path out_dir: cache folder
    :return: DataFrame with columns ghi, dni, dhi, temp_air, ws10
    :rtype: pd.DataFrame
    """
    path = Path(out_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Anchor series not found: {path}")
    return pd.read_csv(path, sep=";")


if __name__ == "__main__":
    build()
