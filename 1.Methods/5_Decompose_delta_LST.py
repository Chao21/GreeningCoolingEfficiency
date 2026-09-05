#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Decompose LAI-change-induced LST effects using consistently masked inputs.

Author
------
Chao Zhang, National University of Singapore

Date
----
2026-05-22

Purpose
-------
Aggregate LAI and LST sensitivity from 0.05 to 1 degree using a common valid
mask, decompose cumulative LST effects into LAI- and sensitivity-driven terms,
estimate Theil-Sen trends, and summarize the results regionally.

Notes
-----
For each interval, the total LST increment is the LAI change multiplied by the
mean sensitivity at the two endpoints. The LAI-driven increment uses a fixed
reference sensitivity, and the sensitivity-driven term is their residual. The
workflow does not separate vegetation types. Set ``VEG_LST_DATA_DIR`` and
``VEG_LST_UTILS_DIR`` to override the default data and utility directories.

Scientific workflow
-----
1. Read 0.05° LAI(t) and sensitivity S(t) = ∂LST/∂LAI.
2. Aggregate LAI and S to 1° using the same valid 0.05° pixels:
       valid = finite(LAI) & finite(S)
   Area-weighted aggregation ensures that both variables represent the
   same subgrid population. Optionally aggregate the 1° data to 4°.
3. Define a fixed reference sensitivity S_ref using either the full-period
   climatology or a selected baseline period. For seasonal and monthly
   analyses, calculate S_ref separately for each season or calendar month.
4. Within each annual, seasonal, or monthly sequence, calculate:
       ΔLAI(t) = LAI(t) - LAI(t-1)
       S_mid(t) = [S(t) + S(t-1)] / 2
5. Decompose the cumulative LAI-induced LST effect into:
       dLST_total(t) = cumsum[ΔLAI(t) * S_mid(t)]
       dLST_LAI_driven(t) = cumsum[ΔLAI(t) * S_ref]
       dLST_Sens_driven(t) = dLST_total(t) - dLST_LAI_driven(t)
6. Calculate pixel-wise Theil–Sen trends and Kendall-tau p values for
   each decomposition component at 1° and 4° resolutions.
7. Aggregate the 1° component time series and trends globally, by climate
   zone, and by country.
"""


import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kendalltau, theilslopes

DATA_ROOT = Path(os.environ.get(
    "VEG_LST_DATA_DIR",
    "/home/energy/chaoz/project/05Veg_LST/data",
))
UTILS_DIR = Path(os.environ.get(
    "VEG_LST_UTILS_DIR",
    "/home/energy/chaoz/code/utils",
))
sys.path.insert(0, str(UTILS_DIR))
from da_utils import region_weighted_mean

LAI_PRODUCT = "GLASS"
TIME_SCALES = ("Annual", "Season")
LST_VARIABLE = "dailymean"  # "dailymean", "day", or "night"
START_YEAR, END_YEAR = 2001, 2024
CLIMATE_NAMES = ("Tropical", "Arid", "Temperate", "Boreal")


# ============================================================
# 0. Basic helpers
# ============================================================
def log(message: str) -> None:
    """Print a timestamped message."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def ensure_dir(path: str | Path) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def provenance_attrs() -> dict[str, str]:
    """Return common provenance metadata for generated NetCDF files."""
    return {
        "creator": "Chao Zhang",
        "institution": "National University of Singapore",
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
    }


def standardize_time_scale(time_scale: str) -> str:
    """Standardize time scale string."""
    ts = time_scale.lower()
    if ts not in {"annual", "season", "month"}:
        raise ValueError("time_scale must be one of {'Annual', 'Season', 'Month'}")
    return ts


def round_latlon(ds_or_da, ndigits: int = 4):
    """Round lat/lon to avoid tiny floating point mismatch during alignment."""
    out = ds_or_da.copy()
    if "lat" in out.coords:
        out = out.assign_coords(lat=np.round(out["lat"].astype(np.float32), ndigits))
    if "lon" in out.coords:
        out = out.assign_coords(lon=np.round(out["lon"].astype(np.float32), ndigits))
    return out


def prep_3d(da: xr.DataArray, start_year: int, end_year: int) -> xr.DataArray:
    """
    Subset a 3D DataArray to the target period and enforce:
        dims = time, lat, lon
    """
    da = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    da = da.transpose("time", "lat", "lon")
    da = round_latlon(da)
    return da


def time_values_to_decimal_year(time_values) -> np.ndarray:
    """Convert array-like datetime values to decimal year."""
    dt = pd.to_datetime(time_values)

    year = dt.year.astype(float)
    doy = dt.dayofyear.astype(float)
    diy = np.where(dt.is_leap_year, 366.0, 365.0)

    return year + (doy - 1.0) / diy


def time_to_decimal_year(time: xr.DataArray) -> np.ndarray:
    """Convert datetime64 time coordinate to decimal year."""
    return time_values_to_decimal_year(time.values)


def get_time_groups(time: xr.DataArray, time_scale: str):
    """
    Return time-group labels for annual / season / month outputs.

    annual:
        one group: annual
    season:
        expects representative months {3,6,9,12}
        spring=3, summer=6, autumn=9, winter=12
    month:
        Jan-Dec
    """
    time_scale = standardize_time_scale(time_scale)
    t = xr.DataArray(time.values, dims="time")
    nt = t.size

    if time_scale == "annual":
        return np.zeros(nt, dtype=int), ["annual"]

    if time_scale == "season":
        m = t.dt.month.values
        labels = np.full(nt, -1, dtype=int)
        labels[m == 3] = 0
        labels[m == 6] = 1
        labels[m == 9] = 2
        labels[m == 12] = 3
        if np.any(labels < 0):
            bad = np.unique(m[labels < 0])
            raise ValueError(
                "time_scale='Season' expects timestamps in months {3,6,9,12}. "
                f"Found unexpected months: {bad}"
            )
        return labels, ["spring", "summer", "autumn", "winter"]

    labels = (t.dt.month.values - 1).astype(int)
    return labels, ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def theil_sen_ci_p(
    x: np.ndarray,
    y: np.ndarray,
    ci: float = 0.95,
) -> tuple[float, float, float, float]:
    """
    Compute Theil-Sen slope, confidence interval, and Kendall-tau p value.

    Returns
    -------
    slope, ci_lo, ci_hi, pval : float
    """
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    xm = x[m]
    ym = y[m]

    if np.unique(xm).size < 2:
        return np.nan, np.nan, np.nan, np.nan

    slope, _, lo, hi = theilslopes(ym, xm, alpha=ci)

    kt = kendalltau(xm, ym, nan_policy="omit")
    pval = float(kt.pvalue) if np.isfinite(kt.pvalue) else np.nan

    return float(slope), float(lo), float(hi), pval
# ============================================================
# 1. 0.05° -> 1° consistent-mask aggregation
# ============================================================
def build_0p05_to_1deg_index_edge_based(lat_0p05: np.ndarray, lon_0p05: np.ndarray):
    """
    Edge-based 0.05° to 1° mapping.

    Output 1° grid:
        lat = 90, 89, ..., -89
        lon = -180, -179, ..., 179

    Interpretation:
        lat coordinate is the upper edge of a 1° cell [lat-1, lat].
        lon coordinate is the left edge of a 1° cell [lon, lon+1).
    """
    lat0 = np.asarray(lat_0p05)
    lon0 = np.asarray(lon_0p05)

    lat1 = np.arange(90, -90, -1)
    lon1 = np.arange(-180, 180, 1)
    nlat1 = lat1.size
    nlon1 = lon1.size

    eps = 1e-10
    lat_bin = np.floor(90.0 - lat0 + eps).astype(np.int32)
    lon_bin = np.floor(lon0 + 180.0 + eps).astype(np.int32)

    lat_bin = np.clip(lat_bin, 0, nlat1 - 1)
    lon_bin = np.clip(lon_bin, 0, nlon1 - 1)

    gid_flat = (
        np.repeat(lat_bin.astype(np.int64), lon_bin.size) * nlon1
        + np.tile(lon_bin.astype(np.int64), lat_bin.size)
    ).astype(np.int32)

    n_1deg = int(nlat1 * nlon1)

    return gid_flat, lat1, lon1, n_1deg


def aggregate_one_field_to_coarse(
    value_flat: np.ndarray,
    valid_flat: np.ndarray,
    weight_flat: np.ndarray,
    gid_flat: np.ndarray,
    n_1deg: int,
):
    """
    Aggregate one fine (0.05°) field to coarse (1°) using area-weighted mean.

    Parameters
    ----------
    value_flat : np.ndarray
        Flattened fine (0.05°) field.
    valid_flat : np.ndarray
        Flattened common mask. The same mask must be used for LAI and S.
    weight_flat : np.ndarray
        Area proxy weights, usually cos(latitude).
    gid_flat : np.ndarray
        Flattened coarse (1°) grid-cell id for each fine (0.05°) pixel.
    n_1deg : int
        Number of coarse (1°) cells.
    """
    val = np.asarray(value_flat, dtype=float)
    valid = np.asarray(valid_flat, dtype=bool)
    w = np.asarray(weight_flat, dtype=float)

    keep = valid & np.isfinite(val) & np.isfinite(w) & (w > 0)

    count = np.bincount(gid_flat[keep], minlength=n_1deg).astype(np.int32)
    wsum = np.bincount(gid_flat[keep], weights=w[keep], minlength=n_1deg).astype(float)
    vsum = np.bincount(
        gid_flat[keep],
        weights=val[keep] * w[keep],
        minlength=n_1deg,
    ).astype(float)

    mean = np.divide(
        vsum,
        wsum,
        out=np.full(n_1deg, np.nan, dtype=float),
        where=wsum > 0,
    )

    return mean.astype(np.float32), count, wsum.astype(np.float32)


def aggregate_LAI_S_to_1deg_consistent_mask(
    LAI: xr.DataArray,
    Sen: xr.DataArray,
    *,
    min_count: int = 10,
) -> xr.Dataset:
    """
    Aggregate 0.05° LAI and sensitivity S to 1° using a consistent mask.

    Common valid mask for each time step:
        valid = finite(LAI) & finite(Sen)

    This ensures that 1° LAI and 1° sensitivity represent the same subgrid
    population before calculating LST* = S * LAI.
    """
    log("Aligning 0.05° LAI and sensitivity...")
    LAI, Sen = xr.align(LAI, Sen, join="inner")

    LAI = LAI.transpose("time", "lat", "lon")
    Sen = Sen.transpose("time", "lat", "lon")

    gid_flat, lat1, lon1, n_1deg = build_0p05_to_1deg_index_edge_based(
        LAI["lat"].values,
        LAI["lon"].values,
    )

    # Area proxy for regular lon-lat grid.
    w_lat = np.cos(np.deg2rad(LAI["lat"].values))
    weight_2d = np.repeat(w_lat[:, None], LAI.sizes["lon"], axis=1).astype(np.float32)
    weight_flat = weight_2d.ravel()

    nt = LAI.sizes["time"]
    times = LAI["time"].values

    LAI_all = np.full((nt, n_1deg), np.nan, dtype=np.float32)
    Sen_all = np.full((nt, n_1deg), np.nan, dtype=np.float32)
    count_all = np.zeros((nt, n_1deg), dtype=np.int32)
    wsum_all = np.full((nt, n_1deg), np.nan, dtype=np.float32)
    coverage_all = np.full((nt, n_1deg), np.nan, dtype=np.float32)

    # Max possible count per 1° cell based on all 0.05° pixels, independent of data validity.
    all_valid = np.ones_like(LAI.isel(time=0).values, dtype=bool).ravel()
    max_count = np.bincount(gid_flat[all_valid], minlength=n_1deg).astype(np.float32)

    for ti in range(nt):
        log(f"Aggregating LAI/S to 1° with common mask: time {ti + 1}/{nt}")

        LAI0 = LAI.isel(time=ti).values
        Sen0 = Sen.isel(time=ti).values

        valid = np.isfinite(LAI0) & np.isfinite(Sen0)
        valid_flat = valid.ravel()

        LAI_mean, count, wsum = aggregate_one_field_to_coarse(
            LAI0.ravel(), valid_flat, weight_flat, gid_flat, n_1deg
        )
        Sen_mean, _, _ = aggregate_one_field_to_coarse(
            Sen0.ravel(), valid_flat, weight_flat, gid_flat, n_1deg
        )

        sparse = count < min_count
        LAI_mean[sparse] = np.nan
        Sen_mean[sparse] = np.nan

        coverage = np.divide(
            count.astype(np.float32),
            max_count,
            out=np.full(n_1deg, np.nan, dtype=np.float32),
            where=max_count > 0,
        )
        coverage[sparse] = np.nan

        LAI_all[ti] = LAI_mean
        Sen_all[ti] = Sen_mean
        count_all[ti] = count
        wsum_all[ti] = wsum
        coverage_all[ti] = coverage

    ds_out = xr.Dataset(
        data_vars={
            "LAI": (("time", "lat", "lon"), LAI_all.reshape(nt, lat1.size, lon1.size)),
            "sens": (("time", "lat", "lon"), Sen_all.reshape(nt, lat1.size, lon1.size)),
            "count_0p05": (("time", "lat", "lon"), count_all.reshape(nt, lat1.size, lon1.size)),
            "weight_sum": (("time", "lat", "lon"), wsum_all.reshape(nt, lat1.size, lon1.size)),
            "coverage": (("time", "lat", "lon"), coverage_all.reshape(nt, lat1.size, lon1.size)),
        },
        coords={
            "time": times,
            "lat": lat1.astype(np.float32),
            "lon": lon1.astype(np.float32),
        },
        attrs={
            "description": (
                "1° LAI and sensitivity aggregated from 0.05° using a "
                "consistent valid mask."
            ),
            "common_mask": "finite(LAI) & finite(sensitivity)",
            "min_count": int(min_count),
        },
    )

    ds_out["LAI"].attrs = {
        "long_name": "Leaf area index aggregated to 1 degree",
        "units": "m2 m-2",
        "aggregation": "area-weighted mean using common valid mask",
    }
    ds_out["sens"].attrs = {
        "long_name": "LST-LAI sensitivity aggregated to 1 degree",
        "units": "K per LAI unit",
        "aggregation": "area-weighted mean using common valid mask",
    }
    ds_out["coverage"].attrs = {
        "long_name": "Valid 0.05 degree pixel fraction within each 1 degree cell",
        "units": "1",
        "formula": "count_valid_0p05 / count_total_0p05",
    }

    return ds_out


# ============================================================
# Optional: aggregate 1° time-series dataset to coarser grids
# ============================================================
def build_1deg_to_coarse_index_edge_based(
    lat_1deg: np.ndarray,
    lon_1deg: np.ndarray,
    *,
    resolution: int = 4,
):
    """
    Build edge-based 1° -> coarse-grid mapping.

    Default 4° grid:
        lat = [90, 86, 82, ..., -86]
        lon = [-180, -176, -172, ..., 176]

    Example 3° grid:
        lat = [90, 87, 84, ..., -87]
        lon = [-180, -177, -174, ..., 177]

    Notes
    -----
    The input 1° grid is assumed to be edge-based:
        lat = [90, 89, ..., -89]
        lon = [-180, -179, ..., 179]
    """
    if 180 % resolution != 0 or 360 % resolution != 0:
        raise ValueError("resolution must exactly divide both 180 and 360, e.g., 2, 3, 4, 5, 6.")

    lat1 = np.asarray(lat_1deg)
    lon1 = np.asarray(lon_1deg)

    lat_coarse = np.arange(90, -90, -resolution)
    lon_coarse = np.arange(-180, 180, resolution)

    nlat_c = lat_coarse.size
    nlon_c = lon_coarse.size

    eps = 1e-10
    lat_bin = np.floor((90.0 - lat1 + eps) / resolution).astype(np.int32)
    lon_bin = np.floor((lon1 + 180.0 + eps) / resolution).astype(np.int32)

    lat_bin = np.clip(lat_bin, 0, nlat_c - 1)
    lon_bin = np.clip(lon_bin, 0, nlon_c - 1)

    gid_flat = (
        np.repeat(lat_bin.astype(np.int64), lon_bin.size) * nlon_c
        + np.tile(lon_bin.astype(np.int64), lat_bin.size)
    ).astype(np.int32)

    n_coarse = int(nlat_c * nlon_c)

    return gid_flat, lat_coarse, lon_coarse, n_coarse


def aggregate_1deg_to_coarse(
    ds_1deg: xr.Dataset,
    *,
    resolution: int = 4,
    value_vars: Sequence[str] = ("LAI", "sens", "coverage"),
    common_valid_vars: Sequence[str] = ("LAI", "sens"),
    min_count: int = 8,
) -> xr.Dataset:
    """
    Aggregate 1° time-series variables to a coarser edge-based grid using
    area-weighted means.

    For scientific consistency, all variables are aggregated using the same
    common 1° valid mask, by default:
        valid = finite(LAI) & finite(sens)

    min_count is the minimum number of valid 1° cells required within each
    coarse cell.
    """
    log(f"Aggregating 1° dataset to {resolution}° grid")

    ds_1deg = ds_1deg.transpose("time", "lat", "lon")

    for v in common_valid_vars:
        if v not in ds_1deg:
            raise KeyError(f"Common-mask variable '{v}' not found in ds_1deg.")

    gid_flat, lat_c, lon_c, n_coarse = build_1deg_to_coarse_index_edge_based(
        ds_1deg["lat"].values,
        ds_1deg["lon"].values,
        resolution=resolution,
    )

    w_lat = np.cos(np.deg2rad(ds_1deg["lat"].values))
    weight_2d = np.repeat(w_lat[:, None], ds_1deg.sizes["lon"], axis=1).astype(np.float32)
    weight_flat = weight_2d.ravel()

    nt = ds_1deg.sizes["time"]
    times = ds_1deg["time"].values

    data_vars = {}
    count_all = np.zeros((nt, n_coarse), dtype=np.int32)
    wsum_all = np.full((nt, n_coarse), np.nan, dtype=np.float32)

    for v in value_vars:
        if v not in ds_1deg:
            log(f"  Skip missing variable: {v}")
            continue

        log(f"  Aggregating variable with common mask: {v}")
        out_all = np.full((nt, n_coarse), np.nan, dtype=np.float32)

        for ti in range(nt):
            arr = ds_1deg[v].isel(time=ti).values

            common_valid = np.ones_like(arr, dtype=bool)
            for mask_var in common_valid_vars:
                common_valid &= np.isfinite(ds_1deg[mask_var].isel(time=ti).values)

            valid_flat = common_valid.ravel()

            mean, count, wsum = aggregate_one_field_to_coarse(
                arr.ravel(),
                valid_flat,
                weight_flat,
                gid_flat,
                n_coarse,
            )

            sparse = count < min_count
            mean[sparse] = np.nan

            out_all[ti] = mean

            if v == value_vars[0]:
                count_all[ti] = count
                wsum_all[ti] = wsum

        data_vars[v] = (("time", "lat", "lon"), out_all.reshape(nt, lat_c.size, lon_c.size))

    data_vars["count_1deg"] = (
        ("time", "lat", "lon"),
        count_all.reshape(nt, lat_c.size, lon_c.size),
    )
    data_vars["weight_sum"] = (
        ("time", "lat", "lon"),
        wsum_all.reshape(nt, lat_c.size, lon_c.size),
    )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "lat": lat_c.astype(np.float32),
            "lon": lon_c.astype(np.float32),
        },
        attrs={
            "description": f"{resolution}° dataset aggregated from 1° using area-weighted mean.",
            "source_resolution": "1 degree",
            "target_resolution": f"{resolution} degree",
            "grid_definition": (
                f"edge-based: lat=np.arange(90,-90,-{resolution}); "
                f"lon=np.arange(-180,180,{resolution})"
            ),
            "aggregation": "area-weighted mean using cos(latitude)",
            "common_mask": " & ".join([f"finite({v})" for v in common_valid_vars]),
            "min_count_1deg": int(min_count),
        },
    )

    for v in value_vars:
        if v in ds_out and v in ds_1deg:
            ds_out[v].attrs.update(ds_1deg[v].attrs)
            ds_out[v].attrs["aggregation_to_coarse"] = (
                f"area-weighted mean from 1° to {resolution}° using common valid mask"
            )

    ds_out["count_1deg"].attrs = {
        "long_name": f"Number of valid 1 degree cells within each {resolution} degree cell",
        "units": "1",
        "valid_definition": ds_out.attrs["common_mask"],
    }

    return ds_out


# ============================================================
# 2. dLST decomposition at 1° or coarser resolution
# ============================================================
def compute_dLST_decomposition(
    ds: xr.Dataset,
    *,
    time_scale: str,
    lai_var: str = "LAI",
    sen_var: str = "sens",
    cov_var: str = "coverage",
    baseline_start: str = "2001-01-01",
    baseline_end: str = "2005-12-31",
    sen_ref_mode: str = "climatology",   # "climatology" or "baseline"
    cumsum_skipna: bool = False,
) -> xr.Dataset:
    """
    Decompose greening/browning-induced LST effect using cumulative year-to-year
    LAI changes and time-varying LST-LAI sensitivity.

    This follows the Li et al. style cumulative formulation, applied to the
    temporal resolution of the input data. The input sensitivity can already be
    annual, seasonal, or monthly.

    Total effect
    ------------
        dLST_total(Y) = Σ[ΔLAI(y) * mean(S(y), S(y-1))]

    where:
        ΔLAI(y) = LAI(y) - LAI(y-1)

    LAI-driven effect
    -----------------
        dLST_LAI_driven(Y) = Σ[ΔLAI(y) * S_ref]

    Sensitivity-driven effect
    -------------------------
        dLST_Sens_driven(Y) = dLST_total(Y) - dLST_LAI_driven(Y)

    Equivalently:
        dLST_Sens_driven(Y)
            = Σ{ΔLAI(y) * [mean(S(y), S(y-1)) - S_ref]}

    Reference sensitivity
    ---------------------
    sen_ref_mode="climatology":
        S_ref is the multi-year mean sensitivity over the full analysis period.

    sen_ref_mode="baseline":
        S_ref is the mean sensitivity over baseline_start to baseline_end.

    For seasonal or monthly input, S_ref is calculated separately for each
    season/month.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing LAI, sensitivity, and coverage variables with
        dimensions time, lat, lon.

    time_scale : str
        Temporal scale of the input data. Must be one of:
            "Annual", "Season", "Month"

    lai_var : str, default "LAI"
        Name of the LAI variable in ds.

    sen_var : str, default "sens"
        Name of the LST-LAI sensitivity variable in ds.

    cov_var : str, default "coverage"
        Name of the coverage variable in ds.

    baseline_start, baseline_end : str
        Baseline period used only when sen_ref_mode="baseline".

    sen_ref_mode : str, default "climatology"
        Method for defining fixed reference sensitivity S_ref.
        Options:
            "climatology"
            "baseline"

    cumsum_skipna : bool, default False
        Missing-value behavior for cumulative sums.

        False:
            Missing year-to-year increments propagate forward in the cumulative
            time series. This is stricter and usually safer for physical
            cumulative effects.

        True:
            Missing increments are skipped during cumulative summation. Use this
            only if missing increments should be treated as absent rather than
            invalidating later cumulative values.

    Returns
    -------
    xr.Dataset
        Dataset containing:
            dLST_total
            dLST_LAI_driven
            dLST_Sens_driven
            coverage
            LAI
            sens
            optional diagnostics copied from input dataset
    """
    time_scale = standardize_time_scale(time_scale)

    sen_ref_mode = sen_ref_mode.lower()
    if sen_ref_mode not in {"climatology", "baseline"}:
        raise ValueError("sen_ref_mode must be 'climatology' or 'baseline'.")

    for var_name, label in [
        (lai_var, "LAI"),
        (sen_var, "Sensitivity"),
        (cov_var, "Coverage"),
    ]:
        if var_name not in ds:
            raise KeyError(f"{label} variable '{var_name}' not found in dataset.")

    LAI = ds[lai_var].transpose("time", "lat", "lon")
    Sen = ds[sen_var].transpose("time", "lat", "lon")
    Cov = ds[cov_var].transpose("time", "lat", "lon")
    time = LAI["time"]

    LAI, Sen, Cov = xr.align(LAI, Sen, Cov, join="inner")

    if LAI.sizes.get("time", 0) < 2:
        raise ValueError(
            "At least two time steps are required for cumulative decomposition."
        )

    # --------------------------------------------------
    # Reference sensitivity source
    # --------------------------------------------------
    if sen_ref_mode == "baseline":
        Sen_ref_src = Sen.sel(time=slice(baseline_start, baseline_end))
        if Sen_ref_src.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No sensitivity data found in baseline period "
                f"{baseline_start} to {baseline_end}."
            )
    else:
        Sen_ref_src = Sen

    # --------------------------------------------------
    # Helper: cumulative decomposition for one grouped series
    # --------------------------------------------------
    def _cumulative_effect(
        LAI_g: xr.DataArray,
        Sen_g: xr.DataArray,
        Sen_ref_g: xr.DataArray,
    ):
        """
        Calculate cumulative LAI-change-induced dLST for one temporal group.

        Parameters
        ----------
        LAI_g, Sen_g : xr.DataArray
            Time series for one annual/seasonal/monthly group with dimensions:
                time, lat, lon

        Sen_ref_g : xr.DataArray
            Fixed reference sensitivity for this group with dimensions:
                lat, lon

        Returns
        -------
        total, lai_driven, sens_driven : xr.DataArray
            Cumulative effects with the same time coordinate as LAI_g.
        """
        LAI_g, Sen_g = xr.align(LAI_g, Sen_g, join="inner")

        if LAI_g.sizes.get("time", 0) != Sen_g.sizes.get("time", 0):
            raise ValueError("LAI_g and Sen_g must have the same number of time steps.")

        if LAI_g.sizes.get("time", 0) < 2:
            nan_arr = xr.full_like(LAI_g, np.nan, dtype=np.float32)
            return nan_arr, nan_arr, nan_arr

        time_new = LAI_g["time"].isel(time=slice(1, None))

        LAI_now = LAI_g.isel(time=slice(1, None))
        LAI_prev = LAI_g.isel(time=slice(0, -1)).assign_coords(time=time_new)

        Sen_now = Sen_g.isel(time=slice(1, None))
        Sen_prev = Sen_g.isel(time=slice(0, -1)).assign_coords(time=time_new)

        dLAI = (LAI_now - LAI_prev).rename("dLAI_increment")
        Sen_mid = ((Sen_now + Sen_prev) / 2.0).rename("Sen_mid")

        inc_total = dLAI * Sen_mid
        inc_lai = dLAI * Sen_ref_g

        total = inc_total.cumsum("time", skipna=cumsum_skipna)
        lai_driven = inc_lai.cumsum("time", skipna=cumsum_skipna)
        sens_driven = total - lai_driven

        zero = xr.zeros_like(LAI_g.isel(time=0)).expand_dims(
            time=[LAI_g["time"].values[0]]
        )

        total = xr.concat([zero, total], dim="time").assign_coords(time=LAI_g["time"])
        lai_driven = xr.concat([zero, lai_driven], dim="time").assign_coords(
            time=LAI_g["time"]
        )
        sens_driven = xr.concat([zero, sens_driven], dim="time").assign_coords(
            time=LAI_g["time"]
        )

        return total, lai_driven, sens_driven

    # --------------------------------------------------
    # Annual / seasonal / monthly processing
    # --------------------------------------------------
    if time_scale == "annual":
        Sen_ref = Sen_ref_src.mean("time", skipna=True)

        dLST_total, dLST_LAI_driven, dLST_Sens_driven = _cumulative_effect(
            LAI,
            Sen,
            Sen_ref,
        )

        Sen_ref_out = Sen_ref

    elif time_scale == "season":
        labels, group_names = get_time_groups(time, time_scale)
        season_values = np.array(group_names, dtype=object)[labels]

        LAI2 = LAI.assign_coords(season=("time", season_values))
        Sen2 = Sen.assign_coords(season=("time", season_values))

        ref_labels, ref_group_names = get_time_groups(Sen_ref_src["time"], time_scale)
        ref_season_values = np.array(ref_group_names, dtype=object)[ref_labels]
        Sen_ref2 = Sen_ref_src.assign_coords(season=("time", ref_season_values))

        total_list = []
        lai_list = []
        sens_list = []
        sen_ref_list = []

        def _drop_coord(da: xr.DataArray, name: str) -> xr.DataArray:
            if name in da.coords:
                return da.reset_coords(name, drop=True)
            return da

        for gname in group_names:
            LAI_g = _drop_coord(LAI2.where(LAI2["season"] == gname, drop=True), "season")
            Sen_g = _drop_coord(Sen2.where(Sen2["season"] == gname, drop=True), "season")
            Sen_ref_g = _drop_coord(
                Sen_ref2.where(Sen_ref2["season"] == gname, drop=True),
                "season",
            ).mean("time", skipna=True)

            total_g, lai_g, sens_g = _cumulative_effect(LAI_g, Sen_g, Sen_ref_g)

            total_list.append(total_g)
            lai_list.append(lai_g)
            sens_list.append(sens_g)
            sen_ref_list.append(Sen_ref_g.expand_dims(group=[gname]))

        dLST_total = xr.concat(total_list, dim="time").sortby("time")
        dLST_LAI_driven = xr.concat(lai_list, dim="time").sortby("time")
        dLST_Sens_driven = xr.concat(sens_list, dim="time").sortby("time")
        Sen_ref_out = xr.concat(sen_ref_list, dim="group")

    elif time_scale == "month":
        total_list = []
        lai_list = []
        sens_list = []
        sen_ref_list = []

        for m in range(1, 13):
            LAI_g = LAI.where(LAI["time"].dt.month == m, drop=True)
            Sen_g = Sen.where(Sen["time"].dt.month == m, drop=True)
            Sen_ref_g = (
                Sen_ref_src.where(Sen_ref_src["time"].dt.month == m, drop=True)
                .mean("time", skipna=True)
            )

            total_g, lai_g, sens_g = _cumulative_effect(LAI_g, Sen_g, Sen_ref_g)

            total_list.append(total_g)
            lai_list.append(lai_g)
            sens_list.append(sens_g)
            sen_ref_list.append(Sen_ref_g.expand_dims(month=[m]))

        dLST_total = xr.concat(total_list, dim="time").sortby("time")
        dLST_LAI_driven = xr.concat(lai_list, dim="time").sortby("time")
        dLST_Sens_driven = xr.concat(sens_list, dim="time").sortby("time")
        Sen_ref_out = xr.concat(sen_ref_list, dim="month")

    else:
        raise ValueError("time_scale must be annual, season, or month")

    dLST_total = dLST_total.rename("dLST_total")
    dLST_LAI_driven = dLST_LAI_driven.rename("dLST_LAI_driven")
    dLST_Sens_driven = dLST_Sens_driven.rename("dLST_Sens_driven")

    data_vars = {
        "dLST_total": dLST_total.astype(np.float32),
        "dLST_LAI_driven": dLST_LAI_driven.astype(np.float32),
        "dLST_Sens_driven": dLST_Sens_driven.astype(np.float32),
        "coverage": Cov.astype(np.float32),
        "LAI": LAI.astype(np.float32),
        "sens": Sen.astype(np.float32),
        "sens_ref": Sen_ref_out.astype(np.float32),
    }

    for optional_var in ["count_0p05", "count_1deg", "weight_sum"]:
        if optional_var in ds:
            data_vars[optional_var] = ds[optional_var]

    ds_out = xr.Dataset(
        data_vars,
        attrs={
            "description": (
                "Decomposition of greening/browning-induced LST effect using "
                "cumulative year-to-year LAI changes and time-varying sensitivity."
            ),
            "decomposition_formula": (
                "dLST_total(Y)=sum[ΔLAI(y)*mean(S(y),S(y-1))]; "
                "dLST_LAI_driven(Y)=sum[ΔLAI(y)*S_ref]; "
                "dLST_Sens_driven(Y)=sum[ΔLAI(y)*(mean(S(y),S(y-1))-S_ref)]"
            ),
            "time_scale": time_scale,
            "sen_ref_mode": sen_ref_mode,
            "baseline_period_for_optional_sen_ref": f"{baseline_start} to {baseline_end}",
            "cumsum_skipna": int(cumsum_skipna),
            "common_mask": ds.attrs.get("common_mask", "finite(LAI) & finite(sens)"),
            "coverage_variable": cov_var,
            "note": (
                "The first time step is set to zero because cumulative effects "
                "are defined from year-to-year LAI increments."
            ),
        },
    )

    ds_out["dLST_total"].attrs = {
        "long_name": "Total cumulative LAI-change-induced LST effect",
        "units": "K",
        "formula": "cumsum(ΔLAI * mean[S(t), S(t-1)])",
        "first_time_step": "zero baseline",
    }

    ds_out["dLST_LAI_driven"].attrs = {
        "long_name": (
            "LAI-driven cumulative LST effect under fixed reference sensitivity"
        ),
        "units": "K",
        "formula": "cumsum(ΔLAI * S_ref)",
        "S_ref_mode": sen_ref_mode,
    }

    ds_out["dLST_Sens_driven"].attrs = {
        "long_name": "Sensitivity-driven cumulative LST effect",
        "units": "K",
        "formula": (
            "dLST_total - dLST_LAI_driven = "
            "cumsum[ΔLAI * (mean[S(t), S(t-1)] - S_ref)]"
        ),
        "S_ref_mode": sen_ref_mode,
    }

    ds_out["coverage"].attrs = ds[cov_var].attrs.copy()
    ds_out["coverage"].attrs["source_variable"] = cov_var

    ds_out["LAI"].attrs = ds[lai_var].attrs.copy()
    ds_out["sens"].attrs = ds[sen_var].attrs.copy()

    ds_out["sens_ref"].attrs = {
        "long_name": "Reference LST-LAI sensitivity used for LAI-driven component",
        "units": ds[sen_var].attrs.get("units", "K per LAI unit"),
        "sen_ref_mode": sen_ref_mode,
        "baseline_period_if_used": f"{baseline_start} to {baseline_end}",
        "description": (
            "For annual data, dimensions are lat, lon. For seasonal data, "
            "dimensions are group, lat, lon. For monthly data, dimensions are "
            "month, lat, lon."
        ),
    }

    return ds_out


# ============================================================
# 3. Pixel-wise trends
# ============================================================
def compute_pixel_trend_adaptive(
    ds_1deg: xr.Dataset,
    out_nc: str | Path,
    *,
    time_scale: str,
    value_var: str,
    cov_var: str = "coverage",
    cov_thresh: float = 0.025,
    min_trend_n: int = 3,
    ci: float = 0.95,
    encoding_zlib_level: int = 5,
) -> xr.Dataset:
    """
    Compute pixel-wise Theil-Sen trend, CI, and p value for a 1° variable.

    For annual:
        output dims = lat, lon
    For seasonal/monthly:
        output dims = group, lat, lon
    """
    time_scale = standardize_time_scale(time_scale)

    V = ds_1deg[value_var]
    C = ds_1deg[cov_var]

    x_all = time_to_decimal_year(V["time"])
    labels, group_names = get_time_groups(V["time"], time_scale)

    nlat = V.sizes["lat"]
    nlon = V.sizes["lon"]
    Vv = V.values
    Cv = C.values

    if time_scale == "annual":
        slope = np.full((nlat, nlon), np.nan, dtype=np.float32)
        lo_ci = np.full((nlat, nlon), np.nan, dtype=np.float32)
        hi_ci = np.full((nlat, nlon), np.nan, dtype=np.float32)
        pval = np.full((nlat, nlon), np.nan, dtype=np.float32)

        for i in range(nlat):
            for j in range(nlon):
                y = Vv[:, i, j]
                cov = Cv[:, i, j]
                m = (cov >= cov_thresh) & np.isfinite(y) & np.isfinite(x_all)
                if m.sum() < min_trend_n:
                    continue
                s, lo, hi, p = theil_sen_ci_p(x_all[m], y[m], ci=ci)
                slope[i, j] = s
                lo_ci[i, j] = lo
                hi_ci[i, j] = hi
                pval[i, j] = p

        ds_out = xr.Dataset(
            data_vars={
                "trend_theilsen": (("lat", "lon"), slope),
                "trend_ci_lo": (("lat", "lon"), lo_ci),
                "trend_ci_hi": (("lat", "lon"), hi_ci),
                "trend_p": (("lat", "lon"), pval),
            },
            coords={"lat": V["lat"].values, "lon": V["lon"].values},
            attrs={
                "time_scale": time_scale,
                "target_var": value_var,
                "coverage_threshold": float(cov_thresh),
                "ci": float(ci),
                "x_axis": "decimal_year",
                "p_value_test": "kendalltau two-sided",
                "trend_units": "K yr-1",
            },
        )

    else:
        G = len(group_names)
        slope = np.full((G, nlat, nlon), np.nan, dtype=np.float32)
        lo_ci = np.full((G, nlat, nlon), np.nan, dtype=np.float32)
        hi_ci = np.full((G, nlat, nlon), np.nan, dtype=np.float32)
        pval = np.full((G, nlat, nlon), np.nan, dtype=np.float32)

        for g, gname in enumerate(group_names):
            idx = labels == g
            xg = x_all[idx]
            log(f"Trend group={gname}, n={idx.sum()}")

            for i in range(nlat):
                for j in range(nlon):
                    y = Vv[idx, i, j]
                    cov = Cv[idx, i, j]
                    m = (cov >= cov_thresh) & np.isfinite(y) & np.isfinite(xg)
                    if m.sum() < min_trend_n:
                        continue
                    s, lo, hi, p = theil_sen_ci_p(xg[m], y[m], ci=ci)
                    slope[g, i, j] = s
                    lo_ci[g, i, j] = lo
                    hi_ci[g, i, j] = hi
                    pval[g, i, j] = p

        ds_out = xr.Dataset(
            data_vars={
                "trend_theilsen": (("group", "lat", "lon"), slope),
                "trend_ci_lo": (("group", "lat", "lon"), lo_ci),
                "trend_ci_hi": (("group", "lat", "lon"), hi_ci),
                "trend_p": (("group", "lat", "lon"), pval),
            },
            coords={
                "group": group_names,
                "lat": V["lat"].values,
                "lon": V["lon"].values,
            },
            attrs={
                "time_scale": time_scale,
                "target_var": value_var,
                "coverage_threshold": float(cov_thresh),
                "ci": float(ci),
                "x_axis": "decimal_year",
                "p_value_test": "kendalltau two-sided",
                "trend_units": "K yr-1",
            },
        )

    ds_out.attrs.update(provenance_attrs())
    enc = {
        v: {"zlib": True, "complevel": encoding_zlib_level}
        for v in ds_out.data_vars
    }
    ds_out.to_netcdf(out_nc, encoding=enc)
    log(f"Saved trend nc: {out_nc}")

    return ds_out



def compute_trend_contribution_and_ratio(
    lai_trend,
    sen_trend,
    out_nc: str | Path,
    *,
    lai_trend_var: str = "trend_theilsen",
    sen_trend_var: str = "trend_theilsen",
    eps: float = 1e-12,
    encoding_zlib_level: int = 5,
) -> xr.Dataset:
    """
    Calculate the relative contribution and signed ratio of LAI-driven and
    sensitivity-driven dLST trends.

    This function is designed to be applied to pixel-wise trend datasets
    generated from:
        dLST_LAI_driven
        dLST_Sens_driven

    Definitions
    -----------
    Let:
        T_lai = trend of dLST_LAI_driven
        T_sen = trend of dLST_Sens_driven

    The absolute contribution of each component is:

        Contri_sen = abs(T_sen) / [abs(T_sen) + abs(T_lai)] * 100

        Contri_lai = abs(T_lai) / [abs(T_sen) + abs(T_lai)] * 100

    These two variables describe the relative magnitude contribution of each
    component, independent of sign. Where both trends are zero or unavailable,
    contributions are set to NaN.

    The signed offset/intensification ratio is calculated where
    abs(T_lai) > eps:

        ratio_sen_to_lai = T_sen / T_lai

    Interpretation
    --------------
    ratio_sen_to_lai > 0:
        Sensitivity-driven and LAI-driven trends have the same sign, so the
        sensitivity-driven trend intensifies the LAI-driven trend.

    ratio_sen_to_lai < 0:
        Sensitivity-driven and LAI-driven trends have opposite signs, so the
        sensitivity-driven trend offsets the LAI-driven trend.

    abs(ratio_sen_to_lai):
        Magnitude of the sensitivity-driven trend relative to the LAI-driven
        trend.

    Parameters
    ----------
    lai_trend : xr.Dataset or str
        Pixel-wise trend dataset, or path to a NetCDF file, for
        dLST_LAI_driven. Must contain lai_trend_var.

    sen_trend : xr.Dataset or str
        Pixel-wise trend dataset, or path to a NetCDF file, for
        dLST_Sens_driven. Must contain sen_trend_var.

    out_nc : str
        Output NetCDF path for the contribution and ratio dataset.

    lai_trend_var : str, default "trend_theilsen"
        Variable name containing the LAI-driven Theil-Sen trend.

    sen_trend_var : str, default "trend_theilsen"
        Variable name containing the sensitivity-driven Theil-Sen trend.

    eps : float, default 1e-12
        Minimum absolute LAI-driven trend required to calculate the ratio.

    encoding_zlib_level : int, default 5
        NetCDF compression level.

    Returns
    -------
    xr.Dataset
        Dataset containing:
            trend_lai_driven
            trend_sens_driven
            abs_trend_lai_driven
            abs_trend_sens_driven
            contribution_lai_driven
            contribution_sens_driven
            ratio_sens_to_lai
    """
    if isinstance(lai_trend, (str, os.PathLike)):
        with xr.open_dataset(lai_trend) as source:
            lai_ds = source.load()
        lai_source = str(lai_trend)
    else:
        lai_ds = lai_trend
        lai_source = lai_ds.attrs.get("source", "in-memory dataset")

    if isinstance(sen_trend, (str, os.PathLike)):
        with xr.open_dataset(sen_trend) as source:
            sen_ds = source.load()
        sen_source = str(sen_trend)
    else:
        sen_ds = sen_trend
        sen_source = sen_ds.attrs.get("source", "in-memory dataset")

    if lai_trend_var not in lai_ds:
        raise KeyError(f"LAI-driven trend variable '{lai_trend_var}' not found.")
    if sen_trend_var not in sen_ds:
        raise KeyError(f"Sensitivity-driven trend variable '{sen_trend_var}' not found.")

    T_lai, T_sen = xr.align(lai_ds[lai_trend_var], sen_ds[sen_trend_var], join="inner")

    abs_lai = np.abs(T_lai)
    abs_sen = np.abs(T_sen)
    denom = abs_lai + abs_sen

    contribution_lai = xr.where(
        denom > 0,
        abs_lai / denom * 100.0,
        np.nan,
    ).astype(np.float32)

    contribution_sen = xr.where(
        denom > 0,
        abs_sen / denom * 100.0,
        np.nan,
    ).astype(np.float32)

    ratio = xr.where(
            np.abs(T_lai) > eps,
            T_sen / T_lai,
            np.nan,
        ).rename("ratio_sens_to_lai")

    ds_out = xr.Dataset(
        data_vars={
            "trend_lai_driven": T_lai.astype(np.float32),
            "trend_sens_driven": T_sen.astype(np.float32),
            "abs_trend_lai_driven": abs_lai.astype(np.float32),
            "abs_trend_sens_driven": abs_sen.astype(np.float32),
            "contribution_lai_driven": contribution_lai,
            "contribution_sens_driven": contribution_sen,
            "ratio_sens_to_lai": ratio,
        },
        attrs={
            "description": (
                "Relative contribution and signed ratio of sensitivity-driven "
                "and LAI-driven dLST trends."
            ),
            "lai_driven_trend_source": str(lai_source),
            "sens_driven_trend_source": str(sen_source),
            "lai_trend_variable": lai_trend_var,
            "sens_trend_variable": sen_trend_var,
            "contribution_formula_sens": (
                "abs(trend_sens_driven) / "
                "[abs(trend_sens_driven) + abs(trend_lai_driven)] * 100"
            ),
            "contribution_formula_lai": (
                "abs(trend_lai_driven) / "
                "[abs(trend_sens_driven) + abs(trend_lai_driven)] * 100"
            ),
            "ratio_formula": (
                "trend_sens_driven / trend_lai_driven where "
                "abs(trend_lai_driven) > eps"
            ),
            "eps": float(eps),
            "ratio_interpretation": (
                "Positive ratio means sensitivity-driven and LAI-driven trends "
                "have the same sign and sensitivity intensifies the LAI-driven "
                "trend. Negative ratio means opposite signs and sensitivity "
                "offsets the LAI-driven trend."
            ),
            "contribution_units": "percent",
            "trend_units": lai_ds.attrs.get("trend_units", "K yr-1"),
        },
    )

    ds_out["trend_lai_driven"].attrs = {
        "long_name": "Theil-Sen trend of LAI-driven dLST component",
        "units": lai_ds.attrs.get("trend_units", "K yr-1"),
        "source_component": "dLST_LAI_driven",
    }

    ds_out["trend_sens_driven"].attrs = {
        "long_name": "Theil-Sen trend of sensitivity-driven dLST component",
        "units": sen_ds.attrs.get("trend_units", "K yr-1"),
        "source_component": "dLST_Sens_driven",
    }

    ds_out["abs_trend_lai_driven"].attrs = {
        "long_name": "Absolute LAI-driven dLST trend",
        "units": lai_ds.attrs.get("trend_units", "K yr-1"),
    }

    ds_out["abs_trend_sens_driven"].attrs = {
        "long_name": "Absolute sensitivity-driven dLST trend",
        "units": sen_ds.attrs.get("trend_units", "K yr-1"),
    }

    ds_out["contribution_lai_driven"].attrs = {
        "long_name": "Relative contribution of LAI-driven trend magnitude",
        "units": "%",
        "valid_range": "0 to 100 where denominator is positive",
    }

    ds_out["contribution_sens_driven"].attrs = {
        "long_name": "Relative contribution of sensitivity-driven trend magnitude",
        "units": "%",
        "valid_range": "0 to 100 where denominator is positive",
    }

    ds_out["ratio_sens_to_lai"].attrs = {
        "long_name": "Signed ratio of sensitivity-driven trend to LAI-driven trend",
        "units": "1",
        "formula": (
            "trend_sens_driven / trend_lai_driven where "
            "abs(trend_lai_driven) > eps"
        ),
        "interpretation": (
            "Positive values indicate intensification; negative values indicate "
            "offsetting. Absolute value gives relative magnitude."
        ),
    }

    ds_out.attrs.update(provenance_attrs())
    enc = {
        v: {"zlib": True, "complevel": encoding_zlib_level}
        for v in ds_out.data_vars
    }
    ds_out.to_netcdf(out_nc, encoding=enc)
    log(f"Saved trend contribution and ratio nc: {out_nc}")

    return ds_out


# ============================================================
# 4. Regional time-series aggregation
# ============================================================
def regional_means_timeseries_long_to_csv(
    ds_1deg: xr.Dataset,
    out_csv: str | Path,
    *,
    time_scale: str,
    region_input=None,
    region_colname_or_names=None,
    value_vars: Sequence[str],
    cov_var: str = "coverage_1deg",
    cov_thresh: float = 0.025,
    global_name: str = "Global",
) -> pd.DataFrame:
    """
    Compute global/regional area-weighted mean time series in LONG format.

    Output columns:
        variable, group, region, time, value

    Coverage filtering is applied before regional averaging:
        value_masked = value.where(coverage >= cov_thresh)
    """
    time_scale = standardize_time_scale(time_scale)

    labels, group_names = get_time_groups(ds_1deg["time"], time_scale)

    frames = []
    for value_var in value_vars:
        V = ds_1deg[value_var]
        C = ds_1deg[cov_var]
        V_masked = V.where(C >= cov_thresh)

        for g, gname in enumerate(group_names):
            idx = labels == g
            tidx = np.where(idx)[0]
            da_g = V_masked.isel(time=tidx)

            log(f"Regional aggregation: var={value_var}, group={gname}")

            df_wide = region_weighted_mean(
                da_g,
                region_input=region_input,
                region_colname_or_names=region_colname_or_names,
                time_dim="time",
                lat_dim="lat",
                lon_dim="lon",
                global_name=global_name,
            )

            df_long = (
                df_wide.reset_index()
                .melt(id_vars="region", var_name="time", value_name="value")
            )
            df_long.insert(0, "group", gname)
            df_long.insert(0, "variable", value_var)
            frames.append(df_long)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all = df_all.sort_values(["variable", "group", "region", "time"]).reset_index(drop=True)

    df_all.to_csv(out_csv, index=False)
    log(f"Saved regional csv: {out_csv}")

    return df_all


def compute_region_trend_table(
    df: pd.DataFrame,
    out_csv: str | Path,
    value_vars: list[str],
    *,
    group: str | None = None,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Compute Theil-Sen trend, confidence interval, and p value for
    multiple variables across regions.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns:
            ['variable', 'region', 'time', 'value']

        If seasonal/monthly output is used, it can also include:
            ['group']

    value_vars : list[str]
        Variables to process, e.g.
            ['dLST_total', 'dLST_LAI_driven', 'dLST_Sens_driven']

    group : str or None
        If None:
            compute trends for all groups.
        If str:
            select one group first, e.g. 'annual', 'spring', 'summer'.

    ci : float
        Confidence level for Theil-Sen CI.

    Returns
    -------
    pd.DataFrame
        Columns:
            variable, group, region, n,
            slope, slope_ci_lo, slope_ci_hi, slope_p,
            slope_decade, slope_decade_ci_lo, slope_decade_ci_hi
    """
    dff = df.copy()

    if "variable" not in dff.columns:
        raise ValueError("Input df must include a 'variable' column.")

    dff = dff[dff["variable"].isin(value_vars)].copy()
    dff["time"] = pd.to_datetime(dff["time"])

    if "group" not in dff.columns:
        dff["group"] = "annual"

    if group is not None:
        dff = dff[dff["group"] == group].copy()

    rows = []

    for (var, grp, region), sub in dff.groupby(["variable", "group", "region"]):
        sub = sub.sort_values("time")
        sub = sub.dropna(subset=["value"])

        if len(sub) < 3:
            rows.append({
                "variable": var,
                "group": grp,
                "region": region,
                "n": len(sub),
                "slope": np.nan,
                "slope_ci_lo": np.nan,
                "slope_ci_hi": np.nan,
                "slope_p": np.nan,
                "slope_decade": np.nan,
                "slope_decade_ci_lo": np.nan,
                "slope_decade_ci_hi": np.nan,
            })
            continue

        x = time_values_to_decimal_year(sub["time"].values)
        y = sub["value"].values

        slope, ci_lo, ci_hi, pval = theil_sen_ci_p(x, y, ci=ci)

        rows.append({
            "variable": var,
            "group": grp,
            "region": region,
            "n": len(sub),
            "slope": slope,
            "slope_ci_lo": ci_lo,
            "slope_ci_hi": ci_hi,
            "slope_p": pval,
            "slope_decade": slope * 10,
            "slope_decade_ci_lo": ci_lo * 10,
            "slope_decade_ci_hi": ci_hi * 10,
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["variable", "group", "region"]).reset_index(drop=True)
    
    df_out.to_csv(out_csv, index=False)
    log(f"Saved regional csv: {out_csv}")

    return df_out


# ============================================================
# 5. One-stop pipeline
# ============================================================
def run_dlst_decomposition_consistent_mask_pipeline(
    *,
    base_dir: str | Path,
    lai_path: str | Path,
    sens_path: str | Path,
    clim_zone_path: str | Path,
    country_shp_path: str | Path,
    out_dir: str | Path,
    time_scale: str = "Annual",
    lai_var: str = "LAI",
    sens_var: str = "sens",
    start_year: int = 2001,
    end_year: int = 2024,
    min_count: int = 10,
    cov_thresh: float = 0.025,
    cumsum_skipna: bool = False,
    clim_names: tuple[str, ...] = ("Tropical", "Arid", "Temperate", "Boreal"),
    country_name_col: str = "NAME",
    prefix: str = "dLST",
    components: tuple[str, ...] = (
        "dLST_total",
        "dLST_LAI_driven",
        "dLST_Sens_driven",
    ),
):
    """
    Run the full dLST decomposition workflow.

    Workflow
    --------
    1. Read 0.05° LAI and LST-LAI sensitivity datasets.
    2. Aggregate LAI and sensitivity from 0.05° to 1° using the same valid
       fine-pixel mask:
           valid = finite(LAI) & finite(sensitivity)
    3. Aggregate the 1° LAI/sensitivity dataset to 4° using a common 1° mask.
       In the current setup, 4° cells are retained when at least 8 valid 1°
       cells are available.
    4. Compute cumulative dLST decomposition at both 1° and 4°:
           dLST_total = cumsum[dLAI(t) * mean(S(t), S(t-1))]
           dLST_LAI_driven = cumsum[dLAI(t) * S_ref]
           dLST_Sens_driven = dLST_total - dLST_LAI_driven
       where dLAI(t) is the change from the preceding time step within the
       annual, seasonal, or monthly sequence and S_ref is the climatological
       sensitivity by default.
    5. Compute pixel-wise Theil-Sen trends for each requested component.
    6. Aggregate 1° component time series to global, climate-zone, and country
       scales, then compute regional Theil-Sen trend tables.

    Parameters
    ----------
    base_dir : str or pathlib.Path
        Base directory against which relative input and output paths are
        resolved. The function does not change the process working directory.

    lai_path : str
        Path to the input 0.05° LAI NetCDF file. Can be absolute or relative to
        base_dir. The file must contain the variable named by lai_var and have
        dimensions compatible with time, lat, and lon.

    sens_path : str
        Path to the input 0.05° LST-LAI sensitivity NetCDF file. Can be absolute
        or relative to base_dir. The file must contain the variable named by
        sens_var and have dimensions compatible with time, lat, and lon.

    clim_zone_path : str
        Path to the climate-zone mask file, usually a 1° DataArray. This is used
        for regional aggregation by climate zone.

    country_shp_path : str
        Path to the country shapefile used for country-level regional
        aggregation. Antarctica is removed if a CONTINENT or continent column is
        available.

    out_dir : str
        Output directory for all NetCDF and CSV products. Can be absolute or
        relative to base_dir. The directory is created if it does not exist.

    time_scale : str, default "Annual"
        Temporal scale of the input data and decomposition. Must be one of:
        "Annual", "Season", or "Month".
        For "Season", timestamps are expected to use representative months
        {3, 6, 9, 12}, corresponding to spring, summer, autumn, and winter.

    lai_var : str, default "LAI"
        Variable name of LAI in the input LAI NetCDF file.

    sens_var : str, default "sens"
        Variable name of LST-LAI sensitivity in the input sensitivity NetCDF
        file.

    start_year : int, default 2001
        First year included in the analysis period.

    end_year : int, default 2024
        Last year included in the analysis period.

    min_count : int, default 10
        Minimum number of valid 0.05° pixels required within each 1° grid cell
        when aggregating LAI and sensitivity from 0.05° to 1°.
        For a typical 1° cell containing 20 x 20 = 400 fine pixels,
        min_count=10 corresponds to about 0.025 coverage.

    cov_thresh : float, default 0.025
        Coverage threshold used for 1° pixel-wise trends and 1° regional time
        series aggregation. Values with coverage below this threshold are
        excluded before trend or regional averaging.
        In the current 4° trend workflow, cov_thresh is set to 0 because the
        4° validity criterion is controlled by the 1° -> 4° min_count threshold.

    clim_names : tuple of str, default ("Tropical", "Arid", "Temperate", "Boreal")
        Names assigned to climate-zone classes during climate-zone regional
        aggregation. The order must match the coding/order expected by
        region_weighted_mean() for the supplied climate-zone mask.

    country_name_col : str, default "NAME"
        Column name in the country shapefile that contains country names.

    prefix : str, default "dLST"
        Prefix used in all output file names.

    components : tuple of str
        dLST decomposition variables for which pixel-wise trends, regional time
        series, and regional trend tables are produced. By default:
            dLST_total
            dLST_LAI_driven
            dLST_Sens_driven

    Returns
    -------
    dict
        Dictionary containing selected in-memory datasets/dataframes and output
        paths. Main entries include:
            ds_1deg
                Aggregated 1° LAI/sensitivity dataset.
            ds_decomp
                1° dLST decomposition dataset.
            df_global, df_clim, df_country
                Regional time-series tables.
            df_global_trend, df_clim_trend, df_country_trend
                Regional trend tables.
            paths
                Dictionary of generated NetCDF and CSV output paths.
    """
    
    base_dir = Path(base_dir)

    def resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else base_dir / path

    lai_path = resolve(lai_path)
    sens_path = resolve(sens_path)
    clim_zone_path = resolve(clim_zone_path)
    country_shp_path = resolve(country_shp_path)
    out_dir = resolve(out_dir)
    ensure_dir(out_dir)

    ts = standardize_time_scale(time_scale)
    tag = f"{prefix}_{time_scale}_{start_year}_{end_year}"

    path_agg_1deg = out_dir / f"{tag}_LAI_S_1deg_consistent_mask.nc"
    path_agg_4deg = out_dir / f"{tag}_LAI_S_4deg_consistent_mask.nc"
    path_decomp_1deg = out_dir / f"{tag}_Decomp_1deg_consistent_mask.nc"
    path_decomp_4deg = out_dir / f"{tag}_Decomp_4deg_consistent_mask.nc"
    path_contribution_1deg = (
        out_dir / f"{tag}_TrendContributionRatio_1deg.nc"
    )
    path_contribution_4deg = (
        out_dir / f"{tag}_TrendContributionRatio_4deg.nc"
    )
    path_global_csv = out_dir / f"{tag}_Global_TS.csv"
    path_clim_csv = out_dir / f"{tag}_ClimateZone_TS.csv"
    path_country_csv = out_dir / f"{tag}_Country_TS.csv"
    path_global_trend_csv = out_dir / f"{tag}_Global_Trend.csv"
    path_clim_trend_csv = out_dir / f"{tag}_ClimateZone_Trend.csv"
    path_country_trend_csv = out_dir / f"{tag}_Country_Trend.csv"

    # Step 1. Aggregate LAI and sensitivity to 1° consistently.
    log("Step 1: aggregate LAI and sensitivity to 1° using common mask")
    with xr.open_dataset(lai_path) as lai_ds, xr.open_dataset(sens_path) as sens_ds:
        lai = prep_3d(lai_ds[lai_var], start_year, end_year)
        sens = prep_3d(sens_ds[sens_var], start_year, end_year)
        ds_1deg = aggregate_LAI_S_to_1deg_consistent_mask(
            lai,
            sens,
            min_count=min_count,
        )

    ds_1deg.attrs.update({
        **provenance_attrs(),
        "title": (
            "1 degree LAI and LST-LAI sensitivity using a consistent "
            "0.05 degree mask"
        ),
        "analysis_period": f"{start_year}-{end_year}",
        "time_scale": ts,
        "lai_source": str(lai_path),
        "sensitivity_source": str(sens_path),
    })

    enc = {v: {"zlib": True, "complevel": 5} for v in ds_1deg.data_vars}
    ds_1deg.to_netcdf(path_agg_1deg, encoding=enc)
    log(f"Saved 1° LAI/S dataset: {path_agg_1deg}")

    # --------------------------------------------------------
    # Step 2. Aggregate LAI and sensitivity to 4°.
    # --------------------------------------------------------
    log("Step 1.2: aggregate LAI and sensitivity to 4°")

    ds_4deg = aggregate_1deg_to_coarse(
        ds_1deg,
        resolution=4,
        value_vars=("LAI", "sens", "coverage"),
        common_valid_vars=("LAI", "sens"),
        min_count=8,
    )

    ds_4deg.attrs.update({
        **provenance_attrs(),
        "title": (
            "4 degree LAI and LST-LAI sensitivity aggregated from "
            "1 degree data"
        ),
        "analysis_period": f"{start_year}-{end_year}",
        "time_scale": ts,
        "lai_source": str(lai_path),
        "sensitivity_source": str(sens_path),
        "source_1deg_dataset": str(path_agg_1deg),
        "note": (
            "4 degree cells are retained when at least 8 valid 1 degree "
            "cells are available."
        ),
    })

    enc = {v: {"zlib": True, "complevel": 5} for v in ds_4deg.data_vars}
    ds_4deg.to_netcdf(path_agg_4deg, encoding=enc)
    log(f"Saved 4° LAI/S dataset: {path_agg_4deg}")
    
    # --------------------------------------------------------
    # Step 2.1. Decompose LAI-induced LST effect at 1°.
    # --------------------------------------------------------
    log("Step 2.1: compute 1° dLST decomposition")
    ds_decomp_1deg = compute_dLST_decomposition(
        ds_1deg,
        time_scale=ts,
        lai_var="LAI",
        sen_var="sens",
        cov_var="coverage",
        cumsum_skipna=cumsum_skipna,
    )

    ds_decomp_1deg.attrs.update({
        **provenance_attrs(),
        "source_1deg_dataset": str(path_agg_1deg),
        "analysis_period": f"{start_year}-{end_year}",
    })

    enc = {v: {"zlib": True, "complevel": 5} for v in ds_decomp_1deg.data_vars}
    ds_decomp_1deg.to_netcdf(path_decomp_1deg, encoding=enc)
    log(f"Saved dLST decomposition dataset: {path_decomp_1deg}")

    # --------------------------------------------------------
    # Step 2.2. Decompose LAI-induced LST effect at 4°.
    # --------------------------------------------------------
    log("Step 2.2: compute 4° dLST decomposition")
    
    ds_decomp_4deg = compute_dLST_decomposition(
        ds_4deg,
        time_scale=ts,
        lai_var="LAI",
        sen_var="sens",
        cov_var="coverage",
        cumsum_skipna=cumsum_skipna,
    )

    ds_decomp_4deg.attrs.update({
        **provenance_attrs(),
        "source_4deg_dataset": str(path_agg_4deg),
        "analysis_period": f"{start_year}-{end_year}",
    })

    enc = {v: {"zlib": True, "complevel": 5} for v in ds_decomp_4deg.data_vars}
    ds_decomp_4deg.to_netcdf(path_decomp_4deg, encoding=enc)
    log(f"Saved 4° dLST decomposition dataset: {path_decomp_4deg}")

    # --------------------------------------------------------
    # Step 3.1. Pixel-wise trends at 1° for each component.
    # --------------------------------------------------------
    trend_paths_1deg = {}

    for comp in components:
        out_nc = out_dir / f"{tag}_Trend_1deg_{comp}.nc"
        log(f"Step 3.1: compute 1° pixel-wise trend for {comp}")

        compute_pixel_trend_adaptive(
            ds_decomp_1deg,
            out_nc,
            time_scale=ts,
            value_var=comp,
            cov_var="coverage",
            cov_thresh=cov_thresh,
            ci=0.95,
            encoding_zlib_level=5,
        )

        trend_paths_1deg[comp] = out_nc

    # --------------------------------------------------------
    # Step 3.2. Pixel-wise trends at 4° for each component.
    # --------------------------------------------------------
    trend_paths_4deg = {}

    for comp in components:
        out_nc = out_dir / f"{tag}_Trend_4deg_{comp}.nc"
        log(f"Step 3.2: compute 4° pixel-wise trend for {comp}")

        compute_pixel_trend_adaptive(
            ds_decomp_4deg,
            out_nc,
            time_scale=ts,
            value_var=comp,
            cov_var="coverage",
            cov_thresh=0,
            ci=0.95,
            encoding_zlib_level=5,
        )

        trend_paths_4deg[comp] = out_nc

    
    # --------------------------------------------------------
    # Step 3.3. Contribution and offset/intensification ratio.
    # --------------------------------------------------------
    log("Step 3.3: compute 1° trend contribution and ratio")

    ds_contribution_1deg = compute_trend_contribution_and_ratio(
        trend_paths_1deg["dLST_LAI_driven"],
        trend_paths_1deg["dLST_Sens_driven"],
        path_contribution_1deg,
        eps=1e-12,
        encoding_zlib_level=5,
    )

    log("Step 3.4: compute 4° trend contribution and ratio")

    ds_contribution_4deg = compute_trend_contribution_and_ratio(
        trend_paths_4deg["dLST_LAI_driven"],
        trend_paths_4deg["dLST_Sens_driven"],
        path_contribution_4deg,
        eps=1e-12,
        encoding_zlib_level=5,
    )
    # --------------------------------------------------------
    # Step 4. Regional time-series aggregation.
    # --------------------------------------------------------
    log("Step 4: aggregate component time series to global, climate zones, and countries")
    with xr.open_dataarray(clim_zone_path) as source:
        clim_zone = source.load()
    countries = gpd.read_file(country_shp_path)

    for col in ["CONTINENT", "continent"]:
        if col in countries.columns:
            countries = countries[countries[col].astype(str).str.lower() != "antarctica"]
            break
    
    df_global = regional_means_timeseries_long_to_csv(
        ds_decomp_1deg,
        path_global_csv,
        time_scale=ts,
        region_input=None,
        region_colname_or_names=None,
        value_vars=list(components),
        cov_var="coverage",
        cov_thresh=cov_thresh,
        global_name="Global",
    )

    df_clim = regional_means_timeseries_long_to_csv(
        ds_decomp_1deg,
        path_clim_csv,
        time_scale=ts,
        region_input=clim_zone,
        region_colname_or_names=list(clim_names),
        value_vars=list(components),
        cov_var="coverage",
        cov_thresh=cov_thresh,
        global_name="Global",
    )

    df_country = regional_means_timeseries_long_to_csv(
        ds_decomp_1deg,
        path_country_csv,
        time_scale=ts,
        region_input=countries,
        region_colname_or_names=country_name_col,
        value_vars=list(components),
        cov_var="coverage",
        cov_thresh=cov_thresh,
        global_name="Global",
    )

    df_global_trend = compute_region_trend_table(
        df_global,
        path_global_trend_csv,
        value_vars=list(components),
    )
    df_clim_trend = compute_region_trend_table(
        df_clim,
        path_clim_trend_csv,
        value_vars=list(components),
    )
    df_country_trend = compute_region_trend_table(
        df_country,
        path_country_trend_csv,
        value_vars=list(components),
    )


    log("Pipeline finished.")

    return {
        "ds_1deg": ds_1deg,
        "ds_decomp": ds_decomp_1deg,
        "df_global": df_global,
        "df_clim": df_clim,
        "df_country": df_country,
        "df_global_trend": df_global_trend,
        "df_clim_trend": df_clim_trend,
        "df_country_trend": df_country_trend,
        "ds_contribution_1deg": ds_contribution_1deg,
        "ds_contribution_4deg": ds_contribution_4deg,
        "paths": {
            "lai_sens_1deg": path_agg_1deg,
            "lai_sens_4deg": path_agg_4deg,
            "decomp_1deg": path_decomp_1deg,
            "decomp_4deg": path_decomp_4deg,
            "global_csv": path_global_csv,
            "climate_zone_csv": path_clim_csv,
            "country_csv": path_country_csv,
            "global_trend_csv": path_global_trend_csv,
            "climate_zone_trend_csv": path_clim_trend_csv,
            "country_trend_csv": path_country_trend_csv,
            "trend_paths_1deg": trend_paths_1deg,
            "trend_paths_4deg": trend_paths_4deg,
            "contribution_ratio_1deg": path_contribution_1deg,
            "contribution_ratio_4deg": path_contribution_4deg,
        },
    }


# ============================================================
# 6. Main execution
# ============================================================
def main() -> None:
    """Run the manuscript decomposition for annual and seasonal inputs."""
    for time_scale in TIME_SCALES:
        suffix = "_NSharmonized" if time_scale == "Season" else ""
        lai_path = (
            f"LAI/LAI_{LAI_PRODUCT}_{time_scale}_0.05d_"
            f"{START_YEAR}_{END_YEAR}{suffix}.nc"
        )
        sens_path = (
            f"processed/Sensitivity_20260208/{LAI_PRODUCT}/"
            f"Sensitivity_{time_scale}_LST{LST_VARIABLE}_LAI_"
            f"{LAI_PRODUCT}_0p05{suffix}.nc"
        )
        out_dir = (
            "processed/dLST_Decompose_20260223/ConsistentMask_20260603/"
            f"{LAI_PRODUCT}_{time_scale}_LST{LST_VARIABLE}"
        )

        results = run_dlst_decomposition_consistent_mask_pipeline(
            base_dir=DATA_ROOT,
            lai_path=lai_path,
            sens_path=sens_path,
            clim_zone_path="koppen_geiger_4class_1d.nc",
            country_shp_path="shapefile/countries.shp",
            out_dir=out_dir,
            time_scale=time_scale,
            lai_var="LAI",
            sens_var="sens",
            start_year=START_YEAR,
            end_year=END_YEAR,
            min_count=10,
            cov_thresh=0.025,
            cumsum_skipna=True,
            clim_names=CLIMATE_NAMES,
            country_name_col="NAME",
            prefix=f"dLST{LST_VARIABLE}_{LAI_PRODUCT}",
        )

        log("Generated output files:")
        for name, path in results["paths"].items():
            log(f"{name}: {path}")


if __name__ == "__main__":
    main()
