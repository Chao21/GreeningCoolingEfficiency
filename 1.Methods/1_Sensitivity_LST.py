#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Calculate global gridded LST sensitivity to LAI using space-for-time substitution.

Author
------
Chao Zhang, National University of Singapore

Date
----
2026-09-05

Purpose
-------
Estimate the local Theil-Sen slope between LST and LAI after controlling for
land-cover class, land-cover fraction, and elevation. An adaptive spatial
window supplies neighboring samples that satisfy the QC-1 criteria.

Notes
-----
The calculation uses Numba acceleration and longitude-block multiprocessing
with halos. It returns the slope, valid-neighbor count, and selected window
size; it does not estimate slope uncertainty. Input data are expected on the
same 0.05-degree global grid. Set ``VEG_LST_DATA_DIR`` to override the default
data directory.
"""

import math
import os
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import xarray as xr
from numba import njit


DATA_ROOT = Path(os.environ.get(
    "VEG_LST_DATA_DIR",
    "/home/energy/chaoz/project/05Veg_LST/data",
))

# ---------------------------
# 0) Config
# ---------------------------

@dataclass(frozen=True)
class SensitivityConfig:
    n_min: int = 10                 # minimum number of valid neighbor pixels
    lai_min: float = 0              # minimum LAI to consider pixel as vegetated
    lai_diff_min: float = 0.05      # minimum |ΔLAI| between target and neighbor
    dem_diff_max: float = 100.0     # maximum elevation difference (m)
    lc_frac_diff_max: float = 10.0  # maximum land-cover fraction difference (%)
    valid_lc: tuple[int, ...] = (1, 2, 3, 4)
    verbose: bool = False


CFG = SensitivityConfig()


def cfg_to_attrs(cfg: SensitivityConfig) -> dict:
    """Write configuration into NetCDF attributes."""
    out = {}
    for k, v in cfg.__dict__.items():
        out[f"cfg_{k}"] = int(v) if isinstance(v, bool) else v
    return out


# ---------------------------
# 1) Utilities
# ---------------------------

def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _open_decode_time(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=True)
    if "time" not in ds.coords:
        raise ValueError(f"No 'time' coordinate found in {path}. dims={ds.dims}")
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        raise ValueError(f"'time' is not datetime64 in {path}. dtype={ds['time'].dtype}")
    return ds


def normalize_latlon(da: xr.DataArray, decimals: int = 2) -> xr.DataArray:
    """Round latitude and longitude coordinates."""
    return da.assign_coords(
        lat=np.round(da.lat.values, decimals),
        lon=np.round(da.lon.values, decimals),
    )


def validate_grid(reference: xr.DataArray, **arrays: xr.DataArray) -> None:
    """Require all inputs to use the reference latitude-longitude grid."""
    for name, da in arrays.items():
        for coord in ("lat", "lon"):
            if not da[coord].equals(reference[coord]):
                raise ValueError(f"{name} has a different {coord} coordinate")


# ---------------------------
# 2) Block split / merge (lon-wise, with halo)
# ---------------------------

def split_lon_with_halo(arr, block_number=12, halo=20):
    """
    Split array along lon with halo padding.

    arr: (..., lon)
    returns list of (subarr, core0, core1, g0, g1)
    """
    X = arr.shape[-1]
    out = []
    for i in range(block_number):
        core0 = int(i * X / block_number)
        core1 = int((i + 1) * X / block_number)

        g0 = max(core0 - halo, 0)
        g1 = min(core1 + halo, X)

        sub = arr[..., g0:g1]
        out.append((sub, core0, core1, g0, g1))
    return out


def merge_lon_from_blocks(block_results, X, fill_value=np.nan):
    """
    Merge block results back along lon.

    block_results: list of (subres, core0, core1, g0, g1)
    subres shape: (..., g1-g0)
    """
    merged = None
    for (subres, core0, core1, g0, g1) in block_results:
        if merged is None:
            merged = np.full(subres.shape[:-1] + (X,), fill_value, dtype=subres.dtype)

        l0 = core0 - g0
        l1 = l0 + (core1 - core0)
        merged[..., core0:core1] = subres[..., l0:l1]

    return merged


# ---------------------------
# 3) Numba-friendly offsets
# ---------------------------

def precompute_offsets_numba(lat_res, lon_res, init_win=0.5, max_win=1.0, step=0.1):
    wins = np.arange(init_win, max_win + 1e-9, step).astype(np.float32)

    offsets_list = []
    counts = []
    for w in wins:
        di = int(round((w / 2) / lat_res))
        dj = int(round((w / 2) / lon_res))

        ii, jj = np.meshgrid(
            np.arange(-di, di + 1),
            np.arange(-dj, dj + 1),
            indexing="ij",
        )
        off = np.stack([ii.ravel(), jj.ravel()], axis=1)
        off = off[~((off[:, 0] == 0) & (off[:, 1] == 0))]
        offsets_list.append(off.astype(np.int16))
        counts.append(off.shape[0])

    max_n = max(counts)
    nwin = len(wins)

    offsets_arr = np.zeros((nwin, max_n, 2), dtype=np.int16)
    counts_arr = np.zeros(nwin, dtype=np.int16)

    for k, off in enumerate(offsets_list):
        offsets_arr[k, :off.shape[0], :] = off
        counts_arr[k] = off.shape[0]

    return wins, offsets_arr, counts_arr

# ---------------------------
# 4) Numba kernel: Theil-Sen slope
# ---------------------------

@njit(cache=True)
def theil_sen_slope(x, y, n, min_dx=1e-2):
    """
    Return:
      slope (median pairwise slopes)
    """
    m = n * (n - 1) // 2
    if m <= 0:
        return math.nan

    slopes = np.empty(m, dtype=np.float32)
    idx = 0

    for i in range(n):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = x[j] - xi
            if abs(dx) >= min_dx:
                slopes[idx] = (y[j] - yi) / dx
                idx += 1

    if idx < 5:
        return math.nan

    slopes = slopes[:idx]
    slopes.sort()

    mid = idx // 2
    return slopes[mid] if idx % 2 == 1 else 0.5 * (slopes[mid - 1] + slopes[mid])


@njit(cache=True)
def pixel_sensitivity_numba(
    lst, lai, dem, lc_type, lc_frac,
    i, j,
    wins, offsets_arr, counts_arr,
    n_min, lai_min, lai_diff_min, dem_diff_max, lc_frac_diff_max,
):
    nlat, nlon = lst.shape

    lst0 = lst[i, j]
    lai0 = lai[i, j]
    dem0 = dem[i, j]
    lct0 = lc_type[i, j]
    lcf0 = lc_frac[i, j]

    if (not math.isfinite(lst0)) or (not math.isfinite(lai0)) or (lai0 < lai_min):
        return math.nan, 0, math.nan
    if (not math.isfinite(dem0)) or (not math.isfinite(lct0)) or (not math.isfinite(lcf0)):
        return math.nan, 0, math.nan

    for k in range(len(wins)):
        cnt = counts_arr[k]
        offs = offsets_arr[k]

        d_lai = np.empty(cnt, dtype=np.float32)
        d_lst = np.empty(cnt, dtype=np.float32)
        n_valid = 0

        for p in range(cnt):
            ii = i + offs[p, 0]
            jj = j + offs[p, 1]
            if ii < 0 or ii >= nlat or jj < 0 or jj >= nlon:
                continue

            nb_lst = lst[ii, jj]
            nb_lai = lai[ii, jj]
            if (not math.isfinite(nb_lst)) or (not math.isfinite(nb_lai)) or (nb_lai < lai_min):
                continue

            nb_dem = dem[ii, jj]
            nb_lct = lc_type[ii, jj]
            nb_lcf = lc_frac[ii, jj]
            if (
                not math.isfinite(nb_dem)
                or not math.isfinite(nb_lct)
                or not math.isfinite(nb_lcf)
            ):
                continue

            if nb_lct != lct0:
                continue
            if abs(nb_dem - dem0) > dem_diff_max:
                continue
            if abs(nb_lcf - lcf0) > lc_frac_diff_max:
                continue

            dlai = nb_lai - lai0
            if abs(dlai) < lai_diff_min:
                continue

            d_lai[n_valid] = dlai
            d_lst[n_valid] = nb_lst - lst0
            n_valid += 1

        if n_valid >= n_min:
            slope = theil_sen_slope(
                d_lai, d_lst, n_valid, min_dx=0.01
            )
            return slope, n_valid, wins[k]

    return math.nan, 0, math.nan


# ---------------------------
# 5) QC-0 target mask (outside Numba)
# ---------------------------

def build_target_mask(
    lst, lai, dem, lc_type, lc_frac,
    valid_lc=(1, 2, 3, 4), lai_min=0.05,
):
    mask = (
        np.isfinite(lst) &
        np.isfinite(lai) &
        np.isfinite(dem) &
        np.isfinite(lc_type) &
        np.isfinite(lc_frac)
    )
    mask &= np.isin(lc_type, valid_lc)
    mask &= (lai >= lai_min)
    return mask


# ---------------------------
# 6) One time-step (Numba, single block)
# ---------------------------

def compute_sensitivity_one_time_numba(
    lst_2d, lai_2d, lc_type_2d, lc_frac_2d, dem_2d,
    wins, offsets_arr, counts_arr,
    cfg: SensitivityConfig,
):
    nlat, nlon = lst_2d.shape

    sens = np.full((nlat, nlon), np.nan, dtype=np.float32)
    n_valid = np.zeros((nlat, nlon), dtype=np.int16)
    win = np.full((nlat, nlon), np.nan, dtype=np.float32)

    tmask = build_target_mask(
        lst_2d, lai_2d, dem_2d, lc_type_2d, lc_frac_2d,
        valid_lc=cfg.valid_lc, lai_min=cfg.lai_min
    )
    idx = np.argwhere(tmask)

    if cfg.verbose:
        print(f"  target pixels: {idx.shape[0]} / {nlat*nlon}")

    lstA = np.asarray(lst_2d, dtype=np.float32)
    laiA = np.asarray(lai_2d, dtype=np.float32)
    demA = np.asarray(dem_2d, dtype=np.float32)

    # Avoid RuntimeWarning: invalid value encountered in cast
    # We force NaNs to -1 (invalid class) before int cast.
    lctA = np.asarray(np.nan_to_num(lc_type_2d, nan=-1.0), dtype=np.int16)
    lcfA = np.asarray(lc_frac_2d, dtype=np.float32)

    for i, j in idx:
        s, n_v, w = pixel_sensitivity_numba(
            lstA, laiA, demA, lctA, lcfA,
            int(i), int(j),
            wins, offsets_arr, counts_arr,
            cfg.n_min, cfg.lai_min, cfg.lai_diff_min, cfg.dem_diff_max, cfg.lc_frac_diff_max
        )
        if math.isfinite(s):
            sens[i, j] = s
            n_valid[i, j] = np.int16(n_v)
            win[i, j] = w

    return sens, n_valid, win


# ---------------------------
# 7) Multiprocessing over lon-blocks for one time step
# ---------------------------

def _run_one_block(args):
    (
        lst_b, lai_b, lct_b, lcf_b, dem_b,
        core0, core1, g0, g1,
        wins, offsets_arr, counts_arr, cfg
    ) = args

    sens, n_valid, win = compute_sensitivity_one_time_numba(
        lst_b, lai_b, lct_b, lcf_b, dem_b,
        wins, offsets_arr, counts_arr,
        cfg=cfg
    )

    return sens, n_valid, win, core0, core1, g0, g1


def compute_sensitivity_one_time_numba_mp(
    pool,
    lst_2d, lai_2d, lc_type_2d, lc_frac_2d, dem_2d,
    wins, offsets_arr, counts_arr,
    cfg: SensitivityConfig,
    block_number=12,
    halo=20,
):
    X = lst_2d.shape[-1]

    blocks_lst = split_lon_with_halo(lst_2d, block_number, halo)
    blocks_lai = split_lon_with_halo(lai_2d, block_number, halo)
    blocks_lct = split_lon_with_halo(lc_type_2d, block_number, halo)
    blocks_lcf = split_lon_with_halo(lc_frac_2d, block_number, halo)
    blocks_dem = split_lon_with_halo(dem_2d, block_number, halo)

    tasks = []
    for b in range(block_number):
        lst_b, core0, core1, g0, g1 = blocks_lst[b]
        tasks.append((
            lst_b,
            blocks_lai[b][0],
            blocks_lct[b][0],
            blocks_lcf[b][0],
            blocks_dem[b][0],
            core0, core1, g0, g1,
            wins, offsets_arr, counts_arr, cfg
        ))

    results = pool.map(_run_one_block, tasks)

    def collect(idx):
        return [(r[idx], r[3], r[4], r[5], r[6]) for r in results]

    sens = merge_lon_from_blocks(collect(0), X)
    n_valid = merge_lon_from_blocks(collect(1), X, fill_value=0)
    win = merge_lon_from_blocks(collect(2), X)

    return sens, n_valid, win


def _select_lc_for_year(da: xr.DataArray, year: int) -> xr.DataArray:
    """
    Landcover is typically annual. We select all timesteps in the same year,
    and take the first one (robust even if it is stored monthly by mistake).
    """
    t = da["time"]
    sub = da.sel(time=(t.dt.year == year))
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"No LC data found for year={year} in variable '{da.name}'.")
    return sub.isel(time=0)


# ---------------------------
# 8) Main run
# ---------------------------

def main_run(
    lst_path, lai_path, lc_type_path, lc_frac_path, dem_path, output_path,
    time_scale, start_year=2001, end_year=2020,
) -> None:
    """Calculate LST sensitivity to LAI for one product and time scale."""
    log("Loading data")
    period = slice(f"{start_year}-01-01", f"{end_year}-12-31")
    lst = _open_decode_time(lst_path)["Ts"].sel(time=period)
    lai = _open_decode_time(lai_path)["LAI"].sel(time=period)
    lc_type = _open_decode_time(lc_type_path)["LC_type_broad"].sel(
        time=period
    )
    lc_frac = _open_decode_time(lc_frac_path)["LC_frac_broad"].sel(
        time=period
    )
    dem = xr.open_dataset(dem_path)["elevation"]

    lst = normalize_latlon(lst).transpose("time", "lat", "lon")
    lai = normalize_latlon(lai).transpose("time", "lat", "lon")
    lc_type = normalize_latlon(lc_type).transpose("time", "lat", "lon")
    lc_frac = normalize_latlon(lc_frac).transpose("time", "lat", "lon")
    dem = normalize_latlon(dem).transpose("lat", "lon")
    lst, lai = xr.align(lst, lai, join="exact")
    validate_grid(lst, lc_type=lc_type, lc_frac=lc_frac, dem=dem)

    wins, offsets_arr, counts_arr = precompute_offsets_numba(
        lat_res=0.05, lon_res=0.05,
        init_win=0.5, max_win=1.0, step=0.1,
    )

    # Populate the Numba cache before spawning worker processes.
    pixel_sensitivity_numba(
        np.zeros((2, 2), np.float32),
        np.ones((2, 2), np.float32),
        np.zeros((2, 2), np.float32),
        np.ones((2, 2), np.int16),
        np.ones((2, 2), np.float32),
        0, 0, wins, offsets_arr, counts_arr,
        2, 0.05, 0.05, 100.0, 10.0,
    )

    times = lst.time.values
    nt, nlat, nlon = lst.shape
    sens = np.full((nt, nlat, nlon), np.nan, np.float32)
    n_valid = np.zeros((nt, nlat, nlon), np.int16)
    win_out = np.full((nt, nlat, nlon), np.nan, np.float32)

    block_number, halo, nproc = 12, 20, 12
    years = np.unique(
        xr.DataArray(times, dims="time").dt.year.values
    ).astype(int)
    lc_cache = {
        year: (
            _select_lc_for_year(lc_type, year).values,
            _select_lc_for_year(lc_frac, year).values,
        )
        for year in years
    }

    ctx = get_context("spawn")
    with ctx.Pool(processes=nproc) as pool:
        for ti, timestamp in enumerate(times):
            year = int(str(np.datetime64(timestamp, "Y")))
            lct_t, lcf_t = lc_cache[year]
            date = np.datetime_as_string(timestamp, unit="D")
            log(f"[{ti + 1}/{nt}] {date}")

            result = compute_sensitivity_one_time_numba_mp(
                pool,
                lst.sel(time=timestamp).values,
                lai.sel(time=timestamp).values,
                lct_t, lcf_t, dem.values,
                wins, offsets_arr, counts_arr,
                cfg=CFG, block_number=block_number, halo=halo,
            )
            sens[ti], n_valid[ti], win_out[ti] = result

    log("Writing results")
    output = xr.Dataset(
        data_vars={
            "sens": (
                ("time", "lat", "lon"), sens,
                {
                    "long_name": (
                        f"{time_scale} sensitivity of LST to LAI "
                        "(Theil-Sen slope)"
                    ),
                    "units": "K per unit LAI",
                },
            ),
            "n_valid": (
                ("time", "lat", "lon"), n_valid,
                {
                    "long_name": "Number of valid neighboring pixels",
                    "units": "count",
                },
            ),
            "win_deg": (
                ("time", "lat", "lon"), win_out,
                {
                    "long_name": "Selected adaptive-window width",
                    "units": "degree",
                },
            ),
        },
        coords={"time": times, "lat": lst.lat, "lon": lst.lon},
        attrs={
            "method": (
                "Space-for-time Theil-Sen regression with QC-1, adaptive "
                "windows, and longitude-block multiprocessing with halos"
            ),
            "time_scale": time_scale,
            "lc_handling": (
                "The first land-cover snapshot in each year is applied to "
                "all observations in that year"
            ),
            "init_win_deg": float(wins[0]),
            "max_win_deg": float(wins[-1]),
            "block_number": block_number,
            "halo_pixels": halo,
            "mp_start_method": "spawn",
            **cfg_to_attrs(CFG),
        },
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 5}
        for name in output.data_vars
    }
    output.to_netcdf(output_path, encoding=encoding)
    log(f"Saved: {output_path}")


if __name__ == "__main__":
    os.chdir(DATA_ROOT)
    log("Starting sensitivity calculation")

    lc_type_path = "Land_Cover/LC_MCD12C1_type_broad_2001_2024.nc"
    lc_frac_path = "Land_Cover/LC_MCD12C1_frac_broad_2001_2024.nc"
    dem_path = "DEM_0.05d_180.nc"
    time_scales = ["Annual"]
    lst_variables = ["dailymean"]
    lai_products = ["GLASS"]
    end_years = {"GLASS": 2024, "GlobMapv3": 2023, "GIMMS4g": 2020}

    for time_scale in time_scales:
        suffix = "_NSharmonized" if time_scale == "Season" else ""
        for lst_variable in lst_variables:
            for lai_product in lai_products:
                end_year = end_years[lai_product]
                lai_path = (
                    f"LAI/LAI_{lai_product}_{time_scale}_0.05d_"
                    f"2001_{end_year}{suffix}.nc"
                )
                lst_path = (
                    f"LST_MOD11C3/0.05d/LST_MOD11C3_{time_scale}_"
                    f"{lst_variable}_0.05d_2001_2025{suffix}.nc"
                )
                output_path = (
                    f"processed/Sensitivity_20260208/{lai_product}/"
                    f"MOD11C3/LAI_diffmin{CFG.lai_diff_min}/"
                    f"Sensitivity_{time_scale}_LST{lst_variable}_"
                    f"LAI_{lai_product}_0p05{suffix}.nc"
                )
                main_run(
                    lst_path, lai_path, lc_type_path, lc_frac_path,
                    dem_path, output_path, time_scale,
                    start_year=2001, end_year=end_year,
                )

    log("Sensitivity calculation complete")
