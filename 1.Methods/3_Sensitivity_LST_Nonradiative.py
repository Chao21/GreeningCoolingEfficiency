#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Attribute temporal changes in LST sensitivity to environmental predictors.

Author
------
Chao Zhang, National University of Singapore

Date
----
2026-09-05

Purpose
-------
Apply standardized ridge regression separately in snow-affected and snow-free
regions to the total, radiative, and nonradiative LST sensitivities to LAI.
Report regression coefficients, diagnostic statistics, predictor trends, and
their estimated physical contributions to the target trend.

Notes
-----
The active analysis uses ridge regression with alpha=0.1 and retains temporal
trends (``DO_DETREND=False``). Snow cover is included only in snow-affected
regions. Ridge coefficient p-values and the model F-test are approximate
diagnostics. Physical contributions are calculated as the standardized
coefficient multiplied by the standardized predictor trend and target standard
deviation. Set ``VEG_LST_DATA_DIR`` to override the default data directory.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.stats as stats
import xarray as xr
from scipy.signal import detrend
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_ROOT = Path(os.environ.get(
    "VEG_LST_DATA_DIR",
    "/home/energy/chaoz/project/05Veg_LST/data",
))

# -----------------------------------------------------------------------------
# 0. GLOBAL SETTINGS
# -----------------------------------------------------------------------------
DO_DETREND = False
WINDOW_SIZE = 1
MIN_VALID_N = 15 if WINDOW_SIZE == 1 else 25
SNOW_THRESHOLD = 1.0  # percent
REGRESSION_METHOD = "ridge"  # "ridge", "ols", or "pls"
RIDGE_ALPHA = 0.1


def log(message: str) -> None:
    """Print a timestamped message."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)

# -----------------------------------------------------------------------------
# 1. UTILITIES
# -----------------------------------------------------------------------------
def detrend_2d_columns(arr_2d, do_detrend: bool = False):
    """
    Detrend a 2D array column by column while allowing NaNs.

    Parameters
    ----------
    arr_2d : np.ndarray, shape (time, n_series)
        Each column is treated as one local pixel time series.
    do_detrend : bool
        If True, remove the linear trend from each column.
        If False, return the original values unchanged.

    Returns
    -------
    np.ndarray, shape (time, n_series)
        Detrended or original series. Columns with <2 valid values are returned as NaN
        when do_detrend=True.

    Notes
    -----
    This function is used by the moving-window attribution function.
    When window_size > 1, each local pixel inside the window is processed separately
    before pooled samples are stacked.
    """
    arr_2d = np.asarray(arr_2d, dtype=float)

    # Keep original data when detrending is disabled.
    # This is the default for CO2-related analyses.
    if not do_detrend:
        return arr_2d.copy()

    out = np.full_like(arr_2d, np.nan, dtype=float)

    n_series = arr_2d.shape[1]
    for j in range(n_series):
        y = arr_2d[:, j]
        m = np.isfinite(y)
        if m.sum() < 2:
            continue
        try:
            out[m, j] = detrend(y[m], type="linear")
        except Exception:
            continue

    return out


def combine_masked_results(res_a, res_b, mask_a):
    """
    Combine two attribution result datasets using a boolean mask.

    Parameters
    ----------
    res_a, res_b : xr.Dataset
        Two result datasets with the same variable structure.
    mask_a : xr.DataArray, shape (lat, lon)
        Pixels where mask_a == True use res_a; otherwise use res_b.

    Returns
    -------
    xr.Dataset
        Combined dataset.

    Notes
    -----
    This is used to merge:
      - attribution for snow-affected pixels
      - attribution for snow-free pixels
    """
    out = xr.Dataset()
    for v in res_a.data_vars:
        out[v] = xr.where(mask_a, res_a[v], res_b[v])
        out[v].attrs = res_a[v].attrs.copy()

    out.attrs = res_a.attrs.copy()
    out.attrs["combination_rule"] = "xr.where(mask_a, result_a, result_b)"
    return out


# -----------------------------------------------------------------------------
# 2. CORE REGRESSION FUNCTION
# -----------------------------------------------------------------------------
def apply_pixel_attribution(
    y_window,
    X_window,
    alpha=0.1,
    regression_method="ridge",
    pls_n_components=2,
    pls_n_permutations=200,
    pls_random_seed=42,
    std_threshold=1e-6,
    min_valid_n=15,
    do_detrend=False,
):
    """
    Perform standardized regression for one central pixel using samples pooled
    from a moving spatial window.

    Regression methods
    ------------------
    regression_method="ridge":
        Ridge regression with penalty alpha.

    regression_method="ols":
        Ordinary least squares multiple linear regression.

    regression_method="pls":
        Partial least squares regression using sklearn.cross_decomposition.
        PLS coefficient p-values are estimated by permutation of y.

    Output format is identical for all methods:
        [betas(n), pvals(n), r2(1), model_pval(1), vif(n)]

    Parameters
    ----------
    y_window : np.ndarray, shape (time, lat_win, lon_win)
        Target variable inside the moving window around the central pixel.

    X_window : np.ndarray, shape (time, feature, lat_win, lon_win)
        Predictor variables inside the same moving window.

    alpha : float, default 0.1
        Ridge penalty. Used only when regression_method="ridge".

    regression_method : {"ridge", "ols", "pls"}, default "ridge"
        Regression method.

    pls_n_components : int, default 2
        Number of PLS latent components. Effective value is clipped to:
            min(pls_n_components, n_active_predictors, n_valid_samples - 1)

    pls_n_permutations : int, default 200
        Number of y-permutations used for empirical PLS coefficient p-values.

    pls_random_seed : int, default 42
        Random seed for PLS permutation p-values.

    std_threshold : float
        Threshold below which target/predictor variability is considered inactive.

    min_valid_n : int
        Minimum number of pooled valid samples required after flattening time x space.

    do_detrend : bool
        If True, detrend each local time series before standardization/regression.
        If False, use original standardized time series.

    Returns
    -------
    np.ndarray
        1D output:
            [betas(n), pvals(n), r2(1), model_pval(1), vif(n)]

    Notes
    -----
    PLS p-values:
        For each permutation, y is randomly shuffled while X is fixed. The PLS
        model is refit and coefficient magnitudes are stored. The empirical
        two-sided p-value is:

            p_j = (1 + count(|beta_perm_j| >= |beta_obs_j|)) / (n_perm + 1)

        This is computationally heavier than Ridge/OLS.
    """
    n_features = X_window.shape[1]
    out_size = n_features * 3 + 2
    fail_output = np.full(out_size, np.nan, dtype=float)

    regression_method = str(regression_method).lower()

    if regression_method not in ["ridge", "ols", "pls"]:
        return fail_output

    # --------------------------------------------------
    # 1. Reshape window data
    # --------------------------------------------------
    nt, nwy, nwx = y_window.shape
    n_pix = nwy * nwx

    y_2d = y_window.reshape(nt, n_pix)
    X_3d = X_window.reshape(nt, n_features, n_pix)

    # --------------------------------------------------
    # 2. Optional detrending for each neighborhood pixel
    # --------------------------------------------------
    y_dt_2d = detrend_2d_columns(y_2d, do_detrend=do_detrend)

    X_dt_3d = np.full_like(X_3d, np.nan, dtype=float)
    for k in range(n_features):
        X_dt_3d[:, k, :] = detrend_2d_columns(X_3d[:, k, :], do_detrend=do_detrend)

    # --------------------------------------------------
    # 3. Flatten pooled samples across time x neighborhood pixels
    # --------------------------------------------------
    y_pool = y_dt_2d.reshape(nt * n_pix)
    X_pool = np.moveaxis(X_dt_3d, 1, -1).reshape(nt * n_pix, n_features)

    # --------------------------------------------------
    # 4. Row-wise cleaning
    # --------------------------------------------------
    valid_mask = np.isfinite(y_pool) & np.isfinite(X_pool).all(axis=1)
    n_valid = valid_mask.sum()
    if n_valid < min_valid_n:
        return fail_output

    y_clean = y_pool[valid_mask]
    X_clean = X_pool[valid_mask, :]

    # --------------------------------------------------
    # 5. Variance screening
    # --------------------------------------------------
    if np.std(y_clean) < std_threshold:
        return fail_output

    active_mask = np.std(X_clean, axis=0) > std_threshold
    if not np.any(active_mask):
        return fail_output

    X_active = X_clean[:, active_mask]
    n_active = X_active.shape[1]

    if len(y_clean) <= n_active + 2:
        return fail_output

    # --------------------------------------------------
    # 6. Standardization
    # --------------------------------------------------
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_std = scaler_x.fit_transform(X_active)
    y_std = scaler_y.fit_transform(y_clean.reshape(-1, 1)).ravel()

    # --------------------------------------------------
    # 7. Fit model
    # --------------------------------------------------
    try:
        if regression_method == "ols":
            model = LinearRegression()
            model.fit(X_std, y_std)

            betas_active = np.asarray(model.coef_, dtype=float).reshape(-1)
            y_pred = np.asarray(model.predict(X_std), dtype=float).reshape(-1)

        elif regression_method == "ridge":
            model = Ridge(alpha=alpha)
            model.fit(X_std, y_std)

            betas_active = np.asarray(model.coef_, dtype=float).reshape(-1)
            y_pred = np.asarray(model.predict(X_std), dtype=float).reshape(-1)

        elif regression_method == "pls":
            n_comp = min(int(pls_n_components), n_active, len(y_std) - 1)
            if n_comp < 1:
                return fail_output

            model = PLSRegression(n_components=n_comp, scale=False)
            model.fit(X_std, y_std.reshape(-1, 1))

            betas_active = np.asarray(model.coef_, dtype=float).reshape(-1)
            y_pred = np.asarray(model.predict(X_std), dtype=float).reshape(-1)

        # --------------------------------------------------
        # 8. Model diagnostics
        # --------------------------------------------------
        resid = y_std - y_pred
        rss = np.sum(resid ** 2)
        tss = np.sum((y_std - np.mean(y_std)) ** 2)
        r2 = 1 - rss / tss if tss > 0 else np.nan

        df_model = n_active
        df_resid = len(y_std) - n_active - 1
        if df_resid <= 0:
            return fail_output

        # Model-level F test.
        # OLS: standard.
        # Ridge/PLS: approximate diagnostic, not strict inferential p-value.
        if np.isfinite(r2) and (1 - r2) > 0:
            f_stat = (r2 / df_model) / ((1 - r2) / df_resid)
            model_pval = 1 - stats.f.cdf(f_stat, df_model, df_resid)
        else:
            model_pval = np.nan

        # --------------------------------------------------
        # 9. Coefficient p-values
        # --------------------------------------------------
        if regression_method in ["ols", "ridge"]:
            mse = rss / df_resid
            XtX = X_std.T @ X_std

            if regression_method == "ols":
                cov_beta = mse * np.linalg.pinv(XtX)
            else:
                # Approximate only for Ridge.
                cov_beta = mse * np.linalg.pinv(XtX + alpha * np.eye(n_active))

            se_beta = np.sqrt(np.diag(cov_beta))
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = betas_active / se_beta

            pvals_active = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

        elif regression_method == "pls":
            n_perm = int(pls_n_permutations)

            if n_perm <= 0:
                pvals_active = np.full(n_active, np.nan, dtype=float)
            else:
                rng = np.random.default_rng(pls_random_seed)

                beta_perm_abs = np.full((n_perm, n_active), np.nan, dtype=float)

                n_comp = min(int(pls_n_components), n_active, len(y_std) - 1)

                for b in range(n_perm):
                    y_perm = rng.permutation(y_std)

                    try:
                        perm_model = PLSRegression(n_components=n_comp, scale=False)
                        perm_model.fit(X_std, y_perm.reshape(-1, 1))
                        beta_perm_abs[b, :] = np.abs(
                            np.asarray(perm_model.coef_, dtype=float).reshape(-1)
                        )
                    except Exception:
                        continue

                obs_abs = np.abs(betas_active)
                valid_perm = np.isfinite(beta_perm_abs).all(axis=1)

                if valid_perm.sum() < max(20, int(0.2 * n_perm)):
                    pvals_active = np.full(n_active, np.nan, dtype=float)
                else:
                    beta_perm_abs = beta_perm_abs[valid_perm, :]
                    pvals_active = (
                        1.0
                        + np.sum(beta_perm_abs >= obs_abs[None, :], axis=0)
                    ) / (beta_perm_abs.shape[0] + 1.0)

        # --------------------------------------------------
        # 10. VIF
        # --------------------------------------------------
        vif_active = np.full(n_active, np.nan, dtype=float)
        for i in range(n_active):
            try:
                vif_active[i] = variance_inflation_factor(X_std, i)
            except Exception:
                vif_active[i] = np.nan

        # --------------------------------------------------
        # 11. Restore inactive predictors
        # --------------------------------------------------
        betas_full = np.zeros(n_features, dtype=float)
        pvals_full = np.full(n_features, np.nan, dtype=float)
        vif_full = np.full(n_features, np.nan, dtype=float)

        betas_full[active_mask] = betas_active
        pvals_full[active_mask] = pvals_active
        vif_full[active_mask] = vif_active

        return np.concatenate([betas_full, pvals_full, [r2], [model_pval], vif_full])

    except Exception:
        return fail_output


# -----------------------------------------------------------------------------
# 3. CONTRIBUTION CALCULATION
# -----------------------------------------------------------------------------
def calculate_physical_contribution(
    beta,
    predictor_da,
    target_da,
    time_dim="time",
    min_valid_n=15,
    regression_method="ridge",
    do_detrend=False,
):
    """
    Calculate physical contribution and associated predictor/target trends.

    Returned variables
    ------------------
    physical_contribution : contribution of predictor trend to target trend
    predictor_trend       : raw trend of predictor per decade
    predictor_std_trend   : trend of standardized predictor per decade
    target_trend          : raw trend of target per decade
    target_std            : standard deviation of target over time

    Formula
    -------
    physical_contribution = beta * predictor_std_trend * target_std

    Notes
    -----
    1. beta is estimated from standardized regression. Whether beta is based on
       detrended or non-detrended data is controlled in run_attribution_workflow.
    2. predictor_std_trend is computed from the original standardized predictor series.
    3. predictor_std_trend is already in units of SD per decade, so no extra x10 is
       needed in the final contribution formula.
    """

    def safe_linregress_slope_decade(y):
        """Return linear slope per decade against integer time index."""
        y = np.asarray(y, dtype=float)
        m = np.isfinite(y)
        if m.sum() < 2:
            return np.nan
        x = np.arange(len(y))[m]
        y = y[m]
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope * 10.0

    # --------------------------------------------------
    # 1. Valid counts and common valid mask
    # --------------------------------------------------
    n_valid_x = predictor_da.notnull().sum(time_dim)
    n_valid_y = target_da.notnull().sum(time_dim)
    valid_mask = (n_valid_x >= min_valid_n) & (n_valid_y >= min_valid_n)

    x = predictor_da.where(valid_mask)
    y = target_da.where(valid_mask)

    # --------------------------------------------------
    # 2. Safe mean and sample std, manually computed to avoid nanstd warnings
    # --------------------------------------------------
    x_mean = x.mean(time_dim, skipna=True)
    y_mean = y.mean(time_dim, skipna=True)

    x_anom2 = (x - x_mean) ** 2
    y_anom2 = (y - y_mean) ** 2

    x_denom = xr.where(n_valid_x > 1, n_valid_x - 1, np.nan)
    y_denom = xr.where(n_valid_y > 1, n_valid_y - 1, np.nan)

    x_var = x_anom2.sum(time_dim, skipna=True) / x_denom
    y_var = y_anom2.sum(time_dim, skipna=True) / y_denom

    x_stdv = np.sqrt(x_var)
    sigma_y = np.sqrt(y_var)

    x_stdv = xr.where(np.isfinite(x_stdv) & (x_stdv > 0), x_stdv, np.nan)
    sigma_y = xr.where(np.isfinite(sigma_y) & (sigma_y > 0), sigma_y, np.nan)

    # Standardized predictor used for predictor_std_trend.
    x_std = (x - x_mean) / x_stdv

    # --------------------------------------------------
    # 3. Predictor and target trends
    # --------------------------------------------------
    predictor_trend = xr.apply_ufunc(
        safe_linregress_slope_decade,
        x,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).where(valid_mask)
    predictor_trend.attrs["long_name"] = "Predictor trend"
    predictor_trend.attrs["units"] = f"{predictor_da.attrs.get('units', '')} per decade"

    predictor_std_trend = xr.apply_ufunc(
        safe_linregress_slope_decade,
        x_std,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).where(valid_mask)
    predictor_std_trend.attrs["long_name"] = "Standardized predictor trend"
    predictor_std_trend.attrs["units"] = "SD per decade"

    target_trend = xr.apply_ufunc(
        safe_linregress_slope_decade,
        y,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).where(valid_mask)
    target_trend.attrs["long_name"] = "Target trend"
    target_trend.attrs["units"] = f"{target_da.attrs.get('units', '')} per decade"

    sigma_y = sigma_y.where(valid_mask)
    sigma_y.attrs["long_name"] = "Target standard deviation"
    sigma_y.attrs["units"] = target_da.attrs.get("units", "")

    # --------------------------------------------------
    # 4. Physical contribution
    # --------------------------------------------------
    contrib = (beta * predictor_std_trend) * sigma_y
    contrib = contrib.where(valid_mask)
    contrib.attrs["long_name"] = "Physical contribution"
    contrib.attrs["units"] = f"{target_da.attrs.get('units', '')} per decade"
    contrib.attrs["note"] = f"Contribution per decade; valid years >= {min_valid_n}"

    ds_out = xr.Dataset(
        {
            "physical_contribution": contrib,
            "predictor_trend": predictor_trend,
            "predictor_std_trend": predictor_std_trend,
            "target_trend": target_trend,
            "target_std": sigma_y,
        }
    )

    contrib.attrs["note"] = (
        f"Contribution per decade; valid years >= {min_valid_n}; "
        f"beta estimated using {regression_method}; "
        f"regression detrend={int(do_detrend)}"
    )
    
    ds_out.attrs["note"] = (
        "physical_contribution = beta * predictor_std_trend * target_std; "
        f"all outputs masked where valid years < {min_valid_n}; "
        f"beta estimated using {regression_method}; "
        "for PLS, contribution should be interpreted as a PLS-projected contribution."
    )

    return ds_out


# -----------------------------------------------------------------------------
# 4. WORKFLOW WRAPPER
# -----------------------------------------------------------------------------
def run_attribution_workflow(
    ds,
    target_var,
    predictors,
    alpha=0.1,
    regression_method="ridge",  # "ridge", "ols", or "pls"
    pls_n_components=2,
    pls_n_permutations=200,
    pls_random_seed=42,
    mask=None,
    time_dim="time",
    lat_dim="lat",
    lon_dim="lon",
    window_size=1,
    std_threshold=1e-6,
    min_valid_n=15,
    do_detrend=False,
):
    """
    Run moving-window attribution analysis.

    Regression methods
    ------------------
    regression_method="ridge":
        Ridge regression using alpha.

    regression_method="ols":
        Ordinary least squares multiple linear regression.

    regression_method="pls":
        Partial least squares regression using PLSRegression.
        Coefficient p-values are estimated by permutation inside
        apply_pixel_attribution().

    Returns
    -------
    xr.Dataset
        Attribution results for all grid cells.
    """
    regression_method = str(regression_method).lower()

    if regression_method not in ["ridge", "ols", "pls"]:
        raise ValueError("regression_method must be one of {'ridge', 'ols', 'pls'}.")

    if (window_size < 1) or (window_size % 2 == 0):
        raise ValueError("window_size must be a positive odd integer, e.g. 1, 3, 5, ...")

    log(f"Processing attribution for: {target_var}")
    log(f"Predictors: {predictors}")
    log(f"regression_method = {regression_method}")
    log(f"alpha = {alpha}")
    if regression_method == "pls":
        log(f"pls_n_components = {pls_n_components}")
        log(f"pls_n_permutations = {pls_n_permutations}")
    log(f"window_size = {window_size}")
    log(f"do_detrend = {do_detrend}")

    ds_sub = ds.where(mask) if mask is not None else ds

    y_input = ds_sub[target_var].transpose(time_dim, lat_dim, lon_dim)

    X_input = xr.concat([ds_sub[p] for p in predictors], dim="feature")
    X_input = X_input.assign_coords(feature=predictors)
    X_input = X_input.transpose(time_dim, "feature", lat_dim, lon_dim)

    y_window = (
        y_input
        .rolling({lat_dim: window_size, lon_dim: window_size}, center=True, min_periods=1)
        .construct({lat_dim: "lat_win", lon_dim: "lon_win"})
        .transpose(lat_dim, lon_dim, time_dim, "lat_win", "lon_win")
    )

    X_window = (
        X_input
        .rolling({lat_dim: window_size, lon_dim: window_size}, center=True, min_periods=1)
        .construct({lat_dim: "lat_win", lon_dim: "lon_win"})
        .transpose(lat_dim, lon_dim, time_dim, "feature", "lat_win", "lon_win")
    )

    n = len(predictors)

    res_raw = xr.apply_ufunc(
        apply_pixel_attribution,
        y_window,
        X_window,
        input_core_dims=[
            [time_dim, "lat_win", "lon_win"],
            [time_dim, "feature", "lat_win", "lon_win"],
        ],
        output_core_dims=[["out_dim"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={
            "alpha": alpha,
            "regression_method": regression_method,
            "pls_n_components": pls_n_components,
            "pls_n_permutations": pls_n_permutations,
            "pls_random_seed": pls_random_seed,
            "std_threshold": std_threshold,
            "min_valid_n": min_valid_n,
            "do_detrend": do_detrend,
        },
        dask_gufunc_kwargs={"output_sizes": {"out_dim": n * 3 + 2}},
    )

    res = xr.Dataset()

    res["beta"] = (
        res_raw.isel(out_dim=slice(0, n))
        .rename({"out_dim": "feature"})
        .assign_coords(feature=predictors)
    )
    res["beta"].attrs = {
        "long_name": "Standardized regression coefficient",
        "units": "unitless",
        "regression_method": regression_method,
    }

    res["pval"] = (
        res_raw.isel(out_dim=slice(n, 2 * n))
        .rename({"out_dim": "feature"})
        .assign_coords(feature=predictors)
    )
    res["pval"].attrs = {
        "long_name": "Coefficient p-value",
        "units": "unitless",
        "note": (
            "OLS p-values are standard; Ridge p-values are approximate; "
            "PLS p-values are empirical permutation p-values."
        ),
        "pls_n_permutations": (
            int(pls_n_permutations) if regression_method == "pls" else -1
        ),
    }

    res["r2"] = res_raw.isel(out_dim=2 * n)
    res["r2"].attrs = {
        "long_name": "Model R-squared",
        "units": "unitless",
        "regression_method": regression_method,
    }

    res["model_pval"] = res_raw.isel(out_dim=2 * n + 1)
    res["model_pval"].attrs = {
        "long_name": "Model F-test p-value",
        "units": "unitless",
        "note": (
            "Standard for OLS; approximate diagnostic for Ridge and PLS."
        ),
    }

    res["vif"] = (
        res_raw.isel(out_dim=slice(2 * n + 2, 3 * n + 2))
        .rename({"out_dim": "feature"})
        .assign_coords(feature=predictors)
    )
    res["vif"].attrs = {
        "long_name": "Variance inflation factor",
        "units": "unitless",
        "note": (
            "Predictor collinearity diagnostic calculated from standardized "
            "active predictors"
        ),
    }

    phys_contrib = xr.Dataset()
    pred_trend = xr.Dataset()
    pred_std_trend = xr.Dataset()
    target_trend = xr.Dataset()
    target_std = xr.Dataset()

    for p in predictors:
        tmp = calculate_physical_contribution(
            res["beta"].sel(feature=p),
            ds_sub[p],
            ds_sub[target_var],
            time_dim=time_dim,
            min_valid_n=min_valid_n,
            regression_method=regression_method,
            do_detrend=do_detrend,
        )

        phys_contrib[p] = tmp["physical_contribution"]
        pred_trend[p] = tmp["predictor_trend"]
        pred_std_trend[p] = tmp["predictor_std_trend"]
        target_trend[p] = tmp["target_trend"]
        target_std[p] = tmp["target_std"]

    res["physical_contribution"] = (
        phys_contrib.to_array(dim="feature")
        .assign_coords(feature=predictors)
    )
    res["physical_contribution"].attrs = {
        "long_name": "Physical contribution per decade",
        "note": (
            "Calculated as beta * predictor_std_trend * target_std. "
            "For PLS, beta is the PLS coefficient in standardized space."
        ),
    }

    res["predictor_trend"] = (
        pred_trend.to_array(dim="feature")
        .assign_coords(feature=predictors)
    )
    res["predictor_trend"].attrs = {"long_name": "Predictor trend per decade"}

    res["predictor_std_trend"] = (
        pred_std_trend.to_array(dim="feature")
        .assign_coords(feature=predictors)
    )
    res["predictor_std_trend"].attrs = {
        "long_name": "Standardized predictor trend per decade"
    }

    res["target_trend"] = (
        target_trend.to_array(dim="feature")
        .assign_coords(feature=predictors)
    )
    res["target_trend"].attrs = {"long_name": "Target trend per decade"}

    res["target_std"] = (
        target_std.to_array(dim="feature")
        .assign_coords(feature=predictors)
    )
    res["target_std"].attrs = {"long_name": "Target standard deviation"}

    total_phys = np.abs(res["physical_contribution"]).sum("feature")
    eps = 1e-12
    res["percent_contribution"] = xr.where(
        np.abs(total_phys) > eps,
        res["physical_contribution"] / total_phys * 100.0,
        np.nan,
    )
    res["percent_contribution"].attrs = {
        "long_name": "Percent contribution",
        "units": "%",
        "note": "physical_contribution / sum(abs(physical_contribution)) * 100",
    }

    if regression_method == "ridge":
        regression_label = f"Ridge(alpha={alpha})"
    elif regression_method == "ols":
        regression_label = "OLS"
    else:
        regression_label = f"PLSRegression(n_components={pls_n_components})"

    res.attrs = {
        "target_variable": target_var,
        "predictors": ", ".join(predictors),
        "regression_method": regression_method,
        "regression_type": regression_label,
        "alpha": float(alpha),
        "pls_n_components": (
            int(pls_n_components) if regression_method == "pls" else -1
        ),
        "pls_n_permutations": (
            int(pls_n_permutations) if regression_method == "pls" else -1
        ),
        "pls_random_seed": (
            int(pls_random_seed) if regression_method == "pls" else -1
        ),
        "window_size": int(window_size),
        "detrend": int(do_detrend),
        "window_note": (
            "window_size=1 means pixel-wise temporal regression; "
            "window_size=3 means pooled 3x3 moving-window regression"
        ),
        "note": (
            "Regression performed on "
            + ("detrended " if do_detrend else "non-detrended ")
            + "standardized time series. Output includes beta, p-value, R2, "
            "model p-value, VIF, physical contribution, predictor trend, "
            "standardized predictor trend, target trend, and target standard deviation."
        ),
    }

    return res


# -----------------------------------------------------------------------------
# 5. PUBLIC REUSABLE FUNCTION: SNOW-SPLIT ATTRIBUTION
# -----------------------------------------------------------------------------
def run_snow_split_attribution(
    ds,
    target_var,
    preds_snow,
    preds_nosnow,
    is_snow,
    alpha=0.1,
    regression_method="ridge",
    pls_n_components=2,
    pls_n_permutations=200,
    pls_random_seed=42,
    time_dim="time",
    lat_dim="lat",
    lon_dim="lon",
    window_size=1,
    std_threshold=1e-6,
    min_valid_n=15,
    do_detrend=False,
):
    """
    Run attribution for one response variable by splitting the globe into
    snow-affected and snow-free regions.

    Regression methods
    ------------------
    regression_method="ridge":
        Ridge regression using alpha.

    regression_method="ols":
        Ordinary least squares multiple linear regression.

    regression_method="pls":
        Partial least squares regression using PLSRegression. Coefficient
        p-values are estimated by permutation inside apply_pixel_attribution().

    Notes
    -----
    This function:
      1. runs attribution in snow-affected pixels
      2. runs attribution in snow-free pixels
      3. harmonizes the snow-free result to the snow-feature template
      4. merges the two outputs back to one global dataset
    """
    regression_method = str(regression_method).lower()
    if regression_method not in ["ridge", "ols", "pls"]:
        raise ValueError("regression_method must be one of {'ridge', 'ols', 'pls'}.")

    is_nosnow = ~is_snow

    log(f"Running snow-split attribution for: {target_var}")
    log(f"regression_method = {regression_method}")
    if regression_method == "pls":
        log(f"pls_n_components = {pls_n_components}")
        log(f"pls_n_permutations = {pls_n_permutations}")

    # -----------------------------
    # Snow-affected attribution
    # -----------------------------
    res_snow = run_attribution_workflow(
        ds,
        target_var=target_var,
        predictors=preds_snow,
        alpha=alpha,
        regression_method=regression_method,
        pls_n_components=pls_n_components,
        pls_n_permutations=pls_n_permutations,
        pls_random_seed=pls_random_seed,
        mask=is_snow,
        time_dim=time_dim,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        window_size=window_size,
        std_threshold=std_threshold,
        min_valid_n=min_valid_n,
        do_detrend=do_detrend,
    )

    # -----------------------------
    # Snow-free attribution
    # -----------------------------
    res_nosnow = run_attribution_workflow(
        ds,
        target_var=target_var,
        predictors=preds_nosnow,
        alpha=alpha,
        regression_method=regression_method,
        pls_n_components=pls_n_components,
        pls_n_permutations=pls_n_permutations,
        pls_random_seed=pls_random_seed,
        mask=is_nosnow,
        time_dim=time_dim,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        window_size=window_size,
        std_threshold=std_threshold,
        min_valid_n=min_valid_n,
        do_detrend=do_detrend,
    )

    # -----------------------------
    # Harmonize snow-free result to snow-feature template
    # -----------------------------
    template_features = preds_snow
    res_nosnow_h = xr.Dataset()

    feature_vars = [
        "beta",
        "pval",
        "vif",
        "physical_contribution",
        "percent_contribution",
        "predictor_trend",
        "predictor_std_trend",
        "target_trend",
        "target_std",
    ]

    for v in feature_vars:
        res_nosnow_h[v] = res_nosnow[v].reindex(feature=template_features)
        res_nosnow_h[v].attrs = res_nosnow[v].attrs.copy()

    scalar_vars = ["r2", "model_pval"]
    for v in scalar_vars:
        res_nosnow_h[v] = res_nosnow[v]
        res_nosnow_h[v].attrs = res_nosnow[v].attrs.copy()

    # -----------------------------
    # Merge back to global result
    # -----------------------------
    res_all = combine_masked_results(res_snow, res_nosnow_h, is_snow)

    res_all.attrs.update(res_snow.attrs)
    res_all.attrs["snow_split_target"] = target_var
    res_all.attrs["snow_predictors"] = ", ".join(preds_snow)
    res_all.attrs["snowfree_predictors"] = ", ".join(preds_nosnow)
    res_all.attrs["regression_method"] = regression_method
    res_all.attrs["alpha"] = float(alpha)
    res_all.attrs["pls_n_components"] = (
        int(pls_n_components) if regression_method == "pls" else -1
    )
    res_all.attrs["pls_n_permutations"] = (
        int(pls_n_permutations) if regression_method == "pls" else -1
    )
    res_all.attrs["pls_random_seed"] = (
        int(pls_random_seed) if regression_method == "pls" else -1
    )
    res_all.attrs["detrend"] = int(do_detrend)

    return res_all


# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    os.chdir(DATA_ROOT)
    log("Reading datasets...")

    ds_mean_lst = xr.open_dataset(
        "processed/Sensitivity_20260208/GLASS/MOD11C3/"
        "Sensitivity_Annual_LSTdailymean_LAI_GLASS_1d.nc"
    )
    ds_mean_lst_rad = xr.open_dataset(
        "processed/Sensitivity_LST_Energy_LAI_20260325/Albedo/Annual/"
        "Sensitivity_Annual_LSTdailymean_LAI_from_Albedo_1d_gapfilled.nc"
    )
    ds_mean_lst_nonrad = xr.open_dataset(
        "processed/Sensitivity_LST_Energy_LAI_20260325/LEH/Annual/"
        "Sensitivity_Annual_LSTdailymean_LAI_from_LEH_FAO56_1d.nc"
    )
    ds_mean_lai = xr.open_dataset("LAI/LAI_GLASS_Annual_1d_2001_2024.nc")
    ds_mean_era = xr.open_dataset(
        "ERA5_Land/ERA5_Land_Annual_1d_2000_2024.nc"
    )
    ds_mean_co2 = xr.open_dataset("CO2/CO2_Annual_1d_2001_2024.nc")

    # Ensure ERA5-Land uses 2001-2024 and has consistent dimension order.
    ds_mean_era = (
        ds_mean_era
        .sel(time=slice("2001-01-01", "2024-12-31"))
        .transpose("time", "lat", "lon")
    )

    # Align all datasets exactly. If this fails, check lat/lon/time coordinates.
    (
        ds_mean_lst,
        ds_mean_lst_rad,
        ds_mean_lst_nonrad,
        ds_mean_lai,
        ds_mean_era,
        ds_mean_co2,
    ) = xr.align(
        ds_mean_lst,
        ds_mean_lst_rad,
        ds_mean_lst_nonrad,
        ds_mean_lai,
        ds_mean_era,
        ds_mean_co2,
        join="exact",
    )

    # Output directory. Add a suffix to make detrended/non-detrended results explicit.
    mode_tag = "Detrend" if DO_DETREND else "NoDetrend"
    subdir = (
        Path("processed/Attribution_20260827")
        / f"{REGRESSION_METHOD}Regression_{mode_tag}"
    )
    subdir.mkdir(parents=True, exist_ok=True)

    log("Building combined dataset...")

    ds_mean = xr.Dataset()
    ds_mean["lst_mean"] = ds_mean_lst["sens_1deg_mean"]
    ds_mean["lst_rad_mean"] = ds_mean_lst_rad["sens_1deg_mean"]
    ds_mean["lst_nonrad_mean"] = ds_mean_lst_nonrad["sens_1deg_mean"]

    ds_mean["t2m_mean"] = ds_mean_era["t2m"]
    ds_mean["ssrd_mean"] = ds_mean_era["ssrd"]
    ds_mean["swvl1_mean"] = ds_mean_era["swvl1"]
    ds_mean["vpd_mean"] = ds_mean_era["vpd"]
    ds_mean["snowc_mean"] = ds_mean_era["snowc"]
    ds_mean["lai_mean"] = ds_mean_lai["LAI_1deg_mean"]
    ds_mean["co2_mean"] = ds_mean_co2["co2"]

    # -----------------------------------------------------------------------------
    # Snow / no-snow mask
    # -----------------------------------------------------------------------------
    snowc_clim = ds_mean["snowc_mean"].mean("time")
    is_snow = snowc_clim > SNOW_THRESHOLD

    # -----------------------------------------------------------------------------
    # Predictor sets for snow-split analysis
    # -----------------------------------------------------------------------------
    # Snow-affected regions include snow cover as a predictor.
    # Snow-free regions exclude snow cover because it has no physical meaning there.
    # CO2 is included to examine long-term physiological/structural influence.
    preds_snow = [
        "co2_mean", "t2m_mean", "swvl1_mean", "ssrd_mean", "vpd_mean",
        "snowc_mean",
    ]
    preds_nosnow = [
        "co2_mean", "t2m_mean", "swvl1_mean", "ssrd_mean", "vpd_mean",
    ]

    # -----------------------------------------------------------------------------
    # Run the same snow-split workflow for all three response factors
    # -----------------------------------------------------------------------------
    response_vars = [
        "lst_mean",
        "lst_nonrad_mean",
        "lst_rad_mean",
    ]

    results_all = {}

    for target_var in response_vars:
        results_all[target_var] = run_snow_split_attribution(
            ds=ds_mean,
            target_var=target_var,
            preds_snow=preds_snow,
            preds_nosnow=preds_nosnow,
            is_snow=is_snow,
            alpha=RIDGE_ALPHA,
            regression_method=REGRESSION_METHOD,
            time_dim="time",
            lat_dim="lat",
            lon_dim="lon",
            window_size=WINDOW_SIZE,
            std_threshold=1e-6,
            min_valid_n=MIN_VALID_N,
            do_detrend=DO_DETREND,
        )
        results_all[target_var].attrs.update({
            "creator": "Chao Zhang",
            "institution": "National University of Singapore",
            "creation_time_utc": datetime.now(timezone.utc).isoformat(),
            "snow_threshold_percent": SNOW_THRESHOLD,
            "minimum_valid_samples": MIN_VALID_N,
            "detrend": int(DO_DETREND),
        })

    # -----------------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------------
    log("Saving outputs...")

    outfiles = {
        "lst_mean": subdir / (
            f"Attribution_LSTTotal_Annual_winsize{WINDOW_SIZE}.nc"
        ),
        "lst_nonrad_mean": subdir / (
            f"Attribution_LSTNonRadiative_Annual_winsize{WINDOW_SIZE}.nc"
        ),
        "lst_rad_mean": subdir / (
            f"Attribution_LSTRadiative_Annual_winsize{WINDOW_SIZE}.nc"
        ),
    }

    for target_var, ds_out in results_all.items():
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
        ds_out.to_netcdf(outfiles[target_var], encoding=encoding)
        log(f"Saved: {outfiles[target_var]}")

    log("Done.")


if __name__ == "__main__":
    main()
