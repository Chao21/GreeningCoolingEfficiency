#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Create Figure 5: decomposition of LAI-change-induced LST effects.

Author
------
Chao Zhang, National University of Singapore

Date
----
2026-09-05

Purpose
-------
Present cumulative LAI-induced LST changes, their LAI- and sensitivity-driven
components, regional trends, spatial patterns, and the dominant trend term.

Notes
-----
Map trends are converted from per year to per decade for display. Distribution
and categorical insets are drawn in parent-axis coordinates to ensure reliable
SVG export. Set ``VEG_LST_DATA_DIR`` and ``VEG_LST_UTILS_DIR`` to override the
default project directories.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import linregress, t


DATA_ROOT = Path(os.environ.get(
    "VEG_LST_DATA_DIR",
    "/home/energy/chaoz/project/05Veg_LST/data",
))
UTILS_DIR = Path(os.environ.get(
    "VEG_LST_UTILS_DIR",
    "/home/energy/chaoz/code/utils",
))
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from plot_utils import get_truncated_cs, plot_settings  # noqa: E402


@dataclass(frozen=True)
class FigureConfig:
    base_dir: Path = DATA_ROOT
    output_figure: Path = DATA_ROOT.parent / "figure/Fig.05_dLST_Decomposition.svg"
    folder: Path = Path(
        "processed/dLST_Decompose_20260223/ConsistentMask_20260603/"
        "GLASS_Annual_LSTdailymean"
    )
    prefix: str = "dLSTdailymean_GLASS_Annual_2001_2024"
    regions: tuple[str, ...] = ("Global", "Tropical", "Temperate", "Arid", "Boreal")
    colors: tuple[str, ...] = ("#000000", "#3f87ba", "#cc5349")
    trend_vmin: float = -0.06
    trend_vmax: float = 0.06


LABEL_DELTA = {
    "Total": r"$\delta\mathrm{LST}^{*}$",
    "LAI-driven": r"$\delta\mathrm{LST}^{*}_{\mathrm{LAI}}$",
    "Sens-driven": r"$\delta\mathrm{LST}^{*}_{\mathrm{EFF}}$",
}
LABEL_SIGMA = {
    "Total": r"$\sigma\mathrm{LST}^{*}$",
    "LAI-driven": r"$\sigma\mathrm{LST}^{*}_{\mathrm{LAI}}$",
    "Sens-driven": r"$\sigma\mathrm{LST}^{*}_{\mathrm{EFF}}$",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_inputs(cfg: FigureConfig) -> dict[str, object]:
    """Load regional time series, regional trends, and gridded trend maps."""
    folder, prefix = cfg.base_dir / cfg.folder, cfg.prefix
    paths = {
        "df_region": folder / f"{prefix}_ClimateZone_TS.csv",
        "ds_total": folder / f"{prefix}_Trend_1deg_dLST_total.nc",
        "ds_lai": folder / f"{prefix}_Trend_1deg_dLST_LAI_driven.nc",
        "ds_sens": folder / f"{prefix}_Trend_1deg_dLST_Sens_driven.nc",
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))
    return {
        key: (
            pd.read_csv(path, parse_dates=["time"])
            if key == "df_region"
            else xr.load_dataset(path)
        )
        for key, path in paths.items()
    }


def select_series(
    df: pd.DataFrame,
    *,
    variable: str | None = None,
    region: str = "Global",
    group: str = "annual",
) -> pd.DataFrame:
    """Filter a long regional time-series table."""
    out = df[(df["group"] == group) & (df["region"] == region)].copy()
    if variable is not None:
        out = out[out["variable"] == variable].copy()
    return out.sort_values("time")


def decimal_year(time_values: Sequence[pd.Timestamp] | pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Convert datetime-like values to decimal years."""
    idx = pd.DatetimeIndex(pd.to_datetime(time_values))
    year = idx.year.astype(float)
    doy = idx.dayofyear.astype(float)
    days_in_year = np.array([
        366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
        for year in idx.year
    ])
    return year + (doy - 1.0) / days_in_year


def linear_trend(time_values, values, alpha: float = 0.05):
    """Linear trend against decimal year with 95% CI."""
    x = decimal_year(time_values)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan)

    lr = linregress(x[valid], y[valid])
    dfree = valid.sum() - 2
    if dfree > 0 and np.isfinite(lr.stderr):
        tcrit = t.ppf(1 - alpha / 2, dfree)
        ci = (lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr)
    else:
        ci = (np.nan, np.nan)

    return lr.slope, lr.pvalue, lr.intercept, lr.stderr, ci


def p_to_star(pval: float) -> str:
    if not np.isfinite(pval):
        return ""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def build_region_slope_table(
    df_region: pd.DataFrame,
    *,
    regions: Sequence[str],
    group: str = "annual",
) -> pd.DataFrame:
    """Compute regional trends for total, LAI-driven, and sensitivity-driven components."""
    variable_map = {
        "total": "dLST_total",
        "lai": "dLST_LAI_driven",
        "sens": "dLST_Sens_driven",
    }
    rows = []

    for region in regions:
        row = {"region": region}
        for short_name, variable in variable_map.items():
            sub = select_series(df_region, variable=variable, region=region, group=group)
            slope, pval, _, _, ci = linear_trend(sub["time"], sub["value"])
            row[f"{short_name}_slope"] = slope
            row[f"{short_name}_p"] = pval
            row[f"{short_name}_ci_lo"] = ci[0]
            row[f"{short_name}_ci_hi"] = ci[1]
        rows.append(row)

    return pd.DataFrame(rows)


def plot_global_timeseries(ax, df_region: pd.DataFrame, *, colors: Sequence[str]):
    """Plot global total, LAI-driven, and sensitivity-driven dLST* time series."""
    series = [
        ("Total", "dLST_total", colors[0]),
        ("LAI-driven", "dLST_LAI_driven", colors[1]),
        ("Sens-driven", "dLST_Sens_driven", colors[2]),
    ]

    annotation_lines = []
    for label, variable, color in series:
        df = select_series(df_region, variable=variable, region="Global")
        time = pd.DatetimeIndex(df["time"])
        x = decimal_year(time)
        y = df["value"].to_numpy(dtype=float)

        ax.plot(
            time.year, y, marker="o", color=color, alpha=0.5,
            linestyle="--", linewidth=1.2, markersize=3.5,
            label=LABEL_SIGMA[label],
        )

        slope, pval, intercept, stderr, _ = linear_trend(time, y)
        if np.isfinite(slope):
            ax.plot(time.year, intercept + slope * x, color=color, linewidth=2.2)
            slope_text = f"–{abs(slope) * 10:.3f}" if slope<0 else f"{slope * 10:.3f}"
            annotation_lines.append((
                label,
                f"{slope_text} K decade$^{{-1}}${p_to_star(pval)}",
                color,
            ))
            print(f"{label}: slope={slope * 10:.4f}, p={pval:.4f}, stderr={stderr * 10:.4f}")
        else:
            annotation_lines.append((label, "NA", color))

    ax.set_title("Global decomposition")
    ax.set_ylabel('Change in LST$^{*}$ [K]')
    ax.set_ylim(-0.055,0.03)
    ax.tick_params(direction="in")
    ax.legend(
        frameon=False, ncol=3, columnspacing=1, loc="lower right",
        bbox_to_anchor=(0.98, -0.33),
    )

    bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.5}
    for y, (label, text, color) in zip((0.37, 0.25, 0.13), annotation_lines):
        ax.text(
            0.04,
            y,
            f"{LABEL_DELTA[label]} = {text}",
            color=color,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=bbox,
        )

    return ax


def plot_climate_zone_decomposition(
    ax,
    df: pd.DataFrame,
    *,
    colors: Sequence[str],
    ylim=(-0.07, 0.07),
    ratio_digits: int = 0,
    show_legend: bool = True,
):
    """Plot regional trends as total/LAI/sensitivity dots with confidence intervals."""
    regions = df["region"].tolist()
    x = np.arange(len(regions))
    dx = 0.22
    scale = 10.0
    marker_size = 4.8

    color_map = {"total": colors[0], "lai": colors[1], "sens": colors[2]}

    for i, region in enumerate(regions):
        if region in {"Global", "Temperate", "Boreal"}:
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, facecolor="0.92", edgecolor="none", zorder=0)
    ax.axhline(0, linewidth=0.8, color="k", zorder=1)

    def values(prefix: str):
        slope = df[f"{prefix}_slope"].to_numpy() * scale
        ci_lo = df[f"{prefix}_ci_lo"].to_numpy() * scale
        ci_hi = df[f"{prefix}_ci_hi"].to_numpy() * scale
        yerr = np.vstack([slope - ci_lo, ci_hi - slope])
        pval = df[f"{prefix}_p"].to_numpy()
        return slope, yerr, pval

    def draw_points(xpos, yval, yerr, pval, color, label, zorder):
        ax.errorbar(
            xpos, yval, yerr=yerr, fmt="none", ecolor=color,
            elinewidth=0.9, capsize=2, alpha=0.9, zorder=zorder,
        )
        pval = np.asarray(pval)
        sig = np.isfinite(pval) & (pval < 0.05)
        nonsig = np.isfinite(pval) & (pval >= 0.05)
        missing = ~np.isfinite(pval)
        ax.plot(
            xpos[sig], yval[sig], "o", markerfacecolor=color,
            markeredgecolor=color, markersize=marker_size, linestyle="none",
            label=label, zorder=zorder + 1,
        )
        ax.plot(
            xpos[nonsig], yval[nonsig], "o", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.0,
            markersize=marker_size, linestyle="none", zorder=zorder + 1,
        )
        ax.plot(
            xpos[missing], yval[missing], "o", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=0.8, alpha=0.45,
            markersize=marker_size, linestyle="none", zorder=zorder + 1,
        )

    lai, lai_yerr, lai_p = values("lai")
    sens, sens_yerr, sens_p = values("sens")
    total, total_yerr, total_p = values("total")

    draw_points(x - dx, lai, lai_yerr, lai_p, color_map["lai"], LABEL_DELTA["LAI-driven"], 3)
    draw_points(x + dx, sens, sens_yerr, sens_p, color_map["sens"], LABEL_DELTA["Sens-driven"], 3)
    draw_points(x, total, total_yerr, total_p, color_map["total"], LABEL_DELTA["Total"], 4)

    ax.set_ylim(ylim)
    ax.set_yticks([-0.06,-0.03,0,0.03,0.06])
    ratio_y = ylim[1] - 0.10 * (ylim[1] - ylim[0])
    for i in range(len(regions)):
        if np.isfinite(lai[i]) and np.isfinite(sens[i]) and abs(lai[i]) > 1e-12:
            ratio = sens[i] / lai[i] * 100
            ratio_text = (
                f"–{abs(ratio):.{ratio_digits}f}%"
                if ratio < 0
                else f"{abs(ratio):.{ratio_digits}f}%"
            )
            ratio_color = "#b0202e" if ratio > 0 else "#6aa8cc"
        else:
            ratio_text = "NA"
            ratio_color = "0.35"
        ax.text(
            x[i], ratio_y, ratio_text, color=ratio_color, ha="center",
            va="center", fontsize=12, clip_on=False, zorder=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=12)
    ax.set_ylabel(r"Trend in LST$^{*}$ [K decade$^{-1}$]")
    ax.set_title("Decomposition by climate zone")
    ax.tick_params(direction="in")

    ax.add_patch(Rectangle(
        (-0.4, 0.048), 4.8, 0.018, facecolor="none", edgecolor="gray",
        linestyle="--", linewidth=1.0,
    ))
    ax.annotate(
        f"{LABEL_DELTA['Sens-driven']} / {LABEL_DELTA['LAI-driven']}",
        xy=(1.2, 0.048),
        xytext=(-0.5, 0.025),
        ha="left",
        va="bottom",
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "0.3", "lw": 0.8, "shrinkA": 0, "shrinkB": 0},
        zorder=5,
    )

    if show_legend:
        color_handles = [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=color_map["total"],
                markeredgecolor=color_map["total"], markersize=5,
                label=LABEL_DELTA["Total"],
            ),
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=color_map["lai"],
                markeredgecolor=color_map["lai"], markersize=5,
                label=LABEL_DELTA["LAI-driven"],
            ),
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=color_map["sens"],
                markeredgecolor=color_map["sens"], markersize=5,
                label=LABEL_DELTA["Sens-driven"],
            ),
        ]
        sig_handles = [
            Line2D(
                [0], [0], marker="o", color="black",
                markerfacecolor="black", markeredgecolor="black",
                linestyle="none", markersize=5, label="$p$<0.05",
            ),
            Line2D(
                [0], [0], marker="o", color="black",
                markerfacecolor="white", markeredgecolor="black",
                linestyle="none", markersize=5, label="$p$>=0.05",
            ),
        ]
        leg = ax.legend(handles=color_handles, frameon=False, ncol=3, handletextpad=0.45,
                        loc="lower left", bbox_to_anchor=(-0.02, -0.33))
        ax.add_artist(leg)
        ax.legend(
            handles=sig_handles, frameon=False, ncol=1, handlelength=0.8,
            handletextpad=0.45, loc="lower right", bbox_to_anchor=(1.0, 0),
        )

    return ax


def integer_percentages(counts: np.ndarray) -> np.ndarray:
    """Convert counts to integer percentages that sum exactly to 100."""
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        return np.zeros_like(counts, dtype=int)
    raw = counts / counts.sum() * 100
    pct = np.floor(raw).astype(int)
    order = np.argsort(-(raw - pct), kind="stable")
    pct[order[:100 - pct.sum()]] += 1
    return pct


def add_pdf_inset(ax, trend: xr.DataArray, *, cmap, norm,
                  inset_bbox=(0.07, 0.05, 0.22, 0.22), bins=20):
    """Draw a distribution directly on the map, without a child Axes."""
    values = np.asarray(trend.values)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return []

    vmin, vmax = norm.vmin, norm.vmax
    edges = np.linspace(vmin, vmax, bins + 1)
    density, _ = np.histogram(values, bins=edges, density=True)
    if not np.isfinite(density).any() or np.nanmax(density) <= 0:
        return []

    x0, y0, width, height = inset_bbox
    x_edges = x0 + width * (edges - vmin) / (vmax - vmin)
    heights = 0.72 * height * density / np.nanmax(density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    artists = []
    for left, right, h, center in zip(x_edges[:-1], x_edges[1:], heights, centers):
        bar = Rectangle(
            (left, y0), right - left, h, transform=ax.transAxes,
            facecolor=cmap(norm(center)), edgecolor="none", alpha=0.9,
            zorder=20, clip_on=False,
        )
        ax.add_patch(bar)
        artists.append(bar)

    curve_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    artists += ax.plot(curve_x, y0 + heights, color="k", lw=1,
                       transform=ax.transAxes, zorder=21, clip_on=False)
    artists += ax.plot([x0, x0 + width], [y0, y0], color="k", lw=1,
                       transform=ax.transAxes, zorder=21, clip_on=False)
    for x, value, color, ha in (
        (x0, np.mean(values < 0), "tab:blue", "left"),
        (x0 + width, np.mean(values > 0), "tab:red", "right"),
    ):
        artists.append(ax.text(
            x, y0 + 0.88 * height, f"{value * 100:.0f}%", color=color,
            transform=ax.transAxes, ha=ha, va="top", fontsize=11,
            zorder=22, clip_on=False,
        ))
    return artists


def plot_trend_map(
    ax,
    ds: xr.Dataset,
    *,
    title: str,
    cmap,
    norm,
    step: int = 3,
    add_inset: bool = False,
):
    """Plot a gridded Theil-Sen trend map with significance stippling."""
    trend = ds["trend_theilsen"] * 10.0
    trend.attrs = ds["trend_theilsen"].attrs.copy()
    trend.attrs["display_units"] = "K decade-1"
    pval = ds["trend_p"]
    trend.plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
        add_colorbar=False, rasterized=True,
    )
    ax.coastlines(linewidth=0.5, color="0.3")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, edgecolor="0.4")
    ax.set_title(title)
    ax.set_extent([-180, 180, -60, 90]) #, crs=ccrs.PlateCarree()

    sig_sparse = (pval < 0.05).isel(lat=slice(0, None, step), lon=slice(0, None, step))
    yy, xx = np.meshgrid(sig_sparse["lat"].values, sig_sparse["lon"].values, indexing="ij")
    mask = sig_sparse.values.astype(bool)
    ax.scatter(
        xx[mask], yy[mask], s=0.3, color="k", marker=".", alpha=0.7,
        transform=ccrs.PlateCarree(), zorder=10,
    )

    if add_inset:
        add_pdf_inset(ax, trend, cmap=cmap, norm=norm)
    return ax


def plot_dominance_map(ax, ds_lai: xr.Dataset, ds_sens: xr.Dataset, *, eps: float = 0.0):
    """Plot dominant driver of trend terms: LAI-driven or sensitivity-driven."""
    lai = ds_lai["trend_theilsen"]
    sens = ds_sens["trend_theilsen"]
    lai, sens = xr.align(lai, sens, join="inner")
    valid = np.isfinite(lai) & np.isfinite(sens)

    sens_dom = (np.abs(sens) > (1 + eps) * np.abs(lai)) & valid
    lai_dom = (np.abs(lai) > (1 + eps) * np.abs(sens)) & valid

    cat = xr.full_like(lai, np.nan, dtype=np.float32)
    cat = xr.where(lai_dom & (lai < 0), 0, cat)
    cat = xr.where(lai_dom & (lai >= 0), 1, cat)
    cat = xr.where(sens_dom & (sens >= 0), 2, cat)
    cat = xr.where(sens_dom & (sens < 0), 3, cat)

    colors = ["#4682b4", "#87ceeb", "#ffd700", "#f08080"]
    labels = ["LAI –", "LAI +", "EFF +", "EFF –"]
    cmap = ListedColormap(colors)
    cmap.set_bad("none")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    im = ax.pcolormesh(
        cat["lon"], cat["lat"], cat, transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, shading="auto", rasterized=True,
    )
    ax.coastlines(linewidth=0.5, color="0.3")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3, edgecolor="0.4")
    ax.set_title("Dominance of trend terms")
    ax.set_extent([-180, 180, -60, 90]) #, crs=ccrs.PlateCarree()

    counts = np.array([int((cat == k).sum().values) for k in range(4)], dtype=float)
    pct = integer_percentages(counts)

    # SVG-safe categorical key drawn in parent-axis coordinates.
    x0, y0, width, height = (0, 0.04, 0.39, 0.39)
    background = Rectangle(
        (x0, y0), width, height, transform=ax.transAxes,
        facecolor="none", edgecolor="none", alpha=0.95,
        zorder=20, clip_on=False,
    )
    ax.add_patch(background)
    for i, (color, label) in enumerate(zip(colors, labels)):
        yc = y0 + height * (0.85 - 0.24 * i)
        bar = Rectangle(
            (x0 + 0.35 * width, yc - 0.05 * height),
            0.08 * width, 0.14 * height, transform=ax.transAxes,
            facecolor=color, edgecolor="none", zorder=21, clip_on=False,
        )
        ax.add_patch(bar)
        ax.text(x0 + 0.32 * width, yc, f"{pct[i]}%", transform=ax.transAxes,
                va="center", ha="right", fontsize=12, color="0.15",
                zorder=22, clip_on=False)
        ax.text(x0 + 0.46 * width, yc, label, transform=ax.transAxes,
                va="center", ha="left", fontsize=12, color="0.15",
                zorder=22, clip_on=False)

    return im


def move_axis(ax, *, dx=0.0, dy=0.0, dw=0.0, dh=0.0):
    """Move or resize an axis in figure coordinates."""
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width + dw, pos.height + dh])
    return ax


def add_colorbar_near_axis(
    ref_ax,
    fig,
    *,
    cmap,
    norm,
    ticks,
    ticklabels,
    orientation="vertical",
    label="",
):
    """Add a colorbar positioned relative to another axis."""
    pos = ref_ax.get_position()
    if orientation == "horizontal":
        bounds = [pos.x0 + 0.15, pos.y0 + 0.015, pos.width / 3, 0.008]
    else:
        bounds = [pos.x0 + 0.03, pos.y0 + 0.01, 0.01, 0.07]
    cbar = mpl.colorbar.ColorbarBase(
        fig.add_axes(bounds), cmap=cmap, norm=norm, orientation=orientation,
        extend="both", format=mticker.FuncFormatter(lambda x, _: f"{x:g}"),
    )
    cbar.set_ticks(ticks, labels=ticklabels)
    if label:
        cbar.set_label(label, rotation=270 if orientation == "vertical" else 0,
                       labelpad=15)
    cbar.ax.tick_params(direction="in", length=3, width=1, color="white")
    cbar.outline.set_edgecolor("white")
    return cbar


def build_figure(cfg: FigureConfig) -> plt.Figure:
    """Create and save Figure 5."""
    data = load_inputs(cfg)
    df_region = data["df_region"]
    region_stats = build_region_slope_table(df_region, regions=cfg.regions)

    plot_settings()
    fig = plt.figure(figsize=(12, 11.5))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=(4, 4, 4),
        width_ratios=(5, 5), wspace=0.10, hspace=0.18,
    )

    cmap_trend, norm_trend = get_truncated_cs(
        "RdBu_r", skip_middle=0.15,
        vmin=cfg.trend_vmin, vmax=cfg.trend_vmax,
    )

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_total = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
    ax_dom = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
    ax_lai = fig.add_subplot(gs[2, 0], projection=ccrs.Robinson())
    ax_sens = fig.add_subplot(gs[2, 1], projection=ccrs.Robinson())

    plot_global_timeseries(ax_ts, df_region, colors=cfg.colors)
    plot_climate_zone_decomposition(
        ax_bar, region_stats, colors=cfg.colors,
        ylim=(-0.07, 0.07), show_legend=True,
    )

    titles = {
        "total": f"{LABEL_DELTA['Total']} [K decade$^{{-1}}$]",
        "lai": f"{LABEL_DELTA['LAI-driven']} [K decade$^{{-1}}$]",
        "sens": f"{LABEL_DELTA['Sens-driven']} [K decade$^{{-1}}$]",
    }
    plot_trend_map(
        ax_total, data["ds_total"], title=titles["total"],
        cmap=cmap_trend, norm=norm_trend, step=3,
    )
    plot_dominance_map(ax_dom, data["ds_lai"], data["ds_sens"])
    plot_trend_map(
        ax_lai, data["ds_lai"], title=titles["lai"],
        cmap=cmap_trend, norm=norm_trend, step=3,
    )
    plot_trend_map(
        ax_sens, data["ds_sens"], title=titles["sens"],
        cmap=cmap_trend, norm=norm_trend, step=3,
    )

    move_axis(ax_lai, dy=0.05)
    move_axis(ax_sens, dy=0.05)

    ticks = [-0.06, -0.03, 0.0, 0.03, 0.06]
    ticklabels = ["–0.06", "–0.03", "0", "0.03", "0.06"]
    for ax in (ax_total, ax_lai, ax_sens):
        add_colorbar_near_axis(ax, fig, cmap=cmap_trend, norm=norm_trend, ticks=ticks,
                               ticklabels=ticklabels, orientation="vertical")

    ax_ts.set_position([ax_total.get_position().x0 + 0.05,
                        ax_ts.get_position().y0 + 0.02,
                        ax_total.get_position().width - 0.05,
                        ax_ts.get_position().height - 0.05])
    ax_bar.set_position([ax_bar.get_position().x0 + 0.05,
                         ax_bar.get_position().y0 + 0.02,
                         ax_bar.get_position().width - 0.05,
                         ax_bar.get_position().height - 0.05])

    for ax, label, x in [
        (ax_ts, "a", -0.14),
        (ax_bar, "b", -0.14),
        (ax_total, "c", 0.02),
        (ax_dom, "d", 0.02),
        (ax_lai, "e", 0.02),
        (ax_sens, "f", 0.02),
    ]:
        ax.text(
            x, 1.1, label, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="right",
        )

    output = (
        cfg.output_figure
        if cfg.output_figure.is_absolute()
        else cfg.base_dir / cfg.output_figure
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    log(f"Saved figure: {output.resolve()}")
    return fig


def main() -> None:
    cfg = FigureConfig()
    build_figure(cfg)


if __name__ == "__main__":
    main()
