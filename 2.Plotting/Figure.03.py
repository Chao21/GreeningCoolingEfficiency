#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Create Figure 3: radiative and nonradiative LST-sensitivity changes.

Author
------
Chao Zhang, National University of Singapore

Date
----
2026-09-05

Purpose
-------
Compare temporal and spatial changes in the radiative and nonradiative
components of LST sensitivity to LAI across annual and seasonal scales.

Notes
-----
Regional trends use ordinary least-squares regression and are reported per
decade. Map stippling denotes p < 0.05, and distribution insets are drawn
directly in parent-axis coordinates for reliable SVG export. Set
``VEG_LST_DATA_DIR`` and ``VEG_LST_UTILS_DIR`` to override default paths.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.stats import linregress


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

from plot_utils import get_truncated_cs, plot_settings


@dataclass(frozen=True)
class FigureConfig:
    root: Path = (
        DATA_ROOT / "processed/Sensitivity_LST_Energy_LAI_20260325"
    )
    output_figure: Path = DATA_ROOT.parent / "figure/Fig.03_Decomposition_Energy.svg"

    lst_var: str = "dailymean"

    regions: tuple[str, ...] = ("Global", "Tropical", "Temperate", "Arid", "Boreal")
    seasons: tuple[str, ...] = ("annual", "spring", "summer", "autumn", "winter")
    colors: tuple[str, ...] = ("#000000", "#cb2f2d", "#436aab")

    trend_vmin: float = -0.12
    trend_vmax: float = 0.12


LABEL_DELTA = {
    "Radiative": r"$\delta(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{radiative}}$",
    "Non-Radiative": (
        r"$\delta(\partial\mathrm{LST}/\partial\mathrm{LAI})_"
        r"{\mathrm{nonradiative}}$"
    ),
}
LABEL_SENS = {
    "Radiative": r"$(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{radiative}}$",
    "Non-Radiative": r"$(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{nonradiative}}$",
}

def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def input_paths(cfg: FigureConfig) -> dict[str, Path]:
    """Build all input paths used by Figure 3."""
    root, lst = cfg.root, cfg.lst_var

    return {
        "rad_annual_csv": root / "Albedo/Annual" / (
            f"Sensitivity_Annual_LST{lst}_LAI_from_Albedo_"
            "regionAgg_gapfilled.csv"
        ),
        "rad_season_csv": root / "Albedo/Season" / (
            f"Sensitivity_Season_LST{lst}_LAI_from_Albedo_"
            "regionAgg_gapfilled_NSharmonized.csv"
        ),
        "rad_trend_nc": root / "Albedo/Annual" / (
            f"Sens_trend_Annual_LST{lst}_LAI_from_Albedo_1d_gapfilled.nc"
        ),
        "nonrad_annual_csv": root / "LEH/Annual" / (
            f"Sensitivity_Annual_LST{lst}_LAI_from_LEH_FAO56_regionAgg.csv"
        ),
        "nonrad_season_csv": root / "LEH/Season" / (
            f"Sensitivity_Season_LST{lst}_LAI_from_LEH_FAO56_"
            "regionAgg_NSharmonized.csv"
        ),
        "nonrad_trend_nc": root / "LEH/Annual" / (
            f"Sens_trend_Annual_LST{lst}_LAI_from_LEH_FAO56_1d.nc"
        ),
    }


def load_inputs(cfg: FigureConfig) -> dict[str, object]:
    """Load annual/seasonal regional time series and gridded trend datasets."""
    paths = input_paths(cfg)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))

    return {
        "df_rad_annual": pd.read_csv(paths["rad_annual_csv"], parse_dates=["time"]),
        "df_rad_season": pd.read_csv(paths["rad_season_csv"], parse_dates=["time"]),
        "df_nonrad_annual": pd.read_csv(paths["nonrad_annual_csv"], parse_dates=["time"]),
        "df_nonrad_season": pd.read_csv(paths["nonrad_season_csv"], parse_dates=["time"]),
        "rad_trend": xr.load_dataset(paths["rad_trend_nc"]),
        "nonrad_trend": xr.load_dataset(paths["nonrad_trend_nc"]),
    }


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


def linear_trend(time_values, values):
    """Return OLS slope, p value, intercept, and standard error per year."""
    x = decimal_year(time_values)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return (np.nan,) * 4

    lr = linregress(x[valid], y[valid])
    return lr.slope, lr.pvalue, lr.intercept, lr.stderr


def p_to_star(pval: float) -> str:
    if not np.isfinite(pval):
        return ""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def get_region_series(
    df: pd.DataFrame,
    region: str = "Global",
    group: str = "annual",
) -> pd.DataFrame:
    """Filter a long regional time-series table to one region and one group."""
    out = df[(df["group"] == group) & (df["region"] == region)].copy()
    return out.sort_values("time")


def build_region_slope_table(
    df_rad: pd.DataFrame,
    df_nonrad: pd.DataFrame,
    *,
    regions: Sequence[str],
    group: str = "annual",
) -> pd.DataFrame:
    """Compute regional radiative and non-radiative trends for one time group."""
    rows = []

    for region in regions:
        rad = get_region_series(df_rad, region, group)
        nonrad = get_region_series(df_nonrad, region, group)

        s_rad, p_rad, *_ = linear_trend(rad["time"], rad["value"])
        s_nonrad, p_nonrad, *_ = linear_trend(nonrad["time"], nonrad["value"])

        rows.append(
            {
                "region": region,
                "radiative_slope": s_rad,
                "radiative_p": p_rad,
                "non_radiative_slope": s_nonrad,
                "non_radiative_p": p_nonrad,
            }
        )

    return pd.DataFrame(rows)


def plot_global_trend(
    ax,
    df: pd.DataFrame,
    *,
    ylabel: str,
    title: str,
    color: str,
    ylim: tuple[float, float] | None = None,
):
    """Plot one global component time series with linear trend and annotation."""
    df = df.sort_values("time")
    time = pd.DatetimeIndex(df["time"])
    x = decimal_year(time)
    y = df["value"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    sns.regplot(
        x=x[valid],
        y=y[valid],
        ax=ax,
        scatter_kws={"s": 12, "alpha": 1.0, "color": color},
        line_kws={"color": color, "lw": 1.5, "ls": "-"},
        ci=95,
        truncate=True,
    )
    ax.plot(x, y, linestyle="dashdot", linewidth=1.5, color=color, alpha=0.3)

    slope, pval, _, stderr = linear_trend(time, y)
    if np.isfinite(slope):
        slope_text = f"–{abs(slope) * 10:.3f}" if slope<0 else f"{slope * 10:.3f}"
        p_text = "$p<0.001$" if pval < 0.001 else f"$p={pval:.3f}$"
        ax.text(0.02, 0.97, f"Slope = {slope_text} decade$^{{-1}}$, {p_text}",
            color=color, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, bbox={"facecolor": "none", "edgecolor": "none", "alpha": 0.55},
        )
        print(f"{title}: slope={slope * 10:.4f}, p={pval:.4f}, stderr={stderr * 10:.4f}")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.tick_params(axis="both", direction="in",labelsize=12)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


def plot_climate_zone_bars(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    colors: dict[str, str],
    ylim: tuple[float, float] = (-0.3, 0.3),
    show_ylabel: bool = False,
    anno_flag: bool = False
):
    """Plot radiative and non-radiative regional trends as overlaid bars."""
    regions = df["region"].tolist()
    x = np.arange(len(regions))
    width = 0.6
    scale = 10.0

    radiative = df["radiative_slope"].to_numpy() * scale
    nonrad = df["non_radiative_slope"].to_numpy() * scale

    ax.axhline(0, linewidth=0.8, color="k")
    ax.bar(
        x, radiative, width=width, color=colors["radiative"], alpha=0.70,
        label=LABEL_DELTA["Radiative"], zorder=2,
    )
    ax.bar(
        x, nonrad, width=width, color=colors["non_radiative"], alpha=0.90,
        label=LABEL_DELTA["Non-Radiative"], zorder=2,
    )

    offset = 0.005 * (ylim[1] - ylim[0])
    for i in range(len(regions)):
        for value, pval in [
            (radiative[i], df["radiative_p"].iloc[i]),
            (nonrad[i], df["non_radiative_p"].iloc[i]),
        ]:
            star = p_to_star(pval)
            if star:
                ax.text(
                    x[i],
                    value + offset*0.5 if value >= 0 else value - 4 * offset,
                    star,
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=12,
                )

        if np.isfinite(radiative[i]) and np.isfinite(nonrad[i]) and abs(nonrad[i]) > 1e-12:
            ratio = radiative[i] / nonrad[i] * 100
            ratio_text = f"–{abs(ratio):.0f}%" if ratio<0 else f"{abs(ratio):.0f}%"
        else:
            ratio_text = "NA"
        ax.text(
            x[i], ylim[1] - 0.01, ratio_text, ha="center", va="top",
            color="0.65", rotation=90, fontsize=12,
        )

    if anno_flag:
        ax.add_patch(Rectangle(
            (-0.4, 0.23), 4.8, 0.12, facecolor="none", edgecolor="gray",
            linestyle="--", linewidth=1.0,
        ))
        ax.annotate(
            "radiative / nonradiative",
            xy=(1.2, 0.23),
            xytext=(-0.4, 0.16),
            ha="left",
            va="bottom",
            fontsize=12,
            arrowprops={"arrowstyle": "-", "color": "0.3", "lw": 0.8, "shrinkA": 0, "shrinkB": 0},
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(1, len(regions) + 1)], fontsize=12)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.tick_params(direction="in")

    if show_ylabel:
        ax.set_ylabel(r"K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$")
        ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2], ["-0.2", "-0.1", "0", "0.1", "0.2"])
        for i, region in enumerate(regions):
            ax.text(
                x[i], ylim[0] + 0.02, region, ha="center", va="bottom",
                rotation=90, fontsize=12,
            )
    else:
        ax.set_yticklabels([])

    return ax


def add_pdf_inset(
    ax,
    trend: xr.DataArray,
    *,
    cmap,
    norm,
    inset_bbox=(0.065, 0.06, 0.25, 0.26),
    bins: int = 20,
):
    """Draw a histogram directly on the map without creating an inset Axes."""
    values = np.asarray(trend.values)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return []

    p_neg = np.mean(values < 0) * 100
    p_pos = np.mean(values > 0) * 100
    vmin = getattr(norm, "vmin", np.nanmin(values))
    vmax = getattr(norm, "vmax", np.nanmax(values))
    edges = np.linspace(vmin, vmax, bins + 1)
    density, _ = np.histogram(values, bins=edges, density=True)
    if not np.isfinite(density).any() or np.nanmax(density) <= 0:
        return []

    x0, y0, width, height = inset_bbox
    x_edges = x0 + width * (edges - vmin) / (vmax - vmin)
    bar_heights = 0.72 * height * density / np.nanmax(density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    artists = []

    for left, right, bar_height, center in zip(
        x_edges[:-1], x_edges[1:], bar_heights, centers
    ):
        patch = Rectangle(
            (left, y0), right - left, bar_height,
            transform=ax.transAxes, facecolor=cmap(norm(center)),
            edgecolor="none", alpha=0.9, zorder=20, clip_on=False,
        )
        ax.add_patch(patch)
        artists.append(patch)

    zero_x = x0 + width * (0 - vmin) / (vmax - vmin)
    artists += ax.plot(
        [x0, x0 + width], [y0, y0], color="k", lw=1,
        transform=ax.transAxes, zorder=21, clip_on=False,
    )
    artists += ax.plot(
        [zero_x, zero_x], [y0, y0 + 0.78 * height],
        color="k", ls="--", lw=0.6, transform=ax.transAxes,
        zorder=21, clip_on=False,
    )
    artists += [
        ax.text(x0+0.04, y0 + 0.82 * height, f"{p_neg:.0f}%", color="tab:blue",
                transform=ax.transAxes, ha="left", va="top", fontsize=12, zorder=22),
        ax.text(x0 + width-0.04, y0 + 0.82 * height, f"{p_pos:.0f}%", color="tab:red",
                transform=ax.transAxes, ha="right", va="top", fontsize=12, zorder=22),
    ]
    return artists


def plot_map_slope(
    ax,
    ds: xr.Dataset,
    *,
    cmap,
    norm,
    title: str,
    significance_level: float = 0.05,
    stipple_step: int = 2,
):
    """Plot gridded trend map with significance stippling and PDF inset."""
    trend = ds["trend_theilsen"]
    pval = ds["trend_p"]

    trend.plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
        add_colorbar=False, rasterized=True,
    )
    ax.set_extent([-180, 180, -60, 90])
    ax.coastlines(edgecolor="gray", linewidth=0.5)
    ax.set_title(title)

    sig_sparse = (pval < significance_level).isel(
        lat=slice(0, None, stipple_step),
        lon=slice(0, None, stipple_step),
    )
    lat = sig_sparse["lat"].values
    lon = sig_sparse["lon"].values
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    mask = np.asarray(sig_sparse.values, dtype=bool)
    ax.scatter(
        xx[mask],
        yy[mask],
        s=8,
        color="k",
        marker=".",
        edgecolor="none",
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    add_pdf_inset(ax, trend, cmap=cmap, norm=norm)
    return ax



def move_axis(ax, *, dx=0.0, dy=0.0, dw=0.0, dh=0.0):
    """Move or resize an axis in figure coordinates."""
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width + dw, pos.height + dh])
    return ax


def build_figure(cfg: FigureConfig) -> plt.Figure:
    """Create and save Figure 3."""
    data = load_inputs(cfg)
    colors = {"total": cfg.colors[0], "radiative": cfg.colors[1], "non_radiative": cfg.colors[2]}

    cmap_trend, norm_trend = get_truncated_cs(
        "RdYlBu_r",
        skip_middle=0.15,
        vmin=cfg.trend_vmin,
        vmax=cfg.trend_vmax,
    )

    region_stats = {
        season: build_region_slope_table(
            data["df_rad_annual"] if season == "annual" else data["df_rad_season"],
            data["df_nonrad_annual"] if season == "annual" else data["df_nonrad_season"],
            regions=cfg.regions,
            group=season,
        )
        for season in cfg.seasons
    }

    plot_settings()
    fig = plt.figure(figsize=(12, 10.5))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=(0.75, 1.15, 1.1),
        wspace=0.16, hspace=0.4,
    )

    ax_rad_ts = fig.add_subplot(gs[0, 0])
    ax_nonrad_ts = fig.add_subplot(gs[0, 1])
    ax_rad_map = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
    ax_nonrad_map = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())

    gs_bars = gs[2, 0:2].subgridspec(1, 5, wspace=0.08)
    bar_axes = [fig.add_subplot(gs_bars[0, i]) for i in range(5)]

    ylabel = r"K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$"
    plot_global_trend(
        ax_rad_ts, get_region_series(data["df_rad_annual"], "Global"),
        ylabel=ylabel, title=LABEL_SENS["Radiative"],
        color=colors["radiative"], ylim=(0.68, 1.03),
    )
    plot_global_trend(
        ax_nonrad_ts, get_region_series(data["df_nonrad_annual"], "Global"),
        ylabel="", title=LABEL_SENS["Non-Radiative"],
        color=colors["non_radiative"], ylim=(-1.43, -1.05),
    )

    plot_map_slope(
        ax_rad_map, data["rad_trend"], cmap=cmap_trend, norm=norm_trend,
        title=LABEL_DELTA["Radiative"],
    )
    plot_map_slope(
        ax_nonrad_map, data["nonrad_trend"], cmap=cmap_trend,
        norm=norm_trend, title=LABEL_DELTA["Non-Radiative"],
    )
    move_axis(ax_rad_map, dx=-0.025, dy=-0.01, dw=0.05, dh=0.045)
    move_axis(ax_nonrad_map, dx=-0.025, dy=-0.01, dw=0.05, dh=0.045)

    season_titles = ("Annual", "Spring", "Summer", "Autumn", "Winter")
    for ax, season, title in zip(bar_axes, cfg.seasons, season_titles):
        plot_climate_zone_bars(
            ax,
            region_stats[season],
            title=title,
            colors=colors,
            ylim=(-0.3, 0.35),
            show_ylabel=(season == "annual"),
            anno_flag=(title == "Winter"),
        )

    sm = mpl.cm.ScalarMappable(norm=norm_trend, cmap=cmap_trend)
    cbar = fig.colorbar(
        sm, ax=[ax_rad_map, ax_nonrad_map], orientation="horizontal",
        fraction=0.045, pad=0.06, aspect=45, extend="both",
    )
    cbar.set_label(r"K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$", fontsize=12)

    handles, labels = bar_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.05),
        frameon=False, ncol=2, columnspacing=1.2, handletextpad=0.6,
    )

    panel_axes = [ax_rad_ts, ax_nonrad_ts, ax_rad_map, ax_nonrad_map, *bar_axes]
    for ax, label in zip(panel_axes, list("abcdefghi")):
        x = 0.0
        ax.text(x, 1.02, label, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom", ha="left")

    output = (
        cfg.output_figure
        if cfg.output_figure.is_absolute()
        else cfg.base_dir / cfg.output_figure
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    log(f"Saved figure: {output.resolve()}")
    return fig


def main() -> None:
    cfg = FigureConfig()
    build_figure(cfg)


if __name__ == "__main__":
    main()
