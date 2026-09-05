#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Create Figure 1: regional LST-LAI sensitivity time series and seasonal trends.

Author
------
Chao Zhang, National University of Singapore

Date
----
2026-09-05

Purpose
-------
Plot annual LST-LAI sensitivity for five climate regions and summarize the
corresponding seasonal means and trends in compact inset panels.

Notes
-----
Regional trends are estimated using ordinary least-squares regression against
decimal year and reported per decade. Set ``VEG_LST_DATA_DIR`` and
``VEG_LST_UTILS_DIR`` to override the default project directories.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
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
    base_dir: Path = DATA_ROOT
    product: str = "GLASS"
    time_scale: str = "Annual"
    lst_var: str = "dailymean"
    output_figure: Path = DATA_ROOT.parent / "figure/Fig.01_PixelTrend_Annual.svg"

    regions: tuple[str, ...] = ("Global", "Tropical", "Temperate", "Arid", "Boreal")
    seasons: tuple[str, ...] = ("spring", "summer", "autumn", "winter")
    sensitivity_ylim: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "Global": (-0.63, -0.30),
            "Tropical": (-0.52, -0.37),
            "Temperate": (-0.52, -0.37),
            "Arid": (-1.77, -1.09),
            "Boreal": (0.17, 0.43),
        }
    )
    trend_vmin: float = -0.09
    trend_vmax: float = 0.09


def sensitivity_paths(cfg: FigureConfig) -> dict[str, Path]:
    prefix = f"Sensitivity_{cfg.time_scale}_LST{cfg.lst_var}_LAI_{cfg.product}"
    season_prefix = f"Sensitivity_Season_LST{cfg.lst_var}_LAI_{cfg.product}"
    root = cfg.base_dir / "processed/Sensitivity_20260208" / cfg.product / "MOD11C3"

    return {
        "annual_region": root / f"{prefix}_regionAgg.csv",
        "season_region": root / f"{season_prefix}_regionAgg_NSharmonized.csv",
    }


def load_inputs(cfg: FigureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the data used by the figure."""
    paths = sensitivity_paths(cfg)
    df_annual = pd.read_csv(paths["annual_region"], parse_dates=["time"])
    df_season = pd.read_csv(paths["season_region"], parse_dates=["time"])
    return df_annual, df_season


def select_region_group(df: pd.DataFrame, *, group: str, region: str) -> pd.DataFrame:
    """Select one region and one temporal group from the regional time-series table."""
    out = df[(df["group"] == group) & (df["region"] == region)].copy()
    return out.sort_values("time")


def p_to_star(pval: float) -> str:
    """Return a compact significance marker."""
    if not np.isfinite(pval):
        return ""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


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


def linear_trend(time_values: Sequence[pd.Timestamp], values: Sequence[float]):
    """
    Linear trend against decimal year.

    Returns
    -------
    slope, pvalue, intercept, rvalue, stderr
        Slope is in data units per year.
    """
    x = decimal_year(time_values)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    lr = linregress(x[valid], y[valid])
    return lr.slope, lr.pvalue, lr.intercept, lr.rvalue, lr.stderr


def summarize_mean_trend_by_region(
    df: pd.DataFrame,
    *,
    regions: Iterable[str],
    groups: Iterable[str],
) -> pd.DataFrame:
    """
    Summarize mean sensitivity and linear trend for each region and season.

    The output slope is converted from per-year to per-decade.
    """
    rows = []

    for group in groups:
        for region in regions:
            sub = select_region_group(df, group=group, region=region)

            if sub.empty:
                rows.append(
                    {
                        "group": group,
                        "region": region,
                        "mean": np.nan,
                        "slope": np.nan,
                        "pval": np.nan,
                        "intercept": np.nan,
                        "star": "",
                    }
                )
                continue

            slope, pval, intercept, _, _ = linear_trend(sub["time"], sub["value"])
            rows.append(
                {
                    "group": group,
                    "region": region,
                    "mean": float(sub["value"].mean()),
                    "slope": slope * 10.0,
                    "pval": pval,
                    "intercept": intercept,
                    "star": p_to_star(pval),
                }
            )

    return pd.DataFrame(rows)


def marker_size_from_abs_mean(
    value: float,
    *,
    bins: Sequence[float] = (0, 0.5, 1.0, 1.5, 2.0, np.inf),
    sizes: Sequence[float] = (100, 160, 230, 310, 400),
) -> float:
    """Map absolute mean sensitivity to marker area."""
    if not np.isfinite(value):
        return sizes[0]

    abs_value = abs(value)
    for i, size in enumerate(sizes):
        if bins[i] <= abs_value < bins[i + 1]:
            return size
    return sizes[-1]


def plot_sensitivity_timeseries(
    ax,
    df_region: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    color: str = '0.2',
    ylim: tuple[float, float] | None = None,
):
    """Plot annual regional sensitivity time series with linear trend."""
    if df_region.empty:
        raise ValueError(f"No data available for panel: {title}")

    df_region = df_region.sort_values("time").copy()
    time = pd.DatetimeIndex(df_region["time"])
    x = decimal_year(time)
    y = df_region["value"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() >= 3:
        sns.regplot(
            x=x[valid],
            y=y[valid],
            ax=ax,
            scatter_kws={"s": 12, "alpha": 0.65, "color": color},
            line_kws={"color": 'k', "lw": 1.5, "ls": "-"},
            ci=95,
            truncate=True,
        )

        slope, pval, intercept, _, stderr = linear_trend(time, y)
        y_fit = intercept + slope * x
        # ax.plot(x, y_fit, color=color, lw=1.8)
        y_mean = np.nanmean(y)
        proportion = np.abs(slope * 10 / y_mean * 100)
        _slope = f"–{abs(slope)}" if slope<0 else slope
        trend_text = (
            f"Trend = {_slope * 10:.3f} ({proportion:.1f}%) "
            f"decade$^{{-1}}${p_to_star(pval)}"
        )
    else:
        trend_text = "Trend = NA"

    ax.plot(x, y, color=color, lw=1.2, ls="dashdot", alpha=0.8)

    y_mean = np.nanmean(y)
    ax.text(
        0.02,
        0.94,
        (
            f"Climatological mean = –{abs(y_mean):.3f}"
            if y_mean < 0
            else f"Climatological mean = {y_mean:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=12,
        fontweight='normal',
        va="top",
        ha="left",
    )
    ax.text(
        0.02,
        0.80,
        trend_text,
        transform=ax.transAxes,
        fontsize=12,
        color='#c3282c',
        fontweight='normal',
        va="top",
        ha="left",
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.tick_params(axis="both", direction="in", length=3, width=1)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel('')

    return ax


def plot_region_season_box(
    ax_ref,
    df_summary: pd.DataFrame,
    *,
    region: str,
    cmap,
    norm,
    season_order: Sequence[str] = ("spring", "summer", "autumn", "winter"),
    show_ns: bool = False,
    show_size_legend: bool = False,
):
    """
    Add a 2 x 2 seasonal inset to a regional time-series panel.

    Marker direction:
        upward triangle   = positive mean sensitivity
        downward triangle = negative mean sensitivity

    Marker color:
        seasonal trend.

    Marker size:
        absolute mean sensitivity.
    """
    bounds = [0.62, 0.02, 0.6, 0.6] if region == "Global" else [0.65, 0.02, 0.46, 0.46]
    ax = ax_ref.inset_axes(bounds)
    ax.patch.set_alpha(0)

    df_region = df_summary[df_summary["region"].str.lower() == region.lower()].copy()
    if df_region.empty:
        raise ValueError(f"No seasonal summary found for region: {region}")

    positions = {
        "spring": (1, 1),
        "summer": (0, 1),
        "autumn": (0, 0),
        "winter": (1, 0),
    }

    for season in season_order:
        row = df_region[df_region["group"] == season]
        if row.empty:
            raise ValueError(f"Missing {season} summary for region: {region}")

        row = row.iloc[0]
        x0, y0 = positions[season]

        ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor="none", edgecolor="none", linewidth=0.1))

        mean_value = row["mean"]
        marker = "^" if mean_value >= 0 else "v"
        marker_size = marker_size_from_abs_mean(mean_value)

        ax.scatter(
            x0 + 0.5,
            y0 + 0.5,
            s=marker_size,
            marker=marker,
            c=[row["slope"]],
            cmap=cmap,
            norm=norm,
            edgecolor="k",
            linewidth=0.6,
            zorder=3,
        )

        label = row["star"] if row["star"] else ("n.s." if show_ns else "")
        if label:
            ax.text(x0 + 0.5, y0 + 0.25, label, ha="center", va="bottom", fontsize=12, zorder=4)

        if region == "Global":
            ax.text(x0 + 0.5, y0, season, ha="center", va="bottom", fontsize=12)

    if show_size_legend:
        size_values = (100, 160, 230, 310, 400)
        size_labels = ("<0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">2.0")
        handles = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="white",
                markeredgecolor="0.15",
                markersize=np.sqrt(size),
                linestyle="None",
                label=label,
            )
            for size, label in zip(size_values, size_labels)
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(-1.5, -0.5),
            ncol=len(handles),
            frameon=False,
            handletextpad=0.2,
            columnspacing=0.4,
            borderaxespad=0.0,
        )
        ax.text(
            -1.5,
            -1.12,
            r"Seasonal |mean| [K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$]",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([1, 1], [0, 2], color="0.2", lw=0.5)
    ax.plot([0, 2], [1, 1], color="0.2", lw=0.5)

    # for spine in ax.spines.values():
    #     spine.set_linewidth(0.5)
    #     spine.set_color("0")

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def add_colorbar(
    ax,
    *,
    cmap,
    norm,
    ticks: Sequence[float],
    ticklabels: Sequence[str],
    orientation: str = "horizontal",
    label: str | None = None,
):
    """Add a compact colorbar to an explicitly provided axis."""
    colorbar = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation=orientation,
        extend="both",
        format=mticker.FuncFormatter(lambda x, _: f"{x:g}"),
    )
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(ticklabels)

    if label:
        if orientation == "vertical":
            colorbar.set_label(label, rotation=270, labelpad=35)
        else:
            colorbar.set_label(label)

    colorbar.ax.tick_params(direction="in", length=3, width=1, color="white")
    colorbar.outline.set_edgecolor("white")
    return colorbar


def build_figure(cfg: FigureConfig) -> plt.Figure:
    """Create and save the final figure."""
    df_annual, df_season = load_inputs(cfg)

    df_all = pd.concat([df_annual, df_season], ignore_index=True)
    df_summary = summarize_mean_trend_by_region(df_all, regions=cfg.regions, groups=cfg.seasons)

    plot_settings()
    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.18, hspace=0.35)

    axes = {
        "Global": fig.add_subplot(gs[0, 0:2]),
        "Tropical": fig.add_subplot(gs[1, 0]),
        "Temperate": fig.add_subplot(gs[1, 1]),
        "Arid": fig.add_subplot(gs[2, 0]),
        "Boreal": fig.add_subplot(gs[2, 1]),
    }

    cmap_trend, norm_trend = get_truncated_cs(
        "RdYlBu_r",
        skip_middle=0.15,
        vmin=cfg.trend_vmin,
        vmax=cfg.trend_vmax,
    )

    ylabel = r"$\partial$LST / $\partial$LAI [K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$]"

    panel_letters = ("a", "b", "c", "d", "e")
    for region, letter in zip(cfg.regions, panel_letters):
        ax = axes[region]
        df_region = select_region_group(df_annual, group="annual", region=region)

        plot_sensitivity_timeseries(
            ax,
            df_region,
            title=region,
            ylabel=ylabel if region in ("Global", "Tropical", "Arid") else "",
            color='k',
            ylim=cfg.sensitivity_ylim[region],
        )

        plot_region_season_box(
            ax,
            df_summary,
            region=region,
            cmap=cmap_trend,
            norm=norm_trend,
            show_size_legend=(region == "Arid"),
        )

        ax.text(
            -0.01,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )

    # Colorbar below the Boreal panel.
    ax_boreal = axes["Boreal"]
    cax = fig.add_axes(
        [
            ax_boreal.get_position().x0 + 0.01,
            ax_boreal.get_position().y0 - 0.055,
            ax_boreal.get_position().width - 0.02,
            0.014,
        ]
    )
    ticks = [-0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09]
    add_colorbar(
        cax,
        cmap=cmap_trend,
        norm=norm_trend,
        ticks=ticks,
        ticklabels=[f"{tick:.2f}".rstrip("0").rstrip(".") if tick else "0" for tick in ticks],
        orientation="horizontal",
        label=r"Seasonal trend [K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$]",
    )

    cfg.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output_figure, bbox_inches="tight")
    print(f"Saved figure: {cfg.output_figure}")
    return fig


def main() -> None:
    cfg = FigureConfig()
    build_figure(cfg)


if __name__ == "__main__":
    main()
