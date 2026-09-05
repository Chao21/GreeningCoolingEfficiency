#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Create Figure 2: spatial patterns of LST-LAI sensitivity and its trend.

Author
------
Chao Zhang, National University of Singapore

Date
----
2026-09-05

Purpose
-------
Map the mean and temporal trend of LST-LAI sensitivity, summarize their
climate-zone distributions, and relate both fields to climate space.

Notes
-----
The figure contains a global trend map, an inset mean map, regional box plots,
and precipitation-temperature hexbins. Significant map trends are stippled at
p < 0.05. Set ``VEG_LST_DATA_DIR`` and ``VEG_LST_UTILS_DIR`` to override the
default project directories.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.patches import Ellipse


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
    product: str = "GLASS"
    time_scale: str = "Annual"
    lst_var: str = "dailymean"
    output_figure: Path = DATA_ROOT.parent / "figure/Fig.02_PixelTrend_Map.svg"

    mean_vmin: float = -3.0
    mean_vmax: float = 3.0
    trend_vmin: float = -0.09
    trend_vmax: float = 0.09


def sensitivity_paths(cfg: FigureConfig) -> dict[str, Path]:
    prefix = f"Sensitivity_{cfg.time_scale}_LST{cfg.lst_var}_LAI_{cfg.product}"
    root = cfg.base_dir / "processed/Sensitivity_20260208" / cfg.product / "MOD11C3"

    return {
        "sensitivity_1deg": root / f"{prefix}_1d.nc",
        "sensitivity_trend": root / (
            f"Sens_trend_{cfg.time_scale}_LST{cfg.lst_var}_"
            f"LAI_{cfg.product}_1d.nc"
        ),
        "climate_zone": cfg.base_dir / "koppen_geiger_4class_1d.nc",
        "climate_data": cfg.base_dir / "ERA5-Land_Annual_2000_2024_1deg.nc",
    }


def load_inputs(cfg: FigureConfig) -> dict[str, object]:
    """Load all data used by the figure."""
    paths = sensitivity_paths(cfg)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))

    with xr.open_dataset(paths["sensitivity_1deg"]) as ds:
        sens = ds["sens_1deg_mean"].load()
    trend = xr.load_dataset(paths["sensitivity_trend"])
    climate_zone = xr.load_dataarray(paths["climate_zone"])
    with xr.open_dataset(paths["climate_data"]) as ds:
        climate = ds.sel(time=slice("2001", "2024")).mean(
            "time", skipna=True
        ).load()

    return {
        "sens": sens,
        "trend": trend,
        "climate_zone": climate_zone,
        "precip": climate["tp"],
        "temperature": climate["t2m"],
    }


def latitude_weights_like(da: xr.DataArray) -> xr.DataArray:
    """Return cosine-latitude area weights broadcast to a DataArray grid."""
    weights = np.cos(np.deg2rad(da["lat"]))
    return weights.broadcast_like(da)


def plot_map_mean(ax, da: xr.DataArray, *, cmap, norm, title: str):
    """Plot mean sensitivity on a Cartopy axis."""
    da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        rasterized=True,
    )
    ax.set_extent([-180, 180, -60, 90])
    ax.coastlines(edgecolor="gray", linewidth=0.5)
    ax.set_title(title)
    return ax


def plot_map_slope(
    ax,
    ds_trend: xr.Dataset,
    *,
    cmap,
    norm,
    title: str,
    significance_level: float = 0.05,
    stipple_step: int = 2,
):
    """Plot trend map and stipple statistically significant pixels."""
    slope = ds_trend["trend_theilsen"]
    pval = ds_trend["trend_p"]

    slope.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        rasterized=True,
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
        s=2,
        color="k",
        marker=".",
        alpha=1,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    return ax


def add_colorbar(
    ax,
    *,
    cmap,
    norm,
    ticks: Sequence[float],
    ticklabels: Sequence[str | float],
    orientation: str = "horizontal",
    label: str | None = None,
    label_position: str | None = None,
):
    """Add a compact colorbar to an explicitly positioned axis."""
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
            colorbar.set_label(label, rotation=90, labelpad=5, fontsize=10)
            colorbar.ax.yaxis.set_label_position(label_position or "left")
        else:
            colorbar.set_label(label, fontsize=10)
            colorbar.ax.xaxis.set_label_position(label_position or "top")

    colorbar.ax.tick_params(direction="in", length=3, width=1, color="white")
    colorbar.outline.set_edgecolor("white")
    return colorbar


def move_axis(ax, *, dx: float = 0.0, dy: float = 0.0, dw: float = 0.0, dh: float = 0.0):
    """Move or resize an existing Matplotlib axis in figure coordinates."""
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width + dw, pos.height + dh])
    return ax


def climate_trend_sign_proportions(
    sens_mean: xr.DataArray,
    sens_trend: xr.Dataset,
    *,
    trend_var: str = "trend_theilsen",
    weights: xr.DataArray | None = None,
) -> dict[str, float]:
    """
    Compute proportions of the four mean/trend sign combinations.

    Keys:
        "--": mean < 0 and trend < 0
        "-+": mean < 0 and trend >= 0
        "+-": mean >= 0 and trend < 0
        "++": mean >= 0 and trend >= 0
    """
    trend = sens_trend[trend_var]
    sens_mean, trend = xr.align(sens_mean, trend, join="inner")
    valid = np.isfinite(sens_mean) & np.isfinite(trend)

    masks = {
        "--": valid & (sens_mean < 0) & (trend < 0),
        "-+": valid & (sens_mean < 0) & (trend >= 0),
        "+-": valid & (sens_mean >= 0) & (trend < 0),
        "++": valid & (sens_mean >= 0) & (trend >= 0),
    }

    if weights is None:
        total = float(valid.sum().item())
        return {
            key: float(mask.sum().item()) / total * 100
            if total > 0
            else np.nan
            for key, mask in masks.items()
        }

    weights, valid = xr.align(weights, valid, join="inner")
    total = float(weights.where(valid).sum(skipna=True).item())
    return {
        key: float(weights.where(mask).sum(skipna=True).item()) / total * 100
        if total > 0
        else np.nan
        for key, mask in masks.items()
    }


def region_values_and_stats(
    data: xr.DataArray,
    region_mask: xr.DataArray,
    weights: xr.DataArray,
    *,
    region_id: int | None = None,
    pval: xr.DataArray | None = None,
    significance_level: float = 0.05,
):
    """
    Extract regional values and calculate distribution diagnostics.

    Returns
    -------
    values, pos_pct, neg_pct, sig_pos_pct, sig_neg_pct, median, weighted_mean
    """
    if region_id is None:
        region_data = data
        region_weights = weights
        region_pval = pval
    else:
        region_data = data.where(region_mask == region_id)
        region_weights = weights.where(region_mask == region_id)
        region_pval = pval.where(region_mask == region_id) if pval is not None else None

    values = region_data.values.ravel()
    w = region_weights.values.ravel()
    valid = np.isfinite(values) & np.isfinite(w)

    values = values[valid]
    w = w[valid]

    if values.size == 0:
        return values, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan

    pos_pct = np.mean(values > 0) * 100
    neg_pct = np.mean(values < 0) * 100
    median = np.nanmedian(values)
    weighted_mean = np.nansum(values * w) / np.nansum(w)

    sig_pos_pct = 0.0
    sig_neg_pct = 0.0
    if region_pval is not None:
        p = region_pval.values.ravel()[valid]
        sig_pos_pct = np.mean((values > 0) & (p < significance_level)) * 100
        sig_neg_pct = np.mean((values < 0) & (p < significance_level)) * 100

    return values, pos_pct, neg_pct, sig_pos_pct, sig_neg_pct, median, weighted_mean


def plot_regional_distribution_boxes(
    ax,
    sens_mean: xr.DataArray,
    sens_trend: xr.Dataset,
    climate_zone: xr.DataArray,
    *,
    legend: bool = False,
    title: str = "",
    ylabel_mean: str = (
        r"$\partial$LST / $\partial$LAI mean"
        "\n"
        r"[K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$]"
    ),
    ylabel_trend: str = (
        r"$\partial$LST / $\partial$LAI trend"
        "\n"
        r"[K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$]"
    ),
):
    """Draw paired boxplots of mean sensitivity and sensitivity trend by climate zone."""
    colors = {
        "mean": sns.color_palette("Blues", 3)[1],
        "trend": sns.color_palette("YlOrBr", 3)[1],
    }
    region_ids = {
        "Global": None,
        "Tropical": 1,
        "Temperate": 3,
        "Arid": 2,
        "Boreal": 4,
    }
    region_names = list(region_ids)

    weights_mean = latitude_weights_like(sens_mean)
    trend_decade = sens_trend["trend_theilsen"] * 10.0
    weights_trend = weights_mean.broadcast_like(trend_decade)

    ax_mean = ax
    ax_trend = ax_mean.twinx()

    centers = np.arange(len(region_names)) * 1.3
    offset = 0.22
    width = 0.28
    pos_mean = centers - offset
    pos_trend = centers + offset

    for i, name in enumerate(region_names):
        region_id = region_ids[name]

        mean_vals, *_ = region_values_and_stats(
            sens_mean,
            climate_zone,
            weights_mean,
            region_id=region_id,
        )
        trend_vals, trend_pos, trend_neg, sig_pos, sig_neg, *_ = region_values_and_stats(
            trend_decade,
            climate_zone,
            weights_trend,
            region_id=region_id,
            pval=sens_trend["trend_p"],
        )

        bp_mean = ax_mean.boxplot(
            mean_vals,
            positions=[pos_mean[i]],
            widths=width,
            patch_artist=True,
            showcaps=False,
            showfliers=False,
            whis=(5, 95),
        )
        bp_trend = ax_trend.boxplot(
            trend_vals,
            positions=[pos_trend[i]],
            widths=width,
            patch_artist=True,
            showcaps=False,
            showfliers=False,
            whis=(5, 95),
        )

        for patch in bp_mean["boxes"]:
            patch.set(facecolor=colors["mean"], edgecolor="k", linewidth=0.7)
        for patch in bp_trend["boxes"]:
            patch.set(facecolor=colors["trend"], edgecolor="k", linewidth=0.7)
        for element in ("medians", "whiskers"):
            for item in bp_mean[element] + bp_trend[element]:
                item.set(
                    color="k" if element == "medians" else "gray",
                    linewidth=0.7 if element == "medians" else 0.5,
                )

        ax_trend.text(
            pos_trend[i] - 0.08,
            0.40,
            f"+{trend_pos:.1f}%",
            color=colors["trend"],
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax_trend.text(
            pos_trend[i] - 0.08,
            -0.40,
            f"-{trend_neg:.1f}%",
            color=colors["trend"],
            ha="center",
            va="top",
            fontsize=10,
        )

    for region in ("Tropical", "Arid"):
        idx = region_names.index(region)
        left = centers[idx] - 0.65
        right = centers[idx] + 0.65
        ax_mean.axvspan(left, right, color="0.92", zorder=0)

    ax_mean.axhline(0, color="gray", linestyle="--", lw=1.5, alpha=0.6, zorder=1)
    ax_mean.grid(axis="y", linestyle=":", alpha=0.5)
    ax_mean.set_xticks(centers)
    ax_mean.set_xticklabels(region_names, rotation=30)
    ax_mean.set_xlim(centers[0] - 0.6, centers[-1] + 0.6)
    ax_mean.set_ylim(-3, 3)
    ax_trend.set_ylim(-0.5, 0.5)

    ax_mean.set_ylabel(ylabel_mean, color=colors["mean"], fontsize=12)
    ax_trend.set_ylabel(
        ylabel_trend,
        color=colors["trend"],
        fontsize=12,
        rotation=270,
        labelpad=30,
    )
    ax_mean.tick_params(axis="y", colors=colors["mean"], direction="in")
    ax_trend.tick_params(axis="y", colors=colors["trend"], direction="in")
    ax_mean.tick_params(axis="x", direction="in")

    ax_mean.spines["left"].set_color(colors["mean"])
    ax_mean.spines["right"].set_color("none")
    ax_trend.spines["left"].set_color("none")
    ax_trend.spines["right"].set_color(colors["trend"])
    sns.despine(ax=ax_mean, top=True, right=False)

    if legend:
        handles = [
            mpatches.Patch(color=colors["mean"], label="Mean"),
            mpatches.Patch(color=colors["trend"], label="Trend"),
        ]
        ax_mean.legend(
            handles=handles,
            loc="lower left",
            frameon=False,
            fontsize=12,
            ncol=2,
            columnspacing=1.5,
            handlelength=1.5,
            bbox_to_anchor=(0.1, -0.55),
        )

    ax_mean.set_title(title, fontsize=16, pad=30, fontweight="bold")
    return ax_mean


def plot_climate_hexbin(
    ax,
    target: xr.DataArray,
    precip: xr.DataArray,
    temperature: xr.DataArray,
    *,
    cmap,
    norm,
    title: str | None = None,
    min_count: int = 4,
    ellipse_kwargs: dict | None = None,
):
    """Plot target values in precipitation-temperature climate space."""
    target, precip, temperature = xr.align(target, precip, temperature, join="inner")

    x = precip.values.ravel()
    if np.nanquantile(x, 0.95) < 10:
        x = x * 1000.0 * 12.0

    y = temperature.values.ravel()
    z = target.values.ravel()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    hb = ax.hexbin(
        x[valid],
        y[valid],
        C=z[valid],
        gridsize=(60, 20),
        cmap=cmap,
        norm=norm,
        reduce_C_function=np.mean,
        mincnt=min_count,
        edgecolors="gray",
        linewidths=0.2,
    )

    if title:
        ax.set_title(title, fontweight="bold", pad=10)

    ax.set_xlabel(r"Precipitation [mm yr$^{-1}$]", fontsize=12)
    ax.set_ylabel("Temperature [K]", fontsize=12)
    ax.set_xlim(0, 4000)
    ax.set_ylim(255, 305)
    ax.tick_params(direction="in", length=4)
    ax.tick_params(axis="both", which="major", labelsize=12)

    if ellipse_kwargs is not None:
        ax.add_patch(Ellipse(**ellipse_kwargs))

    return hb


def add_sign_combination_labels(ax, props: dict[str, float]):
    """Add mean/trend sign-combination proportions beside the main map."""
    labels = {
        "-+": f"−  +: {props.get('-+', np.nan):.1f}%",
        "--": f"−  −: {props.get('--', np.nan):.1f}%",
        "+-": f"+  −: {props.get('+-', np.nan):.1f}%",
        "++": f"+  +: {props.get('++', np.nan):.1f}%",
    }

    ax.text(-0.053, 0.84, "Mean", transform=ax.transAxes, fontsize=12, rotation=90)
    ax.text(-0.030, 0.84, "Trend", transform=ax.transAxes, fontsize=12, rotation=90)

    for y, key in zip((0.76, 0.68, 0.60, 0.52), ("-+", "--", "+-", "++")):
        ax.text(
            -0.05,
            y,
            labels[key],
            transform=ax.transAxes,
            backgroundcolor="w",
            fontsize=12,
        )


def build_figure(cfg: FigureConfig) -> plt.Figure:
    """Create and save Figure 2."""
    data = load_inputs(cfg)
    sens_mean = data["sens"].mean("time", skipna=True)
    trend = data["trend"]
    climate_zone = data["climate_zone"].where(np.isfinite(sens_mean))

    cmap_mean, norm_mean = get_truncated_cs(
        "RdYlBu_r",
        skip_middle=0.15,
        vmin=cfg.mean_vmin,
        vmax=cfg.mean_vmax,
    )
    cmap_trend, norm_trend = get_truncated_cs(
        "RdYlBu_r",
        skip_middle=0.15,
        vmin=cfg.trend_vmin,
        vmax=cfg.trend_vmax,
    )

    plot_settings()
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=(0.7, 0.3),
        width_ratios=(0.5, 0.25, 0.25),
        wspace=0.5,
        hspace=0.05,
    )

    ax_map = fig.add_subplot(gs[0, 0:3], projection=ccrs.Robinson())
    ax_boxes = fig.add_subplot(gs[1, 0])
    ax_hex_mean = fig.add_subplot(gs[1, 1])
    ax_hex_trend = fig.add_subplot(gs[1, 2])

    plot_map_slope(
        ax_map,
        trend,
        cmap=cmap_trend,
        norm=norm_trend,
        title=r"$\partial$LST / $\partial$LAI trend",
        stipple_step=2,
    )

    props = climate_trend_sign_proportions(sens_mean, trend)
    add_sign_combination_labels(ax_map, props)

    ax_inset = ax_map.inset_axes([-0.1, 0.0, 0.4, 0.4], projection=ccrs.Robinson())
    plot_map_mean(
        ax_inset,
        sens_mean,
        cmap=cmap_mean,
        norm=norm_mean,
        title=r"$\partial$LST / $\partial$LAI mean",
    )

    plot_regional_distribution_boxes(ax_boxes, sens_mean, trend, climate_zone)
    move_axis(ax_boxes, dx=-0.03)

    ellipse_kwargs = {
        "xy": (780, 268),
        "width": 1000,
        "height": 18,
        "angle": 0.5,
        "edgecolor": "k",
        "facecolor": "none",
        "linewidth": 1.1,
        "linestyle": "--",
        "zorder": 6,
    }
    plot_climate_hexbin(
        ax_hex_mean,
        sens_mean,
        data["precip"],
        data["temperature"],
        cmap=cmap_mean,
        norm=norm_mean,
        ellipse_kwargs=ellipse_kwargs,
    )
    plot_climate_hexbin(
        ax_hex_trend,
        trend["trend_theilsen"],
        data["precip"],
        data["temperature"],
        cmap=cmap_trend,
        norm=norm_trend,
        ellipse_kwargs=ellipse_kwargs,
    )
    move_axis(ax_hex_mean, dx=0.015)

    # Main map trend colorbar.
    cax_map = fig.add_axes(
        [
            ax_map.get_position().x0 + 0.3,
            ax_map.get_position().y0 + 0.03,
            0.3,
            0.015,
        ]
    )
    add_colorbar(
        cax_map,
        cmap=cmap_trend,
        norm=norm_trend,
        ticks=(-0.08, -0.04, 0.0, 0.04, 0.08),
        ticklabels=("-0.08", "-0.04", "0", "0.04", "0.08"),
        orientation="horizontal",
        label=r"K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$",
    )

    # Inset mean map colorbar.
    cax_inset = fig.add_axes(
        [
            ax_inset.get_position().x0 + 0.05,
            ax_inset.get_position().y0 + 0.065,
            0.01,
            0.08,
        ]
    )
    add_colorbar(
        cax_inset,
        cmap=cmap_mean,
        norm=norm_mean,
        ticks=(-2, 0, 2),
        ticklabels=("-2", "0", "2"),
        orientation="vertical",
        label=r"K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$",
    )

    # Climate-space panel colorbars.
    pos_mean = ax_hex_mean.get_position()
    cax_mean = fig.add_axes([pos_mean.x0 + 0.11, pos_mean.y0, 0.01, 0.13])
    add_colorbar(
        cax_mean,
        cmap=cmap_mean,
        norm=norm_mean,
        ticks=(-2, -1, 0, 1, 2),
        ticklabels=("-2", "-1", "0", "1", "2"),
        orientation="vertical",
        label="Mean",
    )

    pos_trend = ax_hex_trend.get_position()
    cax_trend = fig.add_axes([pos_trend.x0 + 0.11, pos_trend.y0, 0.01, 0.13])
    add_colorbar(
        cax_trend,
        cmap=cmap_trend,
        norm=norm_trend,
        ticks=(-0.08, -0.04, 0, 0.04, 0.08),
        ticklabels=("-8", "-4", "0", "4", "8"),
        orientation="vertical",
        label="Trend",
    )

    for ax, label in [
        (ax_map, "a"),
        (ax_inset, "b"),
        (ax_boxes, "c"),
        (ax_hex_mean, "d"),
        (ax_hex_trend, "e"),
    ]:
        ax.text(-0.01, 1.05, label, transform=ax.transAxes, fontsize=16, fontweight="bold")

    cfg.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output_figure, dpi=500, bbox_inches="tight")
    print(f"Saved figure: {cfg.output_figure}")
    return fig


def main() -> None:
    cfg = FigureConfig()
    build_figure(cfg)


if __name__ == "__main__":
    main()
