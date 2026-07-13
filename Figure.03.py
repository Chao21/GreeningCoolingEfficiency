"""
Figure 3: radiative and non-radiative decomposition of LST-LAI sensitivity.

This script is a cleaned, reproducible version of the original Figure03 code.
It preserves the final figure layout:
    a. global radiative sensitivity time series,
    b. global non-radiative sensitivity time series,
    c. radiative sensitivity trend map,
    d. non-radiative sensitivity trend map,
    e-i. annual and seasonal regional trend bars.

The script assumes it is executed on the project machine where the Linux input
paths under BASE_DIR are available.
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress, t


UTILS_DIR = Path("/home/energy/chaoz/code/utils")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from plot_utils import get_truncated_cs, plot_settings  # noqa: E402


@dataclass(frozen=True)
class FigureConfig:
    base_dir: Path = Path("/home/energy/chaoz/project/05Veg_LST/data")
    root: Path = Path("/home/energy/chaoz/project/05Veg_LST/data/processed/Sensitivity_LST_Energy_LAI_20260325")
    output_figure: Path = Path("/home/energy/chaoz/project/05Veg_LST/figure/202606/Fig.03_Decomposition_Energy.png")

    lai_product: str = "GLASS"
    lst_var: str = "dailymean"
    le_product: str = "MOD16A2GF"

    regions: tuple[str, ...] = ("Global", "Tropical", "Temperate", "Arid", "Boreal")
    seasons: tuple[str, ...] = ("annual", "spring", "summer", "autumn", "winter")
    colors: tuple[str, ...] = ("#000000", "#cb2f2d", "#436aab")

    trend_vmin: float = -0.12
    trend_vmax: float = 0.12


LABEL_DELTA = {
    "Radiative": r"$\delta(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{radiative}}$",
    "Non-Radiative": r"$\delta(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{nonradiative}}$",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def input_paths(cfg: FigureConfig) -> dict[str, Path]:
    """Build all input paths used by Figure 3."""
    root = cfg.root
    le = cfg.le_product
    lst = cfg.lst_var

    return {
        "rad_annual_csv": root / "Albedo/Annual" / f"Sensitivity_Annual_LST{lst}_LAI_from_Albedo_regionAgg_gapfilled.csv",
        "rad_season_csv": root / "Albedo/Season" / f"Sensitivity_Season_LST{lst}_LAI_from_Albedo_regionAgg_gapfilled_NSharmonized.csv",
        "rad_mean_nc": root / "Albedo/Annual" / f"Sensitivity_Annual_LST{lst}_LAI_from_Albedo_1d_gapfilled.nc",
        "rad_trend_nc": root / "Albedo/Annual" / f"Sens_trend_Annual_LST{lst}_LAI_from_Albedo_1d_gapfilled.nc",
        "nonrad_annual_csv": root / "NonRadiative/Annual" / f"Sensitivity_Annual_LST{lst}_LAI_from_NonRadiative_{le}_regionAgg_gapfilled.csv",
        "nonrad_season_csv": root / "NonRadiative/Season" / f"Sensitivity_Season_LST{lst}_LAI_from_NonRadiative_{le}_regionAgg_gapfilled_NSharmonized.csv",
        "nonrad_mean_nc": root / "NonRadiative/Annual" / f"Sensitivity_Annual_LST{lst}_LAI_from_NonRadiative_{le}_1d_gapfilled.nc",
        "nonrad_trend_nc": root / "NonRadiative/Annual" / f"Sens_trend_Annual_LST{lst}_LAI_from_NonRadiative_{le}_1d_gapfilled.nc",
        "total_region_csv": cfg.base_dir
        / "processed/Sensitivity_20260208"
        / cfg.lai_product
        / f"Sensitivity_Annual_LST{lst}_LAI_{cfg.lai_product}_regionAgg.csv",
    }


def load_inputs(cfg: FigureConfig) -> dict[str, object]:
    """Load annual/seasonal regional time series and gridded trend datasets."""
    paths = input_paths(cfg)
    os.chdir(cfg.base_dir)

    return {
        "df_rad_annual": pd.read_csv(paths["rad_annual_csv"], parse_dates=["time"]),
        "df_rad_season": pd.read_csv(paths["rad_season_csv"], parse_dates=["time"]),
        "df_nonrad_annual": pd.read_csv(paths["nonrad_annual_csv"], parse_dates=["time"]),
        "df_nonrad_season": pd.read_csv(paths["nonrad_season_csv"], parse_dates=["time"]),
        "df_total_annual": pd.read_csv(paths["total_region_csv"], parse_dates=["time"]),
        "rad_mean": xr.open_dataset(paths["rad_mean_nc"])["sens_1deg_mean"].mean("time", skipna=True),
        "nonrad_mean": xr.open_dataset(paths["nonrad_mean_nc"])["sens_1deg_mean"].mean("time", skipna=True),
        "rad_trend": xr.open_dataset(paths["rad_trend_nc"]),
        "nonrad_trend": xr.open_dataset(paths["nonrad_trend_nc"]),
    }


def decimal_year(time_values: Sequence[pd.Timestamp] | pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Convert datetime-like values to decimal years."""
    idx = pd.DatetimeIndex(pd.to_datetime(time_values))
    year = idx.year.astype(float)
    doy = idx.dayofyear.astype(float)
    days_in_year = np.array([366 if pd.Timestamp(f"{yy}-12-31").is_leap_year else 365 for yy in idx.year])
    return year + (doy - 1.0) / days_in_year


def linear_trend(time_values, values, alpha: float = 0.05):
    """
    Linear trend against decimal year.

    Returns slope, p value, intercept, standard error, and slope confidence
    interval. Slope is in data units per year.
    """
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
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def get_region_series(df: pd.DataFrame, region: str = "Global", group: str = "annual") -> pd.DataFrame:
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

        s_rad, p_rad, _, _, ci_rad = linear_trend(rad["time"], rad["value"])
        s_nonrad, p_nonrad, _, _, ci_nonrad = linear_trend(nonrad["time"], nonrad["value"])

        rows.append(
            {
                "region": region,
                "radiative_slope": s_rad,
                "radiative_p": p_rad,
                "radiative_ci_lo": ci_rad[0],
                "radiative_ci_hi": ci_rad[1],
                "non_radiative_slope": s_nonrad,
                "non_radiative_p": p_nonrad,
                "non_radiative_ci_lo": ci_nonrad[0],
                "non_radiative_ci_hi": ci_nonrad[1],
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

    slope, pval, intercept, stderr, _ = linear_trend(time, y)
    if np.isfinite(slope):
        slope_text = f"{slope * 10:+.3f}"
        ax.text(
            0.02,
            0.97,
            f"Slope={slope_text} decade$^{{-1}}$, $p$={pval:.3f}",
            color=color,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "none", "edgecolor": "none", "alpha": 0.55},
        )
        print(f"{title}: slope={slope * 10:.4f}, p={pval:.4f}, stderr={stderr * 10:.4f}")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.tick_params(axis="both", direction="in")
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
):
    """Plot radiative and non-radiative regional trends as overlaid bars."""
    regions = df["region"].tolist()
    x = np.arange(len(regions))
    width = 0.6
    scale = 10.0

    radiative = df["radiative_slope"].to_numpy() * scale
    nonrad = df["non_radiative_slope"].to_numpy() * scale

    ax.axhline(0, linewidth=0.8, color="k")
    ax.bar(x, radiative, width=width, color=colors["radiative"], alpha=0.70, label=LABEL_DELTA["Radiative"], zorder=2)
    ax.bar(x, nonrad, width=width, color=colors["non_radiative"], alpha=0.90, label=LABEL_DELTA["Non-Radiative"], zorder=2)

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
                    value + offset if value >= 0 else value - 4 * offset,
                    star,
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=9,
                )

        if np.isfinite(radiative[i]) and np.isfinite(nonrad[i]) and abs(nonrad[i]) > 1e-12:
            ratio = radiative[i] / nonrad[i] * 100
            ratio_text = f"{ratio:+.0f}%"
        else:
            ratio_text = "NA"
        ax.text(x[i], ylim[1] - 0.01, ratio_text, ha="center", va="top", color="0.65", rotation=90, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(1, len(regions) + 1)], fontsize=12)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.tick_params(direction="in")

    if show_ylabel:
        ax.set_ylabel(r"[K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$]")
        ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2], ["-0.2", "-0.1", "0", "0.1", "0.2"])
        for i, region in enumerate(regions):
            ax.text(x[i], ylim[0] + 0.02, region, ha="center", va="bottom", rotation=90, fontsize=10)
        ax.legend(
            frameon=False,
            ncol=2,
            columnspacing=1.0,
            handlelength=1.5,
            handletextpad=0.9,
            loc="lower left",
            bbox_to_anchor=(1.5, -0.28),
        )
    else:
        ax.set_yticklabels([])

    return ax


def count_significance_categories(trend: xr.DataArray, pval: xr.DataArray, pthr: float = 0.05):
    """Count trend pixels by sign and significance."""
    trend, pval = xr.align(trend, pval, join="inner")
    valid = np.isfinite(trend) & np.isfinite(pval)
    total = int(valid.sum().values)
    if total == 0:
        return {"sig_neg": 0, "ns_neg": 0, "sig_pos": 0, "ns_pos": 0}, 0

    return {
        "sig_neg": int(((pval < pthr) & (trend < 0) & valid).sum().values),
        "ns_neg": int(((pval >= pthr) & (trend < 0) & valid).sum().values),
        "sig_pos": int(((pval < pthr) & (trend > 0) & valid).sum().values),
        "ns_pos": int(((pval >= pthr) & (trend > 0) & valid).sum().values),
    }, total


def add_pdf_inset(
    ax,
    trend: xr.DataArray,
    pval: xr.DataArray,
    *,
    cmap,
    norm,
    inset_bbox=(0.065, 0.06, 0.25, 0.26),
    bins: int = 20,
):
    """Add a small trend-value distribution inset to a map axis."""
    ax_in = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=inset_bbox,
        bbox_transform=ax.transAxes,
        loc="lower left",
        borderpad=0,
    )
    ax_in.set_facecolor("none")
    ax_in.patch.set_alpha(0.0)

    values = trend.values
    values = values[np.isfinite(values)]
    if values.size < 10:
        ax_in.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_in.transAxes, fontsize=8)
        ax_in.set_axis_off()
        return ax_in

    p_neg = np.mean(values < 0) * 100
    p_pos = np.mean(values > 0) * 100

    vmin = getattr(norm, "vmin", np.nanmin(values))
    vmax = getattr(norm, "vmax", np.nanmax(values))
    edges = np.linspace(vmin, vmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    density, _ = np.histogram(values, bins=edges, density=True)

    ax_in.bar(centers, density, width=widths, align="center", color=cmap(norm(centers)), edgecolor="none", alpha=0.9)
    ax_in.axhline(0, color="k", linewidth=1.0)
    ax_in.axvline(0, color="k", linestyle="--", linewidth=0.6)
    ax_in.text(0.05, 0.88, f"{p_neg:.0f}%", color="tab:blue", transform=ax_in.transAxes, ha="left", va="top", fontsize=11)
    ax_in.text(0.95, 0.88, f"{p_pos:.0f}%", color="tab:red", transform=ax_in.transAxes, ha="right", va="top", fontsize=11)
    ax_in.set_xlim(vmin, vmax)
    ax_in.set_ylim(0, density.max() * 1.25)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(False)

    return ax_in


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

    trend.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False, rasterized=True)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines(edgecolor="gray", linewidth=0.5)
    ax.set_title(title)

    sig_sparse = (pval < significance_level).isel(lat=slice(0, None, stipple_step), lon=slice(0, None, stipple_step))
    lat = sig_sparse["lat"].values
    lon = sig_sparse["lon"].values
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    mask = np.asarray(sig_sparse.values, dtype=bool)
    ax.scatter(
        xx[mask],
        yy[mask],
        s=5,
        color="k",
        marker=".",
        edgecolor="none",
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    add_pdf_inset(ax, trend, pval, cmap=cmap, norm=norm)
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
        colorbar.set_label(label, fontsize=10)
        colorbar.ax.xaxis.set_label_position('top')
    colorbar.ax.tick_params(direction="in", length=3, width=1, color="white")
    colorbar.outline.set_edgecolor("white")
    return colorbar


def move_axis(ax, *, dx=0.0, dy=0.0, dw=0.0, dh=0.0):
    """Move or resize an axis in figure coordinates."""
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width + dw, pos.height + dh])
    return ax


def build_figure(cfg: FigureConfig):
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
    fig = plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.15, hspace=0.2)

    ax_rad_ts = fig.add_subplot(gs[0, 0])
    ax_nonrad_ts = fig.add_subplot(gs[0, 1])
    ax_rad_map = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
    ax_nonrad_map = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())

    gs_bars = gs[2, 0:2].subgridspec(1, 5, wspace=0.08)
    bar_axes = [fig.add_subplot(gs_bars[0, i]) for i in range(5)]

    ylabel = r"K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$"
    plot_global_trend(
        ax_rad_ts,
        get_region_series(data["df_rad_annual"], "Global"),
        ylabel=ylabel,
        title=r"$(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{radiative}}$",
        color=colors["radiative"],
        ylim=(0.68, 1.03),
    )
    plot_global_trend(
        ax_nonrad_ts,
        get_region_series(data["df_nonrad_annual"], "Global"),
        ylabel="",
        title=r"$(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{nonradiative}}$",
        color=colors["non_radiative"],
        ylim=(-1.43, -1.05),
    )

    plot_map_slope(ax_rad_map, data["rad_trend"], cmap=cmap_trend, norm=norm_trend, title=LABEL_DELTA["Radiative"])
    plot_map_slope(ax_nonrad_map, data["nonrad_trend"], cmap=cmap_trend, norm=norm_trend, title=LABEL_DELTA["Non-Radiative"])

    for ax, season, title in zip(bar_axes, cfg.seasons, ("Annual", "Spring", "Summer", "Autumn", "Winter")):
        plot_climate_zone_bars(
            ax,
            region_stats[season],
            title=title,
            colors=colors,
            ylim=(-0.3, 0.3),
            show_ylabel=(season == "annual"),
        )

    move_axis(ax_rad_ts, dw=-0.01, dh=-0.05)
    move_axis(ax_nonrad_ts, dw=-0.01, dh=-0.05)

    for ax in (ax_rad_map, ax_nonrad_map):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.15, pos.y0 - 0.01, 0.15, 0.01])
        add_colorbar(
            cax,
            cmap=cmap_trend,
            norm=norm_trend,
            ticks=(-0.12, -0.06, 0.0, 0.06, 0.12),
            ticklabels=("-0.12", "-0.06", "0", "0.06", "0.12"),
            orientation="horizontal",
            label=r"K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$",
        )

    panel_axes = [ax_rad_ts, ax_nonrad_ts, ax_rad_map, ax_nonrad_map, *bar_axes]
    for ax, label in zip(panel_axes, list("abcdefghi")):
        x = 0.0 if label >= "e" else -0.05
        ax.text(x, 1.10, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

    cfg.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output_figure, dpi=300, bbox_inches="tight")
    log(f"Saved figure: {cfg.output_figure}")
    return fig


def main():
    cfg = FigureConfig()
    build_figure(cfg)


if __name__ == "__main__":
    main()
