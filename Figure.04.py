"""
Figure 4: attribution effects and SEM pathways.

This script is a cleaned, reproducible version of the original Figure04 code.
It generates a 3 x 2 figure:
    a-b. standardized coefficients for snow-affected and snow-free pixels,
    c-d. physical contributions for snow-affected and snow-free pixels,
    e-f. SEM pathway diagrams for the same two groups.

The script assumes it is executed on the project machine where the Linux input
paths under BASE_DIR are available.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


UTILS_DIR = Path("/home/energy/chaoz/code/utils")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from plot_utils import plot_settings  # noqa: E402


@dataclass(frozen=True)
class FigureConfig:
    base_dir: Path = Path("/home/energy/chaoz/project/05Veg_LST/data")
    output_root: Path = Path("../figure/202606")
    panel_mode: str = "humid"  # "snow", "dry", or "humid"
    window_size: int = 1
    attribution_dir: Path = Path("processed/Attribution_20260531/ridgeRegression_NoDetrend")
    sem_dir: Path = Path("processed/Attribution_20260531/PLSEM/regional_summary_tables_20260531")
    pval_threshold: float | None = 0.1
    mean_mode: str = "weighted"


RESPONSE_ORDER = ("LSTtotal", "LSTrad", "LSTnonrad")
RESPONSE_COLORS = {
    "LSTtotal": "#000000",
    "LSTrad": "#cc5349",
    "LSTnonrad": "#3f87ba",
}
RESPONSE_LABELS = {
    "LSTtotal": r"$\partial\mathrm{LST}/\partial\mathrm{LAI}$",
    "LSTrad": r"$(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{radiative}}$",
    "LSTnonrad": r"$(\partial\mathrm{LST}/\partial\mathrm{LAI})_{\mathrm{nonradiative}}$",
}

FEATURE_LABELS = {
    "ssrd_mean": "Rad",
    "snowc_mean": "Snow",
    "tp_mean": "P",
    "t2m_mean": "Ta",
    "swvl1_mean": "SM",
    "vpd_mean": "VPD",
    "co2_mean": r"CO$_2$",
}
FEATURE_ORDER = (
    "co2_mean",
    "t2m_mean",
    "ssrd_mean",
    "swvl1_mean",
    "vpd_mean",
    "snowc_mean",
)


def output_path(panel_mode: str) -> Path:
    if panel_mode == "snow":
        return Path("../figure/202606/Fig.04_Attribution_Global.svg")
    if panel_mode == "dry":
        return Path("../figure/202606/Fig.04_Attribution_Drylands.svg")
    if panel_mode == "humid":
        return Path("../figure/202606/Fig.04_Attribution_Humidlands.svg")
    raise ValueError("panel_mode must be one of {'snow', 'dry', 'humid'}")


def sem_path(panel_mode: str) -> Path:
    if panel_mode == "snow":
        filename = "semcoeff_global_by_snow_winsize1.csv"
    elif panel_mode in {"dry", "humid"}:
        filename = "semcoeff_dryhumid_by_snow_winsize1.csv"
    else:
        raise ValueError("panel_mode must be one of {'snow', 'dry', 'humid'}")
    return Path("processed/Attribution_20260531/PLSEM/regional_summary_tables_20260531") / filename


def sem_region_name(panel_mode: str) -> str:
    return {"snow": "Global", "dry": "Dry", "humid": "Humid"}[panel_mode]


def panel_titles(panel_mode: str) -> tuple[str, str, str]:
    if panel_mode == "snow":
        return "global", "Snow-affected", "Snow-free"
    if panel_mode == "dry":
        return "dry", "Dry snow-affected", "Dry snow-free"
    if panel_mode == "humid":
        return "humid", "Humid snow-affected", "Humid snow-free"
    raise ValueError("panel_mode must be one of {'snow', 'dry', 'humid'}")


def load_attribution_datasets(cfg: FigureConfig) -> dict[str, xr.Dataset]:
    """Load attribution result datasets for total, radiative, and non-radiative responses."""
    w = cfg.window_size
    root = cfg.attribution_dir
    return {
        "LSTtotal": xr.open_dataset(root / f"Attribution_LSTTotal_Annual_winsize{w}.nc"),
        "LSTrad": xr.open_dataset(root / f"Attribution_LSTRadiative_Annual_winsize{w}.nc"),
        "LSTnonrad": xr.open_dataset(root / f"Attribution_LSTNonRadiative_Annual_winsize{w}.nc"),
    }


def build_snow_masks(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Build snow-affected and snow-free masks from availability of snow coefficient."""
    snow_mask = ds["beta"].sel(feature="snowc_mean").notnull()
    return snow_mask, ~snow_mask


def build_dry_humid_masks(aridity: xr.DataArray, ds_ref: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Build dryland and humid masks using AI < 0.65 as dryland."""
    aridity = aridity.sel(lat=ds_ref.lat, lon=ds_ref.lon, method="nearest")
    return aridity < 0.65, aridity >= 0.65


def build_focus_mask(
    ds: xr.Dataset,
    *,
    snow_group: str,
    focus_region: str,
    dry_mask: xr.DataArray | None = None,
    humid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Build the final spatial subset mask."""
    snow_mask, nosnow_mask = build_snow_masks(ds)

    if snow_group == "snow":
        mask = snow_mask
    elif snow_group == "nosnow":
        mask = nosnow_mask
    else:
        raise ValueError("snow_group must be one of {'snow', 'nosnow'}")

    if focus_region == "global":
        return mask
    if focus_region == "dry":
        if dry_mask is None:
            raise ValueError("dry_mask is required for focus_region='dry'")
        return mask & dry_mask.sel(lat=ds.lat, lon=ds.lon, method="nearest")
    if focus_region == "humid":
        if humid_mask is None:
            raise ValueError("humid_mask is required for focus_region='humid'")
        return mask & humid_mask.sel(lat=ds.lat, lon=ds.lon, method="nearest")

    raise ValueError("focus_region must be one of {'global', 'dry', 'humid'}")


def field_to_long_df(
    da_field: xr.DataArray,
    region_mask: xr.DataArray,
    *,
    source_name: str,
    region_id_to_name: Mapping[int, str],
    value_name: str,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> pd.DataFrame:
    """Convert a feature-lat-lon field to a long DataFrame."""
    frames = []
    lat_weights = np.cos(np.deg2rad(da_field[lat_dim]))
    lat_weight_2d = xr.broadcast(lat_weights, da_field.isel(feature=0, drop=True))[0]
    region_mask = region_mask.sel({lat_dim: da_field[lat_dim], lon_dim: da_field[lon_dim]}, method="nearest")

    for feature in da_field.feature.values:
        da_feature = da_field.sel(feature=feature, drop=True)
        ds_tmp = xr.Dataset(
            {
                value_name: da_feature,
                "lat_weight": lat_weight_2d,
                "region_id": region_mask,
            }
        )
        df = ds_tmp.to_dataframe().reset_index()
        df = df[np.isfinite(df[value_name]) & np.isfinite(df["region_id"])].copy()
        if df.empty:
            continue

        df["region_id"] = df["region_id"].astype(int)
        df = df[df["region_id"].isin(region_id_to_name)].copy()
        if df.empty:
            continue

        df["source"] = source_name
        df["feature"] = feature
        df["region"] = df["region_id"].map(region_id_to_name)
        frames.append(
            df[
                [
                    "source",
                    "feature",
                    "region",
                    lat_dim,
                    lon_dim,
                    value_name,
                    "lat_weight",
                ]
            ].rename(columns={lat_dim: "lat", lon_dim: "lon"})
        )

    columns = ["source", "feature", "region", "lat", "lon", value_name, "lat_weight"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def build_long_df_for_dataset(
    ds: xr.Dataset,
    *,
    variable: str,
    snow_group: str,
    focus_region: str,
    source_name: str,
    dry_mask: xr.DataArray | None = None,
    humid_mask: xr.DataArray | None = None,
    pval_threshold: float | None = 0.1,
) -> pd.DataFrame:
    """Build one long dataframe for beta or physical contribution."""
    if variable not in {"beta", "physical_contribution"}:
        raise ValueError("variable must be 'beta' or 'physical_contribution'")

    pval_mask = xr.ones_like(ds[variable], dtype=bool) if pval_threshold is None else ds["pval"] < pval_threshold
    subset_mask = build_focus_mask(
        ds,
        snow_group=snow_group,
        focus_region=focus_region,
        dry_mask=dry_mask,
        humid_mask=humid_mask,
    )
    single_region_mask = xr.where(subset_mask, 1, np.nan)

    if focus_region == "global":
        region_name = "Snow-affected" if snow_group == "snow" else "Snow-free"
    elif focus_region == "dry":
        region_name = "Dry snow-affected" if snow_group == "snow" else "Dry snow-free"
    elif focus_region == "humid":
        region_name = "Humid snow-affected" if snow_group == "snow" else "Humid snow-free"
    else:
        raise ValueError("focus_region must be one of {'global', 'dry', 'humid'}")

    value_name = "beta" if variable == "beta" else "contribution"
    return field_to_long_df(
        ds[variable].where(pval_mask).where(subset_mask),
        single_region_mask,
        source_name=source_name,
        region_id_to_name={1: region_name},
        value_name=value_name,
    )


def summarize_effect_by_feature(
    df: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    value_name: str,
    mean_mode: str = "weighted",
) -> pd.DataFrame:
    """Summarize mean and approximate 95% CI by feature."""
    if mean_mode not in {"simple", "weighted"}:
        raise ValueError("mean_mode must be 'simple' or 'weighted'")

    rows = []
    for feature in feature_order:
        sub = df[df["feature"] == feature].copy()
        if sub.empty:
            continue

        values = sub[value_name].to_numpy(dtype=float)
        if mean_mode == "simple":
            values = values[np.isfinite(values)]
            n = values.size
            if n == 0:
                continue
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else np.nan
            se = std / np.sqrt(n) if n > 1 else np.nan
        else:
            weights = sub["lat_weight"].to_numpy(dtype=float)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            values = values[valid]
            weights = weights[valid]
            n = values.size
            if n == 0:
                continue
            mean = np.sum(values * weights) / np.sum(weights)
            var = np.sum(weights * (values - mean) ** 2) / np.sum(weights)
            std = np.sqrt(var)
            n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
            se = std / np.sqrt(n_eff) if n_eff > 1 else np.nan

        rows.append(
            {
                "feature": feature,
                "n": n,
                "mean": mean,
                "std": std,
                "se": se,
                "ci95": 1.96 * se if np.isfinite(se) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_effect_tables(
    ds_dict: Mapping[str, xr.Dataset],
    *,
    focus_region: str,
    dry_mask: xr.DataArray | None,
    humid_mask: xr.DataArray | None,
    pval_threshold: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build combined beta and contribution long tables for both snow groups."""
    beta_frames = []
    contribution_frames = []

    for response_name in RESPONSE_ORDER:
        ds = ds_dict[response_name]
        for variable, frames in [
            ("beta", beta_frames),
            ("physical_contribution", contribution_frames),
        ]:
            for snow_group in ("snow", "nosnow"):
                frames.append(
                    build_long_df_for_dataset(
                        ds,
                        variable=variable,
                        snow_group=snow_group,
                        focus_region=focus_region,
                        source_name=response_name,
                        dry_mask=dry_mask,
                        humid_mask=humid_mask,
                        pval_threshold=pval_threshold,
                    )
                )

    return pd.concat(beta_frames, ignore_index=True), pd.concat(contribution_frames, ignore_index=True)


def plot_effect_panel(
    ax,
    df: pd.DataFrame,
    *,
    panel_name: str,
    value_name: str,
    ylim: tuple[float, float],
    ylabel: str | None,
    title: str | None = None,
    show_legend: bool = False,
    mean_mode: str = "weighted",
):
    """Plot one coefficient or contribution panel."""
    x_base = np.arange(1, len(FEATURE_ORDER) + 1, dtype=float)
    offsets = np.linspace(-0.2, 0.2, len(RESPONSE_ORDER))

    for idx in (1, 3, 5):
        ax.axvspan(idx - 0.5, idx + 0.5, color="0.92", zorder=0)
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1, zorder=1)

    handles = []
    for i, response in enumerate(RESPONSE_ORDER):
        sub = df[(df["region"] == panel_name) & (df["source"] == response)].copy()
        stats = summarize_effect_by_feature(
            sub,
            feature_order=FEATURE_ORDER,
            value_name=value_name,
            mean_mode=mean_mode,
        )

        y = []
        yerr = []
        for feature in FEATURE_ORDER:
            row = stats[stats["feature"] == feature]
            if row.empty:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                y.append(row["mean"].iloc[0])
                yerr.append(row["ci95"].iloc[0])

        color = RESPONSE_COLORS[response]
        ax.errorbar(
            x_base + offsets[i],
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5,
            capsize=3,
            linestyle="none",
            zorder=3,
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                linestyle="none",
                markersize=6,
                label=RESPONSE_LABELS[response],
            )
        )

    ax.set_ylim(ylim)
    ax.set_xlim(x_base[0] - 0.5, x_base[-1] + 0.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in FEATURE_ORDER], fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    ax.tick_params(direction="in")

    if show_legend:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(1.1, 1.3),
            ncol=len(handles),
            frameon=False,
            fontsize=13,
        )

    return ax


def parse_sig_bool(value) -> bool:
    """Parse significance values stored as bool, number, or string."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def coef_dict_from_table(df: pd.DataFrame, region_name: str, component: str) -> dict[str, dict[str, float | bool]]:
    """Extract SEM coefficient dictionary for one region and snow component."""
    sub = df[(df["region_name"] == region_name) & (df["component"] == component)].copy()
    out = {}
    for _, row in sub.iterrows():
        out[str(row["path"])] = {
            "mean": row["mean"],
            "sig_0.05": parse_sig_bool(row["sig_0.05"]) if "sig_0.05" in row.index else True,
        }
    return out


def path_aliases(path: str) -> list[str]:
    """Return likely aliases for SEM path strings."""
    arrow = "\u2192"
    aliases = {
        "SM->VPD": ["SM->VPD", f"SM{arrow}VPD"],
        "Ta->SM": ["Ta->SM", "TA->SM", "Tair->SM", f"Ta{arrow}SM", f"TA{arrow}SM"],
        "Ta->VPD": ["Ta->VPD", "TA->VPD", "Tair->VPD", f"Ta{arrow}VPD"],
        "Ta->LAI": ["Ta->LAI", "TA->LAI", "Tair->LAI", f"Ta{arrow}LAI"],
        "Ta->SNOW": ["Ta->SNOW", "Ta->Snow", "TA->SNOW", f"Ta{arrow}SNOW", f"Ta{arrow}Snow"],
        "SM->LAI": ["SM->LAI", f"SM{arrow}LAI"],
        "VPD->LAI": ["VPD->LAI", f"VPD{arrow}LAI"],
        "CO2->LAI": ["CO2->LAI", f"CO2{arrow}LAI"],
        "RAD->LAI": ["RAD->LAI", "Rad->LAI", f"RAD{arrow}LAI", f"Rad{arrow}LAI"],
        "SNOW->LAI": ["SNOW->LAI", "Snow->LAI", f"SNOW{arrow}LAI", f"Snow{arrow}LAI"],
        "CO2->SEN_NR": ["CO2->SEN_NR", f"CO2{arrow}SEN_NR"],
        "RAD->SEN_NR": ["RAD->SEN_NR", "Rad->SEN_NR", f"RAD{arrow}SEN_NR"],
        "LAI->SEN_NR": ["LAI->SEN_NR", f"LAI{arrow}SEN_NR"],
        "RAD->SEN_R": ["RAD->SEN_R", "Rad->SEN_R", f"RAD{arrow}SEN_R"],
        "SNOW->SEN_R": ["SNOW->SEN_R", "Snow->SEN_R", f"SNOW{arrow}SEN_R"],
        "LAI->SEN_R": ["LAI->SEN_R", f"LAI{arrow}SEN_R"],
        "SM->SEN_NR": ["SM->SEN_NR", f"SM{arrow}SEN_NR"],
        "VPD->SEN_NR": ["VPD->SEN_NR", f"VPD{arrow}SEN_NR"],
        "SM->SEN_R": ["SM->SEN_R", f"SM{arrow}SEN_R"],
        "SEN_NR->SEN_TOT": ["SEN_NR->SEN_TOT", f"SEN_NR{arrow}SEN_TOT"],
        "SEN_R->SEN_TOT": ["SEN_R->SEN_TOT", f"SEN_R{arrow}SEN_TOT"],
    }
    return aliases.get(path, [path])


def get_path_entry(coef: Mapping[str, object], path: str):
    for alias in path_aliases(path):
        if alias in coef:
            return coef[alias]
    return None


def get_beta(coef: Mapping[str, object], path: str) -> float:
    entry = get_path_entry(coef, path)
    if entry is None:
        return np.nan
    if isinstance(entry, Mapping):
        return float(entry.get("mean", np.nan))
    return float(entry)


def get_sig(coef: Mapping[str, object], path: str) -> bool:
    entry = get_path_entry(coef, path)
    if entry is None or not isinstance(entry, Mapping):
        return True
    return parse_sig_bool(entry.get("sig_0.05", True))


def draw_node(ax, xy, label, color, width=0.16, height=0.085):
    """Draw one SEM node."""
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018",
        facecolor=color,
        edgecolor="k",
        linewidth=1,
        zorder=10,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=12, zorder=30)


def node_edge_point(center, toward, half_width=0.095, half_height=0.060):
    """Find an approximate edge point of a rectangular node."""
    dx = toward[0] - center[0]
    dy = toward[1] - center[1]
    if dx == 0 and dy == 0:
        return center
    scale = max(abs(dx) / half_width, abs(dy) / half_height)
    return center[0] + dx / scale, center[1] + dy / scale


def draw_arrow(
    ax,
    pos: Mapping[str, tuple[float, float]],
    source: str,
    target: str,
    beta: float,
    *,
    label_offset=(0.0, 0.0),
    label_rotation=0,
    arrowstyle="-|>",
    linestyle="-",
):
    """Draw one SEM path arrow."""
    if beta is None or not np.isfinite(beta):
        return
    source_center = pos[source]
    target_center = pos[target]
    start = node_edge_point(source_center, target_center)
    end = node_edge_point(target_center, source_center)
    color = "#cb2f2d" if beta >= 0 else "#436aab"
    linewidth = 0.8 + 4.2 * min(abs(beta), 1.0)

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=arrowstyle,
            mutation_scale=15,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            zorder=20,
        )
    )
    xm = (source_center[0] + target_center[0]) / 2 + label_offset[0]
    ym = (source_center[1] + target_center[1]) / 2 + label_offset[1]
    ax.text(xm, ym, f"{beta:.2f}", fontsize=12, ha="center", va="center", rotation=label_rotation, zorder=31)


def draw_routed_arrow(
    ax,
    points: Sequence[tuple[float, float]],
    beta: float,
    *,
    label_xy,
    label_rotation=0,
    linestyle="-",
):
    """Draw a routed SEM arrow through intermediate points."""
    if beta is None or not np.isfinite(beta):
        return
    color = "#cb2f2d" if beta >= 0 else "#436aab"
    linewidth = 0.8 + 4.2 * min(abs(beta), 1.0)

    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=linewidth, linestyle=linestyle, zorder=8)

    end = node_edge_point(points[-1], points[-2])
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            zorder=20,
        )
    )
    ax.text(label_xy[0], label_xy[1], f"{beta:.2f}", fontsize=12, ha="center", va="center", rotation=label_rotation)


def plot_sem_panel(ax, coef: Mapping[str, object], title: str = "", *, show_snow: bool = True):
    """Plot one PLS-SEM pathway panel."""
    pos = {
        "VPD": (0.14, 0.74),
        "SM": (0.50, 0.89),
        "Ta": (0.86, 0.74),
        "CO2": (0.14, 0.49),
        "LAI": (0.50, 0.49),
        "Snow": (0.86, 0.49),
        "SEN_NR": (0.14, 0.20),
        "Rad": (0.50, 0.20),
        "SEN_R": (0.86, 0.20),
        "SEN_TOT": (0.50, -0.01),
    }
    if not show_snow:
        pos.pop("Snow")

    node_colors = {name: "#f5efe2" for name in pos}
    node_colors.update({"LAI": "#b9d6ae", "SEN_NR": "#a9c8e4", "SEN_R": "#f19991", "SEN_TOT": "lightgray"})
    labels = {"SEN_NR": r"Sen$_{nr}$", "SEN_R": r"Sen$_r$", "SEN_TOT": r"Sen$_{tot}$"}

    direct_paths = [
        ("SM", "VPD", "SM->VPD", (-0.015, 0.03), 18),
        ("Ta", "SM", "Ta->SM", (0.03, 0.020), -10),
        ("Ta", "VPD", "Ta->VPD", (0.05, 0.035), 0),
        ("Ta", "LAI", "Ta->LAI", (0.015, 0.04), 27),
        ("Ta", "Snow", "Ta->SNOW", (0.035, 0.00), -90),
        ("SM", "LAI", "SM->LAI", (-0.03, -0.02), -90),
        ("VPD", "LAI", "VPD->LAI", (-0.02, 0.045), -28),
        ("CO2", "LAI", "CO2->LAI", (0.00, 0.035), 0),
        ("Rad", "LAI", "RAD->LAI", (0.03, 0.00), 88),
        ("Snow", "LAI", "SNOW->LAI", (0.00, 0.035), 0),
        ("CO2", "SEN_NR", "CO2->SEN_NR", (-0.045, 0.00), -90),
        ("Rad", "SEN_NR", "RAD->SEN_NR", (0.00, 0.035), 0),
        ("LAI", "SEN_NR", "LAI->SEN_NR", (-0.045, 0.00), 25),
        ("Rad", "SEN_R", "RAD->SEN_R", (0.00, 0.035), 0),
        ("Snow", "SEN_R", "SNOW->SEN_R", (0.035, 0.00), -90),
        ("LAI", "SEN_R", "LAI->SEN_R", (0.04, 0.00), -30),
    ]
    for source, target, path, offset, rotation in direct_paths:
        if source not in pos or target not in pos:
            continue
        draw_arrow(
            ax,
            pos,
            source,
            target,
            get_beta(coef, path),
            label_offset=offset,
            label_rotation=rotation,
            arrowstyle="<|-|>" if path == "SM->VPD" else "-|>",
            linestyle="-" if get_sig(coef, path) else "--",
        )

    routed_paths = [
        ([(pos["SM"][0], pos["SM"][1] + 0.02), (0.00, 0.91), (0.00, 0.20), pos["SEN_NR"]], "SM->SEN_NR", (0.24, 0.94), 0),
        ([(pos["VPD"][0] - 0.08, pos["VPD"][1]), (0.02, 0.74), (0.02, 0.20), pos["SEN_NR"]], "VPD->SEN_NR", (0.04, 0.61), -90),
        ([(pos["SM"][0], pos["SM"][1] + 0.02), (1.02, 0.91), (1.02, 0.20), pos["SEN_R"]], "SM->SEN_R", (0.74, 0.94), 0),
        ([(pos["SEN_NR"][0], pos["SEN_NR"][1] - 0.06), (0.14, -0.01), pos["SEN_TOT"]], "SEN_NR->SEN_TOT", (0.28, 0.03), 0),
        ([(pos["SEN_R"][0], pos["SEN_R"][1] - 0.06), (0.86, -0.01), pos["SEN_TOT"]], "SEN_R->SEN_TOT", (0.70, 0.03), 0),
    ]
    for points, path, label_xy, rotation in routed_paths:
        if path.startswith("SM->SEN_R") and "SEN_R" not in pos:
            continue
        draw_routed_arrow(
            ax,
            points,
            get_beta(coef, path),
            label_xy=label_xy,
            label_rotation=rotation,
            linestyle="-" if get_sig(coef, path) else "--",
        )

    for node, xy in pos.items():
        draw_node(ax, xy, labels.get(node, node), node_colors[node])

    ax.set_title(title, fontsize=13, pad=2)
    ax.set_xlim(-0.01, 1.03)
    ax.set_ylim(-0.11, 1.00)
    ax.axis("off")
    return ax


def move_axis(ax, *, dx=0.0, dy=0.0, dw=0.0, dh=0.0):
    """Move or resize an axis in figure coordinates."""
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width + dw, pos.height + dh])
    return ax


def build_figure(cfg: FigureConfig):
    """Create and save Figure 4."""
    os.chdir(cfg.base_dir)
    focus_region, left_label, right_label = panel_titles(cfg.panel_mode)
    ds_dict = load_attribution_datasets(cfg)

    aridity = xr.open_dataarray("AI_180x360.nc")
    dry_mask, humid_mask = build_dry_humid_masks(aridity, ds_dict["LSTtotal"])

    df_beta, df_contribution = build_effect_tables(
        ds_dict,
        focus_region=focus_region,
        dry_mask=dry_mask,
        humid_mask=humid_mask,
        pval_threshold=cfg.pval_threshold,
    )

    sem_df = pd.read_csv(sem_path(cfg.panel_mode))
    region = sem_region_name(cfg.panel_mode)
    coef_snow = coef_dict_from_table(sem_df, region, "snow_affected")
    coef_nosnow = coef_dict_from_table(sem_df, region, "snow_free")

    plot_settings()
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), height_ratios=[1, 1, 1.2])
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    axes = axes.ravel()

    plot_effect_panel(
        axes[0],
        df_beta,
        panel_name=left_label,
        value_name="beta",
        ylim=(-0.7, 0.7),
        ylabel="Effect on interannual variability\n(unitless std. coef.)",
        title=left_label,
        show_legend=True,
        mean_mode=cfg.mean_mode,
    )
    plot_effect_panel(
        axes[1],
        df_beta,
        panel_name=right_label,
        value_name="beta",
        ylim=(-0.7, 0.7),
        ylabel=None,
        title=right_label,
        mean_mode=cfg.mean_mode,
    )
    plot_effect_panel(
        axes[2],
        df_contribution,
        panel_name=left_label,
        value_name="contribution",
        ylim=(-0.35, 0.35),
        ylabel=r"Contribution to temporal trend" "\n" r"[K ($\mathrm{m}^2\,\mathrm{m}^{-2}$)$^{-1}$ decade$^{-1}$]",
        mean_mode=cfg.mean_mode,
    )
    plot_effect_panel(
        axes[3],
        df_contribution,
        panel_name=right_label,
        value_name="contribution",
        ylim=(-0.35, 0.35),
        ylabel=None,
        mean_mode=cfg.mean_mode,
    )

    plot_sem_panel(axes[4], coef_snow, "", show_snow=True)
    plot_sem_panel(axes[5], coef_nosnow, "", show_snow=False)
    move_axis(axes[4], dx=-0.02, dy=-0.02, dw=0.04, dh=0.04)
    move_axis(axes[5], dx=-0.02, dy=-0.02, dw=0.04, dh=0.04)

    for ax, label in zip(axes, "abcdef"):
        x = -0.02 if label in "ef" else -0.05
        y = 1.00 if label in "ef" else 1.08
        ax.text(x, y, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    outfig = output_path(cfg.panel_mode)
    outfig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfig, bbox_inches="tight")
    print(f"Saved figure: {outfig}")
    return fig


def main():
    cfg = FigureConfig(panel_mode="humid")
    build_figure(cfg)


if __name__ == "__main__":
    main()
