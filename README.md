# Greening Cooling Efficiency

Code supporting the manuscript **“Biophysical cooling benefit of Earth greening halved by declining efficiency under rising atmospheric carbon dioxide”**.

This repository contains the core processing and plotting workflows used to
quantify changes in vegetation cooling efficiency during 2001–2024. Cooling
efficiency is represented by the sensitivity of land surface temperature
(LST) to leaf area index (LAI), `∂LST/∂LAI`.

> **Publication status:** The manuscript is under review. The final citation,
> DOI, data archive, and software license will be added upon publication.

## Overview

The workflow:

1. estimates local `∂LST/∂LAI` using space-for-time substitution and a
   Theil–Sen estimator;
2. separates total sensitivity into radiative and nonradiative components
   using the surface energy balance;
3. attributes temporal variability and trends to environmental drivers using
   ridge regression and partial least-squares methods; and
4. decomposes the cumulative greening-induced LST effect into LAI-driven and
   cooling-efficiency-driven components.

The main analysis uses annual, seasonal, and monthly satellite and reanalysis
data on a common 0.05° grid. Products used for trend and attribution analyses
are aggregated to 1°.

## Repository structure

```text
GreeningCoolingEfficiency/
├── 1.Methods/
│   ├── 1_Sensitivity_LST.py
│   ├── 2_Sensitivity_LST_Radiative.py
│   ├── 3_Sensitivity_LST_Nonradiative.py
│   ├── 4_Ridge_regression.py
│   ├── 5_Decompose_delta_LST.py
│   └── 6_PLS_SEM_pixel.R
└── 2.Plotting/
    ├── Figure.01.py
    ├── Figure.02.py
    ├── Figure.03.py
    ├── Figure.04.py
    └── Figure.05.py
```

### Processing scripts

| Script | Purpose |
|---|---|
| [`1_Sensitivity_LST.py`](1.Methods/1_Sensitivity_LST.py) | Estimates total LST sensitivity to LAI using adaptive-window space-for-time substitution and the Theil–Sen slope. |
| [`2_Sensitivity_LST_Radiative.py`](1.Methods/2_Sensitivity_LST_Radiative.py) | Estimates albedo sensitivity to LAI and converts it to the radiative contribution to LST sensitivity. |
| [`3_Sensitivity_LST_Nonradiative.py`](1.Methods/3_Sensitivity_LST_Nonradiative.py) | Constructs `LE+H = Rn-G`, estimates its sensitivity to LAI, and converts it to the nonradiative contribution to LST sensitivity. |
| [`4_Ridge_regression.py`](1.Methods/4_Ridge_regression.py) | Attributes total, radiative, and nonradiative sensitivity changes to environmental predictors using standardized ridge regression; PLS regression is included as a robustness test. |
| [`5_Decompose_delta_LST.py`](1.Methods/5_Decompose_delta_LST.py) | Decomposes cumulative LAI-induced LST change into LAI-driven and sensitivity-driven components and calculates Theil–Sen trends. |
| [`6_PLS_SEM_pixel.R`](1.Methods/6_PLS_SEM_pixel.R) | Fits pixel-wise PLS path models for snow-affected and snow-free regions and maps direct, indirect, and total effects. |

### Plotting scripts

| Script | Reproduces |
|---|---|
| [`Figure.01.py`](2.Plotting/Figure.01.py) | Regional annual time series and seasonal changes in total LST sensitivity to LAI. |
| [`Figure.02.py`](2.Plotting/Figure.02.py) | Spatial patterns, climate-zone distributions, and climate-space relationships of mean sensitivity and its trend. |
| [`Figure.03.py`](2.Plotting/Figure.03.py) | Temporal and spatial changes in radiative and nonradiative sensitivity components. |
| [`Figure.04.py`](2.Plotting/Figure.04.py) | Ridge-regression attribution and PLS-SEM pathways. |
| [`Figure.05.py`](2.Plotting/Figure.05.py) | Greening-induced LST changes and their LAI-driven and efficiency-driven components. |

## Methods summary

### LST sensitivity to LAI

For every target 0.05° pixel and time step, neighboring pixels are retained
when they:

- have the same aggregated land-cover class;
- differ in land-cover fraction by less than 10 percentage points;
- differ in elevation by no more than 100 m; and
- differ in LAI by at least 0.05 m² m⁻².

The search begins with a 0.5° window and expands by 0.1° up to 1° until at
least 10 valid neighbors are available. Sensitivity is the median of the
valid pairwise slopes:

```math
S = \frac{\partial \mathrm{LST}}{\partial \mathrm{LAI}}
  = \mathrm{median}_i
    \left(\frac{\Delta \mathrm{LST}_i}{\Delta \mathrm{LAI}_i}\right).
```

Global 0.5th–99.5th percentile filtering is applied before aggregation to 1°.
Only adequately covered 1° cells are retained. Temporal trends are estimated
with the Theil–Sen estimator, and significance is evaluated using a two-sided
Kendall rank test.

### Energy-balance decomposition

Net radiation and turbulent heat flux are calculated as

```math
R_n = SW_{\downarrow}(1-\alpha)
    + \varepsilon LW_{\downarrow}
    - \varepsilon\sigma \mathrm{LST}^4,
```

```math
LE+H = R_n-G.
```

The radiative and nonradiative LST-equivalent sensitivities are obtained from
the albedo and `LE+H` sensitivities using the linearized longwave response:

```math
\lambda_0 = \frac{1}{4\varepsilon\sigma\mathrm{LST}^3},
```

```math
S_{\mathrm{rad}} =
-\lambda_0 SW_{\downarrow}\frac{\partial\alpha}{\partial\mathrm{LAI}},
\qquad
S_{\mathrm{nonrad}} =
-\lambda_0\frac{\partial(LE+H)}{\partial\mathrm{LAI}}.
```

The nonradiative workflow supports alternative ground-heat-flux treatments,
including no ground heat flux, FLDAS, and FAO-56.

### Statistical attribution

At each 1° grid cell, standardized ridge regression (`alpha = 0.1`) relates
total, radiative, and nonradiative sensitivity to air temperature, soil
moisture, vapor-pressure deficit, incoming shortwave radiation, atmospheric
CO₂, and snow cover where applicable. Snow-affected and snow-free regions are
analyzed separately; snow cover is included only in the snow-affected model.

The physical trend contribution of predictor `i` is calculated as

```math
C_i = \beta_i\,\delta(X_i^*)\,\sigma_S,
```

where `βᵢ` is its standardized coefficient, `δ(Xᵢ*)` is its standardized
trend, and `σS` is the temporal standard deviation of the target sensitivity.
PLS regression and pixel-wise PLS path modelling provide complementary
robustness and pathway analyses.

### Greening-induced LST decomposition

LAI and sensitivity are first aggregated using the same valid-pixel mask and
cosine-latitude weights. For each interval,

```math
\Delta \mathrm{LAI}(t)=\mathrm{LAI}(t)-\mathrm{LAI}(t-1),
\qquad
S_{\mathrm{mid}}(t)=\frac{S(t)+S(t-1)}{2}.
```

The cumulative total, LAI-driven, and sensitivity-driven effects are

```math
\mathrm{LST}^*(T)=
\sum_{t=2}^{T}\Delta\mathrm{LAI}(t)S_{\mathrm{mid}}(t),
```

```math
\mathrm{LST}_{\mathrm{LAI}}^*(T)=
\sum_{t=2}^{T}\Delta\mathrm{LAI}(t)S_{\mathrm{ref}},
```

```math
\mathrm{LST}_{\mathrm{EFF}}^*(T)=
\mathrm{LST}^*(T)-\mathrm{LST}_{\mathrm{LAI}}^*(T).
```

For seasonal analyses, Southern Hemisphere seasons are shifted by six months
to align phenological phases with Northern Hemisphere seasons.

## Input data

The large input datasets are not distributed in this repository. Users must
download and preprocess them according to the manuscript and configure the
paths in the scripts.

| Dataset | Main use | Source |
|---|---|---|
| GLASS LAI V6 | Main LAI product | [GLASS](https://glass.hku.hk/) |
| GLOBMAP LAI V3 | LAI robustness analysis | [Zenodo](https://zenodo.org/records/12698637) |
| GIMMS LAI4g | LAI robustness analysis | [Zenodo](https://zenodo.org/records/8281930) |
| MOD11C3.061 | LST and emissivity | [NASA LP DAAC](https://doi.org/10.5067/MODIS/MOD11C3.061) |
| MCD43C3.061 | Black-sky albedo | [NASA LP DAAC](https://doi.org/10.5067/MODIS/MCD43C3.061) |
| MOD16A2GF.061 | Latent heat flux robustness analysis | [NASA LP DAAC](https://doi.org/10.5067/MODIS/MOD16A2GF.061) |
| MCD12C1.006 | Land cover and land-cover fractions | [NASA LP DAAC](https://doi.org/10.5067/MODIS/MCD12C1.006) |
| ERA5-Land | Climate and radiation variables | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means) |
| TerraClimate | Incoming shortwave radiation | [Climatology Lab](https://www.climatologylab.org/terraclimate.html) |
| NOAA atmospheric CO₂ | Attribution predictor | [NOAA GML](https://gml.noaa.gov/ccgg/trends/data.html) |
| SRTM DEM | Elevation constraint and pressure estimation | [NASA Earthdata](https://www.earthdata.nasa.gov/data/instruments/srtm) |
| PML-V2.2a and GLASS ET | Latent heat flux robustness analyses | GEE collection `projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a` and [GLASS](https://glass.hku.hk/) |

All higher-frequency products must be aggregated to the selected monthly,
seasonal, or annual scale before the sensitivity scripts are run. Inputs to a
given calculation must share the same temporal coordinates and 0.05° grid.
The current repository starts from these preprocessed inputs.

## Software environment

The Python workflows were developed for Python 3.10. A representative setup is:

```bash
conda create -n greening-cooling -c conda-forge \
  python=3.10 numpy pandas xarray dask netcdf4 scipy numba \
  scikit-learn statsmodels geopandas matplotlib seaborn cartopy
conda activate greening-cooling
```

The PLS-SEM workflow requires R and these packages:

```r
install.packages(c("terra", "ncdf4", "dplyr", "tidyr", "plspm"))
```

The plotting and regional aggregation scripts also import two project helper
modules, `plot_utils.py` and `da_utils.py`. Place them in a local utility
directory and set `VEG_LST_UTILS_DIR` accordingly.

## Configuration

Most Python scripts accept environment variables instead of requiring edits to
the source:

```bash
export VEG_LST_DATA_DIR=/path/to/GreeningCoolingEfficiency_data
export VEG_LST_UTILS_DIR=/path/to/project_utils
```

Review the configuration block near the top of every script before execution,
especially the selected LAI/LST products, temporal scale, years, worker count,
input filenames, and output directory. In `6_PLS_SEM_pixel.R`, set `root`,
`window_size`, and `detrend` explicitly.

## Running the workflow

Run the processing scripts as standalone programs from the repository root:

```bash
python 1.Methods/1_Sensitivity_LST.py
python 1.Methods/2_Sensitivity_LST_Radiative.py
python 1.Methods/3_Sensitivity_LST_Nonradiative.py
python 1.Methods/4_Ridge_regression.py
python 1.Methods/5_Decompose_delta_LST.py
Rscript 1.Methods/6_PLS_SEM_pixel.R
```

After the required intermediate products have been generated, reproduce the
main figures with:

```bash
for n in 01 02 03 04 05; do
  python "2.Plotting/Figure.${n}.py"
done
```

Global 0.05° processing is computationally and I/O intensive. The sensitivity
scripts use Numba and block-based multiprocessing; they should be executed as
scripts on a machine with sufficient memory and fast storage. Increasing the
worker count beyond the storage bandwidth may reduce performance.

## Reproducibility notes

- Trend variables are generally stored per year; plotting scripts may multiply
  them by 10 for presentation per decade. Check NetCDF metadata before reuse.
- Radiation and ground heat flux follow the sign conventions documented in the
  processing scripts. Do not combine inputs with inconsistent conventions.
- Seasonal inputs are pre-aggregated. The sensitivity scripts do not infer or
  repair temporal alignment.
- Intermediate NetCDF and CSV filenames are part of the plotting-script
  configuration and must be retained or updated consistently.
- The scripts preserve exact spatial and temporal alignment checks to avoid
  silent coordinate reassignment.

## Citation

If you use this code, please cite the associated paper. Complete bibliographic
information will be added here after publication.

```text
Zhang, C., et al. Biophysical cooling benefit of Earth greening halved by declining efficiency under rising atmospheric carbon dioxide. [Journal, volume, pages, DOI to be added].
```

## Contact

For code-related questions, please open an issue in this repository or contact
[Chao Zhang]([chaoz.geo@gmail.com](https://github.com/Chao21)).

<!--
Before public archival release:
1. Add plot_utils.py and da_utils.py, or remove those external dependencies.
2. Add a pinned environment.yml or requirements.txt.
3. Add a LICENSE file and replace the citation placeholder.
4. Add the permanent processed-data archive DOI and checksums, if available.
5. Confirm that every default path is portable and contains no private data.
-->
