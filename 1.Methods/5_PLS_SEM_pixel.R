# ============================================================
# Formal PLS-PM + Spatial Mapping Hybrid Framework in R
# Author: Chao Zhang
# Updated: 2026-05-13 (CO2 + TA, no P, moving window retained)
# using package: plspm
#
# Main features in this version:
#   (1) Retains 3x3 moving-window pooling to increase valid samples
#   (2) Uses regime-consistent windows: snow pixels only borrow from snow pixels;
#       no-snow pixels only borrow from no-snow pixels
#   (3) Does NOT detrend variables; plspm standardizes retained samples with scaled=TRUE
#   (4) Removes precipitation (P)
#   (5) Keeps air temperature (TA) pathways:
#         TA -> SM, VPD, LAI, and SNOW [snow model only]
#   (6) Adds CO2 pathways:
#         CO2 -> LAI and CO2 -> SEN_NR
#   (7) Computes SM <-> VPD correlation outside plspm
#   (8) Computes direct, indirect, and total effects for SEN_NR, SEN_R, and SEN_TOT
#   (9) Computes manual single-indicator GoF = sqrt(mean(R2_endogenous))
#  (10) Writes one combined NetCDF with complete variable metadata
#
# Model structure:
#
#   Upstream climate / forcing:
#       TA  -> SM
#       TA  -> VPD
#       TA  -> SNOW           [snow model only]
#       CO2 -> LAI
#       TA  -> LAI
#       SM  -> LAI
#       VPD -> LAI
#       RAD -> LAI
#       SNOW-> LAI            [snow model only]
#
#   Non-radiative branch:
#       CO2 -> SEN_NR
#       SM  -> SEN_NR
#       VPD -> SEN_NR
#       RAD -> SEN_NR
#       LAI -> SEN_NR
#
#   Radiative branch:
#       RAD  -> SEN_R
#       SNOW -> SEN_R         [snow model only]
#       LAI  -> SEN_R
#       SM   -> SEN_R
#
#   Total sensitivity:
#       SEN_NR -> SEN_TOT
#       SEN_R  -> SEN_TOT
#
#   Separate non-structural bidirectional association:
#       SM <-> VPD, computed as correlation, not fitted in plspm
# ============================================================

# -----------------------------
# 0. Packages
# -----------------------------
library(terra)
library(ncdf4)
library(dplyr)
library(tidyr)
library(plspm)
library(parallel)

# -----------------------------
# 1. User settings
# -----------------------------
root <- "E:/00MyWork/00_2PostDoc/01PilotStudy/04Greenning-T/data"
outdir <- file.path(root, "processed/PLS_PM_R_20260828_CO2_TA_noP_MovingWindow")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

fp_sen_tot <- file.path(root, "processed/Sensitivity_20260208/Sensitivity_Annual_LSTdailymean_LAI_1d_gapfilled.nc")
fp_sen_r   <- file.path(root, "processed/Sensitivity_LST_Energy_LAI_20260325/Albedo/Annual/Sensitivity_Annual_LSTdailymean_LAI_from_Albedo_1d.nc")
fp_sen_nr  <- file.path(root, "processed/Sensitivity_LST_Energy_LAI_20260325/NonRadiative/Annual/Sensitivity_Annual_LSTdailymean_LAI_from_LEH_FAO56_1d.nc")
fp_lai     <- file.path(root, "LAI_GLASS/1d/LAI_GLASS_Annual_1d_2001_2024.nc")
fp_era     <- file.path(root, "ERA5_Land/ERA5_Land_Annual_1d_2000_2024.nc")
fp_co2     <- file.path(root, "CO2/CO2_Annual_1d_2001_2024.nc")
co2_varname <- "co2"

# Moving-window setting.
# window_size = 3 means 3x3 moving-window pooling. With 24 years, the maximum sample size is 216.
# window_size = 1 means pure pixel-wise fitting.

window_size <- 1
# Detrending option.
# TRUE  = detrend each annual pixel series before PLS-PM
# FALSE = use original annual series
detrend <- FALSE # TRUE, FALSE

if (window_size == 1) {
  min_valid_samples <- 15
} else if (window_size >= 3) {
  min_valid_samples <- 25
} else {
  stop("window_size must be 1 or >= 3.")
}

# Output nc file name by specifying the window size
detrend_tag <- ifelse(detrend, "detrend", "nodetrend")

nc_file <- file.path(
  outdir,
  sprintf("pixelwise_plspm_maps_winsize%d_%s.nc", window_size, detrend_tag)
)

# Regime and fitting settings
snow_threshold <- 1.0
std_thresh <- 1e-8
n_cores <- max(1, detectCores() - 1)
set.seed(42)

# -----------------------------
# 2. Utility functions
# -----------------------------
log_msg <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
}

get_var_3d <- function(ncfile, varname, time_start = NULL, time_end = NULL) {
  r <- rast(ncfile, subds = varname)
  
  if (!is.null(time_start) && !is.null(time_end)) {
    r <- r[[time_start:time_end]]
  }
  
  ntime <- nlyr(r)
  ny <- nrow(r)
  nx <- ncol(r)
  
  arr <- array(NA_real_, dim = c(ntime, ny, nx))
  
  for (tt in seq_len(ntime)) {
    # terra cell values are row-wise; restore to [lat row, lon col]
    arr[tt, , ] <- matrix(
      values(r[[tt]], mat = FALSE),
      nrow = ny,
      ncol = nx,
      byrow = TRUE
    )
  }
  
  list(arr = arr, template = r[[1]], raster = r)
}

detrend_array_3d <- function(arr) {
  nt <- dim(arr)[1]
  ny <- dim(arr)[2]
  nx <- dim(arr)[3]
  
  out <- array(NA_real_, dim = dim(arr))
  time_idx <- seq_len(nt)
  
  for (j in seq_len(ny)) {
    for (i in seq_len(nx)) {
      x <- arr[, j, i]
      ok <- is.finite(x)
      
      if (sum(ok) < 3) next
      if (sd(x[ok], na.rm = TRUE) < 1e-12) {
        out[ok, j, i] <- 0
        next
      }
      
      fit <- lm(x[ok] ~ time_idx[ok])
      x_detrended <- x
      x_detrended[ok] <- residuals(fit)
      out[, j, i] <- x_detrended
    }
  }
  
  out
}



safe_sd <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  sd(x, na.rm = TRUE)
}

safe_cor <- function(x, y, method = "pearson") {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  if (sd(x[ok]) < 1e-12 || sd(y[ok]) < 1e-12) return(NA_real_)
  suppressWarnings(cor(x[ok], y[ok], method = method))
}

build_path_matrix <- function(latent_vars, model_formulas) {
  path_mat <- matrix(
    0,
    nrow = length(latent_vars),
    ncol = length(latent_vars),
    dimnames = list(latent_vars, latent_vars)
  )
  
  for (end_var in names(model_formulas)) {
    rhs_vars <- all.vars(model_formulas[[end_var]])
    path_mat[end_var, rhs_vars] <- 1
  }
  
  # plspm requires the matrix to be lower triangular.
  # We stop rather than silently deleting upper-triangle paths.
  if (any(path_mat[upper.tri(path_mat, diag = TRUE)] != 0)) {
    stop("Path matrix is not lower triangular. Reorder latent_vars so causes precede effects.")
  }
  path_mat
}

extract_path <- function(pls_obj, from, to) {
  pc <- pls_obj$path_coefs
  if (is.null(pc)) return(NA_real_)
  if (!(to %in% rownames(pc)) || !(from %in% colnames(pc))) return(NA_real_)
  as.numeric(pc[to, from])
}

extract_r2 <- function(pls_obj, lv) {
  tab <- pls_obj$inner_summary
  if (is.null(tab)) return(NA_real_)
  if (!(lv %in% rownames(tab))) return(NA_real_)
  if (!("R2" %in% colnames(tab))) return(NA_real_)
  as.numeric(tab[lv, "R2"])
}

compute_manual_gof <- function(r2_vec) {
  r2_vec <- as.numeric(r2_vec)
  r2_vec <- r2_vec[is.finite(r2_vec)]
  if (length(r2_vec) == 0) return(NA_real_)
  sqrt(mean(r2_vec))
}

extract_effect_generic <- function(pls_obj, from, to, which = c("direct", "indirect", "total")) {
  which <- match.arg(which)
  eff <- pls_obj$effects
  if (is.null(eff)) return(NA_real_)
  
  if ("relationships" %in% colnames(eff)) {
    key <- paste(from, "->", to)
    sub <- eff[eff$relationships == key, , drop = FALSE]
    if (nrow(sub) == 0) return(NA_real_)
    if (!(which %in% colnames(sub))) return(NA_real_)
    return(as.numeric(sub[[which]][1]))
  }
  
  if (all(c("from", "to") %in% colnames(eff))) {
    sub <- eff[eff$from == from & eff$to == to, , drop = FALSE]
    if (nrow(sub) == 0) return(NA_real_)
    if (!(which %in% colnames(sub))) return(NA_real_)
    return(as.numeric(sub[[which]][1]))
  }
  
  NA_real_
}

extract_direct_effect   <- function(pls_obj, from, to) extract_effect_generic(pls_obj, from, to, "direct")
extract_indirect_effect <- function(pls_obj, from, to) extract_effect_generic(pls_obj, from, to, "indirect")
extract_total_effect    <- function(pls_obj, from, to) extract_effect_generic(pls_obj, from, to, "total")

set_raster_metadata <- function(r, varname, longname, unit = "") {
  names(r) <- as.character(varname)[1]
  varnames(r) <- as.character(varname)[1]
  longnames(r) <- as.character(longname)[1]
  units(r) <- as.character(unit)[1]
  r
}

fix_netcdf_lonlat_names <- function(ncfile) {
  stopifnot(file.exists(ncfile))
  
  nc <- ncdf4::nc_open(ncfile, write = TRUE)
  vars <- names(nc$var)
  if ("easting" %in% vars) ncdf4::ncvar_rename(nc, "easting", "lon")
  if ("northing" %in% vars) ncdf4::ncvar_rename(nc, "northing", "lat")
  ncdf4::nc_close(nc)
  
  nc <- ncdf4::nc_open(ncfile, write = TRUE)
  on.exit(try(ncdf4::nc_close(nc), silent = TRUE), add = FALSE)
  if ("lon" %in% names(nc$var)) {
    ncdf4::ncatt_put(nc, "lon", "long_name", "longitude")
    ncdf4::ncatt_put(nc, "lon", "standard_name", "longitude")
    ncdf4::ncatt_put(nc, "lon", "units", "degrees_east")
    ncdf4::ncatt_put(nc, "lon", "axis", "X")
  }
  if ("lat" %in% names(nc$var)) {
    ncdf4::ncatt_put(nc, "lat", "long_name", "latitude")
    ncdf4::ncatt_put(nc, "lat", "standard_name", "latitude")
    ncdf4::ncatt_put(nc, "lat", "units", "degrees_north")
    ncdf4::ncatt_put(nc, "lat", "axis", "Y")
  }
  invisible(TRUE)
}

write_model_netcdf <- function(outfile, var_names, out_list, template, meta, global_atts) {
  r_list <- vector("list", length(var_names))
  names(r_list) <- var_names
  
  for (k in seq_along(var_names)) {
    nm <- var_names[k]
    r <- rast(template)
    values(r) <- as.vector(t(out_list[[nm]]))
    
    this_meta <- meta %>% filter(varname == nm)
    if (nrow(this_meta) == 0) {
      this_meta <- tibble(varname = nm, longname = nm, unit = "1", description = "No description provided.")
    }
    
    r <- set_raster_metadata(
      r,
      varname = this_meta$varname[1],
      longname = this_meta$longname[1],
      unit = this_meta$unit[1]
    )
    r_list[[k]] <- r
  }
  
  sds_out <- sds(r_list)
  
  desc_atts <- paste0(
    "desc_", meta$varname[match(var_names, meta$varname)], "=",
    meta$description[match(var_names, meta$varname)]
  )
  desc_atts <- desc_atts[!is.na(desc_atts)]
  
  writeCDF(
    x = sds_out,
    filename = outfile,
    overwrite = TRUE,
    atts = c(global_atts, desc_atts),
    compression = 4,
    missval = NA,
    prec = "float",
    tags = TRUE
  )
  
  fix_netcdf_lonlat_names(outfile)
}

# -----------------------------
# 3. Read data
# -----------------------------
log_msg("Reading datasets...")

sen_tot_obj <- get_var_3d(fp_sen_tot, "sens_1deg_mean")
template <- sen_tot_obj$template
crs(template) <- "OGC:CRS84"
nx <- ncol(template)
ny <- nrow(template)

co2_raw     <- get_var_3d(fp_co2, co2_varname)$arr
ta_raw      <- get_var_3d(fp_era, "t2m",   2, 25)$arr
rad_raw     <- get_var_3d(fp_era, "ssrd",  2, 25)$arr
sm_raw      <- get_var_3d(fp_era, "swvl1", 2, 25)$arr
vpd_raw     <- get_var_3d(fp_era, "vpd",   2, 25)$arr
snowc_raw   <- get_var_3d(fp_era, "snowc", 2, 25)$arr
lai_raw     <- get_var_3d(fp_lai, "LAI_1deg_mean")$arr
sen_r_raw   <- get_var_3d(fp_sen_r, "sens_1deg_mean")$arr
sen_nr_raw  <- get_var_3d(fp_sen_nr, "sens_1deg_mean")$arr
sen_tot_raw <- sen_tot_obj$arr

snowc_for_mask <- snowc_raw

# Basic dimension check
ref_dim <- dim(sen_tot_raw)
for (nm in c("co2_raw", "ta_raw", "rad_raw", "sm_raw", "vpd_raw", "snowc_raw", "lai_raw", "sen_r_raw", "sen_nr_raw")) {
  if (!all(dim(get(nm)) == ref_dim)) {
    stop(paste("Dimension mismatch:", nm, paste(dim(get(nm)), collapse = "x"), "vs reference", paste(ref_dim, collapse = "x")))
  }
}

# Regime classification by climatological snow cover
# Always use original, non-detrended snow cover for snow/no-snow classification.
snowc_clim <- apply(snowc_for_mask, c(2, 3), function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
})
is_snow_mask <- ifelse(is.na(snowc_clim), FALSE, snowc_clim > snow_threshold)

cat("Snow mask summary:\n")
print(table(is_snow_mask, useNA = "ifany"))

cat("Snow-only pixels:\n")
print(sum(is_snow_mask == TRUE, na.rm = TRUE))

if (detrend) {
  log_msg("Detrending annual pixel series before PLS-PM...")
  
  co2_raw     <- detrend_array_3d(co2_raw)
  ta_raw      <- detrend_array_3d(ta_raw)
  rad_raw     <- detrend_array_3d(rad_raw)
  sm_raw      <- detrend_array_3d(sm_raw)
  vpd_raw     <- detrend_array_3d(vpd_raw)
  snowc_raw   <- detrend_array_3d(snowc_raw)
  lai_raw     <- detrend_array_3d(lai_raw)
  sen_r_raw   <- detrend_array_3d(sen_r_raw)
  sen_nr_raw  <- detrend_array_3d(sen_nr_raw)
  sen_tot_raw <- detrend_array_3d(sen_tot_raw)
} else {
  log_msg("Using non-detrended annual series before PLS-PM...")
}

# ============================================================
# Diagnostic: simple moving-window mean
# Put the test code here
# ============================================================

simple_window_mean <- function(arr, is_snow_mask, window_size = 3) {
  nt <- dim(arr)[1]
  ny <- dim(arr)[2]
  nx <- dim(arr)[3]
  r_win <- floor((window_size - 1) / 2)
  
  out <- matrix(NA_real_, ny, nx)
  
  for (j in seq_len(ny)) {
    for (i in seq_len(nx)) {
      focal_is_snow <- is_snow_mask[j, i]
      if (is.na(focal_is_snow)) next
      
      vals_all <- c()
      
      jj_seq <- max(1, j - r_win):min(ny, j + r_win)
      ii_seq <- max(1, i - r_win):min(nx, i + r_win)
      
      for (jj in jj_seq) {
        for (ii in ii_seq) {
          neigh_snow <- is_snow_mask[jj, ii]
          if (is.na(neigh_snow)) next
          if (neigh_snow != focal_is_snow) next
          
          vals_all <- c(vals_all, arr[, jj, ii])
        }
      }
      
      out[j, i] <- mean(vals_all, na.rm = TRUE)
    }
  }
  
  out
}

# -----------------------------
# 4. Model path matrices
# -----------------------------
latent_snow <- c("CO2", "TA", "RAD", "SM", "VPD", "SNOW", "LAI", "SEN_NR", "SEN_R", "SEN_TOT")
formulas_snow <- list(
  SM      = ~ TA,
  VPD     = ~ TA,
  SNOW    = ~ TA,
  LAI     = ~ CO2 + TA + SM + VPD + RAD + SNOW,
  SEN_NR  = ~ CO2 + SM + VPD + RAD + LAI,
  SEN_R   = ~ RAD + SNOW + LAI + SM,
  SEN_TOT = ~ SEN_NR + SEN_R
)
snow_path_mat <- build_path_matrix(latent_snow, formulas_snow)
snow_blocks <- list(
  c("co2"), c("ta"), c("rad"), c("sm"), c("vpd"), c("snowc"),
  c("lai"), c("sen_nr"), c("sen_r"), c("sen_tot")
)
snow_modes <- rep("A", length(snow_blocks))

latent_nosnow <- c("CO2", "TA", "RAD", "SM", "VPD", "LAI", "SEN_NR", "SEN_R", "SEN_TOT")
formulas_nosnow <- list(
  SM      = ~ TA,
  VPD     = ~ TA,
  LAI     = ~ CO2 + TA + SM + VPD + RAD,
  SEN_NR  = ~ CO2 + SM + VPD + RAD + LAI,
  SEN_R   = ~ RAD + LAI + SM,
  SEN_TOT = ~ SEN_NR + SEN_R
)
nosnow_path_mat <- build_path_matrix(latent_nosnow, formulas_nosnow)
nosnow_blocks <- list(
  c("co2"), c("ta"), c("rad"), c("sm"), c("vpd"),
  c("lai"), c("sen_nr"), c("sen_r"), c("sen_tot")
)
nosnow_modes <- rep("A", length(nosnow_blocks))

# -----------------------------
# 5. Output names
# -----------------------------
shared_names <- c(
  # Upstream paths
  "beta_TA_to_SM",
  "beta_TA_to_VPD",
  "beta_CO2_to_LAI",
  "beta_TA_to_LAI",
  "beta_SM_to_LAI",
  "beta_VPD_to_LAI",
  "beta_RAD_to_LAI",
  
  # Non-radiative paths
  "beta_CO2_to_SEN_NR",
  "beta_SM_to_SEN_NR",
  "beta_VPD_to_SEN_NR",
  "beta_RAD_to_SEN_NR",
  "beta_LAI_to_SEN_NR",
  
  # Radiative paths
  "beta_RAD_to_SEN_R",
  "beta_LAI_to_SEN_R",
  "beta_SM_to_SEN_R",
  
  # Total sensitivity paths
  "beta_SEN_NR_to_SEN_TOT",
  "beta_SEN_R_to_SEN_TOT",
  
  # Direct / indirect / total effects to SEN_NR
  "direct_CO2_to_SEN_NR",
  "indirect_CO2_to_SEN_NR",
  "total_CO2_to_SEN_NR",
  "direct_TA_to_SEN_NR",
  "indirect_TA_to_SEN_NR",
  "total_TA_to_SEN_NR",
  "direct_RAD_to_SEN_NR",
  "indirect_RAD_to_SEN_NR",
  "total_RAD_to_SEN_NR",
  "direct_SM_to_SEN_NR",
  "indirect_SM_to_SEN_NR",
  "total_SM_to_SEN_NR",
  "direct_VPD_to_SEN_NR",
  "indirect_VPD_to_SEN_NR",
  "total_VPD_to_SEN_NR",
  "direct_LAI_to_SEN_NR",
  "indirect_LAI_to_SEN_NR",
  "total_LAI_to_SEN_NR",
  
  # Direct / indirect / total effects to SEN_R
  "direct_CO2_to_SEN_R",
  "indirect_CO2_to_SEN_R",
  "total_CO2_to_SEN_R",
  "direct_TA_to_SEN_R",
  "indirect_TA_to_SEN_R",
  "total_TA_to_SEN_R",
  "direct_RAD_to_SEN_R",
  "indirect_RAD_to_SEN_R",
  "total_RAD_to_SEN_R",
  "direct_SM_to_SEN_R",
  "indirect_SM_to_SEN_R",
  "total_SM_to_SEN_R",
  "direct_VPD_to_SEN_R",
  "indirect_VPD_to_SEN_R",
  "total_VPD_to_SEN_R",
  "direct_LAI_to_SEN_R",
  "indirect_LAI_to_SEN_R",
  "total_LAI_to_SEN_R",
  
  # Total effects to SEN_TOT
  "total_CO2_to_SEN_TOT",
  "total_TA_to_SEN_TOT",
  "total_RAD_to_SEN_TOT",
  "total_SM_to_SEN_TOT",
  "total_VPD_to_SEN_TOT",
  "total_LAI_to_SEN_TOT",
  
  # Correlations and diagnostics
  "corr_SM_VPD",
  "corr_SEN_NR_SEN_TOT",
  "corr_SEN_R_SEN_TOT",
  "R2_SM",
  "R2_VPD",
  "R2_LAI",
  "R2_SEN_NR",
  "R2_SEN_R",
  "R2_SEN_TOT",
  "GOF",
  "n_valid_samples",
  "n_pixels_in_window",
  "fit_status",
  "is_snow"
)

snow_only_names <- c(
  "beta_TA_to_SNOW",
  "beta_SNOW_to_LAI",
  "beta_SNOW_to_SEN_R",
  "direct_SNOW_to_SEN_NR",
  "indirect_SNOW_to_SEN_NR",
  "total_SNOW_to_SEN_NR",
  "direct_SNOW_to_SEN_R",
  "indirect_SNOW_to_SEN_R",
  "total_SNOW_to_SEN_R",
  "total_SNOW_to_SEN_TOT",
  "R2_SNOW"
)

all_out_names <- c(shared_names, snow_only_names)

# -----------------------------
# 6. Parallel moving-window worker
# -----------------------------
process_one_latitude <- function(j,
                                 nx, ny, window_size,
                                 is_snow_mask,
                                 co2_raw, ta_raw, sm_raw, vpd_raw, rad_raw, lai_raw, snowc_raw,
                                 sen_nr_raw, sen_r_raw, sen_tot_raw,
                                 min_valid_samples, std_thresh,
                                 snow_path_mat, snow_blocks, snow_modes,
                                 nosnow_path_mat, nosnow_blocks, nosnow_modes,
                                 all_out_names) {
  
  row_out <- lapply(all_out_names, function(x) matrix(NA_real_, nrow = 1, ncol = nx))
  names(row_out) <- all_out_names
  r_win <- floor((window_size - 1) / 2)
  
  for (i in seq_len(nx)) {
    focal_is_snow <- is_snow_mask[j, i]
    if (is.na(focal_is_snow)) next
    
    row_out$is_snow[1, i] <- as.numeric(focal_is_snow)
    row_out$fit_status[1, i] <- 0
    
    df_list <- list()
    for (jj in max(1, j - r_win):min(ny, j + r_win)) {
      for (ii in max(1, i - r_win):min(nx, i + r_win)) {
        neigh_snow <- is_snow_mask[jj, ii]
        if (is.na(neigh_snow)) next
        if (neigh_snow != focal_is_snow) next
        
        df_list[[length(df_list) + 1]] <- tibble(
          co2     = co2_raw[, jj, ii],
          ta      = ta_raw[, jj, ii],
          rad     = rad_raw[, jj, ii],
          sm      = sm_raw[, jj, ii],
          vpd     = vpd_raw[, jj, ii],
          snowc   = snowc_raw[, jj, ii],
          lai     = lai_raw[, jj, ii],
          sen_nr  = sen_nr_raw[, jj, ii],
          sen_r   = sen_r_raw[, jj, ii],
          sen_tot = sen_tot_raw[, jj, ii]
        )
      }
    }
    
    if (length(df_list) == 0) next
    
    df_pix <- bind_rows(df_list)
    row_out$n_pixels_in_window[1, i] <- length(df_list)
    
    needed_cols <- if (focal_is_snow) {
      c("co2", "ta", "rad", "sm", "vpd", "snowc", "lai", "sen_nr", "sen_r", "sen_tot")
    } else {
      c("co2", "ta", "rad", "sm", "vpd", "lai", "sen_nr", "sen_r", "sen_tot")
    }
    
    df_pix <- df_pix %>% drop_na(all_of(needed_cols))
    row_out$n_valid_samples[1, i] <- nrow(df_pix)
    if (nrow(df_pix) < min_valid_samples) next
    
    sds <- sapply(df_pix %>% select(all_of(needed_cols)), safe_sd)
    if (any(is.na(sds)) || any(sds < std_thresh)) next
    
    row_out$corr_SM_VPD[1, i] <- safe_cor(df_pix$sm, df_pix$vpd)
    row_out$corr_SEN_NR_SEN_TOT[1, i] <- safe_cor(df_pix$sen_nr, df_pix$sen_tot)
    row_out$corr_SEN_R_SEN_TOT[1, i] <- safe_cor(df_pix$sen_r, df_pix$sen_tot)
    
    fit <- try({
      if (focal_is_snow) {
        plspm(
          Data = df_pix %>% select(co2, ta, rad, sm, vpd, snowc, lai, sen_nr, sen_r, sen_tot),
          path_matrix = snow_path_mat,
          blocks = snow_blocks,
          modes = snow_modes,
          scaled = TRUE,
          boot.val = FALSE
        )
      } else {
        plspm(
          Data = df_pix %>% select(co2, ta, rad, sm, vpd, lai, sen_nr, sen_r, sen_tot),
          path_matrix = nosnow_path_mat,
          blocks = nosnow_blocks,
          modes = nosnow_modes,
          scaled = TRUE,
          boot.val = FALSE
        )
      }
    }, silent = TRUE)
    
    if (inherits(fit, "try-error")) next
    
    # Direct structural paths
    row_out$beta_TA_to_SM[1, i] <- extract_path(fit, "TA", "SM")
    row_out$beta_TA_to_VPD[1, i] <- extract_path(fit, "TA", "VPD")
    row_out$beta_CO2_to_LAI[1, i] <- extract_path(fit, "CO2", "LAI")
    row_out$beta_TA_to_LAI[1, i] <- extract_path(fit, "TA", "LAI")
    row_out$beta_SM_to_LAI[1, i] <- extract_path(fit, "SM", "LAI")
    row_out$beta_VPD_to_LAI[1, i] <- extract_path(fit, "VPD", "LAI")
    row_out$beta_RAD_to_LAI[1, i] <- extract_path(fit, "RAD", "LAI")
    
    row_out$beta_CO2_to_SEN_NR[1, i] <- extract_path(fit, "CO2", "SEN_NR")
    row_out$beta_SM_to_SEN_NR[1, i] <- extract_path(fit, "SM", "SEN_NR")
    row_out$beta_VPD_to_SEN_NR[1, i] <- extract_path(fit, "VPD", "SEN_NR")
    row_out$beta_RAD_to_SEN_NR[1, i] <- extract_path(fit, "RAD", "SEN_NR")
    row_out$beta_LAI_to_SEN_NR[1, i] <- extract_path(fit, "LAI", "SEN_NR")
    
    row_out$beta_RAD_to_SEN_R[1, i] <- extract_path(fit, "RAD", "SEN_R")
    row_out$beta_LAI_to_SEN_R[1, i] <- extract_path(fit, "LAI", "SEN_R")
    row_out$beta_SM_to_SEN_R[1, i] <- extract_path(fit, "SM", "SEN_R")
    
    row_out$beta_SEN_NR_to_SEN_TOT[1, i] <- extract_path(fit, "SEN_NR", "SEN_TOT")
    row_out$beta_SEN_R_to_SEN_TOT[1, i] <- extract_path(fit, "SEN_R", "SEN_TOT")
    
    if (focal_is_snow) {
      row_out$beta_TA_to_SNOW[1, i] <- extract_path(fit, "TA", "SNOW")
      row_out$beta_SNOW_to_LAI[1, i] <- extract_path(fit, "SNOW", "LAI")
      row_out$beta_SNOW_to_SEN_R[1, i] <- extract_path(fit, "SNOW", "SEN_R")
    }
    
    # Effects to SEN_NR
    for (drv in c("CO2", "TA", "RAD", "SM", "VPD", "LAI")) {
      row_out[[paste0("direct_", drv, "_to_SEN_NR")]][1, i] <- extract_direct_effect(fit, drv, "SEN_NR")
      row_out[[paste0("indirect_", drv, "_to_SEN_NR")]][1, i] <- extract_indirect_effect(fit, drv, "SEN_NR")
      row_out[[paste0("total_", drv, "_to_SEN_NR")]][1, i] <- extract_total_effect(fit, drv, "SEN_NR")
    }
    
    # Effects to SEN_R
    for (drv in c("CO2", "TA", "RAD", "SM", "VPD", "LAI")) {
      row_out[[paste0("direct_", drv, "_to_SEN_R")]][1, i] <- extract_direct_effect(fit, drv, "SEN_R")
      row_out[[paste0("indirect_", drv, "_to_SEN_R")]][1, i] <- extract_indirect_effect(fit, drv, "SEN_R")
      row_out[[paste0("total_", drv, "_to_SEN_R")]][1, i] <- extract_total_effect(fit, drv, "SEN_R")
    }
    
    # Total effects to SEN_TOT
    for (drv in c("CO2", "TA", "RAD", "SM", "VPD", "LAI")) {
      row_out[[paste0("total_", drv, "_to_SEN_TOT")]][1, i] <- extract_total_effect(fit, drv, "SEN_TOT")
    }
    
    if (focal_is_snow) {
      row_out$direct_SNOW_to_SEN_NR[1, i] <- extract_direct_effect(fit, "SNOW", "SEN_NR")
      row_out$indirect_SNOW_to_SEN_NR[1, i] <- extract_indirect_effect(fit, "SNOW", "SEN_NR")
      row_out$total_SNOW_to_SEN_NR[1, i] <- extract_total_effect(fit, "SNOW", "SEN_NR")
      row_out$direct_SNOW_to_SEN_R[1, i] <- extract_direct_effect(fit, "SNOW", "SEN_R")
      row_out$indirect_SNOW_to_SEN_R[1, i] <- extract_indirect_effect(fit, "SNOW", "SEN_R")
      row_out$total_SNOW_to_SEN_R[1, i] <- extract_total_effect(fit, "SNOW", "SEN_R")
      row_out$total_SNOW_to_SEN_TOT[1, i] <- extract_total_effect(fit, "SNOW", "SEN_TOT")
      row_out$R2_SNOW[1, i] <- extract_r2(fit, "SNOW")
    }
    
    # Diagnostics
    row_out$R2_SM[1, i] <- extract_r2(fit, "SM")
    row_out$R2_VPD[1, i] <- extract_r2(fit, "VPD")
    row_out$R2_LAI[1, i] <- extract_r2(fit, "LAI")
    row_out$R2_SEN_NR[1, i] <- extract_r2(fit, "SEN_NR")
    row_out$R2_SEN_R[1, i] <- extract_r2(fit, "SEN_R")
    row_out$R2_SEN_TOT[1, i] <- extract_r2(fit, "SEN_TOT")
    
    if (focal_is_snow) {
      row_out$GOF[1, i] <- compute_manual_gof(c(
        row_out$R2_SM[1, i], row_out$R2_VPD[1, i], row_out$R2_SNOW[1, i],
        row_out$R2_LAI[1, i], row_out$R2_SEN_NR[1, i],
        row_out$R2_SEN_R[1, i], row_out$R2_SEN_TOT[1, i]
      ))
    } else {
      row_out$GOF[1, i] <- compute_manual_gof(c(
        row_out$R2_SM[1, i], row_out$R2_VPD[1, i], row_out$R2_LAI[1, i],
        row_out$R2_SEN_NR[1, i], row_out$R2_SEN_R[1, i],
        row_out$R2_SEN_TOT[1, i]
      ))
    }
    
    row_out$fit_status[1, i] <- 1
  }
  
  row_out
}

# -----------------------------
# 7. Parallel run
# -----------------------------
log_msg(sprintf("Starting parallel moving-window PLS-PM with window size %d using %d cores...", window_size, n_cores))

cl <- makeCluster(n_cores)
clusterEvalQ(cl, { library(dplyr); library(tidyr); library(plspm); NULL })
clusterExport(cl, varlist = c(
  "safe_sd", "safe_cor", "extract_path", "extract_r2", "compute_manual_gof",
  "extract_effect_generic", "extract_direct_effect", "extract_indirect_effect", "extract_total_effect",
  "process_one_latitude",
  "nx", "ny", "window_size", "is_snow_mask",
  "co2_raw", "ta_raw", "sm_raw", "vpd_raw", "rad_raw", "lai_raw", "snowc_raw",
  "sen_nr_raw", "sen_r_raw", "sen_tot_raw",
  "min_valid_samples", "std_thresh",
  "snow_path_mat", "snow_blocks", "snow_modes",
  "nosnow_path_mat", "nosnow_blocks", "nosnow_modes",
  "all_out_names"
), envir = environment())

row_results <- parLapply(cl, seq_len(ny), function(j) {
  process_one_latitude(
    j = j,
    nx = nx,
    ny = ny,
    window_size = window_size,
    is_snow_mask = is_snow_mask,
    co2_raw = co2_raw,
    ta_raw = ta_raw,
    sm_raw = sm_raw,
    vpd_raw = vpd_raw,
    rad_raw = rad_raw,
    lai_raw = lai_raw,
    snowc_raw = snowc_raw,
    sen_nr_raw = sen_nr_raw,
    sen_r_raw = sen_r_raw,
    sen_tot_raw = sen_tot_raw,
    min_valid_samples = min_valid_samples,
    std_thresh = std_thresh,
    snow_path_mat = snow_path_mat,
    snow_blocks = snow_blocks,
    snow_modes = snow_modes,
    nosnow_path_mat = nosnow_path_mat,
    nosnow_blocks = nosnow_blocks,
    nosnow_modes = nosnow_modes,
    all_out_names = all_out_names
  )
})
stopCluster(cl)

# -----------------------------
# 8. Reassemble outputs
# -----------------------------
log_msg("Reassembling outputs...")
out_list <- lapply(all_out_names, function(x) matrix(NA_real_, nrow = ny, ncol = nx))
names(out_list) <- all_out_names

for (j in seq_len(ny)) {
  for (nm in all_out_names) {
    out_list[[nm]][j, ] <- row_results[[j]][[nm]][1, ]
  }
}

# -----------------------------
# 9. Metadata table
# -----------------------------
make_meta <- function(varname, longname, unit = "1", description = longname) {
  tibble::tibble(varname = varname, longname = longname, unit = unit, description = description)
}

meta <- bind_rows(
  make_meta("beta_TA_to_SM", "Path coefficient: TA to SM", "1", "Standardized direct path from air temperature to soil moisture."),
  make_meta("beta_TA_to_VPD", "Path coefficient: TA to VPD", "1", "Standardized direct path from air temperature to vapor pressure deficit."),
  make_meta("beta_CO2_to_LAI", "Path coefficient: CO2 to LAI", "1", "Standardized direct path from CO2 to LAI."),
  make_meta("beta_TA_to_LAI", "Path coefficient: TA to LAI", "1", "Standardized direct path from air temperature to LAI."),
  make_meta("beta_SM_to_LAI", "Path coefficient: SM to LAI", "1", "Standardized direct path from soil moisture to LAI."),
  make_meta("beta_VPD_to_LAI", "Path coefficient: VPD to LAI", "1", "Standardized direct path from VPD to LAI."),
  make_meta("beta_RAD_to_LAI", "Path coefficient: RAD to LAI", "1", "Standardized direct path from radiation to LAI."),
  make_meta("beta_CO2_to_SEN_NR", "Path coefficient: CO2 to SEN_NR", "1", "Standardized direct physiological path from CO2 to non-radiative sensitivity."),
  make_meta("beta_SM_to_SEN_NR", "Path coefficient: SM to SEN_NR", "1", "Standardized direct path from soil moisture to non-radiative sensitivity."),
  make_meta("beta_VPD_to_SEN_NR", "Path coefficient: VPD to SEN_NR", "1", "Standardized direct path from VPD to non-radiative sensitivity."),
  make_meta("beta_RAD_to_SEN_NR", "Path coefficient: RAD to SEN_NR", "1", "Standardized direct path from radiation to non-radiative sensitivity."),
  make_meta("beta_LAI_to_SEN_NR", "Path coefficient: LAI to SEN_NR", "1", "Standardized direct path from LAI to non-radiative sensitivity."),
  make_meta("beta_RAD_to_SEN_R", "Path coefficient: RAD to SEN_R", "1", "Standardized direct path from radiation to radiative sensitivity."),
  make_meta("beta_LAI_to_SEN_R", "Path coefficient: LAI to SEN_R", "1", "Standardized direct path from LAI to radiative sensitivity."),
  make_meta("beta_SM_to_SEN_R", "Path coefficient: SM to SEN_R", "1", "Standardized direct path from soil moisture to radiative sensitivity."),
  make_meta("beta_SEN_NR_to_SEN_TOT", "Path coefficient: SEN_NR to SEN_TOT", "1", "Standardized direct path from non-radiative sensitivity to total sensitivity."),
  make_meta("beta_SEN_R_to_SEN_TOT", "Path coefficient: SEN_R to SEN_TOT", "1", "Standardized direct path from radiative sensitivity to total sensitivity."),
  make_meta("beta_TA_to_SNOW", "Path coefficient: TA to SNOW", "1", "Standardized direct path from air temperature to snow cover; snow model only."),
  make_meta("beta_SNOW_to_LAI", "Path coefficient: SNOW to LAI", "1", "Standardized direct path from snow cover to LAI; snow model only."),
  make_meta("beta_SNOW_to_SEN_R", "Path coefficient: SNOW to SEN_R", "1", "Standardized direct path from snow cover to radiative sensitivity; snow model only.")
)

# Add generic metadata for effect, correlation, R2, and diagnostic variables not listed above.
missing_meta <- setdiff(all_out_names, meta$varname)
for (nm in missing_meta) {
  if (grepl("^direct_", nm)) {
    meta <- bind_rows(meta, make_meta(nm, paste("Direct effect:", nm), "1", paste("Standardized direct effect variable", nm)))
  } else if (grepl("^indirect_", nm)) {
    meta <- bind_rows(meta, make_meta(nm, paste("Indirect effect:", nm), "1", paste("Standardized indirect effect variable", nm)))
  } else if (grepl("^total_", nm)) {
    meta <- bind_rows(meta, make_meta(nm, paste("Total effect:", nm), "1", paste("Standardized total effect variable", nm)))
  } else if (grepl("^corr_", nm)) {
    meta <- bind_rows(meta, make_meta(nm, paste("Correlation:", nm), "1", paste("Pearson correlation diagnostic", nm)))
  } else if (grepl("^R2_", nm)) {
    meta <- bind_rows(meta, make_meta(nm, paste("R-squared:", nm), "1", paste("Coefficient of determination", nm)))
  } else if (nm == "GOF") {
    meta <- bind_rows(meta, make_meta(nm, "Manual goodness of fit", "1", "Manual single-indicator GoF = sqrt(mean(R2_endogenous))."))
  } else if (nm == "n_valid_samples") {
    meta <- bind_rows(meta, make_meta(nm, "Number of valid samples", "count", "Number of complete annual samples retained from the regime-consistent moving window."))
  } else if (nm == "n_pixels_in_window") {
    meta <- bind_rows(meta, make_meta(nm, "Number of pixels in moving window", "count", "Number of same-regime pixels included in the moving-window sample."))
  } else if (nm == "fit_status") {
    meta <- bind_rows(meta, make_meta(nm, "Model fit status", "0/1", "Pixel-level fit status: 1=success, 0=failure."))
  } else if (nm == "is_snow") {
    meta <- bind_rows(meta, make_meta(nm, "Snow regime mask", "0/1", "Snow-region classification based on climatological annual snow cover threshold."))
  } else {
    meta <- bind_rows(meta, make_meta(nm, nm, "1", nm))
  }
}

# -----------------------------
# 10. Write NetCDF
# -----------------------------
log_msg("Writing NetCDF...")

global_atts <- c(
  "title=Moving-window pixel-wise PLS-PM outputs for LST sensitivity attribution",
  paste0("summary=Regime-consistent moving-window PLS path modeling using ",
         ifelse(detrend, "detrended", "non-detrended"),
         " annual series; variables are standardized internally by plspm with scaled=TRUE."),
  paste0("preprocessing=", ifelse(detrend, "linear detrending applied to each pixel-level annual series", "no detrending"),
         "; no manual z-scoring; plspm scaled=TRUE standardizes retained samples within each local fit"),
  paste0("detrend=", detrend),
  paste0("window_size=", window_size),
  paste0("min_valid_samples=", min_valid_samples),
  paste0("snow_threshold_percent=", snow_threshold),
  "model_upstream=TA -> SM, VPD, LAI, and SNOW [snow only]; CO2 -> LAI; SM+VPD+RAD(+SNOW) -> LAI",
  "model_nonradiative=CO2 + SM + VPD + RAD + LAI -> SEN_NR",
  "model_radiative_snow=RAD + SNOW + LAI + SM -> SEN_R",
  "model_radiative_nosnow=RAD + LAI + SM -> SEN_R",
  "model_total=SEN_NR + SEN_R -> SEN_TOT",
  "note_bidirectional_SM_VPD=The two-headed SM-VPD arrow is represented by Pearson correlation outside the plspm inner model.",
  "note_gof=Manual GoF is computed as sqrt(mean finite R2 of endogenous constructs)); this is a diagnostic, not the native plspm GoF."
)



write_model_netcdf(
  outfile = nc_file,
  var_names = all_out_names,
  out_list = out_list,
  template = template,
  meta = meta,
  global_atts = global_atts
)

log_msg("Done.")
log_msg(paste("File saved to:", nc_file))



sum(out_list$fit_status == 1, na.rm = TRUE)
range(out_list$n_valid_samples, na.rm = TRUE)
sum(is.finite(out_list[["beta_CO2_to_LAI"]]))


