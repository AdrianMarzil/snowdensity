#!/usr/bin/env python
# coding: utf-8

"""
Snow Water Equivalent (SWE) Calculator
--------------------------------------
Calculates SWE from snow depth using empirical power-law regression.

Inputs:
    - snow depth (mm)
    - temperature difference grid (from GDAL-readable file)
    - winter precipitation grid (from GDAL-readable file)
    - lat/lon coordinates (degrees)
    - date (year, month, day)

Author: Christina Aragon (original)
Refactored by: ChatGPT
Updated: 2025-07
"""

import numpy as np
from osgeo import gdal
from scipy.interpolate import RegularGridInterpolator
from datetime import date

# Load gridded data and prepare interpolators
def load_interpolators(td_path, pptwt_path):
    # Open temperature difference grid
    td_ds = gdal.Open(td_path)
    td = td_ds.ReadAsArray()

    # Open winter precipitation grid
    pptwt_ds = gdal.Open(pptwt_path)
    pptwt = pptwt_ds.ReadAsArray()

    # Extract grid geometry
    nrows, ncols = td.shape
    geotransform = td_ds.GetGeoTransform()
    xll = geotransform[0]
    yll = geotransform[3] + geotransform[5] * nrows
    clsz = geotransform[1]

    # Define longitude and latitude arrays
    ln = np.arange(xll, xll + ncols * clsz, clsz)
    lt = np.arange(yll, yll + nrows * clsz, clsz)
    la = np.flipud(lt)  # Flip to match raster orientation

    # Create interpolators
    f_td = RegularGridInterpolator((la, ln), td)
    f_ppt = RegularGridInterpolator((la, ln), pptwt)

    return f_td, f_ppt

# SWE computation function
def compute_swe(Y, M, D, H, LAT, LON, f_td, f_ppt):
    """
    Compute SWE and day-of-water-year (DOY) for single or multiple points.
    
    Inputs: Scalars or NumPy arrays
    Returns: SWE (mm), DOY
    """
    Y = np.array(Y)
    M = np.array(M)
    D = np.array(D)
    H = np.array(H)
    LAT = np.array(LAT)
    LON = np.array(LON)

    # Interpolate TD and PPTWT
    points = np.vstack((LAT, LON)).T
    TD = f_td(points)
    PPTWT = f_ppt(points)

    # Compute DOY
    doy = np.array([(date(y, m, d) - date(y, 9, 30)).days for y, m, d in zip(Y, M, D)])
    doy = np.where(doy < 0, doy + 365, doy)

    # Coefficients
    a = [0.0533, 0.948, 0.1701, -0.1314, 0.2922]  # accumulation
    b = [0.0481, 1.0395, 0.1699, -0.0461, 0.1804]  # ablation

    # SWE calculation
    acc_term = a[0] * H**a[1] * PPTWT**a[2] * TD**a[3] * doy**a[4] * (1 - np.tanh(.01*(doy - 180))) / 2
    abl_term = b[0] * H**b[1] * PPTWT**b[2] * TD**b[3] * doy**b[4] * (1 + np.tanh(.01*(doy - 180))) / 2
    SWE = acc_term + abl_term

    return SWE, doy
