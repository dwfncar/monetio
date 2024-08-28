"""TEMPO L2 NO2 data file reader.

History:


"""

import logging
import os
import sys
from collections import OrderedDict
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


def _open_one_dataset(fname, variable_dict):
    """Read locally stored INTELSAT TEMPO NO2 level 2
    Parameters
    ----------
    fname : str
        Local path to netCDF4 (HDF5) file.
    variable_dict : dict

    Returns
    -------
    ds : xr.Dataset
    """
    print("reading " + fname)

    ds = xr.Dataset()

    dso = Dataset(fname, "r")
    lon_var = dso.groups["geolocation"]["longitude"]
    lat_var = dso.groups["geolocation"]["latitude"]
    time_var = dso.groups["geolocation"]["time"]

    ds["lon"] = (
        ("x", "y"),
        lon_var[:].squeeze(),
        {"long_name": lon_var.long_name, "units": lon_var.units},
    )
    ds["lat"] = (
        ("x", "y"),
        lat_var[:].squeeze(),
        {"long_name": lat_var.long_name, "units": lat_var.units},
    )
    ds["time"] = (
        ("time",),
        time_var[:].squeeze(),
        {"long_name": time_var.long_name, "units": time_var.units},
    )
    ds["lon"] = ds["lon"].fillna(-999.99)
    ds["lat"] = ds["lat"].fillna(-999.99)
    ds = ds.set_coords(["time", "lon", "lat"])

    ds.attrs["reference_time_string"] = dso.time_coverage_start
    ds.attrs["granule_number"] = dso.granule_num
    ds.attrs["scan_num"] = dso.scan_num

    for varname in variable_dict:
        if varname in [
            "main_data_quality_flag",
            "vertical_column_troposphere",
            "vertical_column_stratosphere",
            "vertical_column_troposphere_uncertainty",
        ]:
            values_var = dso.groups["product"][varname]
        elif varname in [
            "latitude_bounds",
            "longitude_bounds",
            "solar_zenith_angle",
            "solar_azimuth_angle",
            "viewing_zenith_angle",
            "viewing_azimuth_angle",
            "relative_azimuth_angle",
        ]:
            values_var = dso.groups["geolocation"][varname]
        elif varname in [
            "vertical_column_total",
            "vertical_column_total_uncertainty",
            "fitted_slant_column",
            "fitted_slant_column_uncertainty",
            "snow_ice_fraction",
            "terrain_height",
            "ground_pixel_quality_flag",
            "surface_pressure",
            "tropopause_pressure",
            "scattering_weights",
            "gas_profile",
            "albedo",
            "temperature_profile",
            "amf_total",
            "amf_diagnositc_flag",
            "eff_cloud_fraction",
            "amf_cloud_fraction",
            "amf_cloud_pressure",
            "amf_troposphere",
            "amf_stratosphere",
        ]:
            values_var = dso.groups["support_data"][varname]
        elif varname in ["fit_rms_residual", "fit_convergence_flag"]:
            values_var = dso.groups["qa_statistics"][varname]
        values = values_var[:].squeeze()
        fv = values_var.getncattr("_FillValue")
        if not np.isfinite(fv):
            values[:][values[:] == fv] = np.nan

        if "scale" in variable_dict[varname]:
            values[:] = variable_dict[varname]["scale"] * values[:]

        if "minimum" in variable_dict[varname]:
            minimum = variable_dict[varname]["minimum"]
            values = np.ma.masked_where(values < minimum, values)

        if "maximum" in variable_dict[varname]:
            maximum = variable_dict[varname]["maximum"]
            values = np.ma.masked_where(values > maximum, values)

        if "corner" in values_var.dimensions:
            ds[varname] = (("x", "y", "corner"), values, values_var.__dict__)
        elif "swt_level" in values_var.dimensions:
            ds[varname] = (("x", "y", "swt_level"), values, values_var.__dict__)
        else:
            ds[varname] = (("x", "y"), values, values_var.__dict__)

        if "quality_flag_max" in variable_dict[varname]:
            ds.attrs["quality_flag"] = varname
            ds.attrs["quality_thresh_max"] = variable_dict[varname]["quality_flag_max"]

    dso.close()
    if "surface_pressure" in list(variable_dict.keys()):
        ds["pressure_in_Pa"] = calculate_pressure(ds)

    return xr.decode_cf(ds)


def calculate_pressure(ds):
    """Calculates pressure at layer and delta_pressure of layer

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    layer_pressure: xr.DataArray
        Pressure at layer in Pa
    delta_pressure: xr.DataArray
        Difference of pressure in layer
    """
    HPA2PA = 100
    surf_pressure = ds["surface_pressure"]
    eta_a = surf_pressure.Eta_A
    eta_b = surf_pressure.Eta_B
    n_layers = len(surf_pressure.Eta_A)
    press = np.zeros((surf_pressure.shape[0], surf_pressure.shape[1], n_layers))
    for k in range(0, n_layers):
        press[:, :, k] = (eta_a[k] + eta_b[k] * surf_pressure.values) * HPA2PA
    pressure = xr.DataArray(
        data=press,
        dims=("x", "y", "swt_level_stagg"),
        coords={
            "lon": (["x", "y"], surf_pressure.lon.values),
            "lat": (["x", "y"], surf_pressure.lat.values),
        },
        attrs={
            "long_name": "pressure",
            "units": "Pa",
            "valid_min": "0",
            "valid_max": 120000,
            "_FillValue": np.nan,
        },
    )
    return pressure


def apply_quality_flag(ds):
    """Mask variables in place based on quality flag

    Parameters
    ----------
    ds : xr.Dataset
    """
    if "quality_flag" in ds.attrs:
        quality_flag = ds[ds.attrs["quality_flag"]]
        quality_thresh_max = ds.attrs["quality_thresh_max"]

        # Apply the quality thersh maximum to all variables in ds
        for varname in ds:
            if varname != ds.attrs["quality_flag"]:
                logging.debug(varname)
                ds[varname] = ds[varname].where(~(quality_flag > quality_thresh_max))


def open_dataset(fnames, variable_dict, debug=False):
    if debug:
        logging_level = logging.DEBUG
        logging.basicConfig(stream=sys.stdout, level=logging_level)

    if isinstance(fnames, Path):
        fnames = fnames.as_posix()
    if isinstance(variable_dict, str):
        variable_dict = {variable_dict: {}}
    elif isinstance(variable_dict, dict):
        pass
    else:  # Assume sequence
        variable_dict = {varname: {} for varname in variable_dict}

    for subpath in fnames.split("/"):
        if "$" in subpath:
            envvar = subpath.replace("$", "")
            envval = os.getenv(envvar)
            if envval is None:
                raise Exception("Environment variable not defined: " + subpath)
            else:
                fnames = fnames.replace(subpath, envval)

    print(fnames)
    files = sorted(glob(fnames))

    granules = OrderedDict()

    for file in files:
        granule = _open_one_dataset(file, variable_dict)
        apply_quality_flag(granule)
        granules[granule.attrs["reference_time_string"]] = granule

    return granules
