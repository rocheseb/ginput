from __future__ import print_function, division

import datetime as dt
from datetime import datetime as dtime
from glob import glob
import numpy as np
import os
import re

import pandas as pd

from ...common_utils import mod_utils


_mydir = os.path.abspath(os.path.dirname(__file__))


class ACEFileError(Exception):
    pass


def find_ace_file(ace_dir, ace_specie):
    ace_files = glob(os.path.join(ace_dir, '*.nc'))
    matching_files = [f for f in ace_files if f.lower().endswith('{}.nc'.format(ace_specie.lower()))]
    if len(matching_files) < 1:
        raise ACEFileError('Could not find an ACE file for specie "{}" in directory {}'.format(ace_specie, ace_dir))
    elif len(matching_files) > 1:
        raise ACEFileError('Found multiple ACE files for specie "{}" in directory {}'.format(ace_specie, ace_dir))
    else:
        return matching_files[0]


def read_ace_var(nc_handle, varname, qflags):
    data = nc_handle.variables[varname][:].filled(np.nan)
    if qflags is not None:
        data[qflags > 1] = np.nan
    # replace fill values
    data[data < -900] = np.nan
    return data


def read_ace_date(ace_nc_handle, out_type=dt.datetime):
    """
    Read datetimes from an ACE-FTS file

    :param ace_nc_handle: the handle to a netCDF4 dataset for an ACE-FTS file. (Must have variables year, month, day,
     and hour.)
    :type ace_nc_handle: :class:`netCDF4.Dataset`

    :param out_type: the type to return the dates as. May be any time that meets two criteria:

        1. Must be able to be called as ``out_type(year, month, day)`` where year, month, and day are integers to
           produce a datetime.
        2. Must be able to be added to a :class:`datetime.timedelta`

    :return: a numpy array of dates, as type ``out_type``.
    :rtype: :class:`numpy.ndarray`
    """
    ace_years = ace_nc_handle.variables['year'][:].filled(np.nan)
    ace_months = ace_nc_handle.variables['month'][:].filled(np.nan)
    ace_days = ace_nc_handle.variables['day'][:].filled(np.nan)

    ace_hours = ace_nc_handle.variables['hour'][:].filled(np.nan)
    ace_hours = ace_hours.astype(np.float64)  # timedelta demands a 64-bit float, can't be 32-bit

    dates = [out_type(y, m, d) + dt.timedelta(hours=h) for y, m, d, h in zip(ace_years, ace_months, ace_days, ace_hours)]
    return np.array(dates)


def read_ace_latlon(ace_nc_handle, clip=True):
    def clip_vec(v, span):
        adj = span[1] - span[0]
        v[v<span[0]] += adj
        v[v>span[1]] -= adj
        return v

    lon = ace_nc_handle.variables['longitude'][:].filled(np.nan)
    lat = ace_nc_handle.variables['latitude'][:].filled(np.nan)

    if clip:
        lon = clip_vec(lon, [-180.0, 180.0])
        lat = clip_vec(lat, [-90.0, 90.0])

    return lon, lat


def read_ace_theta(ace_nc_handle, qflags=None):
    temperature = read_ace_var(ace_nc_handle, 'temperature', qflags=qflags)
    pressure = read_ace_var(ace_nc_handle, 'pressure', qflags=qflags) * 1013.25  # Pressure given in atm, need hPa
    return mod_utils.calculate_potential_temperature(pressure, temperature)


def get_date_lon_lat_from_dirname(dirname):
    dirname = os.path.basename(dirname.rstrip(os.path.sep))
    datestr = re.search(r'^\d{8}', dirname).group()
    lonstr = re.search(r'(?<=_)[\d\.]+[WE]', dirname).group()
    latstr = re.search(r'(?<=_)[\d\.]+[NS]', dirname).group()

    date = dt.datetime.strptime(datestr, '%Y%m%d')
    lon = mod_utils.format_lon(lonstr)
    lat = mod_utils.format_lat(latstr)

    return date, lon, lat


def find_matching_val_profile(prior_lon, prior_lat, prior_date, val_lons, val_lats, val_dates):
    """
    Find the index of the validation profile matching a prior profile

    :param prior_lon: the longitude of the prior profile
    :type prior_lon: float

    :param prior_lat: the latitude of the prior profile
    :type prior_lat: float

    :param prior_date: the date of the prior profile
    :type prior_date: datetime-like

    :param val_lons: vector of longitudes of the available validation profiles
    :type val_lons: array-like(float)

    :param val_lats: vector of latitudes of the available validation profiles
    :type val_lats: array-like(float)

    :param val_dates: vector of dates of the available validation profiles
    :type val_dates: array-like(datetime-like)

    :return: the matching val profile index
    :rtype: int
    :raises RuntimeError: no matching or too many matching profiles were found
    """
    xx = (val_dates >= prior_date) & (val_dates < (prior_date + dt.timedelta(days=1))) \
         & np.isclose(val_lons, prior_lon) & np.isclose(val_lats, prior_lat)
    if np.sum(xx) < 1:
        raise RuntimeError('Could not find a profile at lon/lat {}/{} on {}'.format(
            prior_lon, prior_lat, prior_date
        ))
    elif np.sum(xx) > 1:
        raise RuntimeError('Found multiple profiles matching lon/lat {}/{} on {}'.format(
            prior_lon, prior_lat, prior_date
        ))
    return np.flatnonzero(xx)


def get_matching_val_profiles(prior_lons, prior_lats, prior_dates, prior_alts, val_lons, val_lats, val_dates, val_alts,
                              val_profiles, val_prof_error=None, interp_to_alt=True):
    """
    Create an array of validation profiles matched up with the given prior profiles

    :param prior_lons: vector of longitudes for the prior profiles
    :type prior_lons: 1D array-like of floats

    :param prior_lats: vector of latitudes for the prior profiles
    :type prior_lats: 1D array-like of floats

    :param prior_dates: vector of dates for the prior profiles. Assumed to be midnight times, but more generally will
     match validation profiles that occur any time between [prior_date, prior_date+24 h).
    :type prior_dates: 1D array-like of datetime-like objects

    :param prior_alts: array of altitudes that each prior is defined on. Must be 2D, such that iterating over it returns
     each priors altitude vector in turn.
    :type prior_alts: 2D array-like of floats

    :param val_lons: vector of longitudes for the validation profiles
    :type val_lons: 1D array-like of floats

    :param val_lats: vector of latitudes for the validation profiles
    :type val_lats: 1D array-like of floats

    :param val_dates: vector of datetimes for the validation profiles
    :type val_dates: 1D array-like of datetime-like objects

    :param val_alts: array of altitudes for the validation profiles. Must follow the same format as the priors'
     altitudes.
    :type val_alts: 2D array-like of floats

    :param val_profiles: the profiles used in the validation. Must follow the same format as the altitude arrays.
    :type val_profiles: 2D array-like of floats

    :param val_prof_error: the array of errors in the validation profiles. Must follow the same format as the altitude
     arrays.
    :type val_prof_error: 2D array-like of floats

    :param interp_to_alt: if ``True``, then the validation profiles are interpolated to the same altitudes as the
     priors. If ``False``, they are left on their own altitude levels.
    :type interp_to_alt: bool

    :return: 2D arrays of the matched validation profile concentrations, errors, and altitudes, a 1D array of the
     validation profiles date times. If ``val_prof_error`` is not given, the errors will be NaNs. The altitudes define
     the altitudes of the validation profiles, whether or not they've been interpolated.
    :rtype: 4 :class:`numpy.ndarray` instances, three 2D and one 1D.
    """
    n_profs = np.size(prior_lons)
    n_out_levels = np.shape(prior_alts)[1] if interp_to_alt else np.size(val_alts)
    out_profiles = np.full([n_profs, n_out_levels], np.nan)
    out_prof_errors = np.full([n_profs, n_out_levels], np.nan)
    out_datetimes = np.full([n_profs], None)

    for idx, (this_lon, this_lat, this_date, this_alt) in enumerate(zip(prior_lons, prior_lats, prior_dates, prior_alts)):
        xx = find_matching_val_profile(this_lon, this_lat, this_date, val_lons, val_lats, val_dates)

        this_prof = val_profiles[xx, :]
        this_prof_error = val_prof_error[xx, :] if val_prof_error is not None else np.full(this_prof.shape, np.nan)
        if interp_to_alt:
            this_prof = np.interp(this_alt, val_alts, this_prof.squeeze())
            this_prof_error = np.interp(this_alt, val_alts, this_prof_error.squeeze())

        out_profiles[idx, :] = this_prof
        out_prof_errors[idx, :] = this_prof_error
        out_datetimes[idx] = val_dates[xx]

    if not interp_to_alt:
        prior_alts = np.tile(val_alts.reshape(1, -1), [n_profs, 1])

    return out_profiles, out_prof_errors, prior_alts, out_datetimes


def read_atm_file(filename, limit_to_meas=False):
    """
    Read a .atm file

    :param filename: the path to the .atm file
    :type filename: str

    :param limit_to_meas: if ``True``, values outside the aircraft floor and ceiling altitudes are removed.
    :type limit_to_meas: bool

    :return: the data from the .atm file and the header information
    :rtype: :class:`pandas.DataFrame`, dict
    """
    header_info = dict()
    with open(filename, 'r') as fobj:
        # skip line 1
        fobj.readline()
        line_num = 0
        for line in fobj:
            line_num += 1
            if re.match(r'\-+$', line):
                # Stop reading the header at a line of all dashes
                break
            else:
                k, v = [s.strip() for s in line.split(':', 1)]
                header_info[k] = convert_atm_value(v)

    data = pd.read_csv(filename, header=line_num + 1, na_values='NAN')
    if limit_to_meas:
        bottom_alt = header_info['aircraft_floor_m']
        top_alt = header_info['aircraft_ceiling_m']
        xx = (data['Altitude_m'] >= bottom_alt) & (data['Altitude_m'] <= top_alt)
        data = data[xx]

    return data, header_info


def convert_atm_value(val):
    """
    Convert values in the .atm file header to Python types.

    Currently supports date (yyyy-mm-dd), datetime (yyyy-mm-dd HH:MM:SS), and float values.

    :param val: the value to convert
    :type val: str

    :return: the converted value
    """
    conv_fxns = (lambda v: dtime.strptime(v, '%Y-%m-%d %H:%M:%S'),
                 lambda v: dtime.strptime(v, '%Y-%m-%d'),
                 float)
    for fxn in conv_fxns:
        try:
            new_val = fxn(val)
        except ValueError:
            continue
        else:
            return new_val

    return val