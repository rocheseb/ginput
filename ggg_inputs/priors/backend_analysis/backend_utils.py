from __future__ import print_function, division

import datetime as dt
from glob import glob
import numpy as np
import os
import re
import sys

_mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_mydir, '..'))


import mod_utils


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
        data[qflags != 0] = np.nan
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




