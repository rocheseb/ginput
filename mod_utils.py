"""
Utility functions shared among the mod maker and prior generation code.

Compatibility notes:
    * We have found that Pandas version 0.20 is incompatible with this module. Pandas >= 0.24 works.
"""

from __future__ import print_function, division

import datetime as dt
from dateutil.relativedelta import relativedelta
import netCDF4 as ncdf
import numpy as np
from numpy import ma
import os
import pandas as pd
import re
from scipy.interpolate import interp1d, interp2d
import subprocess
import sys

import mod_constants as const

_std_model_pres_levels = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0, 800.0, 775.0, 750.0, 725.0,
                                   700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 150.0,
                                   100.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.7, 0.5,
                                   0.4, 0.3, 0.1])
# as of 2019-03-14, both GEOS-FP and MERRA-2 use the same standard 42 pressure levels
# (c.f. https://gmao.gsfc.nasa.gov/GMAO_products/documents/GEOS_5_FP_File_Specification_ON4v1_2.pdf page 52 and
# https://gmao.gsfc.nasa.gov/reanalysis/MERRA/docs/MERRA_File_Specification.pdf page 27 for GEOS-FP and MERRA-2
# respectively)
merra_pres_levels = _std_model_pres_levels
geosfp_pres_levels = _std_model_pres_levels
earth_radius = 6371  # kilometers


class ProgressBar(object):
    """
    Create a text-based progress bar

    An instance of this class can be used to print a text progress bar that does not need a new line for each progress
    step. It uses carriage returns to reset to the beginning of each line before printing the next. This therefore
    does not work well if other print statements occur in between calls to :meth:`print_bar`, the progress bar will
    either end up on a new line anyway or potentially overwrite previous print statements if they did not end with a
    newline.

    :param num_symbols: how many steps there should be in the progress bar. In other words, the progress bar will be
     complete when :meth:`print_bar` is called with ``num_symbols-1``.
    :type num_symbols: int

    :param prefix: a string to include before the beginning of each progress bar. The class will ensure that at least
     one space is present between the prefix and the progress bar, but will not add one if one is already present at
     the end of the prefix.
    :type prefix: str

    :param suffix: a string to include at the end of each progress bar. The class will ensure that at least one space
     is present between the progress bar and the suffix, but will not add one if one is already present at the beginning
     of the suffix.
    :type suffix: str

    :param add_one: if ``True``, the number of symbols printed in the progress bar is equal to ``i+1`` where ``i`` is
     the argument to :meth:`print_bar`. This works well with Python loops over ``i in range(n)``, since the last value
     of ``i`` will be ``n-1``, setting ``add_one`` to ``True`` ensures that a full progress bar is printed at the end.
    :type add_one: bool

    :param style: can be either '*' or 'counter'. The former prints a symbolic progress bar of the form:

        [*   ]
        [**  ]
        [*** ]
        [****]

     where the number of *'s is set by ``num_symbols``. The latter will instead print 'i/num_symbols' for each step.
    :type style: str
    """
    def __init__(self, num_symbols, prefix='', suffix='', add_one=True, style='*'):
        """
        See class help.
        """
        if len(prefix) > 0 and not prefix.endswith(' '):
            prefix += ' '
        if len(suffix) > 0 and not suffix.startswith(' '):
            suffix = ' ' + suffix

        if style == '*':
            self._fmt_str = '{pre}[{{pstr:<{n}}}]{suf}'.format(pre=prefix, n=num_symbols, suf=suffix)
        elif style == 'counter':
            self._fmt_str = '{pre}{{i:>{l}}}/{n}{suf}'.format(pre=prefix, n=num_symbols, suf=suffix, l=len(str(num_symbols)))
        else:
            raise ValueError('style "{}" not recognized'.format(style))
        self._add_one = add_one

    def print_bar(self, i):
        """
        Print the iteration of the progress bar corresponding to step ``i``.

        :param i: defines the progress step, either the number of *'s to print with ``style='*'`` or the counter number
         with ``style='counter'``.
        :type i: int
        :return: None, prints to screen.
        """
        if self._add_one:
            i += 1

        pstr = '*' * i
        pbar = self._fmt_str.format(pstr=pstr, i=i)
        sys.stdout.write('\r' + pbar)
        sys.stdout.flush()

    def finish(self):
        """
        Close the progress bar. By default, just prints a newline.
        :return: None
        """
        sys.stdout.write('\n')
        sys.stdout.flush()


def _get_num_header_lines(filename):
    """
    Get the number of header lines in a standard GGG file

    This assumes that the file specified begins with a line with two numbers: the number of header rows and the number
    of data columns.

    :param filename: the file to read
    :type filename: str

    :return: the number of header lines
    :rtype: int
    """
    with open(filename, 'r') as fobj:
        header_info = fobj.readline()

    return int(header_info.split()[0])


def read_mod_file(mod_file, as_dataframes=False):
    """
    Read a TCCON .mod file.

    :param mod_file: the path to the mod file.
    :type mod_file: str

    :param as_dataframes: if ``True``, then the scalar and profile variables will be kept as dataframes. If ``False``
     (default), they are converted to dictionaries of floats and numpy arrays, respectively.
    :type as_dataframes: bool

    :return: a dictionary with keys 'scalar' and 'profile' containing the respective variables. These values will be
     dictionaries or data frames, depending on ``as_dataframes``.
    :rtype: dict
    """
    n_header_lines = _get_num_header_lines(mod_file)
    # Read the constants from the second line of the file. There's no header for these, we just have to rely on the
    # same constants being in the same position.
    constant_vars = pd.read_csv(mod_file, sep='\s+', header=None, nrows=1, skiprows=1,
                                names=('earth_radius', 'ecc2', 'obs_lat', 'surface_gravity',
                                       'profile_base_geometric_alt', 'base_pressure', 'tropopause_pressure'))
    # Read the scalar variables (e.g. surface pressure, SZA, tropopause) first. We just have to assume their headers are
    # on line 3 and values on line 4 of the file, the first number in the first line gives us the line the profile
    # variables start on.
    scalar_vars = pd.read_csv(mod_file, sep='\s+', header=2, nrows=1)

    # Now read the profile vars.
    profile_vars = pd.read_csv(mod_file, sep='\s+', header=n_header_lines-1)

    out_dict = dict()
    if as_dataframes:
        out_dict['constants'] = constant_vars
        out_dict['scalar'] = scalar_vars
        out_dict['profile'] = profile_vars
    else:
        out_dict['constants'] = {k: v.item() for k, v in constant_vars.items()}
        out_dict['scalar'] = {k: v.item() for k, v in scalar_vars.items()}
        out_dict['profile'] = {k: v.values for k, v in profile_vars.items()}

    return out_dict


def read_map_file(map_file, as_dataframes=False, skip_header=False):
    """
    Read a .map file

    :param map_file: the path to the .map file
    :type map_file: str

    :param as_dataframes: set to ``True`` to return the constants and profiles data as Pandas dataframes. By default,
     (``False``) they are returned as dictionaries of numpy arrays.
    :type as_dataframes: bool

    :param skip_header: set to ``True` to avoid reading the header. This is helpful for reading older .map files that
     have a slightly different header format.
    :type skip_header: bool

    :return: a dictionary with keys 'constants' and 'profile' that hold the header values and main profile data,
     respectively. The form of these values depends on ``as_dataframes``.
    :rtype: dict
    """
    n_header_lines = _get_num_header_lines(map_file)
    constants = dict()
    if not skip_header:
        with open(map_file, 'r') as mapf:
            n_skip = 4
            # Skip the first four lines to get to the constants - these should be (1) the number of header lines &
            # columns, (2) filename, (3) version info, and (4) wiki reference.
            for i in range(n_skip):
                mapf.readline()

            # The last two lines of the header are the column names and units; everything between line 5 and that should
            # be physical constants. Start at n_skip+1 to account for 0 indexing vs. number of lines.

            for i in range(n_skip+1, n_header_lines-1):
                line = mapf.readline()
                # Lines have the form Name (units): value - ignore anything in parentheses
                name, value = line.split(':')
                name = re.sub(r'\(.+\)', '', name).strip()
                constants[name] = float(value)

    df = pd.read_csv(map_file, header=n_header_lines-2, skiprows=[n_header_lines-1], na_values='NAN')
    # Sometimes extra space gets kept in the headers - remove that
    df.rename(columns=lambda h: h.strip(), inplace=True)
    if not as_dataframes:
        data = {k: v.values for k, v in df.items()}
    else:

        data = df

    out_dict = dict()
    out_dict['constants'] = constants
    out_dict['profile'] = data
    return out_dict


def write_map_file(map_file, site_lat, trop_eqlat, prof_ref_lat, surface_alt, tropopause_alt, strat_used_eqlat,
                   variables, units, var_order=None):
    """
    Create a .map file

    :param map_file: the full name to save the map file as
    :type map_file: str

    :param site_lat: the geographic latitude of the site.
    :type site_lat: float

    :param trop_eqlat: the equivalent latitude, derived from the GEOS lat vs. theta climatology, used to create the
     tropospheric part of the profiles.
    :type trop_eqlat: float

    :param prof_ref_lat: the constant reference latitude used for the tropospheric age of air and seasonal cycle
     functions.
    :type prof_ref_lat: float

    :param surface_alt: the surface altitude from the .mod file in kilometers
    :type surface_alt: float

    :param tropopause_alt: the altitude of the tropopause for this profile (in kilometers).
    :type tropopause_alt: float

    :param strat_used_eqlat: whether or not the stratospheric part of the profile used PV-derived equivalent latitude.
     ``False`` means that the geographic latitude of the site was used instead.
    :type strat_used_eqlat: bool

    :param variables: a dictionary where the keys will be used as the column names and the values should be 1D
     array-like values to be written to the map file.
    :type variables: dict(str: array-like)

    :param units: a dictionary that must have the same keys as ``variables`` where the values define the units to print
     in the line under the variable names in the .map file
    :type units: dict(str: str)

    :param var_order: optional, if given, a sequence of the keys in ``variables`` and ``units`` that defines what order
     they are to be written to the .map file. If ``variables`` is an OrderedDict, then this is not necessary.
    :type var_order: sequence(str)

    :return: None
    """
    # variables and units must have the same keys
    if var_order is None:
        var_order = list(variables.keys())
    if set(var_order) != set(variables.keys()) or set(var_order) != set(units.keys()):
        raise ValueError('variables and units must be dictionaries with the same keys, and both must match the '
                         'keys in var_order (if given)')

    k1 = var_order[0]
    size_check = np.size(variables[k1])
    for k, v in variables.items():
        if np.ndim(v) != 1:
            raise ValueError('All values in variables must be 1 dimensional. {} is not.'.format(k))
        elif np.size(v) != size_check:
            raise ValueError('All values in variables must have the same shape. {badvar} has a different shape '
                             '({badshape}) than {chkvar} ({chkshape})'.format(badvar=k, badshape=np.shape(v),
                                                                              chkvar=k1, chkshape=size_check))

    header_lines = []
    # Header line 2: file name (no path)
    header_lines.append(os.path.basename(map_file))
    # Header line 3: version info
    hg_parent, hg_branch, hg_date = hg_commit_info()
    header_lines.append('{pgrm:19} {vers:14} ({branch:19}) {date} {author:10}'
                        .format(pgrm='MOD_MAKER.py', vers=hg_parent, branch=hg_branch, date=hg_date, author='SR, MK, JL'))
    # Header line 4: wiki link
    header_lines.append('Please see https://tccon-wiki.caltech.edu for a complete description of this file and its usage.')
    # Header line 5 to (n-2): constants/site lat
    header_lines.append('Avodagro (molecules/mole): {}'.format(const.avogadro))
    header_lines.append('Mass_Dry_Air (kg/mole): {}'.format(const.mass_dry_air))
    header_lines.append('Mass_H2O (kg/mole): {}'.format(const.mass_h2o))
    header_lines.append('Latitude (degrees): {}'.format(site_lat))
    header_lines.append('Trop. eqlat (degrees): {:.2f}'.format(trop_eqlat))
    header_lines.append('Ref. lat (degrees): {}'.format(prof_ref_lat))
    header_lines.append('Surface altitude (km): {}'.format(surface_alt))
    header_lines.append('Tropopause (km): {}'.format(tropopause_alt))
    header_lines.append('Stratosphere used eq lat: {}'.format(int(strat_used_eqlat)))

    # Line 1: number of header lines and variable columns
    # The number of header lines is however many we've made so far, plus this one, the column names, and the column
    # units (3 extra)
    header_lines.insert(0, '{} {}'.format(len(header_lines)+3, len(variables)))

    # Go ahead and write the header to the file
    with open(map_file, 'w') as mapf:
        for line in header_lines:
            mapf.write(line + '\n')

        # Now we write the variable names, units, and values. Need to get a list of keys to make sure the order we
        # iterate through them is the same
        mapf.write(','.join(var_order) + '\n')
        mapf.write(','.join(units[k] for k in var_order) + '\n')

        # Finally write the values.
        for i in range(size_check):
            formatted_values = ['{:.6G}'.format(variables[k][i]) for k in var_order]
            mapf.write(','.join(formatted_values))
            if i < size_check - 1:
                mapf.write('\n')


def hg_commit_info(hg_dir=None):
    if hg_dir is None:
        hg_dir = os.path.dirname(__file__)
    if len(hg_dir) == 0:
        # If in the current directory, then dirname(__file__) gives an empty string, which isn't allowed as the argument
        # to cwd in check_output
        hg_dir = '.'
    summary = subprocess.check_output(['hg', 'log', '-l', '1'], cwd=hg_dir).splitlines()
    log_dict = dict()
    for line in summary:
        splitline = line.split(':', 1)
        if len(splitline) < 2:
            continue
        k, v = splitline
        log_dict[k.strip()] = v.strip()

    parent = re.search(r'(?<=:)\w+', log_dict['changeset']).group()
    branch = log_dict['branch']
    parent_date = log_dict['date']
    return parent, branch, parent_date


def _lrange(*args):
    # Ensure Python 3 compatibility for adding range() calls together
    r = range(*args)
    if not isinstance(r, list):
        r = list(r)
    return r


def format_lat(lat, prec=0):
    fmt = '{{:.{}f}}{{}}'.format(prec)
    direction = 'N' if lat >= 0 else 'S'
    return fmt.format(abs(lat), direction)


def round_to_zero(val):
    sign = np.sign(val)
    return np.floor(np.abs(val)) * sign


def calculate_model_potential_temperature(temp, pres_levels=_std_model_pres_levels):
    if temp.ndim != 4:
        raise ValueError('temp expected to be 4D')

    ntime, nlev, nlat, nlon = temp.shape
    if nlev != pres_levels.size:
        raise ValueError('Number of levels in temp != number of pressure levels defined')
    pres = np.tile(pres_levels[np.newaxis, :, np.newaxis, np.newaxis], (ntime, 1, nlat, nlon))
    return calculate_potential_temperature(pres, temp)


def calculate_potential_temperature(pres, temp):
    """
    Calculate potential temperature.

    :param pres: Pressure in millibars/hPa.
    :param temp: Absolute temperature in Kelvin.
    :return: the potential temperature corresponding to those T/P coordinates.
    """
    return temp * (1000/pres) ** 0.286


def _construct_grid(*part_defs):
    """
    Helper function to construct coordinates for a 1D grid
    :param part_defs: a sequence of tuples (or lists) defining the start, stop, and step for each grid component. Each
     tuple gets expanded as the arguments to :func:`numpy.arange`.
    :return: the coordinates, sorted, and made sure to be unique
    """
    grid_components = [np.arange(*part) for part in part_defs]
    # Keep only unique values and sort them
    return np.unique(np.concatenate(grid_components))


# Compute area of each grid cell and the total area
def calculate_area(lat, lon, lat_res=None, lon_res=None):
    """
    Calculate grid cell area for an equirectangular grid.

    :param lat: the vector of grid cell center latitudes, in degrees
    :param lon: the vector of grid cell center longitudes, in degrees
    :param lat_res: the width of a single grid cell in the latitudinal direction, in degrees. If omitted, will be
     calculated from the lat vector.
    :param lon_res: the width of a single grid cell in the longitudinal direction, in degrees. If omitted, will be
     calculated from the lat vector.
    :return: 2D array of areas, in units of fraction of Earth's surface area.
    """
    def calculate_resolution(coord_vec, coord_name):
        res = np.diff(coord_vec)
        if not np.all(res - res[0] < 0.001):
            raise RuntimeError('Could not determine a unique {} resolution'.format(coord_name))
        return res[0]

    nlat = lat.size
    nlon = lon.size

    if lat_res is None:
        lat_res = calculate_resolution(lat, 'lat')
    if lon_res is None:
        lon_res = calculate_resolution(lon, 'lon')

    lat_half_res = 0.5 * lat_res

    area = np.zeros([nlat, nlon])

    for j in range(nlat):
        Slat = lat[j]-lat_half_res
        Nlat = lat[j]+lat_half_res

        Slat = np.deg2rad(Slat)
        Nlat = np.deg2rad(Nlat)
        for i in range(nlon):
            area[j, i] = np.deg2rad(lon_res)*np.abs(np.sin(Slat)-np.sin(Nlat))

    if abs(np.sum(area) - 4*np.pi) > 0.0001: # ensure proper normalization so the total area of Earth is 4*pi
        print('Total earth area is {:g} not 4pi (difference of {:g}), normalizing to 4pi.'
              .format(np.sum(area), np.sum(area) - 4*np.pi))
        area *= 4*np.pi/np.sum(area)

    return area


def calculate_eq_lat_on_grid(EPV, PT, area):
    """
    Calculate equivalent latitude on a 4D grid.

    :param EPV: the potential vorticity on the 4D grid.
    :type EPV: :class:`numpy.ndarray`

    :param PT: the potential temperature on the 4D grid.
    :type PT: :class:`numpy.ndarray`

    :param area: the 2D grid of surface area (in steradians) that corresponds to the 2D slices of the 4D grid.
    :type area: :class:`numpy.ndarray`

    :return: equivalent latitude
    :rtype: :class:`numpy.ndarray`
    """
    EL = np.full_like(PT, np.nan)

    for itime in range(PT.shape[0]):
        interpolator = calculate_eq_lat(EPV[itime], PT[itime], area)
        # This is probably going to be horrifically slow - but interp2d sometimes gives weird results when called with
        # vectors, so unfortunately we have to call this one element at a time
        pbar = ProgressBar(PT[itime].size, prefix='Calculating eq. lat for time {}/{}:'.format(itime, PT.shape[0]),
                           style='counter')
        for i in range(PT[itime].size):
            pbar.print_bar(i)
            ilev, ilat, ilon = np.unravel_index(i, PT[itime].shape)
            this_pt = PT[itime, ilev, ilat, ilon]
            this_epv = EPV[itime, ilev, ilat, ilon]
            EL[itime, ilev, ilat, ilon] = interpolator(this_epv, this_pt)[0]

    return EL


def calculate_eq_lat(EPV, PT, area):
    """
    Construct an interpolator for equivalent latitude.

    :param EPV: a 3D grid of potential vorticity
    :type EPV: :class:`numpy.ndarray`

    :param PT: a 3D grid of potential temperature
    :type PT:  :class:`numpy.ndarray`

    :param area: the 2D grid of surface area (in steradians) that corresponds to the 2D slices of the 4D grid.
    :type area: :class:`numpy.ndarray`

    :return: a 2D interpolator for equivalent latitude, requires potential vorticity and potential temperature as inputs
    :rtype: :class:`scipy.interpolate.interp2d`

    Note: when querying the interpolator for equivalent latitude, it is often best to call it with scalar values, even
    though that is slower than calling it with the full vector of PV and PT that you wish to get EL for. The problem is
    that scipy 2D interpolators, when given vectors as input, return a grid. This would be fine, except that the values
    corresponding to the vector of PV and PT are not always along the diagonal and so cannot be extracted with
    :func:`numpy.diag`. (I suspect what is happening is that the interpolator sorts the input values when constructing
    the grid, but I have not tested this. -JLL)
    """
    nlev, nlat, nlon = PT.shape
    # Get rid of fill values, this fills the bottom of profiles with the first valid value
    PT[PT > 1e4] = np.nan
    EPV[EPV > 1e8] = np.nan
    for i in range(nlat):
        pd.DataFrame(PT[:, i, :]).fillna(method='bfill', axis=0, inplace=True)
        pd.DataFrame(EPV[:, i, :]).fillna(method='bfill', axis=0, inplace=True)

    # Define a fixed potential temperature grid, with increasing spacing
    # this is done arbitrarily to get sufficient levels for the interpolation to work well, and not too much for the
    # computations to take less time
    if np.min(PT) > 300 or np.max(PT) < 1000:
        raise ValueError('Potential temperature range is smaller than the [300, 1000] K assumed to create the '
                         'interpolation grid')

    theta_grid = _construct_grid((round_to_zero(np.nanmin(PT)), 300.0, 2), (300.0, 350.0, 5.0), (350.0, 500.0, 10.0),
                                 (500.0, 750.0, 20.0), (750.0, 1000.0, 30.0), (1000.0, round_to_zero(np.nanmax(PT)), 100.0))
    new_nlev = np.size(theta_grid)

    # Get PV on the fixed PT levels ~ 2 seconds per date
    new_EPV = np.zeros([new_nlev, nlat, nlon])
    for i in range(nlat):
        for j in range(nlon):
            new_EPV[:, i, j] = np.interp(theta_grid, PT[:, i, j], EPV[:, i, j])

    # Compute equivalent latitudes
    EL = np.zeros([new_nlev, 100])
    EPV_thresh = np.zeros([new_nlev, 100])
    for k in range(new_nlev): # loop over potential temperature levels
        maxPV = np.max(new_EPV[k]) # global max PV
        minPV = np.min(new_EPV[k]) # global min PV

        # define 100 PV values between the min and max PV
        EPV_thresh[k] = np.linspace(minPV,maxPV,100)

        for l,thresh in enumerate(EPV_thresh[k]):
            area_total = np.sum(area[new_EPV[k]>=thresh])
            EL[k,l] = np.arcsin(1-area_total/(2*np.pi))*90.0*2/np.pi

    # Define a fixed potential vorticity grid, with increasing spacing away from 0
    # The last term should ensure that 0 is in the grid
    pv_grid = _construct_grid((round_to_zero(np.nanmin(EPV_thresh-50.0)), -1000.0, 50.0), (-1000.0, -500.0, 20.0), (-500.0, -100.0, 10.0),
                              (-100.0, -10.0, 1.0), (-10.0, -1.0, 0.1), (-1.0, 1.0, 0.01), (1.0, 10.0, 0.1),
                              (10.0, 100.0, 1.0), (100.0, 500.0, 10.0), (500.0, 1000.0, 20.0),
                              (1000.0, round_to_zero(np.nanmax(EPV_thresh))+50.0, 50.0), (0.0, 0.1))

    # Generate interpolating function to get EL for a given PV and PT
    interp_EL = np.zeros([new_nlev,len(pv_grid)])
    for k in range(new_nlev):
        interp_EL[k] = np.interp(pv_grid,EPV_thresh[k],EL[k])

    return interp2d(pv_grid, theta_grid, interp_EL)


def _format_geosfp_name(product, file_type, date_time):
    """
    Create the file name for a GEOS FP or FP-IT file.

    :param product: which GEOS product ('fp' or 'fpit') to use

    :param file_type: which file type ('Np' for profiles, 'Nx' for surface) to use

    :param date_time: the date and time of the desired file. The hour should be a multiple of 3.
    :type date_time: datetime-like

    :return: the file name
    :rtype: str
    """
    product_patterns = {'fp': 'GEOS.fp.asm.inst3_{dim}d_asm_{type}.{date_time}.V01.nc4',
                        'fpit': 'GEOS.fpit.asm.inst3_{dim}d_asm_{type}.GEOS5124.{date_time}.V01.nc4'}
    file_type_dims = {'Np': 3, 'Nx': 2}
    try:
        pattern = product_patterns[product]
    except KeyError:
        raise ValueError('product "{}" has not been defined. Allowed values are: {}'
                         .format(product, ', '.join(product_patterns.keys())))

    try:
        file_dims = file_type_dims[file_type]
    except KeyError:
        raise ValueError('file_type "{}" is not recognized. Allowed values are: {}'
                         .format(file_type, ', '.join(file_type_dims.keys())))

    date_time = date_time.strftime('%Y%m%d_%H%M')
    return pattern.format(dim=file_dims, type=file_type, date_time=date_time)


def read_geos_files(start_date, end_date, geos_path, profile_variables, surface_variables, product='fpit',
                    keep_time_dim=True, concatenate_arrays=False, set_mask_to_nan=False):
    """
    Read GEOS FP or FP-IT files between specified dates.

    :param start_date: the first date to read GEOS files from
    :type start_date: datetime-like

    :param end_date: the last date to read GEOS file from (exclusive). Note that if this datetime is not exactly a time
     that GEOS files are produced on, you will lose an extra file. For example, since GEOS files are produced every 3
     hours, if you specify ``end_date = datetime.datetime(2012, 1, 1, 23)``, you will lose the file from
     21:00 UTC 1 Jan 2012,
    :type end_date: datetime-like

    :param geos_path: the path where the GEOS files are stored. Must have subdirectories 'Np' and 'Nx' for the profile
     and surface files, respectively. Currently, each of these subdirectories must be flat, meaning that all GEOS data
     from all times is stored at the top level, not organized into further subdirectories by year/month etc.
    :type geos_path: str

    :param profile_variables: a list of variables to read from the profile (Np) files. 'lon', 'lat', and 'lev' are
     always read.
    :type profile_variables: list(str)

    :param surface_variables: a list of variables to read from the surface (Nx) files. 'lon' and 'lat' are always read.
    :type surface_variables: list(str)

    :param product: one of the strings 'fp' or 'fpit', determine which GEOS product is being read.
    :type product: str

    :param keep_time_dim: Set to ``True`` to keep the time dimension of the variables. This means the profile and
     surface variables will be 4D and 3D respectively. Set to ``False`` to remove it (so they will be 3D and 2D).
    :type keep_time_dim: bool

    :param concatenate_arrays: Set to ``True`` to concatenate the data from different files into a single array. This
     requires ``keep_time_dim`` to be ``True`` since they are concatenated along the time dimension. If ``False``, then
     the variables are left as lists of numpy arrays, where each array comes from a separate file.
    :type concatenate_arrays: bool

    :param set_mask_to_nan: Set to ``True`` turn the masked arrays read from netCDF files by default into regular numpy
     arrays, where the masked values are replaced with NaNs.
    :type set_mask_to_nan: bool

    :return: dictionaries of profile and surface variables, and a Pandas DateTimeIndex of the file dates. The variable
     dictionaries' values' format depends on the value of ``concatenate_arrays``.
    :rtype: dict, dict, DatetimeIndex
    """
    def read_var_helper(nchandle, varname, keep_time=keep_time_dim):
        if keep_time:
            data = nchandle.variables[varname][:]
        else:
            # This is equivalent to doing nchandle.variables[varname][0,:,:,:] for a 4D variable; omitting the trailing
            # colons makes it general for any size array.
            data = nchandle.variables[varname][0]

        if set_mask_to_nan:
            data = data.filled(np.nan)

        return data

    def read_files_helper(file_list, variables, is_profile):
        var_data = {v: [] for v in variables}
        for fidx, fname in enumerate(file_list):

            with ncdf.Dataset(fname, 'r') as nchandle:
                # Always read lat/lon. If reading a profile file, get the levels too for the vertical coordinate.
                lon = read_var_helper(nchandle, 'lon', keep_time=True)
                lat = read_var_helper(nchandle, 'lat', keep_time=True)
                if is_profile:
                    lev = read_var_helper(nchandle, 'lev', keep_time=True)

                # If on the first file, store the coordinate variables. If on a later file, double check that the
                # coordinates are the same. They should be, and that assumption makes the data easier to work with
                # since we don't have to recheck our indices for each file.
                if fidx == 0:
                    var_data['lon'] = lon
                    var_data['lat'] = lat
                    if is_profile:
                        var_data['lev'] = lev
                else:
                    chk = not ma.allclose(var_data['lon'], lon) or not ma.allclose(var_data['lat'], lat)
                    if is_profile:
                        chk = chk or not ma.allclose(var_data['lev'], lev)

                    if chk:
                        # TODO: replace this with proper GEOS error from the backend analysis
                        raise RuntimeError('lat, lon, and/or lev are inconsistent among the GEOS files')

                for var in variables:
                    var_data[var].append(read_var_helper(nchandle, var))

        if concatenate_arrays:
            for var, data in var_data.items():
                if isinstance(data, list):
                    # The lon/lat/lev variables don't need concatenated, we checked that they don't change with time
                    # so there's only one array for them, not a list.
                    var_data[var] = concatenate(data, axis=0)

        return var_data

    # input checking - we concatenate along the time dimension, so we better keep it
    if concatenate_arrays and not keep_time_dim:
        raise ValueError('concatenate_arrays = True requires keep_time_dim = True')

    # If we're converting the default masked arrays to regular arrays, we have to use np.concatenate because
    # ma.concatenate always returns a masked array. If we're not converting, then the reverse applied.
    if set_mask_to_nan:
        concatenate = np.concatenate
    else:
        concatenate = ma.concatenate

    geos_prof_files, file_dates = geosfp_file_names(product, 'Np', start_date, end_date)
    geos_surf_files, surf_file_dates = geosfp_file_names(product, 'Nx', start_date, end_date)

    # Check that the file lists have the same dates
    if len(file_dates) != len(surf_file_dates) or any(file_dates[i] != surf_file_dates[i] for i in range(len(file_dates))):
        raise RuntimeError('Somehow listed different profile and surface files')
    elif concatenate_arrays:
        file_dates = pd.DatetimeIndex(file_dates)

    geos_prof_files = [os.path.join(geos_path, 'Np', f) for f in geos_prof_files]
    geos_surf_files = [os.path.join(geos_path, 'Nx', f) for f in geos_surf_files]
    prof_data = read_files_helper(geos_prof_files, profile_variables, is_profile=True)
    surf_data = read_files_helper(geos_surf_files, surface_variables, is_profile=False)
    return prof_data, surf_data, file_dates


def geosfp_file_names(product, file_type, start_date, end_date=None):
    """
    List all file names for GEOS FP or FP-IT files for the given date(s).

    :param product: which GEOS product ('fp' or 'fpit') to use
    :type product: str

    :param file_type: which files ('Np' for profile, 'Nx' for surface) to list
    :type file_type: str

    :param start_date: what date to start listing files for. If ``end_date`` is omitted, only the file for this date
     will be listed. Note that the hour must be a multiple of 3, since GEOS files are produced every three hours.
    :type start_date: datetime-like

    :param end_date: what date to stop list files. This is exclusive, and will not itself be included. Can be omitted
     to just list one file.
    :type end_date: None or datetime-like

    :return: the list of file names and an array of file dates
    :rtype: list, :class:`pandas.DatetimeIndex`
    """
    freq = pd.Timedelta(hours=3)
    if start_date.hour % 3 != 0:
        raise ValueError('The hour of start_date must be a multiple of 3')
    if end_date is None:
        end_date = start_date + freq

    geos_file_dates = pd.date_range(start=start_date, end=end_date - freq, freq=freq)
    geos_file_names = []
    for date in geos_file_dates:
        this_name = _format_geosfp_name(product, file_type, date)
        geos_file_names.append(this_name)

    return geos_file_names, geos_file_dates


def geosfp_file_names_by_day(product, file_type, utc_dates, utc_hours=None):
    """
    Create a list of GEOS-FP file names for specified dates

    This differs from :func:`geosfp_file_names` because this function can list files for only specific hours across
    multiple days. For example, if you want only 00:00 UTC FP-IT profile files for all of 2018, you would call this as::

        geosfp_file_names_by_day('fpit', 'Np', pd.date_range('2018-01-01', '2018-12-31', utc_hours=[0])

    :param product: which GEOS-FP product to make names for: "fp" or "fpit"
    :type product: str

    :param file_type: which vertical coordinate type of file to use. Options are "Np" (3D by pressure) or "Nx"
     (2D variables)
    :type file_type: str

    :param utc_dates: Dates (on UTC time) to read files for.
    :type utc_dates: collection(datetime) or collection(datetime-like objects)

    :param utc_hours: Which hours of the day to use (in UTC). If ``None``, then all hours that GEOS is produced on is
     used (every 3 hours). Otherwise, pass a collection of integers to specify a subset of hours to use. (e.g.
     [0, 3, 6, 9] to only use the files from the first half of each day).
    :type utc_hours: None or collection(int)

    :return: a list of GEOS file names
    :rtype: list(str)
    """
    geos_utc_hours = np.arange(0, 24, 3)
    if utc_hours is not None:
        geos_utc_hours = geos_utc_hours[np.isin(geos_utc_hours, utc_hours)]

    geos_file_names = []
    geos_file_dates = []
    for date in utc_dates:
        for hr in geos_utc_hours:
            date_time = dt.datetime(date.year, date.month, date.day, hr)
            this_name = _format_geosfp_name(product, file_type, date_time)
            geos_file_names.append(this_name)
            geos_file_dates.append(date_time)

    return geos_file_names, geos_file_dates


def mod_interpolation_legacy(z_grid, z_met, t_met, val_met, interp_mode=1, met_alt_geopotential=True):
    """
    Legacy interpolation for .mod file profiles onto the TCCON grid

    :param z_grid: the altitude levels (in kilometers) to interpolate the values onto
    :type z_grid: :class:`numpy.ndarray`

    :param z_met: the altitude levels (in kilometers) of the input values
    :type z_met: :class:`numpy.ndarray`

    :param t_met: the absolute temperature (in Kelvin) on the same levels as the input values
    :type t_met: :class:`numpy.ndarray`

    :param val_met: the input values to be interpolated to the ``z_grid`` levels
    :type val_met: :class:`numpy.ndarray`

    :param interp_mode: how to do the interpolation. Mode ``1`` (default) is used in the original GGG code for water
     vapor dry-air mole fraction. Recommended mode for anything relating to a concentration. Mode ``0`` is for
     temperature only. Mode ``2`` is for pressure, or more generally, values with an exponential dependence on altitude.
    :type interp_mode: int

    :param met_alt_geopotential: if ``True``, in met altitudes are assumed to be heights based on geopotential, not
     geometry. Therefore internally, the grid altitudes are slightly modified to be compatible. If ``False``, then the
     met altitudes are assumed to be geometric, and the grid altitudes will not be scaled.
    :type met_alt_geopotential: bool

    :return: the input values, interpolated onto the ``z_grid`` altitudes.
    :rtype: :class:`numpy.ndarray`
    """
    if met_alt_geopotential:
        z_grid = z_grid / (1 + z_grid/earth_radius)

    val_grid = np.full(z_grid.shape, np.nan, dtype=val_met.dtype)
    for i, z in enumerate(z_grid):
        # Find the levels in the met data above and below the current grid level. If we're below the first met level,
        # use the first two levels to extrapolate
        i_below = np.argwhere(z_met < z)
        if i_below.size == 0:
            i_below = 0
        else:
            i_below = np.max(i_below)

        i_above = i_below + 1

        # Calculate beta, which is used as the interpolation factor
        lapse_rate = (t_met[i_above] - t_met[i_below]) / (z_met[i_above] - z_met[i_below])
        if lapse_rate * (t_met[i_above] - t_met[i_below]) > 0.01 * t_met[i_below]:
            beta = np.log(1 + lapse_rate * (z - z_grid[i_below])/t_met[i_below]) / np.log(1 + lapse_rate*(z_met[i_below] - z_met[i_below]) / t_met[i_below])
        else:
            beta = (z - z_met[i_below]) / (z_met[i_above] - z_met[i_below])

        # Different interpolation modes that come from the GGG2014 fortran code.
        if interp_mode == 0:
            # interp_mode = 0 is for temperature only, uses lapse rate directly
            val_grid[i] = val_met[i_below] + lapse_rate * (z - z_met[i_below])
        elif interp_mode == 1:
            # interp_mode = 1 is for species concentrations
            val_grid[i] = val_met[i_below] + beta * (val_met[i_above] - val_met[i_below])
        elif interp_mode == 2:
            # interp_mode = 2 is for pressure
            val_grid[i] = val_met[i_below] * (val_met[i_above] / val_met[i_below])**beta
        else:
            raise ValueError('interp_mode = {} is not allowed. Must be 0, 1, or 2'.format(interp_mode))

    return val_grid


def mod_interpolation_new(z_grid, z_met, vals_met, interp_mode='linear'):
    """
    New method to interpolate met data onto the TCCON prior altitude grid

    :param z_grid: the altitude levels to interpolate to.
    :type z_grid: :class:`numpy.ndarray`

    :param z_met: the altitude levels in the meteorology data
    :type z_met: :class:`numpy.ndarray`

    :param vals_met: the values to be interpolated, on the ``z_met`` levels
    :type vals_met: :class:`numpy.ndarray`

    :param interp_mode: how to do the interpolation:

        * ``'linear'`` will do linear interpolation of ``vals_met`` with respect to z.
        * ``'lin-log'`` linearly interpolate ln(``vals_met``) with respect to z.
        * ``'log-lin'`` linearly interpolated ``vals_met`` with respect to ln(z).
        * ``'log-log'`` linearly interpolate ln(``vals_met``) with respect to ln(z).

     For compatibility with `mod_interpolation_legacy`, ``interp_mode`` may also be an integer. ``0`` or ``1`` are
     aliases for ``'linear'`` and ``2`` is the same as ``'lin-log'``
    :type interp_mode: str or int

    :return: the values interpolated to the TCCON grid.
    :rtype: :class:`numpy.ndarray`
    """
    interp_mode_compat_mapping = {0: 'linear', 1: 'linear', 2: 'lin-log', 3: 'log-lin', 4: 'log-log'}
    err_msg = 'interp_mode = {} is invalid. It must be one of the strings "{}" or one of the integers {}'.format(
        interp_mode, '", "'.join(interp_mode_compat_mapping.values()),
        ', '.join([str(k) for k in interp_mode_compat_mapping.keys()])
    )
    if isinstance(interp_mode, int):
        try:
            interp_mode = interp_mode_compat_mapping[interp_mode]
        except KeyError:
            raise ValueError(err_msg)
    elif interp_mode not in interp_mode_compat_mapping.values():
        raise ValueError(err_msg)

    interp_mode = interp_mode.lower()
    do_log_x = re.match('^log-\w{3}', interp_mode)
    do_log_y = re.match('\w{3}-log$', interp_mode)

    if do_log_x:
        z_grid = np.log(z_grid)
        z_met = np.log(z_met)
    if do_log_y:
        vals_met = np.log(vals_met)

    vals_interp = interp1d(z_met, vals_met, fill_value='extrapolate')
    vals_grid = vals_interp(z_grid)

    # If doing logarithmic interpolation for the y variable need to restore the output values. (x not returned so no
    # need to restore.)
    if do_log_y:
        vals_grid = np.exp(vals_grid)

    return vals_grid


def interp_to_tropopause_height(theta, altitude, theta_trop):
    """
    Given the potential temperature at the tropopause, find the altitude that corresponds to that tropopause.

    :param theta: a vector of potential temperatures
    :type theta: :class:`numpy.ndarray`

    :param altitude: a vector of altitudes corresponding to ``theta``.
    :type altitude: :class:`numpy.ndarray`

    :param theta_trop: the potential temperature at the tropopause.
    :type theta_trop: float

    :return: the altitude of the tropopause

    Note: this function recognizes that interpolation in the .mod files can result in boundary layer theta that
    increases towards the surface and, in some cases, cross the tropopause potential temperature. To avoid accidentally
    finding a tropopause altitude in the boundary layer, only looks for the tropopause potential temperature in the part
    of the theta vector that is monotonically increasing. As a check, this prints a warning if the last level with
    decreasing theta is above 3 km above sea level. Seeing this message does not necessarily mean there is a problem,
    just that you may want to double check that theta profile.
    """
    # Find the last point where theta is decreasing. It should be increasing consistently above the boundary layer
    # This prevents interpolating to a weirdly low tropopause if the boundary layer somehow contains the potential
    # temperature of the tropopause (sometimes happens if it's extrapolated to the surface)

    # np.nonzero returns a tuple, the first element is an array of indices where diff(theta) < 0
    decr_theta = np.diff(theta) < 0
    if not np.any(decr_theta):
        last_decr = 0
    else:
        last_decr = np.max(np.nonzero(decr_theta)[0]) + 1
    if altitude[last_decr] > 3:
        print('Warning: decreasing potential temperature found above 3 km ({} km). This might cause an erroneously high '
              'tropopause height.'.format(altitude[last_decr]))

    # Do the interpolation with just the altitudes where theta is monotonically increasing.
    return mod_interpolation_new(theta_trop, theta[last_decr:], altitude[last_decr:], interp_mode='linear').item()


def age_of_air(lat, z, ztrop, ref_lat=45.0):
    """
    Calculate age of air using a function form from GGG 2014.

    :param lat: the latitude(s) to calculate age of air for
    :type lat: float or :class:`numpy.ndarray`

    :param z: the altitude(s) to calculate age of air for. If both ``z`` and ``lat`` given as a vectors, they must be
     the same shape. Must have units of kilometers.
    :type z: float or :class:`numpy.ndarray`

    :param ztrop: the tropopause altitude, in kilometers.
    :type ztrop: float

    :param ref_lat: the reference latitude for the cycle. This is where the exponential in latitude is maximized. 45N
     was chosen as the default as the center of the northern hemisphere, where most anthropogenic emissions are.
    :type ref_lat: float

    :return: age of air, in years, as a numpy array
    :rtype: :class:`numpy.ndarray`
    """
    # Force z to be a numpy array. This allows us to use numpy indexing for the extra (stratospheric) term below and
    # simultaneously ensures aoa is always a numpy array.
    if not isinstance(z, np.ndarray):
        z = np.array([z])

    fl = lat/22.0
    aoa = 0.313 - 0.085 * np.exp(-((lat-ref_lat)/18)**2) - 0.268*np.exp(-1.42 * z / (z+ztrop)) * fl / np.sqrt(1+fl**2)

    # We limit the calculation to z > ztrop here because that avoids a divide-by-0 warning
    # This term is really only kept in for completeness; in practice, it should never be used because we don't use
    # this term in the stratosphere.
    extra_term = 7.0 * (z[z > ztrop]-ztrop)/z[z > ztrop]
    aoa[z > ztrop] += extra_term
    return aoa


def seasonal_cycle_factor(lat, z, ztrop, fyr, species='co2', ref_lat=45.0):
    """
    Calculate a factor to multiply a concentration by to account for the seasonal cycle.

    :param lat: the latitude(s) to calculate age of air for
    :type lat: float or :class:`numpy.ndarray`

    :param z: the altitude(s) to calculate age of air for. If both ``z`` and ``lat`` given as a vectors, they must be
     the same shape. Must have units of kilometers.
    :type z: float or :class:`numpy.ndarray`

    :param ztrop: the tropopause altitude, in kilometers.
    :type ztrop: float

    :param fyr: the fraction of the year that corresponds to this date. You can convert a date time to this value with
     :func:`date_to_frac_year`.
    :type fyr: float

    :param species: which species seasonal cycle to calculate; affects the amplitude. Options are: 'co2'.
    :type species: str

    :param ref_lat: reference latitude for the age of air. Set to 45N as an approximation of where the NH emissions are.
    :type ref_lat: float

    :return: the seasonal cycle factor as a numpy array. Multiply this by a deseasonalized concentration at (lat, z) to
     get the concentration including the seasonal cycle
    """
    season_cycle_coeffs = {'co2': 0.007}

    aoa = age_of_air(lat, z, ztrop, ref_lat=ref_lat)
    sv = np.sin(2*np.pi *(fyr - 0.834 - aoa))
    svnl = sv + 1.80 * np.exp(-((lat - 74)/41)**2)*(0.5 - sv**2)
    sca = svnl * np.exp(-aoa/0.20)*(1 + 1.33*np.exp(-((lat-76)/48)**2) * (z+6)/(z+1.4))
    return 1 + sca * season_cycle_coeffs[species]


def date_to_decimal_year(date_in):
    """
    Convert a datetime object to a decimal year.

    A decimal year is e.g. 2018.5, where the part before the decimal is the year itself and after the decimal is how
    much of the year has passed.

    :param date_in: the datetime object to convert
    :type date_in: datetime-like

    :return: the decimal year
    :rtype: float
    """
    return date_in.year + date_to_frac_year(date_in)


def day_of_year(date_in):
    return float(date_in.strftime('%j')) - 1


def date_to_frac_year(date_in):
    """
    Convert a datetime object to a fraction of a year.

    The fraction is essentially how much of the year has passed, so June 1st (of any year) becomes ~0.416.

    Note: this function assumes 365.25 days per year. This is an imperfect solution, ideally a fractional year should
    really describe the fraction of an orbit the Earth has completed, so only use this for things where +/- a day is an
    insignificant error.

    :param date_in: the datetime object to convert
    :type date_in: datetime-like

    :return: the fractional year
    :rtype: float
    """
    return day_of_year(date_in) / 365.25  # since there's about and extra quarter of a day per year that gives us leap years


def frac_year_to_doy(yr_in):
    """
    Convert a fractional year to a day of year.

    Internally, this multiplies by 365.25, so see the warning in the docstring for :func:`date_to_frac_year` about
    its precision. Crucially, this is NOT a reliable inverse operation to :func:`date_to_frac_year`.

    :param yr_in: the fractional year to convert
    :type yr_in: float

    :return: the number of days since 1 Jan
    :rtype: float
    """
    return yr_in * 365.25


def frac_years_to_reldelta(frac_year, allow_nans=True):
    """
    Convert a fractional year to a :class:`relativedelta` from dateutils.

    Note: like the other fraction/decimal year functions, this assumes 365.25 days/year internally. Therefore, this
    should function correctly as an inverse operation to date_to_frac_year when added back to Jan 1 of the year in
    question.

    :param frac_year: the fractional year(s) (e.g 2.5 for 2 and a half years) to convert
    :type frac_year: float or a collection of floats

    :param allow_nans: whether to permit NaNs in the decimal years. If ``True``, then NaNs will be retained in the
     output list. If ``False``, an error is raised if NaNs are found in ``dec_year``.
    :type allow_nans: bool

    :return: a list of dateutils :class:`relativedelta` objects or a single :class:`relativedelta` if a scalar
     ``frac_year`` was given.
    :rtype: :class:`relativedelta` or list(:class:`relativedelta`)
    """
    if isinstance(frac_year, float):
        return_scalar = True
        frac_year = [frac_year]
    else:
        return_scalar = False
    if not allow_nans and np.any(np.isnan(frac_year)):
        raise ValueError('NaNs not permitted in frac_year. Either remove them, or set `allow_nans=True`')
    age_years = np.floor(frac_year)
    age_fracs = np.mod(frac_year, 1)
    rdels = [relativedelta(years=y, days=365.25 * d) if not (np.isnan(y) or np.isnan(d)) else np.nan for y, d in zip(age_years, age_fracs)]
    if return_scalar:
        rdels = rdels[0]

    return rdels


def decimal_year_to_date(dec_year, date_type=dt.datetime):
    """
    Convert decimal year or years (e.g. 2018.5) to a datetime-like object

    :param dec_year: the decimal year or years to convert. May be any kind of collection (if passing multiple values) so
     long as it supports iteration and that iteration returns scalar values (i.e. a 2D numpy array will not work because
     iteration returns rows).

    :param date_type: what type to convert the decimal years in to. May be any time that can be called
     ``date_type(year, month, day)``.

    :return: the converted dates in the type ``date_type``. If a single decimal date was passed in, then a single value
     of type ``date_type`` is returned. If a collection was passed in, then the dates will be returned in a 1D numpy
     array.
    """

    if np.any(np.isnan(dec_year)):
        raise NotImplementedError('NaNs in the input decimal years not implemented')

    try:
        dec_year[0]
    except TypeError:
        dec_year = [dec_year]
        return_as_scalar = True
    else:
        return_as_scalar = False

    years = np.array([int(d) for d in dec_year])
    frac_yrs = np.array([d % 1 for d in dec_year])
    dates = np.array([date_type(y, 1, 1) + frac_years_to_reldelta(fr, allow_nans=False) for y, fr in zip(years, frac_yrs)])

    if return_as_scalar:
        return dates[0]
    else:
        return dates


def start_of_month(date_in, out_type=dt.date):
    """
    Get a date-like object corresponding to the beginning of the month of ``date_in``

    :param date_in: Any date-like object that has attributes ``year`` and ``month``
    :param out_type: A type whose constructor accepts the keyword arguments ``year``, ``month``, and ``day``.
    :return: an instance of ``out_type`` set to day 1, 00:00:00 of the month of ``date_in``.
    """
    return out_type(year=date_in.year, month=date_in.month, day=1)
