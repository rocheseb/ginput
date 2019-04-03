from __future__ import print_function, division

import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import pandas as pd
import re
from scipy.interpolate import interp1d, interp2d
import subprocess

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


class InsufficientMetLevelsError(Exception):
    pass


def _get_num_header_lines(filename):
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


def write_map_file(map_file, site_lat, prof_ref_lat, tropopause_alt, strat_used_eqlat, variables, units, var_order=None):
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
    header_lines.append('Ref. lat (degrees): {}'.format(prof_ref_lat))
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
    grid_components = [np.arange(*part) for part in part_defs]
    # Keep only unique values and sort them
    return np.unique(np.concatenate(grid_components))


# Compute area of each grid cell and the total area
def calculate_area(lat, lon, lat_res, lon_res):
    """
    Calculate grid cell area for an equirectangular grid.

    :param lat: the vector of grid cell center latitudes, in degrees
    :param lon: the vector of grid cell center longitudes, in degrees
    :param lat_res: the width of a single grid cell in the latitudinal direction, in degrees
    :param lon_res: the width of a single grid cell in the longitudinal direction, in degrees
    :return: 2D array of areas, in units of fraction of Earth's surface area.
    """
    nlat = lat.size
    nlon = lon.size
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


def calculate_eq_lat(PT, EPV, area):
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


def geosfp_file_names(product, file_type, utc_dates, utc_hours=None):
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

    geos_utc_hours = np.arange(0, 24, 3)
    if utc_hours is not None:
        geos_utc_hours = geos_utc_hours[np.isin(geos_utc_hours, utc_hours)]

    geos_file_names = []
    for date in utc_dates:
        for hr in geos_utc_hours:
            date_time = dt.datetime(date.year, date.month, date.day, hr).strftime('%Y%m%d_%H%M')
            this_name = pattern.format(dim=file_dims, type=file_type, date_time=date_time)
            geos_file_names.append(this_name)

    return geos_file_names


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
        print('Decreasing potential temperature found above 3 km ({} km). This is not expected, and may '
              'cause erroneously high tropopause altitudes to be computed'.format(altitude[last_decr]))

    # Do the interpolation with just the altitudes where theta is monotonically increasing.
    return mod_interpolation_new(theta_trop, theta[last_decr:], altitude[last_decr:], interp_mode='linear').item()


def age_of_air(lat, z, ztrop, ref_lat=0.0):
    fl = lat/22.0
    aoa = 0.313 - 0.085 * np.exp(-((lat-ref_lat)/18)**2) - 0.268*np.exp(-1.42 * z / (z+ztrop)) * fl / np.sqrt(1+fl**2)
    extra_term = 7.0 * (z-ztrop)/z
    aoa[z > ztrop] += extra_term[z > ztrop]
    return aoa


def seasonal_cycle_factor(lat, z, ztrop, fyr, species='co2', ref_lat=0.0):
    season_cycle_coeffs = {'co2': 0.007}

    aoa = age_of_air(lat, z, ztrop, ref_lat=ref_lat)
    sv = np.sin(2*np.pi *(fyr - 0.834 - aoa))
    svnl = sv + 1.80 * np.exp(-((lat -74)/41)**2)*(0.5 - sv**2)
    sca = svnl * np.exp(-aoa/0.20)*(1 + 1.33*np.exp(-((lat-76)/48)**2) * (z+6)/(z+1.4))
    return 1 + sca * season_cycle_coeffs[species]


def date_to_decimal_year(date_in):
    return date_in.year + date_to_frac_year(date_in)


def date_to_frac_year(date_in):
    doy = float(date_in.strftime('%j'))
    return doy / 365.25  # since there's about and extra quarter of a day per year that gives us leap years


def frac_year_to_doy(yr_in):
    return yr_in * 365.25


def frac_years_to_reldelta(frac_year, allow_nans=True):
    if not allow_nans and np.any(np.isnan(frac_year)):
        raise ValueError('NaNs not permitted in frac_year. Either remove them, or set `allow_nans=True`')
    age_years = np.floor(frac_year)
    age_fracs = np.mod(frac_year, 1)
    return [relativedelta(years=y, days=365.25 * d) if not (np.isnan(y) or np.isnan(d)) else np.nan for y, d in zip(age_years, age_fracs)]


def start_of_month(date_in, out_type=dt.date):
    """
    Get a date-like object corresponding to the beginning of the month of ``date_in``

    :param date_in: Any date-like object that has attributes ``year`` and ``month``
    :param out_type: A type whose constructor accepts the keyword arguments ``year``, ``month``, and ``day``.
    :return: an instance of ``out_type`` set to day 1, 00:00:00 of the month of ``date_in``.
    """
    return out_type(year=date_in.year, month=date_in.month, day=1)
