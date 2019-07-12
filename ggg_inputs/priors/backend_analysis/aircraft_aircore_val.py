"""
Validation plots for GGGNext TCCON priors.

Most functions have some common inputs:

    * ax - if a plotting function has an ax argument, that specifies what axis to plot into. If not given, it creates
      a new figure.
    * data_type - either 'aircraft' or 'aircore'
    * prof_type - '2014', 'devel', or 'py' (for GGG2014, GGGdevel, or the new Python priors, respectively)
    * ztype - 'pres' (for pressure on the y-axis), 'alt' (for altitude), 'alt-trop' (altitude relative to tropopause)
    * bin_edges, bin_centers - vectors defining the edges and center points of bins for plots that use binned data.
        bin_centers should have 1 fewer element than bin_edges. Defaults are defined for different ztypes; usually
        only have to pass this if you want to use custom bins.
    * data_root - the top path to a folder containing aircraft and aircore data, along with priors organized as:

      data_root
      |
      -- atm
      |  |
      |  -- aircore
      |  |  |
      |  |  -- *.atm files
      |  -- aircraft
      |     |
      |     -- *.atm files
      |
      -- map
         |
         -- aircore
         |  |
         |  -- map_GGG2014
         |  |  |
         |  |  -- maps_XX.XXN_XXX.XXW
         |  |      |
         |  |     -- xx*.map
         |  -- map_GGGdevel
         |  |  |
         |  |  -- maps_XX.XXN_XXX.XXW
         |  |     |
         |  |     -- xx*.map
         |  -- map_GGGpy
         |     |
         |     -- yyyymmdd_XX.XXN_XX.XXW
         |        |
         |        -- xxXXN_yyyymmdd_HHMM.map
         |
         -- aircraft
            |
            (same structure as aircore)

Most relevant functions are probably:

    * plot_rmse_comparison() - plots a comparison of RMSE for a single prior type among different data roots.
    * plot_profiles_comparison() - plots a comparison of profiles among different data roots.

    Older:

    * plot_binned_rmse_by_type() - plots GGG2014, GGGdevel, and python priors' RMSE binned by z
    * plot_single_prof_comp() - plot comparision of py and GGG2014 priors for one location

"""

from __future__ import print_function, division
import argparse
from glob import glob

from .backend_utils import read_atm_file
from matplotlib import pyplot as plt
import numpy as np
import re
import os
import sys

from ...common_utils import mod_utils


_mydir = os.path.dirname(__file__)

pbin_edges_std = np.array([1030, 1020, 1010, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 750, 700, 650, 600, 550, 500, 400, 300, 200, 100, 50, 25, 10, 5, 1, 0.1], dtype=np.float)
pbin_centers_std = (pbin_edges_std[:-1] + pbin_edges_std[1:])/2.0
z_edges_std = 7.4 * np.log(1013 / pbin_edges_std)
z_centers_std = (z_edges_std[:-1] + z_edges_std[1:])/2.0
zrel_edges_std = z_edges_std - 12.0
zrel_centers_std = (zrel_edges_std[:-1] + zrel_edges_std[1:])/2.0
prof_types_std = ('2014', 'devel', 'py')

_ggg14_label = 'GGG2014'
_gggdev_label = 'GGG-develop'
_gggpy_label = 'Python'
_label_mapping = {'2014': _ggg14_label, 'devel': _gggdev_label, 'py': _gggpy_label}
_color_mapping = {'2014': 'green', 'devel': 'orange', 'py': 'blue'}
_marker_mapping = {'2014': 'o', 'devel': '^', 'py': 'd'}
_yaxis_labels = {'pres': ('Pressure [hPa]', True), 'alt': ('Altitude [km]', False),
                 'alt-trop': ('Altitude rel. to tropopause [km]', False)}


def cl_prof_types(cl_arg):
    prof_types = tuple(a.strip() for a in cl_arg.split(','))
    if any(pt not in prof_types_std for pt in prof_types):
        print('Allow values to include in --prof-types are: {}'.format(', '.join(prof_types_std)), file=sys.stderr)
        sys.exit(1)

    return prof_types


def format_y_axis(ax, ztype):
    label_str, do_invert = _yaxis_labels[ztype]
    ax.set_ylabel(label_str)
    if do_invert:
        ax.invert_yaxis()


def _lon_ew(lon):
    return 'W' if lon < 0 else 'E'


def _lat_ns(lat):
    return 'S' if lat < 0 else 'N'

def bin_data(data, data_z, bin_edges, bin_op=np.nanmean):
    """
    Bin aircraft/aircore data into vertical bins

    :param data: the data to bin
    :type data: 1D array-like

    :param data_z: the z-coordinates of the data
    :type data_z: 1D array like

    :param bin_edges: the edges of the bins. Assumed to abut, so that the edges of bin i are ``bin_edges[i:i+2]``.
     Must be in the same units as ``data_z``.
    :type bin_edges: 1D array like

    :param bin_op: the operation to use to reduce the values in the bin to one number
    :type bin_op: callable

    :return: vector of length ``bin_edges.size - 1`` containing the binned values
    :rtype: 1D array like
    """
    bins_size = (np.size(bin_edges) - 1,)
    bins = np.full(bins_size, np.nan)
    for i in range(np.size(bin_edges) - 1):
        bottom, top = np.sort(bin_edges[i:i + 2])
        zz = (data_z >= bottom) & (data_z < top)
        bins[i] = bin_op(data[zz])

    return bins


def num_atm_files(atm_dir):
    return len(glob(os.path.join(atm_dir, '*')))


def rmse(obs_values, calc_values):
    """
    Calculate root mean squared error between observed and calculated values.

    :param obs_values: observed values
    :type obs_values: array-like

    :param calc_values: calculated values
    :type calc_values: array-like

    :return: root mean squared error
    :rtype: float
    """
    return rmse2(obs_values - calc_values)


def rmse2(diff_values):
    """
    Calculate the RMSE assuming we already have the array of differences between the two sets of values

    :param diff_values: array of differences
    :type diff_values: array like

    :return: root mean squared error
    :rtype: float
    """
    return np.sqrt(np.nanmean((diff_values)**2.0))


def iter_prior_pairs(prior_root_1, prior_root_2, return_type='dict'):
    """
    Iterate over pairs of prior .map files

    Iterates over .map files under ``prior_root_1`` which should have subdirectories by date, lat, and lon. Finds the
    corresponding file under ``prior_root_2``. Yields pairs of these files in a format determined by ``return_type``.

    :param prior_root_1: the directory with the first set of .map files
    :type prior_root_1: str

    :param prior_root_2: the directory with the first set of .map files
    :type prior_root_2: str

    :param return_type: how to return the .map files. Options are "path" (just return the file paths), "dict" (read the
     .map files in as dictionaries), or "df"/"dataframe" (read the .map files in as dataframes).
    :type return_type: str

    :return: iterable over .map pairs
    :raises IOError: if it cannot find a file from ``prior_root_1`` in ``prior_root_2``.
    """
    for prior_dir_1 in glob(os.path.join(prior_root_1, '*')):
        dir_basename = os.path.basename(prior_dir_1.rstrip('/'))
        prior_dir_2 = os.path.join(prior_root_2, dir_basename)
        if not os.path.isdir(prior_dir_2):
            raise IOError('Cannot find {} in {}'.format(dir_basename, prior_root_2))

        for prior_file_1 in glob(os.path.join(prior_dir_1, '*.map')):
            file_basename = os.path.basename(prior_file_1)
            prior_file_2 = os.path.join(prior_dir_2, file_basename)
            if not os.path.isfile(prior_file_2):
                raise IOError('Cannot find {} in {}'.format(file_basename, prior_dir_2))

            if return_type == 'path':
                yield prior_file_1, prior_file_2
            elif return_type in ['dict', 'df', 'dataframe']:
                as_df = return_type in ['df', 'dataframe']
                prior_dat_1 = mod_utils.read_map_file(prior_file_1, as_dataframes=as_df)
                prior_dat_2 = mod_utils.read_map_file(prior_file_2, as_dataframes=as_df)
                yield prior_dat_1, prior_dat_2


def _get_subdirs_by_type(data_root, data_type, prof_type):
    if data_type.lower() == 'aircore':
        atm_dir = os.path.join(data_root, 'atm', 'aircore')
        map_dir = os.path.join(data_root, 'map', 'aircore')
    elif data_type.lower() == 'aircraft':
        atm_dir = os.path.join(data_root, 'atm', 'aircraft')
        map_dir = os.path.join(data_root, 'map', 'aircraft')
    else:
        raise ValueError('data type not recognized')

    if prof_type == '2014':
        map_dir = os.path.join(map_dir, 'map_GGG2014')
    elif prof_type.lower() in ('next', 'devel'):
        map_dir = os.path.join(map_dir, 'map_GGGdevel')
    elif prof_type.lower() == 'py':
        map_dir = os.path.join(map_dir, 'map_GGGpy')
    else:
        raise ValueError('prof type not recognized')

    return atm_dir, map_dir


def _idl_maps_file(lon, ew, lat, ns, date_time):
    subdir = 'maps_{lat:.2f}{ns}_{lon:.2f}{ew}'.format(lat=lat, ns=ns, lon=lon, ew=ew)
    filename = 'xx{}.map'.format(date_time.strftime('%Y%m%d'))
    return os.path.join(subdir, filename)


def py_map_file_subpath(lon, lat, date_time):
    return _py_maps_file(np.abs(lon), _lon_ew(lon), np.abs(lat), _lat_ns(lat), date_time)


def _py_maps_file(lon, ew, lat, ns, date_time):
    # we need the extra round calls b/c in the list of dates/lats/lons, the lat/lons are rounded to 3 decimal places
    # which means that in rare cases the rounding to 2 decimal places in the map directory names is different
    # starting from 3 decimal places or unlimited, e.g. -21.43524898159509 vs -21.435 - the first rounds to -21.44,
    # the second to -21.43
    lon = round(lon, 3)
    lat = round(lat, 3)
    subdir = '{ymd}_{lon:.2f}{ew}_{lat:.2f}{ns}'.format(ymd=date_time.strftime('%Y%m%d'), lon=lon, ew=ew,
                                                        lat=lat, ns=ns)
    # the new files are given every three hours. Need to find the closest one in time
    hrs = np.arange(0, 24, 3)
    i_hr = np.argmin(np.abs(hrs - date_time.hour))
    hr = hrs[i_hr]

    filename = 'xx{lat:.0f}{ns}_{ymd}_{hr:02d}00.map'.format(lat=lat, ns=ns, ymd=date_time.strftime('%Y%m%d'),
                                                             hr=hr)
    return os.path.join(subdir, filename)


def _choose_map_file_function(example_dirname):
    if re.match('maps', example_dirname):
        return _idl_maps_file
    elif re.match(r'\d{8}', example_dirname):
        return _py_maps_file
    else:
        raise RuntimeError('Do not know what format the maps directory "{}" is'.format(example_dirname))


def _get_id_info_from_atm_file(atmf, as_abs=True):
    _, header_info = read_atm_file(atmf)
    lat = header_info['TCCON_site_latitude_N']
    ns = _lat_ns(lat)

    lon = header_info['TCCON_site_longitude_E']
    ew = _lon_ew(lon)

    if as_abs:
        lat = np.abs(lat)
        lon = np.abs(lon)

    date_time = header_info['flight_date']

    return lon, ew, lat, ns, date_time


def iter_matched_data_by_type(data_root, data_type, prof_type, **kwargs):
    """
    Iterate over matched pairs of .map and .atm files assuming a standard directory structure

    Convenience method to iterate over matched pairs of .map and .atm files assuming that the directory structure is:

     * ``data_root`` contains subdirectories "atm" and "map"
     * "atm" and "map" both contain "aircore" and "aircraft" subdirectories
     * The subdirectories in "atm" contain all the .atm files, not organized any further by subdirectories
     * The subdirectories in "map" each have their own trio of subdirectories: "map_GGG2014", "map_GGGdevel", and
       "map_GGGpy".
     * These three map subdirectories are organized into subdirectories by location and, possibly, date.
        * For the GGG2014 and GGGdevel directories, the subdirectories must be named "maps_[y]y.yy{NS}_[xx]x.xx{WE}" e.g.
          "maps_29.00N_70.00W" or "maps_34.60N_117.30W" and contain files named as "xxyyyymmdd.map" e.g. xx20180723.map.
        * For the GGGpy directory, the subdirectories must be named "yyyymmdd_[xx]x.xx{WE}_[y]y.yy{NS}" e.g.
          "20140716_26.10E_67.50N" and contain 8 files named "xxyy{NS}_yyyymmdd_hhmm.map" e.g. "xx37N_20120114_0300.map"

    :param data_root: the top level directory containing the atm and map subdirs
    :type data_root: str

    :param data_type: which data type to iterate over, "aircore" or "aircraft"
    :type data_type: str

    :param prof_type: which prior type to iterate over, "2014", "next"/"devel", or "py"
    :type prof_type: str

    :param kwargs: additional keywords passed through to :func:`iter_matched_data`

    :return: iterator over data, filename, or data+filename pairs.
    """
    atm_dir, map_dir = _get_subdirs_by_type(data_root, data_type, prof_type)
    for i in iter_matched_data(atm_dir, map_dir, **kwargs):
        yield i


def iter_matched_data(atm_dir, map_dir, years=None, months=None, skip_missing_map=False, ret_filenames=False,
                      include_filenames=False, specie=''):
    """
    Iterate over matched pairs of .atm and .map files.

    This is a more generic version of :func:`iter_matched_data_by_type` that doesn't assume as much about the directory
    structure, therefore allowing greater flexibility in which directories are read at the expense of having to specify
    the map and atm directories separately.

    :param atm_dir: the directory containing all .atm files to match with the priors.
    :type atm_dir: str

    :param map_dir: the directory containing subdirectories of .map files to match with the .atm files.  It will
     auto-detect one of two formats for the subdirectories:

     * The subdirectories may be named following the pattern "maps_[y]y.yy{NS}_[xx]x.xx{WE}" e.g. "maps_29.00N_70.00W"
       or "maps_34.60N_117.30W" and contain files named as "xxyyyymmdd.map" e.g. xx20180723.map.  This is used for
       older versions of priors.
     * Alternatively, the subdirectories may be named "yyyymmdd_[xx]x.xx{WE}_[y]y.yy{NS}" e.g. "20140716_26.10E_67.50N"
       and contain 8 files named "xxyy{NS}_yyyymmdd_hhmm.map" e.g. "xx37N_20120114_0300.map". This is the convention
       expected for the GEOS FP-IT derived priors.

    :param years: if given, a list or other collection of which years to include
    :type years: None or list(int)

    :param months: if given, a list or other collection of which months to include
    :type months: None or list(int)

    :param skip_missing_map: set to ``True`` to skip validation profiles that do not have a corresponding .map file.
     Raises and error if ``False`` and there is a missing .map file.
    :type skip_missing_map: bool

    :param ret_filenames: set to ``True`` to return filenames instead of data loaded from those files.
    :type ret_filenames: bool

    :param include_filenames: set to ``True`` to return both data and filenames
    :type include_filenames: bool

    :param specie: which chemical specie to read. Assumes that the .atm filename have this right before the .atm
     extension, e.g. "korus_38.1N_131.2E_20160512_199018_N2O.atm". Default is an empty string, meaning that it will
     match any .atm file in the ``atm_dir``.
    :type specie: str

    :return: data loaded from the .atm and .map files, the .atm and .map file names, or tuples of both
    :rtype: iterator of (:class:`pandas.DataFrame`, :class:`pandas.DataFrame`) OR (str, str) OR
     ((:class:`pandas.DataFrame`, str), (:class:`pandas.DataFrame`, str)) pairs.
    """

    # List the available obs files
    atm_files = sorted(glob(os.path.join(atm_dir, '*{}.atm'.format(specie))))

    # List one map file directory, just to determine which format they are. glob does not return . and ..
    example_map_dir = glob(os.path.join(map_dir, '*'))[0]
    map_file_fxn = _choose_map_file_function(os.path.basename(example_map_dir))

    missing = 0
    for atmf in atm_files:
        _, _, _, _, date_time = _get_id_info_from_atm_file(atmf)
        if years is not None and date_time.year not in years:
            continue
        elif months is not None and date_time.month not in months:
            continue

        map_file = find_map_for_obs(atmf, map_dir, check_file_exists=False, map_file_fxn=map_file_fxn)
        if not os.path.exists(map_file):
            if skip_missing_map:
                print('Could not find {} corresponding to atm file {}'.format(map_file, atmf))
                missing += 1
                continue
            else:
                raise IOError('Could not find {} corresponding to atm file {}'.format(map_file, atmf))
        if ret_filenames:
            yield atmf, map_file
        else:
            obs_data, _ = read_atm_file(atmf)
            map_data = mod_utils.read_map_file(map_file, as_dataframes=True, skip_header=True)
            if include_filenames:
                yield (obs_data, atmf), (map_data['profile'], map_file)
            else:
                yield obs_data, map_data['profile']

    print('missing {} of {} files'.format(missing, len(atm_files)))


def find_map_for_obs_by_type(obs_file, data_type, prof_type, data_root=None, check_file_exists=True):
    """
    Find the map file that corresponds to an observation .atm file

    :param obs_file: the observation .atm file
    :type obs_file: str

    :param data_type: ``data_type`` argument to :func:`iter_matched_data_by_type`
    :type data_type: str

    :param prof_type: ``prof_type`` argument to :func:`iter_matched_data_by_type`
    :type prof_type: str

    :param data_root: ``data_root`` argument to :func:`iter_matched_data_by_type`
    :type data_root: str

    :return: path to .map file or None if cannot find one
    :rtype: str or None
    """
    if data_root is None:
        data_root = os.path.abspath(os.path.join(os.path.dirname(obs_file), '..', '..'))

    atm_dir, map_dir = _get_subdirs_by_type(data_root, data_type, prof_type)
    return find_map_for_obs(obs_file, map_dir, check_file_exists=check_file_exists)


def find_map_for_obs(obs_file, map_dir, check_file_exists=True, map_file_fxn=None):
    """
    Find a .map file that corresponds to a given observation .atm file

    :param obs_file: the .atm file to find the corresponding .map file for
    :type obs_file: str

    :param map_dir: the directory containing the .map subdirectories
    :type map_dir: str

    :param check_file_exists: by default, this function will check that the expected .map file exists and raise an
     error if it does not. However, you can bypass this check by setting this to ``False``, either for speed, or because
     you want to handle missing files specially.
    :type check_file_exists: bool

    :param map_file_fxn: the function that, given the longitude, ew, latitude, ns, and date of the .atm file will
     generate the subdir + filename of the .map file. If not given, it will be chosen automatically based on the format
     of the map subdirectory names.
    :type map_file_fxn: callable

    :return: the path to the .map file
    :rtype: str
    :raises IOError: if ``check_file_exists`` is ``True`` and the .map file does not exist
    """
    lon, ew, lat, ns, date_time = _get_id_info_from_atm_file(obs_file)
    example_map_dir = glob(os.path.join(map_dir, '*'))[0]
    if map_file_fxn is None:
        map_file_fxn = _choose_map_file_function(os.path.basename(example_map_dir))

    map_file = os.path.join(map_dir, map_file_fxn(lon, ew, lat, ns, date_time))
    if check_file_exists and not os.path.isfile(map_file):
        raise IOError('Could not find {} corresponding to atm file {}'.format(map_file, obs_file))

    return map_file


def load_as_array_by_type(data_root, data_type, prof_type, **kwargs):
    atm_dir, map_dir = _get_subdirs_by_type(data_root, data_type, prof_type)
    return load_as_array(atm_dir, map_dir, **kwargs)


def load_as_array(atm_dir, map_dir, ztype='pres', years=None, months=None):
    """
    Load all the observational data and prior profiles as single (1D) arrays

    :param atm_dir: the directory containing the .atm files
    :type atm_dir: str

    :param map_dir: the directory containing the .map file subdirectories
    :type map_dir: str

    :param ztype: which z-coordinate ("pres", "alt", or "alt-trop") to use when matching the data
    :type ztype: str

    :param years: collection of which years to include. If omitted, all are included
    :type years: list(int) or None

    :param months: collection of which months to include. If omitted, all are included.
    :type months: list(int) or None

    :return: 1D arrays of z-coordinate, observational concentrations, and prior concentrations concatenated together.
    """
    obs_co2 = np.array([])
    prof_co2 = np.array([])
    z = np.array([])

    for (obsdat, obsfile), (mapdat, mapfile) in iter_matched_data(atm_dir, map_dir, years=years, months=months,
                                                                  include_filenames=True):
        this_obs_co2, this_prof_co2, this_obs_z = interp_profile_to_obs(obsdat, mapdat, ztype=ztype, obs_file=obsfile,
                                                                        map_dir=map_dir)

        obs_co2 = np.concatenate((obs_co2, this_obs_co2), axis=0)
        prof_co2 = np.concatenate((prof_co2, this_prof_co2), axis=0)
        z = np.concatenate((z, this_obs_z))

    return z, obs_co2, prof_co2


def load_binned_array(atm_dir, map_dir, specie, ztype='pres', bin_edges=None, bin_op=np.nanmean):
    """
    Load the observed and prior profiles as a 2D array.

    This will interpolate the priors to the altitudes in the observed data, then bin both, to ensure that both are
    treated the same. The returned profiles will have been binned in the z-direction.

    :param atm_dir: the directory containing the .atm files
    :type atm_dir: str

    :param map_dir: the directory containing the .map file subdirectories
    :type map_dir: str

    :param ztype: which z-coordinate ("pres", "alt", or "alt-trop") to use when matching the data
    :type ztype: str

    :param bin_edges: see :func:`bin_data`

    :param bin_op: see :func:`bin_data`

    :param specie: see :func:`iter_matched_data`

    :return: observed profiles and prior profiles as 2D arrays with dimensions (n_profiles) x (n_bins)
    :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`
    """
    if bin_edges is None:
        bin_edges, bin_centers, = _get_std_bins(ztype)

    n_profs = num_atm_files(atm_dir)
    n_levels = np.size(bin_edges) - 1
    obs_profiles = np.full([n_profs, n_levels], np.nan)
    prior_profiles = np.full([n_profs, n_levels], np.nan)
    prof_info = {'lon': np.full([n_profs], np.nan), 'lat': np.full([n_profs], np.nan), 'date': np.full([n_profs], None)}

    pbar = mod_utils.ProgressBar(n_profs, prefix='Loading profile', style='counter', suffix=' ')
    for i, ((obsdat, obsfile), (mapdat, mapfile)) in enumerate(iter_matched_data(atm_dir, map_dir, specie=specie.upper(),
                                                                                 include_filenames=True)):
        pbar.print_bar(i)
        prof_info['lon'][i], _, prof_info['lat'][i], _, prof_info['date'][i] = _get_id_info_from_atm_file(obsfile, as_abs=False)
        this_obs_conc, this_prof_conc, this_z = interp_profile_to_obs(obsdat, mapdat, specie.lower(), ztype=ztype,
                                                                      obs_file=obsfile, map_dir=map_dir)
        obs_profiles[i, :] = bin_data(this_obs_conc, this_z, bin_edges=bin_edges, bin_op=bin_op)
        prior_profiles[i, :] = bin_data(this_prof_conc, this_z, bin_edges=bin_edges, bin_op=bin_op)

    return obs_profiles, prior_profiles, bin_centers, prof_info


def interp_profile_to_obs_by_type(obsdat, mapdat, data_type=None, data_root=None, **kwargs):
    """
    Interpolate prior profile data to the observation's altitudes

    :param obsdat: observational data
    :type obsdat: :class:`pandas.DataFrame`

    :param mapdat: priors from .map files
    :type mapdat: :class:`pandas.DataFrame`

    :param data_type: data type argument to :func:`iter_matched_data_by_type`
    :type data_type: str

    :param data_root: data root argument to :func:`iter_matched_data_by_type`
    :type data_root: str

    :param kwargs: additional keyword args to :func:`interp_profile_to_obs`

    :return: observed concentration, prior concentration, z-coordinates
    :rtype: three :class:`numpy.ndarray` instances
    """
    _, map_dir = _get_subdirs_by_type(data_root, data_type, 'py')
    return interp_profile_to_obs(obsdat, mapdat, **kwargs)


def interp_profile_to_obs(obsdat, mapdat, specie, ztype='pres', obs_file=None, map_dir=None, limit_by_zsurf=True):
    """
    Interpolate prior profile data to the observation's altitude

    :param obsdat: observational data
    :type obsdat: :class:`pandas.DataFrame`

    :param mapdat: priors from .map files
    :type mapdat: :class:`pandas.DataFrame`

    :param ztype: which z-coordinate ("pres", "alt", or "alt-trop") to use when matching the data
    :type ztype: str

    :param obs_file: the .atm file to load the observations from
    :type obs_file: str

    :param map_dir: the directory containing the .map file subdirectories
    :type map_dir: str

    :param limit_by_zsurf: if ``True``, only data above the surface altitude for this prior will be used.
    :type limit_by_zsurf: bool

    :return: observed concentration, prior concentration, z-coordinates
    :rtype: three :class:`numpy.ndarray` instances
    """
    def find_obs_z_var(df):
        search_str = 'Pressure' if ztype == 'pres' else 'Altitude'
        var_name = None
        for k in df.keys():
            if k.startswith(search_str) and var_name is None:
                var_name = k
            elif k.startswith(search_str) and var_name is not None:
                raise RuntimeError('Found multiple column names starting with "{}", cannot determine which one is the '
                                   'z-variable'.format(search_str))

        if var_name is None:
            raise RuntimeError('Could not find a column name starting with "{}" to be the z-variable'.format(search_str))

        return var_name

    interp_mode = 'log-log' if ztype == 'pres' else 'linear'
    obs_z_var = find_obs_z_var(obsdat)
    obs_zscale = 1.0 if ztype == 'pres' else 0.001  # must convert meters to kilometers
    prof_z_var = 'Pressure' if ztype == 'pres' else 'Height'

    # get the observed CO2 and vertical coordinate. Assume that the concentration is the last column in the .atm files
    this_obs_conc = obsdat.iloc[:, -1].values
    this_obs_z = obsdat[obs_z_var].values * obs_zscale

    # get the profile CO2, interpolated to the observation points
    this_prof_conc = mapdat[specie].values
    this_prof_z = mapdat[prof_z_var].values

    # if we only want profile levels above the surface, get the surface altitude and the altitude profile (regardless
    # of z_type) and cut down the profile
    if limit_by_zsurf or ztype == 'alt-trop':
        if obs_file is None or map_dir is None:
            raise TypeError('obs_file and map_dir are needed if limit_by_zsurf is True or ztype == "alt-trop"')
        py_map_file = find_map_for_obs(obs_file, map_dir)
        full_py_mapdate = mod_utils.read_map_file(py_map_file)

    if limit_by_zsurf:
        z_surf = full_py_mapdate['constants']['Surface altitude']
        z_alt = mapdat['Height'].values
        zz = z_alt >= z_surf
        this_prof_conc = this_prof_conc[zz]
        this_prof_z = this_prof_z[zz]

    # if we want this to be relative to the tropopause height, we need the python map files for the tropopause height
    if ztype == 'alt-trop':
        z_trop = full_py_mapdate['constants']['Tropopause']
    else:
        z_trop = 0.0

    this_prof_conc = mod_utils.mod_interpolation_new(this_obs_z, this_prof_z, this_prof_conc, interp_mode=interp_mode)
    return this_obs_conc, this_prof_conc, this_obs_z - z_trop


def fmt_lon(lon):
    if lon < 0:
        d = 'W'
    else:
        d = 'E'

    return '{:.1f}{}'.format(abs(lon), d)


def fmt_lat(lat):
    if lat < 0:
        d = 'S'
    else:
        d = 'N'
    return '{:.1f}{}'.format(abs(lat), d)


def plot_single_prof_comp(py_map_file, data_type, ax=None, ztype='pres'):
    data_root = os.path.abspath(os.path.join(os.path.dirname(py_map_file), '..', '..', '..', '..'))
    found_it = False
    for obs_file, mf in iter_matched_data_by_type(data_root, data_type, 'py', ret_filenames=True):
        if os.path.basename(mf) == os.path.basename(py_map_file):
            found_it = True
            break
    if not found_it:
        raise RuntimeError('Did not find obs file matching {}'.format(py_map_file))

    found_it = False
    for of, map_file_14 in iter_matched_data_by_type(data_root, data_type, '2014', ret_filenames=True):
        if os.path.basename(of) == os.path.basename(obs_file):
            found_it = True
            break

    if not found_it:
        raise RuntimeError('Did not find 2014 map file')

    mapdat = mod_utils.read_map_file(py_map_file, as_dataframes=True)['profile']
    mapdat14 = mod_utils.read_map_file(map_file_14, as_dataframes=True, skip_header=True)['profile']
    obsdat, header_info = read_atm_file(obs_file)

    # Since the 2014 map files don't include surface altitude, right now I'm just not limiting the profiles to above
    # the surface
    obs_co2, prof_co2, p = interp_profile_to_obs_by_type(obsdat, mapdat, ztype=ztype, obs_file=obs_file, data_type=data_type, limit_by_zsurf=False)
    _, prof_14_co2, p14 = interp_profile_to_obs_by_type(obsdat, mapdat14, ztype=ztype, limit_by_zsurf=False)

    if ax is None:
        fig = plt.figure()
        fig.set_size_inches([8, 6])
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    ax.plot(obs_co2, p, label='Aircore')
    ax.plot(prof_co2, p, label='Py prior')
    ax.plot(prof_14_co2, p14, label='GGG2014 prior')
    ax.legend()
    ax.set_xlabel('[CO$_2$]')
    format_y_axis(ax, ztype)

    lon = header_info['TCCON_site_longitude_E']
    lat = header_info['TCCON_site_latitude_N']
    date = header_info['aircraft_start_time_UTC']
    site = header_info['TCCON_site_name']

    ax.set_title('{} {} {} ({})'.format(date.strftime('%Y-%m-%d %H:%M'), fmt_lat(lat), fmt_lon(lon), site))

    return fig, ax


def calc_binned_rmses(data_root, bin_edges):
    binned_rmses = dict()
    for prof_type in ('2014', 'devel', 'py'):
        binned_rmses[prof_type] = dict()
        for data_type in ('aircore', 'aircraft'):  # problem with aircraft longitudes, rerunning now
            this_rmse = dict()
            zall, obsall, profall = load_as_array_by_type(data_root, data_type, prof_type)
            this_rmse['rmse'] = bin_data(obsall - profall, zall, bin_edges, bin_op=rmse2)
            binned_rmses[prof_type][data_type] = this_rmse


def plot_binned_rmse_by_type(data_root, bin_edges, bin_centers, ztype='pres', years=None, months=None, title_extra='',
                             prof_types=('2014', 'devel', 'py'), plot_labels=None):
    if plot_labels is not None:
        plot_labels = {k: v for k, v in zip(prof_types, plot_labels)}
    else:
        plot_labels = _label_mapping
    binned_rmses = dict()
    for prof_type in prof_types:
        binned_rmses[prof_type] = dict()
        for data_type in ('Aircore', 'Aircraft'):
            this_rmse = dict()
            zall, obsall, profall = load_as_array_by_type(data_root, data_type.lower(), prof_type, ztype=ztype, years=years, months=months)
            this_rmse['rmse'] = bin_data(obsall - profall, zall, bin_edges, bin_op=rmse2)
            binned_rmses[prof_type][data_type] = this_rmse

    fig = plt.figure()
    fig.set_size_inches([16, 6])
    all_ax = []
    for idx, dtype in enumerate(('Aircore', 'Aircraft')):
        ax = fig.add_subplot(1, 2, idx + 1)
        for prof_type in prof_types:
            ax.plot(binned_rmses[prof_type][dtype]['rmse'], bin_centers, color=_color_mapping[prof_type],
                    label=plot_labels[prof_type])

        ax.set_xlabel('RMSE (ppm)')
        format_y_axis(ax, ztype)
        ax.legend()
        all_ax.append(ax)
        ax.set_title(dtype)
        ax.grid()
        if ztype == 'alt-trop':
            ax.set_xlim(ax.get_xlim())
            ax.plot(ax.get_xlim(), [0, 0], color='k', linestyle='--', linewidth=2)

    return fig, all_ax


def plot_rmse_comparison(data_roots, data_type, bin_edges=None, bin_centers=None, prof_type='py', ztype='pres',
                         labels=None, ax=None):
    """
    Plot RMSE profiles for the priors given in each of the data_roots.

    :param data_roots: a collection of paths giving the different data roots (see module documentation) containing the
     different priors to plot
    :type data_roots: list(str)

    :param data_type: which data type (aircraft or aircore) the obs file is
    :type data_type: str

    :param bin_edges: edges of the bins to use. These are used to bin the data If omitted, default bins are chosen
     based on ztype.
    :type bin_edges: :class:`numpy.ndarray`

    :param bin_centers: centers of the bins to use. Used for plotting. If omitted, default bins are chosen based on
     ztype.
    :type bin_centers: :class:`numpy.ndarray`

    :param prof_type: which prior type (py, 2014, devel) to plot
    :type prof_type: str

    :param ztype: which z-coordinate to use (pres, alt, alt-trop)
    :type ztype: str

    :param labels: a collection of labels to use for each prior. Assumes same order as data_roots. If omitted, the
     legend will not be created.
    :type labels: list(str)

    :param ax: the axis to plot into. If omitted, a new figure is created.
    :type ax: axis

    :return: figure handle, axis handle, and the list of binned RMSEs (one vector per data root).
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    bin_edges, bin_centers = _get_std_bins(ztype, bin_edges, bin_centers)

    all_binned_rmses = []
    for idx, root in enumerate(data_roots):
        z, obs, prof = load_as_array_by_type(root, data_type.lower(), prof_type, ztype=ztype)
        rmse = bin_data(obs - prof, z, bin_edges, bin_op=rmse2)
        all_binned_rmses.append(rmse)

        label_str = labels[idx] if labels is not None else ''
        ax.plot(rmse, bin_centers, label=label_str)

    ax.set_title(data_type.capitalize())
    ax.set_xlabel('RMSE (ppm)')
    format_y_axis(ax, ztype)
    if labels is not None:
        ax.legend()

    return fig, ax, all_binned_rmses


def plot_bias_spaghetti(obs_profiles, prior_profiles, z, ax=None, color=None, mean_color=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if color is None and mean_color is None:
        color = 'gray'
        mean_color = 'black'
    elif color is None:
        color = 'gray'
    elif mean_color is None:
        mean_color = color

    diff = (obs_profiles - prior_profiles).T
    ax.plot(diff, z, color=color, linewidth=0.5)
    ax.plot(np.nanmean(diff, axis=1), z, color=mean_color, linewidth=2)
    ax.set_xlabel('Observations - priors')
    ax.grid()

    return fig, ax


def plot_profiles_comparison(obs_file, data_roots, data_type, prof_type='py', ztype='pres', labels=None,
                             linestyles=None, ax=None):
    """
    Plot multiple different iterations of a single profile type against the observations for a single site.

    :param obs_file: the path to the observation's .atm file
    :type obs_file: str

    :param data_roots: a collection of paths giving the different data roots (see module documentation) containing the
     different priors to plot
    :type data_roots: list(str)

    :param data_type: which data type (aircraft or aircore) the obs file is
    :type data_type: str

    :param prof_type: which prior type (py, 2014, devel) to plot
    :type prof_type: str

    :param ztype: which z-coordinate to use (pres, alt, alt-trop)
    :type ztype: str

    :param labels: a collection of labels to use for each prior. Assumes same order as data_roots. If omitted, the
     legend will not be created.
    :type labels: list(str)

    :param linestyles: a list of line styles recognized by pyplot.plot(). Must be the same length as data_roots,
     specifies the style to use for each data_root's prior. If not given, all priors will be solid lines.
    :type linestyles: list(str) or None

    :param ax: the axis to plot into. If omitted, a new figure is created.
    :type ax: axis

    :return: figure and axis handles
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    obs_dat, _ = read_atm_file(obs_file)
    for i, root in enumerate(data_roots):
        map_file = find_map_for_obs_by_type(obs_file, data_type, prof_type, data_root=root)
        map_dat = mod_utils.read_map_file(map_file, as_dataframes=True)['profile']
        obs_co2, this_co2, z = interp_profile_to_obs_by_type(obs_dat, map_dat, ztype=ztype, obs_file=obs_file, data_type=data_type)
        if i == 0:
            ax.plot(obs_co2, z, label='Obs.')
            check_obs_co2 = obs_co2
        elif not np.allclose(check_obs_co2, obs_co2, equal_nan=True):
            raise RuntimeError('Observations CO2 profile differs between data roots')

        label_str = labels[i] if labels is not None else ''
        l_style = linestyles[i] if linestyles is not None else '-'
        ax.plot(this_co2, z, linestyle=l_style, label=label_str)

    format_y_axis(ax, ztype)
    ax.set_xlabel('[CO$_2$] (ppm)')
    ax.set_title(os.path.basename(obs_file))

    if labels is not None:
        ax.legend()

    return fig, ax


def _get_std_bins(ztype, bin_edges=None, bin_centers=None):
    if (bin_centers is None) != (bin_edges is None):
        raise TypeError('bin_centers and bin_edges must both be given or neither be given')

    if bin_centers is None:
        if ztype == 'pres':
            bin_centers = pbin_centers_std
            bin_edges = pbin_edges_std
        elif ztype == 'alt-trop':
            bin_centers = zrel_centers_std
            bin_edges = zrel_edges_std
        elif ztype == 'alt':
            bin_centers = z_centers_std
            bin_edges = z_edges_std
        else:
            raise ValueError('No standard bins defined for ztype = "{}"'.format(ztype))

    return bin_edges, bin_centers


def make_all_plots(data_root, fig_save_root, bin_centers=None, bin_edges=None, ztype='pres', prof_types=prof_types_std,
                   font_size=14):

    bin_edges, bin_centers = _get_std_bins(ztype, bin_edges, bin_centers)

    old_font_size = plt.rcParams['font.size']
    try:
        plt.rcParams['font.size'] = font_size
        fig_rmse, ax_rmse = plot_binned_rmse_by_type(data_root, bin_edges, bin_centers, ztype=ztype, prof_types=prof_types)
        fig_rmse.savefig(os.path.join(fig_save_root, '1-profile-rmse.png'))
    finally:
        plt.rcParams['font.size'] = old_font_size


def parse_args():
    parser = argparse.ArgumentParser(description='Remake prior vs. aircraft & aircore comparison plots')
    parser.add_argument('data_root', help='The root directory for the data, containing "atm" and "map" subdirectories')
    parser.add_argument('fig_save_root', help='Where to save the figures')
    parser.add_argument('-z', '--ztype', default='pres', choices=tuple(_yaxis_labels.keys()),
                        help='What quantity to use for z')
    parser.add_argument('--prof-types', type=cl_prof_types, default=prof_types_std,
                        help='Which profile types to include')

    return vars(parser.parse_args())


def main():
    args = parse_args()
    make_all_plots(**args)


if __name__ == '__main__':
    main()
