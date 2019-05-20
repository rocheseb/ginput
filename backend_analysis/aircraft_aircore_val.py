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

    * plot_binned_rmse() - plots GGG2014, GGGdevel, and python priors' RMSE binned by z
    * plot_single_prof_comp() - plot comparision of py and GGG2014 priors for one location

"""

from __future__ import print_function, division
import argparse
from datetime import datetime as dtime
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import sys


_mydir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(_mydir, '..')))
import mod_utils


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


def bin_frac_weight(bottom, top, levels, method='total'):
    if top < bottom:
        tmp = bottom
        bottom = top
        top = tmp

    bin_dist = top - bottom
    if method == 'total':
        prof_dist = bin_dist
        min_lev = np.min(levels)
        max_lev = np.max(levels)

        if min_lev > bottom:
            prof_dist -= min_lev - bottom
        if max_lev < top:
            prof_dist -= top - max_lev

        prof_dist = max(prof_dist, 0.0)
    elif method == 'internal':
        # Find the alt/pres levels within the bin, get the first and last
        xx = np.nonzero((levels >= bottom) & (levels < top))[0]
        xx = np.sort(xx)
        xx1 = xx[0]
        xx2 = xx[-1]

        # The weight will be the fraction of the bin range covered by the
        # profile
        prof_dist = np.abs(levels[xx2] - levels[xx1])

    return prof_dist / bin_dist


def bin_data(data, data_z, bin_edges, weight_method='total', bin_op=np.nanmean):
    bins_size = (np.size(bin_edges) - 1,)
    bins = np.full(bins_size, np.nan)
    weights = np.full(bins_size, np.nan)
    for i in range(np.size(bin_edges) - 1):
        bottom, top = np.sort(bin_edges[i:i + 2])
        zz = (data_z >= bottom) & (data_z < top)
        bins[i] = bin_op(data[zz])
        # pass all altitudes to bin_frac_weight b/c the "total" method considers if the profile extends outside this bin
        weights[i] = bin_frac_weight(bottom, top, data_z, method=weight_method)

    return bins, weights


def convert_atm_value(val):
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


def read_atm_file(filename, limit_to_meas=True):
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

    data = pd.read_csv(filename, header=line_num + 1)
    if limit_to_meas:
        bottom_alt = header_info['aircraft_floor_m']
        top_alt = header_info['aircraft_ceiling_m']
        xx = (data['Altitude_m'] >= bottom_alt) & (data['Altitude_m'] <= top_alt)
        data = data[xx]
    return data, header_info


def rmse(obs_values, calc_values):
    return rmse2(obs_values - calc_values)


def rmse2(diff_values):
    return np.sqrt(np.nanmean((diff_values)**2.0))


def iter_matched_data(data_root, data_type, prof_type, ret_filenames=False, include_filenames=False, skip_missing_map=True,
                      years=None, months=None):
    def idl_maps_file(lon, ew, lat, ns, date_time):
        subdir = 'maps_{lat:.2f}{ns}_{lon:.2f}{ew}'.format(lat=lat, ns=ns, lon=lon, ew=ew)
        filename = 'xx{}.map'.format(date_time.strftime('%Y%m%d'))
        return os.path.join(subdir, filename)

    def py_maps_file(lon, ew, lat, ns, date_time):
        subdir = '{ymd}_{lon:.2f}{ew}_{lat:.2f}{ns}'.format(ymd=date_time.strftime('%Y%m%d'), lon=lon, ew=ew, lat=lat,
                                                            ns=ns)
        # the new files are given every three hours. Need to find the closest one in time
        hrs = np.arange(0, 24, 3)
        i_hr = np.argmin(np.abs(hrs - date_time.hour))
        hr = hrs[i_hr]

        filename = 'xx{lat:.0f}{ns}_{ymd}_{hr:02d}00.map'.format(lat=lat, ns=ns, ymd=date_time.strftime('%Y%m%d'),
                                                                 hr=hr)
        return os.path.join(subdir, filename)

    def aircore_id(atmf):
        bname = os.path.basename(atmf)
        latstr = re.search(r'\d+\.\d+[NS]', bname).group()
        lat = float(latstr[:-1])
        ns = latstr[-1]
        lonstr = re.search(r'\d+\.\d+[EW]', bname).group()
        lon = float(lonstr[:-1])
        ew = lonstr[-1]

        _, header_info = read_atm_file(atmf)
        date_time = header_info['aircraft_start_time_UTC']
        return lon, ew, lat, ns, date_time

    def aircraft_id(atmf):
        _, header_info = read_atm_file(atmf)
        lat = header_info['TCCON_site_latitude_N']
        ns = 'S' if lat < 0 else 'N'
        lat = np.abs(lat)

        lon = header_info['TCCON_site_longitude_E']
        ew = 'W' if lon < 0 else 'E'
        lon = np.abs(lon)

        date_time = header_info['aircraft_start_time_UTC']

        return lon, ew, lat, ns, date_time

    if data_type.lower() == 'aircore':
        atm_dir = os.path.join(data_root, 'atm', 'aircore')
        map_dir = os.path.join(data_root, 'map', 'aircore')
        id_fxn = aircore_id
    elif data_type.lower() == 'aircraft':
        atm_dir = os.path.join(data_root, 'atm', 'aircraft')
        map_dir = os.path.join(data_root, 'map', 'aircraft')
        id_fxn = aircraft_id

    if prof_type == '2014':
        map_dir = os.path.join(map_dir, 'map_GGG2014')
        map_file_fxn = idl_maps_file

    elif prof_type.lower() in ('next', 'devel'):
        map_dir = os.path.join(map_dir, 'map_GGGdevel')
        map_file_fxn = idl_maps_file
    elif prof_type.lower() == 'py':
        map_dir = os.path.join(map_dir, 'map_GGGpy')
        map_file_fxn = py_maps_file
    else:
        raise ValueError('prof type not recognized')

    # List the available obs files
    atm_files = sorted(glob(os.path.join(atm_dir, '*CO2.atm')))

    missing = 0
    for atmf in atm_files:
        lon, ew, lat, ns, date_time = id_fxn(atmf)
        if years is not None and date_time.year not in years:
            continue
        elif months is not None and date_time.month not in months:
            continue

        map_file = os.path.join(map_dir, map_file_fxn(lon, ew, lat, ns, date_time))
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


def find_map_for_obs(obs_file, data_type, prof_type, data_root=None):
    if data_root is None:
        data_root = os.path.abspath(os.path.join(os.path.dirname(obs_file), '..', '..'))

    obs_file = os.path.basename(obs_file)

    for obsf, mapf in iter_matched_data(data_root, data_type, prof_type, ret_filenames=True):
        if os.path.basename(obsf) == obs_file:
            return mapf

    return None


def load_as_array(data_root, data_type, prof_type, ztype='pres', years=None, months=None):
    obs_co2 = np.array([])
    prof_co2 = np.array([])
    z = np.array([])

    for (obsdat, obsfile), (mapdat, mapfile) in iter_matched_data(data_root, data_type, prof_type, years=years, months=months,
                                                                  include_filenames=True):
        this_obs_co2, this_prof_co2, this_obs_z = interp_profile_to_obs(obsdat, mapdat, ztype=ztype, obs_file=obsfile,
                                                                        data_type=data_type)

        obs_co2 = np.concatenate((obs_co2, this_obs_co2), axis=0)
        prof_co2 = np.concatenate((prof_co2, this_prof_co2), axis=0)
        z = np.concatenate((z, this_obs_z))

    return z, obs_co2, prof_co2


def interp_profile_to_obs(obsdat, mapdat, ztype='pres', obs_file=None, data_type=None, data_root=None,
                          limit_by_zsurf=True):
    interp_mode = 'log-log' if ztype == 'pres' else 'linear'
    obs_z_var = 'Pressure_hPa' if ztype == 'pres' else 'Altitude_m'
    obs_zscale = 1.0 if ztype == 'pres' else 0.001  # must convert meters to kilometers
    prof_z_var = 'Pressure' if ztype == 'pres' else 'Height'

    # get the observed CO2 and vertical coordinate
    this_obs_co2 = obsdat['CO2_profile_ppm'].values
    this_obs_z = obsdat[obs_z_var].values * obs_zscale

    # get the profile CO2, interpolated to the observation points
    this_prof_co2 = mapdat['co2'].values
    this_prof_z = mapdat[prof_z_var].values

    # if we only want profile levels above the surface, get the surface altitude and the altitude profile (regardless
    # of z_type) and cut down the profile
    if limit_by_zsurf or ztype == 'alt-trop':
        if obs_file is None or data_type is None:
            raise TypeError('obs_file and data_type are needed if limit_by_zsurf is True or ztype == "alt-trop"')
        py_map_file = find_map_for_obs(obs_file, data_type, 'py', data_root=data_root)
        full_py_mapdate = mod_utils.read_map_file(py_map_file)

    if limit_by_zsurf:
        z_surf = full_py_mapdate['constants']['Surface altitude']
        z_alt = mapdat['Height'].values
        zz = z_alt >= z_surf
        this_prof_co2 = this_prof_co2[zz]
        this_prof_z = this_prof_z[zz]

    # if we want this to be relative to the tropopause height, we need the python map files for the tropopause height
    if ztype == 'alt-trop':
        z_trop = full_py_mapdate['constants']['Tropopause']
    else:
        z_trop = 0.0

    this_prof_co2 = mod_utils.mod_interpolation_new(this_obs_z, this_prof_z, this_prof_co2, interp_mode=interp_mode)
    return this_obs_co2, this_prof_co2, this_obs_z - z_trop


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
    for obs_file, mf in iter_matched_data(data_root, data_type, 'py', ret_filenames=True):
        if os.path.basename(mf) == os.path.basename(py_map_file):
            found_it = True
            break
    if not found_it:
        raise RuntimeError('Did not find obs file matching {}'.format(py_map_file))

    found_it = False
    for of, map_file_14 in iter_matched_data(data_root, data_type, '2014', ret_filenames=True):
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
    obs_co2, prof_co2, p = interp_profile_to_obs(obsdat, mapdat, ztype=ztype, obs_file=obs_file, data_type=data_type, limit_by_zsurf=False)
    _, prof_14_co2, p14 = interp_profile_to_obs(obsdat, mapdat14, ztype=ztype, limit_by_zsurf=False)

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
            zall, obsall, profall = load_as_array(data_root, data_type, prof_type)
            this_rmse['rmse'], this_rmse['weight'] = bin_data(obsall - profall, zall, bin_edges, bin_op=rmse2)
            binned_rmses[prof_type][data_type] = this_rmse


def plot_binned_rmse(data_root, bin_edges, bin_centers, ztype='pres', years=None, months=None, title_extra='',
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
            zall, obsall, profall = load_as_array(data_root, data_type.lower(), prof_type, ztype=ztype, years=years, months=months)
            this_rmse['rmse'], this_rmse['weight'] = bin_data(obsall - profall, zall, bin_edges, bin_op=rmse2)
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


def plot_scatter_by_height(data_root, bin_edges, bin_centers, ztype, prof_types=('2014', 'devel', 'py')):
    # Now lets go profile by profile and do a scatter plot of RMSE vs. altitude, with size representing weight
    fig = plt.figure()
    all_ax = []
    for didx, data_type in enumerate(('Aircore', 'Aircraft')):
        for idx, prof_type in enumerate(prof_types):
            ax = fig.add_subplot(2, 3, idx + 1 + didx * 3)
            all_ax.append(ax)
            for (obsdat, obsfile), (profdat, proffile) in iter_matched_data(data_root, data_type.lower(), prof_type,
                                                                            include_filenames=True):
                this_obs_co2, this_prof_co2, this_z = interp_profile_to_obs(obsdat, profdat, ztype=ztype,
                                                                            obs_file=obsfile, data_type=data_type)
                rmse_bins, wt_bins = bin_data(this_obs_co2 - this_prof_co2, this_z, bin_edges, weight_method='total',
                                              bin_op=rmse2)
                ax.scatter(rmse_bins, bin_centers, s=wt_bins * 5)
            ax.set_xlabel('RMSE')
            format_y_axis(ax, ztype)
            ax.set_title('GGG {} vs. {}'.format(prof_type, data_type))
            ax.grid(which='both')

    fig.set_size_inches([len(prof_types)*7, 16])
    plt.subplots_adjust(hspace=0.3)
    return fig, all_ax


def plot_pbl_rmse_vs_grad(data_root, data_type, prof_types):
    def collect_vals(prior_type):
        rmse_vals = []
        grad_ac_vals = []
        for (obsdat, obsfile), (mapdat, mapfile) in iter_matched_data(data_root, data_type, prior_type, include_filenames=True):
            obs_co2, prof_co2, p = interp_profile_to_obs(obsdat, mapdat, ztype='pres', obs_file=obsfile, data_type=data_type)
            zz = p >= 800
            z_surf = np.argmax(p)
            z_400 = np.argmin(np.abs(p - 400))
            rmse_vals.append(rmse(obs_co2[zz], prof_co2[zz]))
            grad_ac_vals.append(obs_co2[z_surf] - obs_co2[z_400])
        return np.array(rmse_vals), np.array(grad_ac_vals)

    rmse_dict = dict()
    grad_ac_dict = dict()
    for prof_type in prof_types:
        rmse_dict[prof_type], grad_ac_dict[prof_type] = collect_vals(prof_type)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for prof_type in prof_types:
        ax.plot(grad_ac_dict[prof_type], rmse_dict[prof_type], color=_color_mapping[prof_type],
                marker=_marker_mapping[prof_type], linestyle='none', label=_label_mapping[prof_type])
    ax.set_ylabel('RMSE for $p \geq 800$ hPa (ppm)')
    ax.set_xlabel('Obs. CO$_2$(surf) - CO$_2$(400 hPa) (ppm)')
    ax.legend()
    fig.set_size_inches([8, 6])
    return fig, ax


def order_by_pbl_rmse(data_root, prof_type, filter_fxn=None):
    if filter_fxn is None:
        filter_fxn = lambda x,y: np.ones_like(x, dtype=np.bool)

    dtypes = ('aircore', 'aircraft')
    map_files = {k: [] for k in dtypes}
    pbl_err = {k: [] for k in dtypes}
    obs_grad = {k: [] for k in dtypes}
    for dtype in dtypes:
        for obsf, mapf in iter_matched_data(data_root, dtype, prof_type, ret_filenames=True):
            obsdat, _ = read_atm_file(obsf)
            mapdat = mod_utils.read_map_file(mapf, as_dataframes=True)

            obs_co2, prof_co2, p = interp_profile_to_obs(obsdat, mapdat['profile'], obs_file=obsf, data_type=dtype)
            zz = p >= 800
            z_surf = np.argmax(p)
            z_400 = np.argmin(np.abs(p - 400))
            map_files[dtype].append(mapf)
            pbl_err[dtype].append(rmse(obs_co2[zz], prof_co2[zz]))
            obs_grad[dtype].append(obs_co2[z_surf] - obs_co2[z_400])

    map_files = {k: np.array(v) for k, v in map_files.items()}
    pbl_err = {k: np.array(v) for k, v in pbl_err.items()}
    obs_grad = {k: np.array(v) for k, v in obs_grad.items()}

    ordered_errs = dict()
    ordered_files = dict()
    ordered_grads = dict()
    for dtype in dtypes:
        xx = ~np.isnan(pbl_err[dtype])
        this_pbl_err = pbl_err[dtype][xx]
        this_map_files = map_files[dtype][xx]
        this_obs_grad = obs_grad[dtype][xx]

        err_order = np.flipud(np.argsort(this_pbl_err))

        ordered_errs[dtype] = this_pbl_err[err_order]
        ordered_files[dtype] = this_map_files[err_order]
        ordered_grads[dtype] = this_obs_grad[err_order]

        xx = filter_fxn(ordered_errs[dtype], ordered_grads[dtype])
        ordered_errs[dtype] = ordered_errs[dtype][xx]
        ordered_grads[dtype] = ordered_grads[dtype][xx]
        ordered_files[dtype] = ordered_files[dtype][xx]

    return ordered_errs, ordered_grads, ordered_files


def plot_top_n_plb_rmses(data_root, data_type, subplot_pattern, filter_fxn=None, ztype='pres'):
    ordered_errs, ordered_grads, ordered_files = order_by_pbl_rmse(data_root, 'py', filter_fxn=filter_fxn)
    nrow, ncol = subplot_pattern
    n = nrow * ncol

    fig, axs = plt.subplots(nrow, ncol)

    for i in range(n):
        if i > np.size(ordered_files[data_type]):
            continue
        inds = np.unravel_index(i, axs.shape)
        plot_single_prof_comp(ordered_files[data_type][i], data_type, ax=axs[inds], ztype=ztype)

    fig.set_size_inches([24, 24])
    return fig, axs


def lat_rmse_plot(new_map_lists, del_rmse_lists, labels=None, vertical_part=''):
    geog_lat = [[] for m in new_map_lists]
    eq_lat = [[] for m in new_map_lists]
    for i, map_files in enumerate(new_map_lists):
        for f in map_files:
            mapdat = mod_utils.read_map_file(f)
            geog_lat[i].append(mapdat['constants']['Latitude'])
            eq_lat[i].append(mapdat['constants']['Trop. eqlat'])

    colors = ['blue', 'orange', 'red', 'green']
    markers = ['o', '^', 'v', 'd']
    markers2 = ['x', '+', '1', '_']

    fig = plt.figure()
    fig.set_size_inches([16, 6])
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('Order (big -> small) {} RMSE increase'.format(vertical_part))
    ax1.set_ylabel('Latitude')
    ax1.grid(axis='y')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('Order (big -> small) {} RMSE increase'.format(vertical_part))
    ax2.set_ylabel(r'$\Delta$ {} RMSE (ppm)'.format(vertical_part))
    ax2.grid(axis='y')

    for i, (geog, eq, rmse) in enumerate(zip(geog_lat, eq_lat, del_rmse_lists)):
        lab_str = labels[i] if labels is not None else ''
        ax1.plot(geog, label='{} (geog. lat.)'.format(lab_str), linestyle='none', marker=markers[i], color=colors[i])
        ax1.plot(eq, label='{} (eq. lat.)'.format(lab_str), linestyle='none', marker=markers2[i], color=colors[i])
        ax2.plot(rmse, linestyle='none', marker=markers[i], color=colors[i], label=lab_str)

    ax1.legend()
    if labels is not None:
        ax2.legend()

    return fig, [ax1, ax2]


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
        z, obs, prof = load_as_array(root, data_type.lower(), prof_type, ztype=ztype)
        rmse, _ = bin_data(obs - prof, z, bin_edges, bin_op=rmse2)
        all_binned_rmses.append(rmse)

        label_str = labels[idx] if labels is not None else ''
        ax.plot(rmse, bin_centers, label=label_str)

    ax.set_title(data_type.capitalize())
    ax.set_xlabel('RMSE (ppm)')
    format_y_axis(ax, ztype)
    if labels is not None:
        ax.legend()

    return fig, ax, all_binned_rmses


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
        map_file = find_map_for_obs(obs_file, data_type, prof_type, data_root=root)
        map_dat = mod_utils.read_map_file(map_file, as_dataframes=True)['profile']
        obs_co2, this_co2, z = interp_profile_to_obs(obs_dat, map_dat, ztype=ztype, obs_file=obs_file, data_type=data_type)
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


def find_top_rmse_increases(old_data_root, new_data_root, data_type, zrange, ztype, prof_type='py'):
    delta_rmses = []
    obs_files = []
    new_map_files = []
    old_map_files = []
    for (obsdat, obsfile), (new_profdat, new_proffile) in iter_matched_data(new_data_root, data_type, prof_type,
                                                                            include_filenames=True):
        old_proffile = find_map_for_obs(os.path.abspath(obsfile), data_type, prof_type, data_root=old_data_root)
        old_profdat = mod_utils.read_map_file(old_proffile, as_dataframes=True)['profile']

        obs_co2, new_co2, z = interp_profile_to_obs(obsdat, new_profdat, ztype=ztype, obs_file=obsfile, data_type=data_type)
        zz_lev = (z >= np.min(zrange)) & (z <= np.max(zrange))
        new_rmse = rmse(obs_co2[zz_lev], new_co2[zz_lev])
        _, old_co2, _ = interp_profile_to_obs(obsdat, old_profdat, ztype=ztype, obs_file=obsfile, data_type=data_type, data_root=old_data_root)
        old_rmse = rmse(obs_co2[zz_lev], old_co2[zz_lev])

        if np.isnan(new_rmse) or np.isnan(old_rmse):
            continue

        delta_rmses.append(np.mean(new_rmse - old_rmse))
        obs_files.append(obsfile)
        new_map_files.append(new_proffile)
        old_map_files.append(old_proffile)

    pbl_results = [np.array(delta_rmses), np.array(obs_files), np.array(new_map_files), np.array(old_map_files)]
    err_order = np.flipud(np.argsort(pbl_results[0]))
    for i, v in enumerate(pbl_results):
        pbl_results[i] = v[err_order]
    return pbl_results


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
        fig_rmse, ax_rmse = plot_binned_rmse(data_root, bin_edges, bin_centers, ztype=ztype, prof_types=prof_types)
        fig_rmse.savefig(os.path.join(fig_save_root, '1-profile-rmse.png'))

        fig_scatter, ax_scatter = plot_scatter_by_height(data_root, bin_edges, bin_centers, ztype=ztype, prof_types=prof_types)
        fig_scatter.savefig(os.path.join(fig_save_root, '2-rmse-scatter.png'))

        fig_err_v_grad_core, ax_err_v_grad_core = plot_pbl_rmse_vs_grad(data_root, 'aircore', prof_types=prof_types)
        fig_err_v_grad_core.savefig(os.path.join(fig_save_root, '3a-err-vs-grad-aircore.png'))
        fig_err_v_grad_craft, ax_err_v_grad_craft = plot_pbl_rmse_vs_grad(data_root, 'aircraft', prof_types=prof_types)
        fig_err_v_grad_craft.savefig(os.path.join(fig_save_root, '3b-err-vs-grad-aircraft.png'))

        # Make plots of the profiles with the top nine PBL RMSEs for aircore and aircraft
        fig_9core, ax_9core = plot_top_n_plb_rmses(data_root, 'aircore', (2, 3), ztype=ztype)
        fig_9core.savefig(os.path.join(fig_save_root, '4a-top-nine-pbl-rmses-aircore.png'))
        fig_9craft, ax_9craft = plot_top_n_plb_rmses(data_root, 'aircraft', (2, 3), ztype=ztype)
        fig_9craft.savefig(os.path.join(fig_save_root, '4b-top-nine-pbl-rmses-aircraft.png'))

        # Make plots of the top nine PBL RMSE with a gradient of < 5 ppm between 400 hPa and the surface
        def low_gradient(ordered_errs, ordered_grads):
            return np.abs(ordered_grads) < 5

        fig_9core2, ax_9core2 = plot_top_n_plb_rmses(data_root, 'aircore', (2, 3), filter_fxn=low_gradient, ztype=ztype)
        fig_9core2.savefig(os.path.join(fig_save_root, '5a-top-nine-pbl-rmses-grad_lt_5-aircore.png'))
        fig_9craft2, ax_9craft2 = plot_top_n_plb_rmses(data_root, 'aircraft', (2, 3), filter_fxn=low_gradient, ztype=ztype)
        fig_9craft2.savefig(os.path.join(fig_save_root, '5a-top-nine-pbl-rmses-grad_lt_5-aircraft.png'))
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
