from ginput.common_utils import sat_utils

from glob import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

from .. import acos_interface as aci
from ...common_utils import mod_utils

_lon_re = re.compile(r'-?\d+\.\d+[WE]')
_lat_re = re.compile(r'-?\d+\.\d+[NS]')

priors_grp = 'priors'
met_grp = 'Meteorology'
sounding_grp = 'SoundingGeometry'
_acos_prof_var_mapping = {'co2': (priors_grp, 'co2_prior'),
                          'eqlat': (priors_grp, 'equivalent_latitude'),
                          'strat_age': (priors_grp, 'age_of_air'),
                          'z': (priors_grp, 'altitude'),
                          'lat': (priors_grp, 'sounding_latitude'),
                          'lon': (priors_grp, 'sounding_longitude')}
_acos_met_var_mapping = {'pv': (met_grp, 'epv_profile_met'),
                         't': (met_grp, 'temperature_profile_met'),
                         'pres': (met_grp, 'vector_pressure_levels_met'),
                         'datetime': (sounding_grp, 'sounding_time_string'),
                         'met_z': (met_grp, 'height_profile_met'),
                         'met_lat': (sounding_grp, 'sounding_latitude'),
                         'met_lon': (sounding_grp, 'sounding_longitude'),
                         'tropp': (met_grp, 'blended_tropopause_pressure_met'),
                         'tropt': (met_grp, 'tropopause_temperature_met'),
                         'surfz': (met_grp, 'gph_met')}
_acos_var_conversions = {'co2': 1e6,
                         'met_z': 1e-3,
                         'pres': 1e-2,
                         'tropp': 1e-2,
                         'surfz': 1e-3,
                         'datetime': aci._convert_acos_time_strings}


prof_grp = 'profile'
scalar_grp = 'scalar'
file_grp = 'file'
_tccon_prof_var_mapping = {'co2': (prof_grp, 'co2'),
                           'strat_age': (prof_grp, 'strat_age_of_air')}
_tccon_met_var_mapping = {'eqlat': (prof_grp, 'EqL'),
                          'pv': (prof_grp, 'EPV'),
                          'pres': (prof_grp, 'Pressure'),
                          't': (prof_grp, 'Temperature'),
                          'theta': (prof_grp, 'PT'),
                          'z': (prof_grp, 'Height'),
                          'tropp': (scalar_grp, 'TROPPB'),
                          'tropt': (scalar_grp, 'TROPT'),
                          'surfz': (scalar_grp, 'Height'),
                          #'datetime': (file_grp, 'datetime'), # couldn't be weighted
                          'lon': (file_grp, 'lon'),
                          'lat': (file_grp, 'lat')}
_tccon_var_conversions = dict()

_acos_out_vars = [k for k in _acos_prof_var_mapping] + [k for k in _acos_met_var_mapping]
_acos_out_vars_set = set(_acos_out_vars)
_tccon_out_vars = [k for k in _tccon_prof_var_mapping] + [k for k in _tccon_met_var_mapping]
_tccon_out_vars_set = set(_tccon_out_vars)

if len(_acos_out_vars) != len(_acos_out_vars_set):
    raise ValueError('Duplicate variable name in ACOS met and prof variable mapping')
elif len(_tccon_out_vars) != len(_tccon_out_vars_set):
    raise ValueError('Duplicate variable name in TCCON met and prof variable mapping')
# No test that the same variables are defined in ACOS and TCCON, could be a case where we want an extra variable from
# one of them.

# Modify the conversion dictionaries so that scaling factors are functions
def _finalize_conversion_dict(dict_in):
    dict_out = dict()
    for k, v in dict_in.items():
        if isinstance(v, (int, float)):
            dict_out[k] = lambda x, scale=v: x * scale
        else:
            dict_out[k] = v

    return dict_out

_acos_var_conversions = _finalize_conversion_dict(_acos_var_conversions)
_tccon_var_conversions = _finalize_conversion_dict(_tccon_var_conversions)


def plot_acos_tccon_comparison(tccon_data, acos_data, tccon_var, acos_var=None, tccon_alt_var='z', acos_alt_var='z',
                               xquantity='', xunit='', reldiff=False, interp='needed'):
    fig, axs = plt.subplots(1, 3, sharey=True)

    acos_var = tccon_var if acos_var is None else acos_var
    tdata = tccon_data[tccon_var].T
    adata = acos_data[acos_var].T
    tz = tccon_data[tccon_alt_var].T
    az = acos_data[acos_alt_var].T

    alts_differ = np.any(np.abs(tz - az) > 0.05)
    if interp == 'always' or (interp == 'needed' and alts_differ):
        import pdb; pdb.set_trace()
        adata_i = np.full_like(tz, np.nan)
        for i in range(adata.shape[1]):
            adata_i[:, i] = np.interp(tz[:, i], az[:, i], adata[:, i])
        adata = adata_i
        az = tz
    elif alts_differ:
        raise NotImplementedError('TCCON and ACOS data on different z-coords')

    if reldiff:
        diff = (adata - tdata) / tdata * 100
        xstr = r'%$\Delta$ {}'.format(xquantity)
    else:
        diff = adata - tdata
        xstr = r'$\Delta$ {} ({})'.format(xquantity, xunit)

    axs[0].plot(diff, tz, color='gray', linewidth=0.5)
    axs[0].plot(np.nanmean(adata - tdata, axis=1), np.nanmean(tz, axis=1), color='k', linewidth=2)
    axs[0].set_ylabel('Altitude (km)')
    axs[0].set_xlabel(xstr)
    axs[0].set_title('Difference (ACOS - TCCON)')

    axs[1].plot(tdata, tz)
    axs[1].set_xlabel(r'{} ({})'.format(xquantity, xunit))
    axs[1].set_title('TCCON')

    axs[2].plot(adata, az)
    axs[2].set_xlabel(r'{} ({})'.format(xquantity, xunit))
    axs[2].set_title('ACOS')

    fig.set_size_inches(18, 6)
    return fig, axs


def match_acos_tccon_profiles(mod_dir, map_dir, acos_met_file, acos_prof_file):
    # Step 1: find the limits of times in the acos file, use this to determine which GEOS times we need
    # Step 2: list the .mod files for the first GEOS time, find the corresponding .mod files for the second GEOS time
    #  and the .map files for both times
    # Step 3: get the ACOS variables for the lat/lons of the .mod files
    # Step 4: iterate through the mod/map files, loading the profiles, and weighting by time.

    # Step 1
    first_geos_time, last_geos_time = _find_acos_time_limits(acos_met_file)

    # Step 2
    first_mod_files = _list_mod_files_for_time(mod_dir, first_geos_time)
    last_mod_files = _list_mod_files_for_time(mod_dir, last_geos_time, mod_file_order=first_mod_files)
    first_map_files = _list_map_files_matching_mod_files(map_dir, first_mod_files)
    last_map_files = _list_map_files_matching_mod_files(map_dir, last_mod_files)

    # Step 3
    mod_lons, mod_lats = _extract_modfile_lat_lon(first_mod_files)
    with h5py.File(acos_met_file, 'r') as f:
        acos_met_data, acos_met_inds = read_acos_for_lat_lons(f, mod_lons, mod_lats, _acos_met_var_mapping,
                                                              var_conversions=_acos_var_conversions)
    with h5py.File(acos_prof_file, 'r') as f:
        acos_prof_data, acos_prof_inds = read_acos_for_lat_lons(f, mod_lons, mod_lats, _acos_prof_var_mapping,
                                                                var_conversions=_acos_var_conversions)

    # Step 4
    acos_times = acos_met_data['datetime']
    tccon_met_data = read_tccon_data(first_files=first_mod_files, last_files=last_mod_files,
                                     read_fxn=mod_utils.read_mod_file, var_ids=_tccon_met_var_mapping,
                                     var_conversions=_tccon_var_conversions, acos_datetimes=acos_times,
                                     first_geos_time=first_geos_time, last_geos_time=last_geos_time)
    tccon_met_data['met_weights'] = tccon_met_data.pop('weights')
    tccon_prof_data = read_tccon_data(first_files=first_map_files, last_files=last_map_files,
                                      read_fxn=mod_utils.read_map_file, var_ids=_tccon_prof_var_mapping,
                                      var_conversions=_tccon_var_conversions, acos_datetimes=acos_times,
                                      first_geos_time=first_geos_time, last_geos_time=last_geos_time)

    # Last bit: combine the met and prof data to return. Also combine the ACOS indices
    tccon_prof_data.update(tccon_met_data)
    acos_prof_data.update(acos_met_data)
    extra_info = {'met': acos_met_inds, 'prior': acos_prof_inds,
                  'mod_files': [pair for pair in zip(first_mod_files, last_mod_files)],
                  'map_files': [pair for pair in zip(first_map_files, last_map_files)]}
    return tccon_prof_data, acos_prof_data, extra_info


def _getvar(container, varid):
    if isinstance(varid, str):
        varid = (varid,)

    for vid in varid:
        container = container[vid]
    return container


def _make_data_dict(varids):
    return {k: [] for k in varids}


def read_acos_var(h5obj, varid, inds=slice(None), is_gosat=False):
    """
    Read a variable from an ACOS .h5 file

    Reads in a variable from either the met resampler file or the priors file, replacing fill values (-999999) with
    NaNs. Optionally, can specify only a small slice of the variable to read.

    :param h5obj: an :class:`h5py.File` instance open to the file to read from
    :type h5obj: :class:`h5py.File`

    :param varid: a string or collection of strings that describe which variable to read. Passing a collection allows
     you to retrieve variables within groups, i.e. if this is ``('a','b')`` then it will read ``h5obj['a']['b']``.
    :type varid: str or collection(str)

    :param inds: optional, if given, any valid way of indexing an h5py Dataset. If given, only the slice indicated by
     these indices is returned, otherwise the full dataset is.

    :return: the array of data with fill values replaced and subset as requested
    """
    var = _getvar(h5obj, varid)
    if is_gosat:
        # For the GOSAT files, there's extra dimensions in the met file that need dealt with
        if 3 <= var.ndim <= 4:
            data = var[inds, 0, 0]
        elif var.ndim == 1:
            data = var[inds]
    else:
        data = var[inds]

    if not np.issubdtype(data.dtype, np.number):
        return data

    if isinstance(data, np.ndarray):
        data[data < aci._fill_val_threshold] = np.nan
    elif data < aci._fill_val_threshold:
        # if not an array must be a scalar.
        data = np.nan
    return data


def read_acos_for_lat_lons(h5obj, lons, lats, varids, var_conversions=dict(), is_gosat=False):
    """
    Read variables from an ACOS file for specific lat/lons

    :param h5obj: an :class:`h5py.File` instance open to the file to read from
    :type h5obj: :class:`h5py.File`

    :param lons: a vector a longitudes

    :param lats: a vector of latitudes

    :param varids: a collection of var ids for read_acos_var. Must be a dictionary with the output names as keys and the
     paths to the datasets in the h5 file as tuples/lists of strings.

    :return: a dictionary with the different variables requested concatentated into arrays that are nlevels-by-nvars
    :rtype: dict
    """
    try:
        # Is this the met resampler file?
        acos_lon = h5obj['SoundingGeometry']['sounding_longitude'][:]
        acos_lat = h5obj['SoundingGeometry']['sounding_latitude'][:]
    except KeyError:
        # No - must be the priors file
        acos_lon = h5obj['priors']['sounding_longitude'][:]
        acos_lat = h5obj['priors']['sounding_latitude'][:]

    data = _make_data_dict(varids)
    acos_inds = []

    for this_lon, this_lat in zip(lons, lats):
        ind = np.nanargmin((acos_lon - this_lon)**2 + (acos_lat - this_lat)**2)
        ind = np.unravel_index(ind, acos_lon.shape)
        acos_inds.append(ind)
        for key, vid in varids.items():
            this_data = read_acos_var(h5obj, vid, inds=ind, is_gosat=is_gosat)
            if key in var_conversions:
                this_data = var_conversions[key](this_data)
            data[key].append(this_data)

    for k, v in data.items():
        # Should be 1D vectors
        data[k] = np.fliplr(np.vstack(v))

    return data, acos_inds


def read_tccon_data(first_files, last_files, read_fxn, var_ids, var_conversions,
                    acos_datetimes, first_geos_time, last_geos_time):
    def get_data(f):
        data_in = read_fxn(f)
        data_out = dict()

        for k, varid in var_ids.items():
            this_data = _getvar(data_in, varid)
            if k in var_conversions:
                this_data = var_conversions[k](this_data)
            data_out[k] = this_data
        return data_out

    final_data = _make_data_dict(var_ids)
    weights = np.full([len(first_files), 1], np.nan)
    for i, firstf, lastf, dtime in zip(range(len(first_files)), first_files, last_files, acos_datetimes):
        firstdat = get_data(firstf)
        lastdat = get_data(lastf)

        w = sat_utils.time_weight_from_datetime(dtime.item(), first_geos_time, last_geos_time)
        weights[i] = w
        for k in var_ids:
            this_combined_dat = firstdat[k] * w + lastdat[k] * (1 - w)
            final_data[k].append(this_combined_dat)

    for k, v in final_data.items():
        final_data[k] = np.vstack(v)
    final_data['weights'] = weights

    return final_data


def _find_acos_time_limits(acos_met_file):
    """
    Find the GEOS times (3 hr intervals) that bookend the ACOS file

    :param acos_met_file: path to the ACOS met resampler file
    :type acos_met_file: str

    :return: the surrounding GEOS times
    :rtype: :class:`pandas.Timestamp`
    """
    def to_3h(dt):
        hr = dt.hour // 3 * 3
        return pd.Timestamp(dt.year, dt.month, dt.day, hr)

    with h5py.File(acos_met_file, 'r') as f:
        datestrs = read_acos_var(f, ['SoundingGeometry', 'sounding_time_string'])

    acos_datetimes = aci._convert_acos_time_strings(datestrs)
    # Fill values sometimes sneak through as times before 1993. Eliminate those.
    acos_datetimes = acos_datetimes[acos_datetimes > pd.Timestamp(1993, 1, 1)]
    start_date = np.min(acos_datetimes)
    end_date = np.max(acos_datetimes)
    start_date = to_3h(start_date)
    end_date = to_3h(end_date)
    if start_date != end_date:
        raise NotImplementedError('Not set up to handle ACOS files that cross a GEOS time')
    end_date += pd.Timedelta(hours=3)
    return start_date, end_date


def _list_mod_files_for_time(mod_dir, datetime, mod_file_order=None):
    """
    List all .mod files for a specific time.

    :param mod_dir: Directory containing the .mod files
    :type mod_dir: str

    :param datetime: the date/time to find files for
    :type datetime: datetime-like

    :param mod_file_order: if given, must be a list of mod files for a different datetime that indicates the order that
     the mod file's lat/lon should follow.
    :type mod_file_order: list(str)

    :return: list of .mod files
    :rtype: list(str)
    """
    if mod_file_order is None:
        pattern = datetime.strftime('%Y%m%d_%H%MZ')
        pattern = '*{}*.mod'.format(pattern)
        return sorted(glob(os.path.join(mod_dir, pattern)))

    else:
        # Given an existing list of .mod files, find .mod files for a different date/time but for the same lat/lons.
        timestr = datetime.strftime('%Y%m%d_%H%MZ')
        mod_files = [re.sub(r'\d{8}_\d\d00Z', timestr, f) for f in mod_file_order]
        missing_files = [f for f in mod_files if not os.path.isfile(f)]
        if len(missing_files) > 0:
            raise IOError('Missing {} mod files:\n{}'.format(len(missing_files), '\n'.join(missing_files)))
        return mod_files


def _extract_modfile_lat_lon(mod_files):
    lats = []
    lons = []
    for modf in mod_files:
        modf = os.path.basename(modf)
        lats.append(mod_utils.find_lat_substring(modf, to_float=True))
        lons.append(mod_utils.find_lon_substring(modf, to_float=True))

    return np.array(lons), np.array(lats)


def _list_map_files_matching_mod_files(map_dir, mod_files):
    """
    List .map files that correspond to the given .mod files

    :param map_dir: top directory for the map files, should have subdirectories by date and lat/lon
    :type map_dir: str

    :param mod_files: list of .mod files to match .map files for
    :type mod_files: list(str)

    :return: list of .map files, in the same order (by time/lat/lon) as the given .mod files
    :rtype: list(str)
    """
    map_files = []
    missing_files = []
    for modf in mod_files:
        modf = os.path.basename(modf)
        mod_datetime = mod_utils.find_datetime_substring(modf, out_type=pd.Timestamp)
        mod_lon = mod_utils.find_lon_substring(modf, to_float=False)
        mod_lat = mod_utils.find_lat_substring(modf, to_float=False)

        map_subdir = '{date}_{lon}_{lat}'.format(date=mod_datetime.strftime('%Y%m%d'), lon=mod_lon, lat=mod_lat)
        map_fname = mod_utils.map_file_name('xx', mod_utils.format_lat(mod_lat), mod_datetime)
        map_fullfile = os.path.join(map_dir, map_subdir, map_fname)
        if not os.path.isfile(map_fullfile):
            missing_files.append('{} (for {})'.format(map_fullfile, modf))
        map_files.append(map_fullfile)

    if len(missing_files) > 0:
        raise IOError('Could not find {} .map files:\n{}'.format(len(missing_files), '\n'.join(missing_files)))

    return map_files


def weight_tccon_vars_by_time(last_tccon, next_tccon, last_datetime, next_datetime, acos_datetimes):
    ## DEPRECATED ##
    last_datenum = sat_utils.datetime2datenum(last_datetime)
    next_datenum = sat_utils.datetime2datenum(next_datetime)
    acos_datenums = np.array([sat_utils.datetime2datenum(d) for d in acos_datetimes])

    weights = sat_utils.time_weight(acos_datenums, last_datenum, next_datenum)
    weights = weights.reshape(-1, 1)
    # allow broadcasting to expand the weights
    weighted_tccon = dict()
    for k in last_tccon.keys():
        weighted_tccon[k] = last_tccon[k] * weights + next_tccon[k] * (1 - weights)

    return weighted_tccon


def make_mod_h5_file(h5file, mod_dir, last_geos_time=pd.Timestamp('2017-05-14 18:00:00'),
                     next_geos_time=pd.Timestamp('2017-05-14 21:00:00'), mod_files=None, dates=None):
    met_group = 'Meteorology'
    sounding_group = 'SoundingGeometry'

    prof_group = 'profile'
    surf_group = 'scalar'
    file_group = 'file'

    var_dict = {(prof_group, 'EPV'): [met_group, 'epv_profile_met'],
                (prof_group, 'Temperature'): [met_group, 'temperature_profile_met'],
                (prof_group, 'Pressure'): [met_group, 'vector_pressure_levels_met'],
                #'date_strings': [sounding_header, 'sounding_time_string'],
                (prof_group, 'Height'): [met_group, 'height_profile_met'],
                (file_group, 'lat'): [sounding_group, 'sounding_latitude'],
                (file_group, 'lon'): [sounding_group, 'sounding_longitude'],
                (surf_group, 'TROPPB'): [met_group, 'blended_tropopause_pressure_met'],
                (surf_group, 'TROPT'): [met_group, 'tropopause_temperature_met'],
                (surf_group, 'Height'): [met_group, 'gph_met'],
                #'quality_flags': [sounding_header, 'sounding_qual_flag']
                }

    var_scaling = {(prof_group, 'Pressure'): 100.0,
                   #(prof_group, 'EPV'): 1e-6,
                   (prof_group, 'Height'): 1e3,
                   (surf_group, 'Height'): 1e3,
                   (surf_group, 'TROPPB'): 100.0}

    data_dict = {k: dict() for k in (met_group, sounding_group)}

    if mod_files is None:
        mod_files = sorted(glob(os.path.join(mod_dir, '*1800Z*.mod')))
    n_files = len(mod_files)

    if dates is None:
        dates = pd.date_range(last_geos_time, next_geos_time - pd.Timedelta(minutes=1), periods=n_files)
        dates = np.array(dates.to_list())
    elif len(dates) != len(mod_files):
        raise ValueError('Length of dates must equal length of mod_files if both are given.')
    datestrs = aci._convert_to_acos_time_strings(dates).reshape(-1, 1)
    data_dict[sounding_group]['sounding_time_string'] = datestrs
    data_dict[sounding_group]['sounding_qual_flag'] = np.zeros(datestrs.shape, dtype=np.int)

    file_number = -1
    for mfile in mod_files:
        file_number += 1
        moddat18 = mod_utils.read_mod_file(mfile)
        mfile21 = mfile.replace('1800Z', '2100Z')
        moddat21 = mod_utils.read_mod_file(mfile21)

        w = sat_utils.time_weight_from_datetime(dates[file_number], last_geos_time, next_geos_time)

        for tccon_path, oco_path in var_dict.items():
            # If on the first file, we need to create the arrays that we'll store the profiles in
            if file_number == 0:
                if tccon_path[0] == prof_group:
                    shape = (n_files, 1, moddat18[tccon_path[0]][tccon_path[1]].size)
                elif tccon_path[0] in (surf_group, file_group):
                    shape = (n_files, 1)
                else:
                    raise NotImplementedError('No shape defined for {} group'.format(tccon_path[0]))
                data_dict[oco_path[0]][oco_path[1]] = np.full(shape, aci._fill_val, dtype=np.float)

            # Weight the profiles based on the datetime
            data18 = moddat18[tccon_path[0]][tccon_path[1]]
            data21 = moddat21[tccon_path[0]][tccon_path[1]]
            data = w * data18 + (1 - w) * data21
            if tccon_path[0] == prof_group:
                data = np.flipud(data)
            if tccon_path in var_scaling:
                data = data * var_scaling[tccon_path]

            data_dict[oco_path[0]][oco_path[1]][file_number, 0] = data

    with h5py.File(h5file, 'w') as f:
        for group_name, group in data_dict.items():
            g = f.create_group(group_name)
            for var_name, var in group.items():
                if np.issubdtype(var.dtype, np.number):
                    this_fill = aci._fill_val
                elif np.issubdtype(var.dtype, np.string_):
                    this_fill = aci._string_fill
                else:
                    raise NotImplementedError('datatype {}'.format(var.dtype))
                dset = g.create_dataset(name=var_name, data=var, fillvalue=this_fill)
