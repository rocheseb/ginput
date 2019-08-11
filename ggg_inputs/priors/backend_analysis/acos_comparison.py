from ggg_inputs.common_utils import sat_utils

from glob import glob
import h5py
import numpy as np
import pandas as pd
import os
import re

from .. import acos_interface as aci
from ...common_utils import mod_utils

_lon_re = re.compile(r'-?\d+\.\d+[WE]')
_lat_re = re.compile(r'-?\d+\.\d+[NS]')


def _getvar(container, varid):
    if isinstance(varid, str):
        varid = (varid,)

    for vid in varid:
        container = container[vid]
    return container


def iter_varids(varids):
    for vid in varids:
        if vid == ('scalar', 'Height'):
            key = 'SurfAlt'
        else:
            key = vid if isinstance(vid, str) else vid[-1]
        yield vid, key


def _make_data_dict(varids):
    return {k: [] for _, k in iter_varids(varids)}


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


def read_acos_for_lat_lons(h5obj, lons, lats, varids, is_gosat=False):
    """
    Read variables from an ACOS file for specific lat/lons

    :param h5obj: an :class:`h5py.File` instance open to the file to read from
    :type h5obj: :class:`h5py.File`

    :param lons: a vector a longitudes

    :param lats: a vector of latitudes

    :param varids: a collection of var ids for read_acos_var. Must be a collection of strings or collection of
     collections.

    :return: a dictionary with the different variables requested concatentated into arrays that are nlevels-by-nvars
     For any varids that are collections, the last element will be used as the key.
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
        for vid, key in iter_varids(varids):
            data[key].append(read_acos_var(h5obj, vid, inds=ind, is_gosat=is_gosat))

    for k, v in data.items():
        # Should be 1D vectors
        data[k] = np.vstack(v)

    return data, acos_inds


def read_acos_matching_mod_files(h5obj, mod_files, varids, is_gosat=False):
    def extract_mod_lon_lat(filename):
        lonstr = _lon_re.search(filename).group()
        lon = mod_utils.format_lon(lonstr)
        latstr = _lat_re.search(filename).group()
        lat = mod_utils.format_lat(latstr)
        return lon, lat

    lons, lats = zip(*[extract_mod_lon_lat(os.path.basename(f)) for f in mod_files])
    return read_acos_for_lat_lons(h5obj, lons, lats, varids, is_gosat=is_gosat)


def read_acos_matching_map_files(h5obj, map_files, varids, is_gosat=False):
    def extract_map_lon_lat(filepath):
        loc_dir = filepath.split(os.path.sep)[-2]
        lonstr = _lon_re.search(loc_dir).group()
        lon = mod_utils.format_lon(lonstr)
        latstr = _lat_re.search(loc_dir).group()
        lat = mod_utils.format_lat(latstr)
        return lon, lat

    lons, lats = zip(*[extract_map_lon_lat(f) for f in map_files])
    return read_acos_for_lat_lons(h5obj, lons, lats, varids, is_gosat=is_gosat)


def _read_concat_tccon(tccon_files, reader_fxn, varids):
    data = _make_data_dict(varids)
    for filename in tccon_files:
        this_data = reader_fxn(filename, as_dataframes=False)
        for vid, key in iter_varids(varids):
            data[key].append(_getvar(this_data, vid))

    for k, v in data.items():
        data[k] = np.vstack(v)

    return data


def pair_acos_tccon_vectors(h5obj, tccon_files, tccon_varids, acos_varids, is_gosat=False):

    if tccon_files[0].endswith('.mod'):
        acos_read_fxn = read_acos_matching_mod_files
        tccon_read_fxn = mod_utils.read_mod_file
        tccon_alt_varid = ('profile', 'Height')
    elif tccon_files[0].endswith('.map'):
        acos_read_fxn = read_acos_matching_map_files
        tccon_read_fxn = mod_utils.read_map_file
        tccon_alt_varid = ('profile', 'Height')
    else:
        ext = os.path.splitext(tccon_files[0])[1]
        raise ValueError('Do not know how to handle tccon files with extension {}'.format(ext))

    # We always need altitude for the acos and tccon data.
    if 'priors' in h5obj.keys():
        acos_alt_varid = ('priors', 'altitude')
        acos_alt_scale = 1.0
    else:
        acos_alt_varid = ('Meteorology', 'height_profile_met')
        acos_alt_scale = 1e-3  # m to km
    tccon_varids = list(tccon_varids)
    if tccon_alt_varid not in tccon_varids:
        tccon_varids.append(tccon_alt_varid)

    acos_varids = list(acos_varids)
    if acos_alt_varid not in acos_varids:
        acos_varids.append(acos_alt_varid)

    acos_data, acos_inds = acos_read_fxn(h5obj, tccon_files, acos_varids, is_gosat=is_gosat)
    tccon_data = _read_concat_tccon(tccon_files, tccon_read_fxn, tccon_varids)

    # Now handle interpolating the ACOS data to TCCON altitudes
    acos_height_key = acos_alt_varid[-1]
    acos_height = acos_data[acos_height_key] * acos_alt_scale
    tccon_height_key = tccon_alt_varid[-1]
    tccon_height = tccon_data[tccon_height_key]

    for acos_key, acos_arr in acos_data.items():
        if acos_arr.shape[1] == 1:
            # 2D var (not profile). Do not interpolate
            continue

        new_acos_arr = np.full_like(tccon_height, np.nan)
        for i in range(tccon_height.shape[0]):
            new_acos_arr[i, :] = np.interp(tccon_height[i, :], np.flipud(acos_height[i, :]), np.flipud(acos_arr[i, :]))

        acos_data[acos_key] = new_acos_arr

    return acos_data, tccon_data, acos_inds


def weight_tccon_vars_by_time(last_tccon, next_tccon, last_datetime, next_datetime, acos_datetimes):
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


def match_met_and_prior_by_latlon(acos_met, acos_prior, *additional_dicts):
    lon_key = 'sounding_longitude'
    lat_key = 'sounding_latitude'
    if lon_key not in acos_met or lon_key not in acos_prior:
        raise ValueError('Both dicts must contain the key "sounding_longitude"')
    elif lat_key not in acos_met or lat_key not in acos_prior:
        raise ValueError('Both dicts must contain the key "sounding_latitude"')

    acos_prior_matched = {k: np.full_like(v, np.nan) for k, v in acos_prior.items()}
    add_dicts_matched = [{k: np.full_like(v, np.nan) for k, v in this_dict.items()} for this_dict in additional_dicts]
    # Loop over the lat/lon values and find the corresponding index in the prior, then copy that value (or profile)
    # into the new matched array
    prior_lon = acos_prior[lon_key]
    prior_lat = acos_prior[lat_key]
    met_lon = acos_met[lon_key]
    met_lat = acos_met[lat_key]

    for new_ind, (lon, lat) in enumerate(zip(met_lon, met_lat)):
        old_ind = np.nanargmin((prior_lon - lon)**2 + (prior_lat - lat)**2)
        for key in acos_prior.keys():
            acos_prior_matched[key][new_ind] = acos_prior[key][old_ind]
        for new_dict, old_dict in zip(add_dicts_matched, additional_dicts):
            for key in new_dict.keys():
                new_dict[key][new_ind] = old_dict[key][old_ind]

    # Returning it this way allows the extra dicts to get expanded into multiple outputs
    return [acos_prior_matched] + add_dicts_matched


def make_mod_h5_file(h5file, mod_dir, last_geos_time=pd.Timestamp('2017-05-14 18:00:00'),
                     next_geos_time=pd.Timestamp('2017-05-14 21:00:00'), mod_files=None):
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
    dates = pd.date_range(last_geos_time, next_geos_time, periods=n_files)
    datestrs = aci._convert_to_acos_time_strings(np.array(dates.to_list())).reshape(-1, 1)
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
