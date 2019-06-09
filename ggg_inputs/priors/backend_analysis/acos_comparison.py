import numpy as np
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
        key = vid if isinstance(vid, str) else vid[-1]
        yield vid, key


def _make_data_dict(varids):
    return {k: [] for _, k in iter_varids(varids)}


def read_acos_var(h5obj, varid, inds=slice(None)):
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
    data = var[inds]
    if not np.issubdtype(data.dtype, np.number):
        return data

    if isinstance(data, np.ndarray):
        data[data < aci._fill_val_threshold] = np.nan
    elif data < aci._fill_val_threshold:
        # if not an array must be a scalar.
        data = np.nan
    return data


def read_acos_for_lat_lons(h5obj, lons, lats, varids):
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

    for this_lon, this_lat in zip(lons, lats):
        ind = np.nanargmin((acos_lon - this_lon)**2 + (acos_lat - this_lat)**2)
        ind = np.unravel_index(ind, acos_lon.shape)
        for vid, key in iter_varids(varids):
            data[key].append(read_acos_var(h5obj, vid, inds=ind))

    for k, v in data.items():
        # Should be 1D vectors
        data[k] = np.vstack(v)

    return data


def read_acos_matching_mod_files(h5obj, mod_files, varids):
    def extract_mod_lon_lat(filename):
        lonstr = _lon_re.search(filename).group()
        lon = mod_utils.format_lon(lonstr)
        latstr = _lat_re.search(filename).group()
        lat = mod_utils.format_lat(latstr)
        return lon, lat

    lons, lats = zip(*[extract_mod_lon_lat(os.path.basename(f)) for f in mod_files])
    return read_acos_for_lat_lons(h5obj, lons, lats, varids)


def read_acos_matching_map_files(h5obj, map_files, varids):
    def extract_map_lon_lat(filepath):
        loc_dir = filepath.split(os.path.sep)[-2]
        lonstr = _lon_re.search(loc_dir).group()
        lon = mod_utils.format_lon(lonstr)
        latstr = _lat_re.search(loc_dir).group()
        lat = mod_utils.format_lat(latstr)
        return lon, lat

    lons, lats = zip(*[extract_map_lon_lat(f) for f in map_files])
    return read_acos_for_lat_lons(h5obj, lons, lats, varids)


def _read_concat_tccon(tccon_files, reader_fxn, varids):
    data = _make_data_dict(varids)
    for filename in tccon_files:
        this_data = reader_fxn(filename, as_dataframes=False)
        for vid, key in iter_varids(varids):
            data[key].append(_getvar(this_data, vid))

    for k, v in data.items():
        data[k] = np.vstack(v)

    return data


def pair_acos_tccon_vectors(h5obj, tccon_files, tccon_varids, acos_varids):

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

    acos_data = acos_read_fxn(h5obj, tccon_files, acos_varids)
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

    return acos_data, tccon_data


def weight_tccon_vars_by_time(last_tccon, next_tccon, last_datetime, next_datetime, acos_datetimes):
    last_datenum = aci.datetime2datenum(last_datetime)
    next_datenum = aci.datetime2datenum(next_datetime)
    acos_datenums = np.array([aci.datetime2datenum(d) for d in acos_datetimes])

    weights = aci.time_weight(acos_datenums, last_datenum, next_datenum)
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
