from __future__ import print_function, division

import datetime as dt
from glob import glob
import netCDF4 as ncdf
import numpy as np
import os

from ...common_utils import mod_utils
from . import backend_utils as butils

_mydir = os.path.abspath(os.path.dirname(__file__))


def _match_input_size(err_msg, *inputs):
    err_msg = 'All inputs must be scalar or have the same first dimension' if err_msg is None else err_msg

    inputs = list(inputs)

    max_size = max([np.shape(v)[0] for v in inputs if np.ndim(v) > 0])

    for idx, val in enumerate(inputs):
        if np.ndim(val) == 0:
            inputs[idx] = np.full([max_size], val)
        elif np.shape(val)[0] == 1:
            desired_shape = np.ones([np.ndim(val)], dtype=np.int)
            desired_shape[0] = max_size
            inputs[idx] = np.tile(val, desired_shape)
        elif np.shape(val)[0] != max_size:
            raise ValueError(err_msg)

    return inputs


def _read_prior_for_time(prior_dir, prior_hour, specie, prior_var=None, z_var=None):
    all_prior_files = glob(os.path.join(prior_dir, '*.map'))
    prior_file = None
    match_str = '{:02d}00.map'.format(prior_hour)
    for f in all_prior_files:
        if f.endswith(match_str):
            prior_file = f
            break

    if prior_file is None:
        raise IOError('Failed to find file for hour {} in {}'.format(prior_hour, prior_dir))

    prior_var = specie.lower() if prior_var is None else prior_var

    prior_data = mod_utils.read_map_file(prior_file)
    prior_alt = prior_data['profile']['Height']
    prior_conc = prior_data['profile'][prior_var]
    if z_var is None:
        prior_z = prior_alt
    else:
        prior_z = prior_data['profile'][z_var]

    return prior_conc, prior_alt, prior_z


def match_ace_prior_profiles(prior_dirs, ace_dir, specie, match_alt=True, prior_var=None, prior_z_var=None, ace_var=None):
    # Gather a list of all the dates, lats, and lon of the directories containing the priors
    prior_dates = []
    prior_lats = []
    prior_lons = []
    for this_prior_dir in prior_dirs:
        this_prior_date, this_prior_lon, this_prior_lat = butils.get_date_lon_lat_from_dirname(this_prior_dir)
        prior_dates.append(this_prior_date)
        prior_lats.append(this_prior_lat)
        prior_lons.append(this_prior_lon)

    prior_dates = np.array(prior_dates)
    prior_lons = np.array(prior_lons)
    prior_lats = np.array(prior_lats)

    # Now find what hour of the day the ACE profile is. Make this into the closest multiple of 3 since we have outputs
    # every 3 hours
    ace_hours = get_matching_ace_hours(prior_lons, prior_lats, prior_dates, ace_dir, specie)
    prior_hours = (np.round(ace_hours/3)*3).astype(np.int)
    # This is not great, but if there's an hour > 21, we need to set it to 21 because 2100 UTC is the last hour we have
    # priors for. What would be better is to go to the next day, but at the moment we don't have priors for the next
    # day. This can be fixed if/when we do the full ACE record.
    prior_hours[prior_hours > 21] = 21

    # Read in the priors. We'll need the altitude regardless of whether we're interpolating the ACE profiles to those
    # altitudes. Also convert the prior dates to datetimes with the correct hour
    priors = []
    prior_alts = []
    prior_zs = []
    prior_datetimes = []
    for pdir, phr, pdate, in zip(prior_dirs, prior_hours, prior_dates):
        prior_datetimes.append(pdate.replace(hour=phr))
        this_prior, this_alt, this_z = _read_prior_for_time(pdir, phr, specie, prior_var=prior_var, z_var=prior_z_var)

        # Reshape to allow concatenation later
        priors.append(this_prior.reshape(1, -1))
        prior_alts.append(this_alt.reshape(1, -1))
        prior_zs.append(this_z.reshape(1, -1))

    priors = np.concatenate(priors, axis=0)
    prior_alts = np.concatenate(prior_alts, axis=0)
    prior_zs = np.concatenate(prior_zs, axis=0)
    prior_datetimes = np.array(prior_datetimes)

    # Read in the ACE data, interpolating to the profile altitudes if requested
    ace_profiles, ace_alts, ace_datetimes = get_matching_ace_profiles(prior_lons, prior_lats, prior_dates, ace_dir,
                                                                      specie, alt=prior_alts if match_alt else None,
                                                                      ace_var=ace_var)

    return {'priors': priors, 'prior_alts': prior_alts, 'prior_datetimes': prior_datetimes, 'prior_zs': prior_zs,
            'ace_profiles': ace_profiles, 'ace_alts': ace_alts, 'ace_datetimes': ace_datetimes}


def _find_matching_ace_profile(this_lon, this_lat, this_date, ace_lons, ace_lats, ace_dates):
    xx = (ace_dates >= this_date) & (ace_dates < (this_date + dt.timedelta(days=1))) \
         & np.isclose(ace_lons, this_lon) & np.isclose(ace_lats, this_lat)
    if np.sum(xx) < 1:
        raise RuntimeError('Could not find a profile at lon/lat {}/{} on {}'.format(
            this_lon, this_lat, this_date
        ))
    elif np.sum(xx) > 1:
        raise RuntimeError('Found multiple profiles matching lon/lat {}/{} on {}'.format(
            this_lon, this_lat, this_date
        ))
    return xx


def get_matching_ace_hours(lon, lat, date, ace_dir, specie):
    lon, lat, date = _match_input_size('lon, lat, and date must have compatible sizes', lon, lat, date)
    ace_file = butils.find_ace_file(ace_dir, specie)
    with ncdf.Dataset(ace_file, 'r') as nch:
        ace_dates = butils.read_ace_date(nch)
        ace_hours = butils.read_ace_var(nch, 'hour', None)
        ace_lons = butils.read_ace_var(nch, 'longitude', None)
        ace_lats = butils.read_ace_var(nch, 'latitude', None)

    matched_ace_hours = np.full([np.size(lon)], np.nan)
    for idx, (this_lon, this_lat, this_date) in enumerate(zip(lon, lat, date)):
        xx = _find_matching_ace_profile(this_lon, this_lat, this_date, ace_lons, ace_lats, ace_dates)
        matched_ace_hours[idx] = ace_hours[xx]

    return matched_ace_hours


def get_matching_ace_profiles(lon, lat, date, ace_dir, specie, alt=None, prior_var=None, ace_var=None):
    """
    Get the ACE profile(s) for a particular species at specific lat/lons

    :param lon: the longitudes of the ACE profiles to load
    :param lat: the latitudes of the ACE profiles to load
    :param date: the dates of the ACE profiles to load
    :param ace_dir: the directory to find the ACE files
    :param specie: which chemical specie to load
    :param alt: if given, altitudes to interpolate ACE data to. MUST be 2D and the altitudes for a single profile must
     go along the second dimension. The first dimension is assumed to be different profiles. If not given the default
     ACE altitudes are used.
    :return:

    ``lon``, ``lat``, and ``date`` can be given as scalars or 1D arrays. If scalars (or arrays with 1 element), they are
    assumed to be the same for all profiles. If arrays with >1 element, then they are taken to be different values for
    each profile. ``alt`` is similar; if it is given and is a 1-by-n array, then those n altitude are used for all
    profiles. If m-by-n, then it is assumed that there are different altitude levels for each file. All inputs that are
    not scalar must have the same first dimension. Example::

        get_matching_ace_profiles([-90.0, -89.0, -88.0], [0.0, 10.0, 20.0], datetime(2012,1,1), 'ace_data', 'CH4')

    will load three profiles from 1 Jan 2012 at the three lon/lats given.
    """

    interp_to_alt = alt is not None
    lon, lat, date, alt = _match_input_size('lon, lat, date, and alt must have compatible sizes', lon, lat, date, alt)

    ace_file = butils.find_ace_file(ace_dir, specie)

    ace_var = specie.upper() if ace_var is None else ace_var

    with ncdf.Dataset(ace_file, 'r') as nch:
        ace_dates = butils.read_ace_date(nch)
        ace_lons = butils.read_ace_var(nch, 'longitude', None)
        ace_lats = butils.read_ace_var(nch, 'latitude', None)
        ace_alts = butils.read_ace_var(nch, 'altitude', None)
        ace_qflags = butils.read_ace_var(nch, 'quality_flag', None)
        ace_profiles = butils.read_ace_var(nch, ace_var, ace_qflags)

    n_profs = np.size(lon)
    n_out_levels = np.shape(alt)[1] if interp_to_alt else np.size(ace_alts)
    out_profiles = np.full([n_profs, n_out_levels], np.nan)
    out_datetimes = np.full([n_profs], None)

    for idx, (this_lon, this_lat, this_date, this_alt) in enumerate(zip(lon, lat, date, alt)):
        xx = _find_matching_ace_profile(this_lon, this_lat, this_date, ace_lons, ace_lats, ace_dates)

        this_prof = ace_profiles[xx, :]
        if interp_to_alt:
            this_prof = np.interp(this_alt, ace_alts, this_prof.squeeze())

        out_profiles[idx, :] = this_prof
        out_datetimes[idx] = ace_dates[xx]

    if not interp_to_alt:
        alt = np.tile(ace_alts.reshape(1, -1), [n_profs, 1])

    return out_profiles, alt, out_datetimes
