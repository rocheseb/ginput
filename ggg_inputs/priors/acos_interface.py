from __future__ import print_function, division

import argparse
import datetime as dt
import h5py
from itertools import repeat, product
from multiprocessing import Pool
import numpy as np

from ..common_utils import mod_utils
from ..common_utils.ggg_logging import logger, setup_logger
from ..mod_maker import mod_maker
from ..priors import tccon_priors


# Values lower than this will be replaced with NaNs when reading in the resampled met data
_fill_val_threshold = -9e5
# NaNs will be replaced with this value when writing the HDF5 file
_fill_val = -999999


def acos_interface_main(met_resampled_file, geos_files, output_file, nprocs=0):
    """
    The primary interface to create CO2 priors for the ACOS algorithm

    :param met_resampled_file: the path to the HDF5 file containing the met data resampled to the satellite footprints
    :type met_resampled_file: str

    :param geos_files: the list of GEOS FP or FP-IT pressure level met files that bracket the satellite observations.
     That is, if the first satellite observation in ``met_resampled_file`` is on 2019-05-22 02:30:00Z and the last is
     at 2019-05-22 03:30:00Z, then the GEOS files for 0, 3, and 6 UTC for 2019-05-22 must be listed.
    :type geos_files: list(str)

    :param output_file: the path to where the priors (and related data) should be written as an HDF5 file.
    :type output_file: str

    :return: None
    """
    met_data = read_resampled_met(met_resampled_file)

    # Reshape the met data from (soundings, footprints, levels) to (profiles, level)
    orig_shape = met_data['pv'].shape
    nlevels = orig_shape[-1]
    pv_array = met_data['pv'].reshape(-1, nlevels)
    theta_array = met_data['theta'].reshape(-1, nlevels)

    datenum_array = met_data['datenums'].reshape(-1)
    eqlat_array = compute_sounding_equivalent_latitudes(pv_array, theta_array, datenum_array, geos_files, nprocs=nprocs)

    met_data['el'] = eqlat_array.reshape(orig_shape)

    # Create the CO2 priors
    co2_record = tccon_priors.CO2TropicsRecord()

    # The keys here define the variable names that will be used in the HDF file. The values define the corresponding
    # keys in the output dictionaries from tccon_priors.generate_single_tccon_prior.
    var_mapping = {'co2_prior': co2_record.gas_name, 'altitude': 'Height', 'pressure': 'Pressure',
                   'equivalent_latitude': 'EL'}

    if nprocs == 0:
        profiles, units = _prior_serial(orig_shape=orig_shape, var_mapping=var_mapping, met_data=met_data,
                                        co2_record=co2_record)
    else:
        profiles, units = _prior_parallel(orig_shape=orig_shape, var_mapping=var_mapping, met_data=met_data,
                                          co2_record=co2_record, nprocs=nprocs)

    # Add latitude and longitude to the priors file
    profiles['sounding_longitude'] = met_data['longitude']
    units['sounding_longitude'] = 'degrees_east'
    profiles['sounding_latitude'] = met_data['latitude']
    units['sounding_latitude'] = 'degrees_north'

    # Write the priors to the file requested.
    write_prior_h5(output_file, profiles, units)


def _prior_helper(i_sounding, i_foot, mod_data, obs_date, co2_record, var_mapping):
    if i_foot == 0:
        logger.info('Processing set of soundings {}'.format(i_sounding + 1))
    profiles = {k: np.full_like(mod_data['profile']['Height'], np.nan) for k in var_mapping.keys()}
    units = {k: '' for k in var_mapping.keys()}

    if obs_date < dt.datetime(1993, 1, 1):
        # In the test met file I was given, one set of soundings had a date set to 20 Dec 1992, while the rest
        # where on 14 May 2017. Since the 1992 date is close to 999999 seconds before 1 Jan 1993 and the dates
        # are in TAI93 time, I assume these are fill values.
        logger.important('Date before 1993 ({}) found, assuming this is a fill value and skipping '
                         '(sounding group/footprint {}/{})'.format(obs_date, i_sounding + 1, i_foot + 1))
        return profiles, None
    elif np.all(np.isnan(mod_data['profile']['Height'])):
        logger.important('Profile at sounding group/footprint {}/{} is all fill values, not calculating prior'
                         .format(i_sounding + 1, i_foot + 1))
        return profiles, None

    try:
        priors_dict, priors_units, priors_constants = tccon_priors.generate_single_tccon_prior(
            mod_data, obs_date, dt.timedelta(hours=0), co2_record,
        )
    except Exception as err:
        raise err.__class__(err.args[0] + ' Occurred at sounding = {}, footprint = {}'.format(i_sounding, i_foot))

    # Convert the CO2 priors from ppm to dry mole fraction.
    priors_dict[co2_record.gas_name] *= 1e-6
    priors_units[co2_record.gas_name] = 'dmf'

    for h5_var, tccon_var in var_mapping.items():
        # The TCCON code returns profiles ordered surface-to-space. ACOS expects space-to-surface
        profiles[h5_var] = np.flipud(priors_dict[tccon_var])
        # Yes this will get set every time but only needs set once. Can optimize later if it's slow.
        units[h5_var] = priors_units[tccon_var]

    return profiles, units


def _make_output_profiles_dict(orig_shape, var_mapping):
    return {k: np.full(orig_shape, np.nan) for k in var_mapping}


def _prior_serial(orig_shape, var_mapping, met_data, co2_record):
    profiles = _make_output_profiles_dict(orig_shape, var_mapping)
    units = None

    for i_sounding in range(orig_shape[0]):
        for i_foot in range(orig_shape[1]):
            mod_data = _construct_mod_dict(met_data, i_sounding, i_foot)
            obs_date = met_data['dates'][i_sounding, i_foot]

            this_profiles, this_units = _prior_helper(i_sounding, i_foot, mod_data, obs_date, co2_record, var_mapping)
            for h5_var, h5_array in profiles.items():
                h5_array[i_sounding, i_foot, :] = this_profiles[h5_var]
            if this_units is not None:
                units = this_units

    return profiles, units


def _prior_parallel(orig_shape, var_mapping, met_data, co2_record, nprocs):
    logger.info('Running CO2 prior calculation in parallel with {} processes'.format(nprocs))

    # Need to prepare iterators of the sounding and footprint indices, as well as the individual met dictionaries
    # and observation dates. We only want to pass the individual dictionary and date to each worker, not the whole
    # met data, because that would probably be slow due to overhead. (Not tested however.)
    sounding_inds, footprint_inds = [x for x in zip(*product(range(orig_shape[0]), range(orig_shape[1])))]
    mod_dicts = map(_construct_mod_dict, repeat(met_data), sounding_inds, footprint_inds)
    obs_dates = [met_data['dates'][isound, ifoot] for isound, ifoot in zip(sounding_inds, footprint_inds)]

    with Pool(processes=nprocs) as pool:
        result = pool.starmap(_prior_helper, zip(sounding_inds, footprint_inds, mod_dicts, obs_dates, repeat(co2_record), repeat(var_mapping)))

    # At this point, result will be a list of tuples of pairs of dicts, the first dict the profiles dict, the second
    # the units dict or None if the prior calculation did not run. We need to combine the profiles into one array per
    # variable and get one valid units dict
    profiles = _make_output_profiles_dict(orig_shape, var_mapping)
    units = None
    for (these_profs, these_units), i_sounding, i_foot in zip(result, sounding_inds, footprint_inds):
        if these_units is not None:
            units = these_units
        for h5var, h5array in profiles.items():
            h5array[i_sounding, i_foot, :] = these_profs[h5var]

    return profiles, units


def compute_sounding_equivalent_latitudes(sounding_pv, sounding_theta, sounding_datenums, geos_files, nprocs=0):
    """
    Compute equivalent latitudes for a collection of OCO soundings

    :param sounding_pv: potential vorticity in units of PVU (1e-6 K * m2 * kg^-1 * s^-1). Must be an array with
     dimensions (profiles, levels). That is, if the data read from the resampled met files have dimensions (soundings,
     footprints, levels), these must be reshaped so that the first two dimensions get collapsed into one.
    :type sounding_pv: :class:`numpy.ndarray`

    :param sounding_theta: potential temperature in units of K. Same shape as ``sounding_pv`` required.
    :type sounding_theta: :class:`numpy.ndarray`

    :param sounding_datenums: date and time of each profile as a date number (a numpy :class:`~numpy.datetime64` value
     converted to a float type, see :func:`datetime2datenum` in this module). Same shape as ``sounding_pv`` required.
    :type sounding_datenums: :class:`numpy.ndarray`

    :param geos_files: a list of the GEOS 3D met files that bracket the times of all the soundings. Need not be absolute
     paths, but must be paths that resolve correctly from the current working directory.
    :type geos_files: list(str)

    :return: an array of equivalent latitudes with dimensions (profiles, levels)
    :rtype: :class:`numpy.ndarray`
    """
    # Create interpolators for each of the GEOS FP files provided. The resulting dictionary will have the files'
    # datetimes as keys
    geos_utc_times = [mod_utils.datetime_from_geos_filename(f) for f in geos_files]
    geos_datenums = np.array([datetime2datenum(d) for d in geos_utc_times])

    eqlat_fxns = mod_maker.equivalent_latitude_functions_from_geos_files(geos_files, geos_utc_times)
    # it will be easier to work with this as a list of the interpolators in the right order.
    eqlat_fxns = [eqlat_fxns[k] for k in geos_utc_times]

    # This part is going to be slow. We need to use the interpolators to get equivalent latitude profiles for each
    # sounding for the two times on either side of the sounding time, then do a further linear interpolation to
    # the actual sounding time.

    if nprocs == 0:
        return _eqlat_serial(sounding_pv, sounding_theta, sounding_datenums, geos_datenums, eqlat_fxns)
    else:
        return _eqlat_parallel(sounding_pv, sounding_theta, sounding_datenums, geos_datenums, eqlat_fxns, nprocs=nprocs)


def _eqlat_helper(idx, pv_vec, theta_vec, datenum, eqlat_fxns, geos_datenums):
    logger.debug('Calculating eq. lat. {}'.format(idx))
    try:
        i_last_geos = _find_helper(geos_datenums <= datenum, order='last')
        i_next_geos = _find_helper(geos_datenums > datenum, order='first')
    except IndexError as err:
        logger.important('Sounding {}: could not find GEOS file by time. Assuming fill value for time'.format(idx))
        return np.full_like(pv_vec, np.nan)

    last_el_profile = _make_el_profile(pv_vec, theta_vec, eqlat_fxns[i_last_geos])
    next_el_profile = _make_el_profile(pv_vec, theta_vec, eqlat_fxns[i_next_geos])

    # Interpolate between the two times by calculating a weighted average of the two profiles based on the sounding
    # time. This avoids another for loop over all levels.
    weight = (datenum - geos_datenums[i_last_geos]) / (geos_datenums[i_next_geos] - geos_datenums[i_last_geos])
    return weight * last_el_profile + (1 - weight) * next_el_profile


def _eqlat_clip(el):
    def trim(bool_ind, replacement_val):
        if np.any(bool_ind):
            # nanmax can't take an empty array so we have to make sure that
            # bool_ind has at least one true value
            maxdiff = np.nanmax(np.abs(el[bool_ind] - replacement_val))
            el[bool_ind] = replacement_val
            return maxdiff
        else:
            return 0.0

    xx1 = el < -90.0
    xx2 = el > 90.0

    max_below = trim(xx1, -90.0)
    max_above = trim(xx2, 90.0)
    n_outside = xx1.sum() + xx2.sum()

    if n_outside > 0:
        logger.warning('{} equivalent latitudes were outside the range [-90,90] (max difference {}). They have been clipped to [-90,90].'
                       .format(n_outside, max(max_below, max_above))) 


def _eqlat_serial(sounding_pv, sounding_theta, sounding_datenums, geos_datenums, eqlat_fxns):
    sounding_eqlat = np.full_like(sounding_pv, np.nan)

    # This part is going to be slow. We need to use the interpolators to get equivalent latitude profiles for each
    # sounding for the two times on either side of the sounding time, then do a further linear interpolation to
    # the actual sounding time.
    logger.info('Running eq. lat. calculation in serial')
    for idx, (pv_vec, theta_vec, datenum) in enumerate(zip(sounding_pv, sounding_theta, sounding_datenums)):
        sounding_eqlat[idx] = _eqlat_helper(idx, pv_vec, theta_vec, datenum, eqlat_fxns, geos_datenums)

    _eqlat_clip(sounding_eqlat)
    return sounding_eqlat


def _eqlat_parallel(sounding_pv, sounding_theta, sounding_datenums, geos_datenums, eqlat_fxns, nprocs):
    logger.info('Running eq. lat. calculation in parallel with {} processes'.format(nprocs))
    with Pool(processes=nprocs) as pool:
        result = pool.starmap(_eqlat_helper, zip(range(sounding_pv.shape[0]), sounding_pv, sounding_theta, sounding_datenums,
                                                 repeat(eqlat_fxns), repeat(geos_datenums)))

    sounding_eqlat = np.array(result)
    _eqlat_clip(sounding_eqlat)
    return sounding_eqlat


def read_resampled_met(met_file):
    """
    Read the required data from the HDF5 file containing the resampled met data.

    :param met_file: the path to the met file
    :type met_file: str

    :return: a dictionary with variables both read directly from the met file and derived from those values. Keys are:

        * "pv" - potential vorticity in PVU
        * "theta" - potential temperature in K
        * "temperature" - temperature profiles in K
        * "pressure" - pressure profiles in hPa
        * "date_strings" - the sounding date/time as a string
        * "dates" - the sounding date/time as a Python :class:`datetime.datetime` object
        * "datenums" - the sounding date/time as a floating point number (see :func:`datetime2datenum`)
        * "altitude" - the altitude profiles in km
        * "latitude" - the sounding latitudes in degrees (south is negative)
        * "trop_pressure" - the blended tropopause pressure in hPa
        * "trop_temperature" - the blended tropopause temperature in K
        * "surf_gph" - surface geopotential height in m^2 s^-2
        * "surf_alt" - the surface altitude, derived from surface geopotential, in km

    :rtype: dict
    """
    met_group = 'Meteorology'
    sounding_group = 'SoundingGeometry'
    var_dict = {'pv': [met_group, 'epv_profile_met'],
                'temperature': [met_group, 'temperature_profile_met'],
                'pressure': [met_group, 'vector_pressure_levels_met'],
                'date_strings': [sounding_group, 'sounding_time_string'],
                'altitude': [met_group, 'height_profile_met'],
                'latitude': [sounding_group, 'sounding_latitude'],
                'longitude': [sounding_group, 'sounding_longitude'],
                'trop_pressure': [met_group, 'blended_tropopause_pressure_met'],
                'trop_temperature': [met_group, 'tropopause_temperature_met'],
                'surf_gph': [met_group, 'gph_met']
                }
    data_dict = dict()
    with h5py.File(met_file, 'r') as h5obj:
        for out_var, (group_name, var_name) in var_dict.items():
            logger.debug('Reading {}/{}'.format(group_name, var_name))
            tmp_data = h5obj[group_name][var_name][:]
            # TODO: verify that -999999 is the only fill value used with Chris/Albert
            if np.issubdtype(tmp_data.dtype, np.number):
                tmp_data[tmp_data < _fill_val_threshold] = np.nan
            data_dict[out_var] = tmp_data

    # Potential temperature needs to be calculated, the date strings need to be converted, and the potential temperature
    # needs scaled to units of PVU

    # pressure in the met file is in Pa, need hPa for the potential temperature calculation
    data_dict['pressure'] *= 0.01  # convert from Pa to hPa
    data_dict['theta'] = mod_utils.calculate_potential_temperature(data_dict['pressure'], data_dict['temperature'])
    data_dict['dates'] = _convert_acos_time_strings(data_dict['date_strings'], format='datetime')
    data_dict['datenums'] = _convert_acos_time_strings(data_dict['date_strings'], format='datenum')
    data_dict['pv'] *= 1e6

    data_dict['altitude'] *= 1e-3  # in meters, need kilometers
    data_dict['trop_pressure'] *= 1e-2  # in Pa, need hPa

    # surf_gph is height derived from geopotential by divided by g0 = 9.80665 m/s^2 according to Chris O'Dell on
    # 21 May 2019. We can use this as the surface altitude, just need to convert from meters to kilometers.
    data_dict['surf_alt'] = 1e-3 * data_dict['surf_gph']

    return data_dict


def write_prior_h5(output_file, profile_variables, units):
    """
    Write the CO2 priors to and HDF5 file.

    :param output_file: the path to the output file.
    :type output_file: str

    :param profile_variables: a dictionary containing the variables to write. The keys will be used as the variable
     names.
    :type profile_variables: dict

    :param units: a dictionary defining the units each variable is in. Must have the same keys as ``profile_variables``.
    :type units: dict(str)

    :return: none, writes to file on disk.
    """
    with h5py.File(output_file, 'w') as h5obj:
        h5grp = h5obj.create_group('priors')
        for var_name, var_data in profile_variables.items():
            # Replace NaNs with numeric fill values
            filled_data = var_data.copy()
            filled_data[np.isnan(filled_data)] = _fill_val

            # Write the data
            var_unit = units[var_name]
            dset = h5grp.create_dataset(var_name, data=var_data, fillvalue=_fill_val)
            dset.attrs['units'] = var_unit


def _convert_acos_time_strings(time_string_array, format='datetime'):
    """
    Convert an array of time strings in format yyyy-mm-ddTHH:MM:SS.dddZ into an array of python datetimes

    :param time_string_array: the array of input strings
    :type: :class:`numpy.ndarray`

    :param format: controls what format the time data are returned in. Options are:

     * "datetime" - returns :class:`datetime.datetime` objects
     * "datenum" - return dates as a linear number. This should be in units of seconds since 1 Jan 1970; however, this
       unit is not guaranteed, so implementations that rely on that unit should be avoided if possible.  See
       :func:`datetime2datenum` for more information.

    :type format: str

    :return: an array, the same size as ``time_string_array``, that contains the times as datetimes.
    """

    # Start with a flat output array for ease of iteration. Reshape to the same shape as the input at the end.
    if format == 'datetime':
        init_val = None
    elif format == 'datenum':
        init_val = np.nan
    else:
        raise NotImplementedError('No initialization value defined for format == "{}"'.format(format))

    output_array = np.full([time_string_array.size], init_val)
    for idx, time_str in enumerate(time_string_array.flat):
        time_str = time_str.decode('utf8')
        datetime_obj = dt.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        if format == 'datetime':
            output_array[idx] = datetime_obj
        elif format == 'datenum':
            output_array[idx] = datetime2datenum(datetime_obj)
        else:
            raise NotImplementedError('No conversion method defined for format == "{}"'.format(format))

    return np.reshape(output_array, time_string_array.shape)


def datetime2datenum(datetime_obj):
    """
    Convert a single :class:`datetime.datetime` object into a date number.

    Internally, this converts the datetime object into a :class:`numpy.datetime64` object with units of seconds, then
    converts that to a float type. Under numpy version 1.16.2, this results in a number that is seconds since 1 Jan
    1970; however, I have not seen any documentation from numpy guaranteeing that behavior. Therefore, any use of these
    date numbers should be careful to verify this behavior. The following assert block can be used to check this::

        assert numpy.isclose(datetime2datenum('1970-01-01'), 0.0) and numpy.isclose(datetime2datenum('1970-01-02'), 86400)

    :param datetime_obj: the datetime to convert. May be any type that :class:`numpy.datetime64` can intepret as a
     datetime.

    :return: the converted date number
    :rtype: :class:`numpy.float`
    """
    return np.datetime64(datetime_obj, 's').astype(np.float)


def _find_helper(bool_vec, order='all'):
    """
    Helper function to find indices of true values within a vector of booleans

    :param bool_vec: the vector to find indices of ``True`` values in.
    :type bool_vec: :class:`numpy.ndarray`

    :param order: a string indicating which indices to return. "all" returns all indices, "first" returns just the
     index of the first true value, and "last" returns the index of the last true value.
    :type order: str

    :return: the index or indices of the true values
    :rtype: int or ndarray(int)
    """
    order = order.lower()
    inds = np.flatnonzero(bool_vec)
    if order == 'last':
        return inds[-1]
    elif order == 'first':
        return inds[0]
    elif order == 'all':
        return inds
    else:
        raise ValueError('"{}" is not an allowed value for order'.format(order))


def _make_el_profile(pv, theta, interpolator):
    """
    Create an equivalent latitude profile from profiles of PV, theta, and one of the eq. lat. intepolators

    This function will create each level of the eq. lat. profile separately. This is safer than calling the interpolator
    with the PV and theta vectors because the latter returns a 2D array of eq. lat., and occasionally the profile is
    not the diagonal of that array. Since I have not figured out under what conditions that happens, I find this
    approach of calculating each level in a loop, safer.

    :param pv: the profile of potential vorticity in PVU (1e-6 K * m2 * kg^-1 * s^-1).
    :type pv: 1D :class:`numpy.ndarray`

    :param theta: the profile of potential temperature in K
    :type theta: 1D :class:`numpy.ndarray`

    :param interpolator: one of the interpolators returned by :func:`mod_utils.equivalent_latitude_functions_from_geos_files`
     that interpolates equivalent latitude to given PV and theta.
    :type interpolator: :class:`scipy.interpolate.interp2d`

    :return: the equivalent latitude profile
    :rtype: 1D :class:`numpy.ndarray`
    """
    el = np.full_like(pv, np.nan)
    for i in range(el.size):
        el[i] = interpolator(pv[i], theta[i])
    return el


def _construct_mod_dict(acos_data_dict, i_sounding, i_foot):
    """
    Create a dictionary akin that mimics a TCCON .mod file from a single sounding's data.

    :param acos_data_dict: A dictionary containing all the data from the ACOS met resample file (or calculated from it)
     necessary for the creation of TCCON CO2 priors. This dictionary must have the following entries:

        * "el" - equivalent latitude profile in degrees
        * "temperature" - temperature profile in K
        * "pressure" - pressure profile in hPa
        * "theta" - potential temperature profile in K
        * "altitude" - altitude profile in km
        * "latitude" - scalar value defining the sounding latitude
        * "trop_temperature" - the scalar temperature of the tropopause in K
        * "trop_pressure" - the scalar pressure at the tropopause in hPa

     Note that this function also assumes that all these arrays have either dimensions (soundings, footprints, levels)
     or (soundings, footprints).
    :type acos_data_dict: dict

    :param i_sounding: the 0-based index for the sounding (the first dimension in the arrays of ``acos_data_dict``).
    :type i_sounding: int

    :param i_foot: the 0-based index for the footprint (the second dimension in the arrays of ``acos_data_dict``).
    :type i_foot: int

    :return: a dictionary suitable for the first argument of :func:`tccon_priors.generate_single_tccon_prior`
    :rtype: dict
    """

    # This dictionary maps the variables names in the acos_data_dict (the keys) to the keys in the mod-like dict. The
    # latter must be a 2-element collection, since that dictionary is a dict-of-dicts. The first level of keys defines
    # whether the variable is 3D ("profile"), 2D ("scalar") or fixed ("constant") and the second is the actual variable
    # name. Note that these must match the expected structure of a .mod file EXACTLY.
    var_mapping = {'el': ['profile', 'EL'],
                   'temperature': ['profile', 'Temperature'],
                   'pressure': ['profile', 'Pressure'],
                   'theta': ['profile', 'PT'],
                   'altitude': ['profile', 'Height'],
                   'surf_alt': ['scalar', 'Height'],  # need to read in
                   'latitude': ['constants', 'obs_lat'],
                   'trop_temperature': ['scalar', 'TROPT'],  # need to read in
                   'trop_pressure': ['scalar', 'TROPPB']}  # need to read in

    subgroups = set([l[0] for l in var_mapping.values()])
    mod_dict = {k: dict() for k in subgroups}

    for acos_var, (mod_group, mod_var) in var_mapping.items():
        # For 3D vars this slicing will create a vector. For 2D vars, it will create a scalar
        tmp_val = acos_data_dict[acos_var][i_sounding, i_foot]
        # The profile variables need flipped b/c for ACOS they are arranged space-to-surface,
        # but the TCCON code expects surface-to-space
        if mod_group == 'profile':
            tmp_val = np.flipud(tmp_val)
        mod_dict[mod_group][mod_var] = tmp_val
    return mod_dict


def parse_args():
    def comma_list(argin):
        return tuple([a.strip() for a in argin.split(',')])

    parser = argparse.ArgumentParser(description='Command line interface to generate CO2 priors for the ACOS algorithm')
    parser.add_argument('geos_files', type=comma_list, help='Comma-separated list of paths to the GEOS FP or FP-IT '
                                                            'files that cover the times of the soundings. For example, '
                                                            'if the soundings span 0100Z to 0200Z on 2018-01-01, '
                                                            'then the GEOS files for 2018-01-01 0000Z and 0300Z must '
                                                            'be listed. If the soundings span 0230Z to 0330Z, then the '
                                                            'GEOS files for 2018-01-01 0000Z, 0300Z, and 0600Z must '
                                                            'be listed.')
    parser.add_argument('met_resampled_file', help='The path to the HDF5 file containing the GEOS meteorology sampled '
                                                   'at the satellite soundings')
    parser.add_argument('output_file', help='The filename to give the output HDF5 file containing the CO2 profiles and '
                                            'any additional variables. Note that this path will be overwritten without '
                                            'any warning.')
    parser.add_argument('-v', '--verbose', dest='log_level', default=0, action='count',
                        help='Increase logging verbosity')
    parser.add_argument('-q', '--quiet', dest='log_level', const=-1, action='store_const',
                        help='Silence all logging except warnings and critical messages')
    parser.add_argument('-l', '--log-file', default=None, help='Use this to define a path for logging messages to be '
                                                               'stored in. Log messages are still printed to stdout if '
                                                               'this is given. NOTE: Python errors are not captured by '
                                                               'the logging machinery and so will still print to '
                                                               'stderr.')
    parser.add_argument('-n', '--nprocs', default=0, type=int, help='Number of processors to use in parallelization')

    return vars(parser.parse_args())


def main():
    args = parse_args()

    log_level = args.pop('log_level')
    log_file = args.pop('log_file')
    setup_logger(log_file=log_file, level=log_level)

    acos_interface_main(**args)


if __name__ == '__main__':
    main()
