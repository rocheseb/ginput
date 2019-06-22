from collections import OrderedDict
from copy import deepcopy
from glob import glob
import numpy as np
import os
import pandas as pd
import re

from . import backend_utils as butils
from ...mod_maker.tccon_sites import tccon_site_info_for_date
from ...common_utils import mod_utils
from ...common_utils.ggg_logging import logger

_atm_date_fmt = '%Y-%m-%d'
_atm_datetime_fmt = '%Y-%m-%d %H:%M:%S'
_atm_filedate_fmt = '%Y%m%d'


def write_atm_files(data_df, metadata_df, common_header_info, profnum_var, lon_var, lat_var, additional_var,
                    altitude_var, pressure_var, temperature_var=None, h2o_var=None, additional_var_atm_name=None,
                    out_dir='.', out_prefix='aircraft', atm_name_fmt='{prefix}_{lat}_{lon}_{date}_{profnum}_{var}.atm',
                    skip_nans=True, min_num_pts=0):
    """
    Write .atm format files for each profile described in a data frame of aircraft data.

    :param data_df: a data frame holding the aircraft data. Must be index with the datetime of the observations, with
     the difference variables as columns.
    :type data_df: :class:`pandas.DataFrame`

    :param metadata_df: a data frame holding additional information about the variables. Must have one row named "unit"
     or "units" and the variable names as columns.
    :type metadata_df: :class:`pandas.DataFrame`

    :param common_header_info: a dictionary containing the header info that will be the same across all of the .atm
     files generated in this call. Required keys are:

      * "description" - a brief description of the observation contained in the .atm file (e.g. "AirCore Observation
        from ObsPack")
      * "platform" - the measurement platform, as a string (e.g. "AirCore", "DC8", etc.)

     Optional keys are:

      * "2sigma" - the 2 sigma error in the additional variable, in the same units as it is in ``data_df``.
      * "surface_source" - source for surface data for the additional variable
      * "h2o_surface_source" - source for surface data for H2O

    :param profnum_var: the column in the DataFrame that holds the profile numbers. The assumption will be made that
     values here > 0 indicate actual profiles, and values <= 0 indicate non-profile data points.
    :type profnum_var: str

    :param altitude_var: the column in the DataFrame that holds the aircraft altitude.
    :type altitude_var: str

    :param pressure_var: the column in the DataFrame that holds the external pressure.
    :type pressure_var: str

    :param temperature_var: the column in the DataFrame that holds the external temperature. If not given, that column
     will be filled with NaNs in the .atm files.
    :type temperature_var: str

    :param h2o_var: the column in the DataFrame that holds the water vapor concentration. If not given, that column will
     be filled with NaNs in the .atm files.

    :param additional_var: column in the DataFrame that holds an additional variable to include in the .atm file. If
     not given, no extra column will be provided.
    :type additional_var: str

    :param additional_var_atm_name: the name to give the additional variable in the .atm file. Will be suffixed with
     "_profile" automatically.
    :type additional_var_atm_name: str

    :param out_dir: the directory to save the .atm files to.
    :type out_dir: str

    :param out_prefix: prefix to use in the .atm file names.
    :type out_prefix: str

    :param atm_name_fmt: a string using curly bracket formatting that specifies the pattern for the .atm file names.
     Keys allowed are ``prefix`` (value of ``out_prefix``), ``lon`` and ``lat`` (mean lon/lat to 1 decimal point),
     ``date`` (the date in yyyymmdd format), ``profnum`` (the profile number in the aircraft data), and ``var`` (the
     value of ``additional_var_atm_name``).
    :type atm_name_fmt: str

    :param skip_nans: avoid writing a value to the .atm file if the additional variable is a NaN
    :type skip_nans: bool

    :param min_num_pts: minimum number of points required in a profile to write out a .atm file. If ``skip_nans`` is
     ``True``, only non-NaN values are counted.

    :return: None, writes files to ``out_dir``.
    """
    unit_in_meta = 'unit' in metadata_df.index
    units_in_meta = 'units' in metadata_df.index
    if unit_in_meta and units_in_meta:
        raise ValueError('metadata_df contains both "unit" and "units" rows, please consolidate to one or the other')
    elif units_in_meta:
        unit_key = 'units'
    elif unit_in_meta:
        unit_key = 'unit'
    else:
        raise ValueError('metadata_df contains neither "unit" not "units" rows, one must be present')

    def atm_colname(atm_name, aircraft_varname, default='?'):
        if aircraft_varname is None:
            return '{}_{}'.format(atm_name, default)
        else:
            return '{}_{}'.format(atm_name, metadata_df.loc[unit_key, aircraft_varname])

    prof_inds = data_df[profnum_var]
    unique_prof_inds = prof_inds.unique()

    # Construct the mapping between data frame column names and .atm column names. The latter are typically formatted
    # {name}_{unit}
    var_mapping = OrderedDict()
    var_mapping['altitude'] = (atm_colname('Altitude', altitude_var), altitude_var)
    var_mapping['pressure'] = (atm_colname('Pressure', pressure_var), pressure_var)
    var_mapping['temperature'] = (atm_colname('Temperature', temperature_var, 'C'), temperature_var)
    var_mapping['h2o'] = (atm_colname('H2O_profile', h2o_var, 'ppm'), h2o_var)
    var_mapping['additional'] = (atm_colname(additional_var_atm_name + '_profile', additional_var), additional_var)

    for uind in unique_prof_inds:
        this_header_info = deepcopy(common_header_info)
        if uind <= 0:
            # Assume any profiles have a number > 0 and that <= 0 indicates a non-profile segment
            continue

        logger.info('Generating .atm file for profile {}'.format(uind))

        pp = prof_inds == uind
        all_data = OrderedDict()
        utc_dates = data_df.index[pp]
        for atm_name, old_name in var_mapping.values():
            if old_name is not None:
                all_data[atm_name] = data_df.loc[pp, old_name]
            else:
                all_data[atm_name] = pd.Series(np.full([pp.sum()], np.nan), index=utc_dates)

        if skip_nans:
            n_points = np.sum(~np.isnan(all_data[var_mapping['additional'][0]]))
        else:
            n_points = np.size(all_data[var_mapping['additional'][0]])

        if n_points < min_num_pts:
            logger.important('Profile {} has < {} valid points, skipping'.format(uind, min_num_pts))
            continue

        # Match to tccon site
        prof_datetime = utc_dates.min()
        prof_date = pd.Timestamp(prof_datetime.year, prof_datetime.month, prof_datetime.day)
        lat = np.nanmean(data_df.loc[pp, lat_var].to_numpy())
        lon = data_df.loc[pp, lon_var].to_numpy()
        lon[lon > 180] -= 360

        dateline_lon_signs = np.sign(lon[np.abs(lon) > 175])
        if dateline_lon_signs.size > 0 and not np.all(dateline_lon_signs == dateline_lon_signs[0]):
            logger.debug('Longitude straddles date line')
            lon[lon < 0] += 360
            lon = np.nanmean(lon)
            if lon > 180:
                lon -= 360
        else:
            lon = np.nanmean(lon)

        tccon_site = _match_to_tccon_site(lon, lat, prof_date)
        this_header_info.update(tccon_site)

        this_atm_file = atm_name_fmt.format(prefix=out_prefix,
                                            lon=mod_utils.format_lon(lon, prec=1),
                                            lat=mod_utils.format_lat(lat, prec=1),
                                            date=prof_date.strftime(_atm_filedate_fmt),
                                            profnum=uind,
                                            var=additional_var_atm_name)
        this_atm_file = os.path.join(out_dir, this_atm_file)
        _write_single_atm_file(this_atm_file, data=all_data, utc_dates=utc_dates, var_mapping=var_mapping,
                               header_info=this_header_info, skip_nans=skip_nans)


def _match_to_tccon_site(lon, lat, date, tolerance=0.5):
    tccon_sites = tccon_site_info_for_date(date)
    lauders = ('Lauder 01', 'Lauder 02')

    matched_site = None
    for site, info in tccon_sites.items():
        distance = (lon - info['lon_180'])**2 + (lat - info['lat'])**2
        if distance < tolerance:
            if matched_site is None:
                matched_site = {'tccon_name': info['name'], 'tccon_lon': info['lon_180'], 'tccon_lat': info['lat']}
            elif matched_site['tccon_name'] in lauders and info['name'] in lauders:
                # Lauder is a special case that there's two instruments. Doesn't matter which one we match, so simplify
                # the name (from "Lauder 01/02" to "Lauder") and keep it's lat/lon.
                matched_site = {'tccon_name': 'Lauder', 'tccon_lon': info['lon_180'], 'tccon_lat': info['lat']}
            else:
                raise RuntimeError('Matched two TCCON sites: {} and {}'.format(matched_site['tccon_name'], info['name']))

    if matched_site is None:
        matched_site = {'tccon_name': 'NA', 'tccon_lon': lon, 'tccon_lat': lat}

    return matched_site


def _write_single_atm_file(filename, data, utc_dates, var_mapping, header_info, skip_nans=True):
    def opt_header(key):
        if key not in header_info:
            return 'NA'
        else:
            return header_info[key]

    with open(filename, 'w') as fobj:
        # Get the times for the bottom of the profile, and the start/stop time of the profile
        alt_key = var_mapping['altitude'][0]
        alt_unit = alt_key.split('_')[-1]
        additional_var = var_mapping['additional'][0]
        specie, unit = re.search(r'(\w+)_profile_(\w+)', additional_var).groups()
        specie = specie.lower()
        unit = unit.lower()

        floor_time = data[alt_key].idxmin()
        floor_date = pd.Timestamp(floor_time.year, floor_time.month, floor_time.day)

        sigma_descr = 'aircraft_{}_error_2sigma_{}'.format(specie, unit)
        max_width = len(sigma_descr)

        def format_header_line(descr, value):
            descr += ': '
            fmt = '{{descr:<{}}}{{value}}\n'.format(max_width + 2)
            return fmt.format(descr=descr, value=value)

        fobj.write('{} on {}\n'.format(header_info['description'], utc_dates.min().strftime(_atm_date_fmt)))
        fobj.write(format_header_line('aircraft_info', header_info['platform']))

        fobj.write(format_header_line('flight_date', floor_date.strftime(_atm_date_fmt)))
        fobj.write(format_header_line('aircraft_floor_time_UTC', floor_time.strftime(_atm_datetime_fmt)))
        fobj.write(format_header_line('aircraft_start_time_UTC', utc_dates.min().strftime(_atm_datetime_fmt)))
        fobj.write(format_header_line('aircraft_stop_time_UTC', utc_dates.max().strftime(_atm_datetime_fmt)))

        fobj.write(format_header_line('altitude_source', var_mapping['altitude'][1]))
        fobj.write(format_header_line('pressure_source', var_mapping['pressure'][1]))
        fobj.write(format_header_line('temperature_source', var_mapping['temperature'][1]))
        fobj.write(format_header_line('h2o_profile_source', var_mapping['h2o'][1]))
        fobj.write(format_header_line('{}_source'.format(specie), var_mapping['additional'][1]))

        fobj.write(format_header_line(sigma_descr, opt_header('2sigma')))
        fobj.write(format_header_line('aircraft_ceiling_{}'.format(alt_unit), '{:.2f}'.format(data[alt_key].max())))
        fobj.write(format_header_line('aircraft_floor_{}'.format(alt_unit), '{:.2f}'.format(data[alt_key].min())))
        fobj.write(format_header_line('{}_surface_source'.format(specie), opt_header('surface_source')))
        fobj.write(format_header_line('h2o_surface_source', opt_header('h2o_surface_source')))

        fobj.write(format_header_line('TCCON_site_name', header_info['tccon_name']))
        fobj.write(format_header_line('TCCON_site_longitude_E', header_info['tccon_lon']))
        fobj.write(format_header_line('TCCON_site_latitude_N', header_info['tccon_lat']))

        fobj.write('-'*51 + '\n')

        # Done with header - onto the actual data
        iterator = range(utc_dates.size)
        if data[alt_key][0] > data[alt_key][-1]:
            iterator = reversed(iterator)

        fobj.write(','.join(k for k in data.keys()) + '\n')
        for i in iterator:
            if skip_nans and np.isnan(data[additional_var][i]):
                continue
            data_row = ['{:.3F}'.format(v[i]) for v in data.values()]
            fobj.write(','.join(data_row) + '\n')


def write_date_lat_lon_list(atm_dir, list_file):
    atm_files = sorted(glob(os.path.join(atm_dir, '*')))

    with open(list_file, 'w') as wobj:
        wobj.write('DATES,LAT,LON\n')
        for f in atm_files:
            _, header_info = butils.read_atm_file(f)
            date = header_info['flight_date']
            lon = header_info['TCCON_site_longitude_E']
            lat = header_info['TCCON_site_latitude_N']

            line = '{date},{lat:.3f},{lon:.3f}\n'.format(date=date.strftime('%Y-%m-%d'), lat=lat, lon=lon)
            wobj.write(line)