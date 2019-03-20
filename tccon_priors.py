from __future__ import print_function, division

import datetime as dt
from dateutil.relativedelta import relativedelta
import netCDF4 as ncdf
import numpy as np
import os
import pandas as pd
import pyproj
from scipy.interpolate import interp2d
from shapely.geometry import shape

# TODO: move all into package and use a proper relative import
import mod_utils

earth_radius = 6371e3  # meters


data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
clams_file = os.path.join(data_dir, 'clams_age_clim_scaled.nc')


class CO2RecordError(Exception):
    pass


def _is_pole(lat):
    return np.abs(np.abs(lat) - 90.0) < 1e-4


def equirect_grid_area(latitudes, longitudes):
    if np.ndim(longitudes) != 1 or np.ndim(latitudes) != 1:
        raise ValueError('longitudes and latitudes are expected to be vectors')

    n_lon = longitudes.size
    n_lat = latitudes.size

    delta_lon = np.unique(np.abs(np.diff(longitudes)))
    delta_lat = np.unique(np.abs(np.diff(latitudes)))
    if delta_lon.size != 1 or delta_lat.size != 1:
        raise NotImplementedError('longitudes and/or latitudes do not have consistent spacing. This has not been '
                                  'implemented yet.')
    else:
        # convert from 1-element arrays to true scalar values and to radians
        delta_lon = np.deg2rad(delta_lon.item())/2.0
        delta_lat = np.deg2rad(delta_lat.item())/2.0

    # not right!
    # For a sufficiently small area on the surface of a sphere the solid angle subtended by a given dlat and dlon is:
    #
    #   d\Omega = sin(\theta) d\theta d\phi
    #
    # where d\Omega is the solid angle, \theta = latitude and \phi = longitude
    # Since the definition of solid angle is:
    #
    #   \Omega = A / r^2
    #
    # where A is the surface area subtended and r is the sphere's radius, then for grid boxes with lon/lat spacing
    # d\phi and d\theta, the area of the grid box is:
    #
    # dA = r^2 * sin(\theta) * d\theta d\phi
    areas_one_lon = 4.0 * earth_radius**2.0 * delta_lon * np.sin(delta_lat) * np.cos(np.deg2rad(latitudes))
    xx_poles = _is_pole(latitudes)
    if np.sum(xx_poles) > 0:
        areas_one_lon[xx_poles] = earth_radius**2.0 * delta_lon * delta_lat**2.0

    areas = np.tile(areas_one_lon.reshape(-1, 1), (1, n_lon))
    return areas


def calculate_equiv_lat_from_pot_vort(pv, lon, lat):

    if np.ndim(lon) != 1 or np.ndim(lat) != 1:
        raise ValueError('lon and lat are expected to be vectors')

    nlon = np.size(lon)
    nlat = np.size(lat)
    if np.ndim(pv) < 3 or np.ndim(pv) > 4:
        raise ValueError('pv expected to be 3 or 4 dimensions')
    elif np.shape(pv)[-1] != nlon or np.shape(pv)[-2] != nlat:
        raise ValueError('pv is expected to be ntimes x nlev x nlat x nlon. One or both of the last two dimensions of '
                         'the given pv are not the same length as lat or lon.')
    elif np.ndim(pv) == 3:
        # assume pv was given without a time dimension, add one
        pv = pv[np.newaxis, :, :, :]

    box_area = equirect_grid_area(lat, lon).flatten()

    eq_lat = np.full_like(pv, np.nan, dtype=np.float)
    for i_time in range(pv.shape[0]):
        for i_lev in range(pv.shape[1]):
            pv_slice = np.squeeze(pv[i_time, i_lev, :, :]).flatten()
            pv_sort = np.argsort(pv_slice)

            cum_pv_area = np.cumsum(box_area[pv_sort])
            rel_area = cum_pv_area/(0.5 * cum_pv_area[-1] - 1)
            eq_lat_slice = np.full((nlon*nlat,), np.nan, dtype=np.float)
            eq_lat_slice[pv_sort] = np.rad2deg(np.arcsin(rel_area))
            eq_lat[i_time, i_lev, :, :] = np.reshape(eq_lat_slice, (nlat, nlon))

    return eq_lat


def add_co2_strat_prior(prof_co2, retrieval_date, prof_theta, prof_eqlat, tropopause_theta, co2_record,
                        co2_lag=relativedelta(months=2), age_window_spread=0.3):
    """
    Add the stratospheric CO2 to a TCCON prior profile

    :param prof_co2: the profile CO2 mixing ratios, in ppm. Will be modified in-place to add the stratospheric
     component.
    :type prof_co2: :class:`numpy.ndarray` (in ppm)

    :param retrieval_date: the UTC date of the retrieval.
    :type retrieval_date: :class:`datetime.datetime`

    :param prof_theta: the theta (potential temperature) coordinates for the TCCON profile.
    :type prof_theta: :class:`numpy.ndarray` (in K)

    :param prof_eqlat: the equivalent latitude coordinates for the TCCON profile.
    :type prof_eqlat: :class:`numpy.ndarray` (in degrees)

    :param tropopause_theta: the potential temperature at the tropopause, according to the input meteorology.
    :type tropopause_theta: float (in K)

    :param co2_record: the Mauna Loa-Samoa CO2 record. Must be a dict with keys 'date', 'co2', 'latency'. 'date' must be
     a sequence of datetimes giving the UTC date of the corresponding CO2 values, 'co2' a sequence of floats giving
     the time series of CO2 measurements, and 'latency' a sequence of ints giving the number of years that that CO2
     point had to be extrapolated (e.g. if the point on 1 June 2019 was calculated by extrapolating from 1 June 2017,
     then this should be 2). If looked up directly, 'latency' for that point will be 0.
    :type co2_record: dict('date': list(datetime.datetime), 'co2': list(float), 'latency': list(int))

    :param co2_lag: the lag between the MLO/SAM record and the CO2 concentration at the tropopause.
    :type co2_lag: :class:`~dateutil.relativedelta.relativedelta` or :class:`datetime.timedelta`

    :return: the updated CO2 profile and the latency profile
    """
    # Next we find the age of air in the stratosphere for points with theta > 380 K. We'll get all levels now and
    # restrict to >= 380 K later.
    xx_overworld = prof_theta >= 380
    retrieval_doy = int(np.round(mod_utils.date_to_frac_year(retrieval_date) * 365.25))
    age_of_air = get_mean_age(prof_theta, prof_eqlat, retrieval_doy, as_timedelta=True)

    # Now, assuming that the CLAMS age is the mean age of the stratospheric air and that we can assume the CO2 has
    # not changed since it entered the stratosphere, we look up the CO2 at the boundary condition. Assume that the
    # CO2 record has daily points, so we just want the date (not datetime). Lastly, we add the lag to the record dates
    # so that, e.g. if we want a lag of 2 months and we're querying for June 1, then the Mar 1 record will have a lagged
    # date of June 1
    #
    # We do a poor man's age spectrum by averaging between, by default +/- 30% of the age of the air. This creates and
    # averaging window that gets broader as the air gets older, which should average out the seasonal cycle by ~2 years
    # Andrews et al. 2001 (JGR Atmos, p. 32,295, see pg. 32,300) found that the seasonal cycle was largely gone for
    # air older than that.

    for i in np.argwhere(xx_overworld).flat:
        age_start = (1 - age_window_spread) * age_of_air[i]
        avg_start_date = retrieval_date + age_start - co2_lag
        age_end = (1 + age_window_spread) * age_of_air[i]
        avg_end_date = retrieval_date + age_end - co2_lag

        prof_co2[i] = co2_record.avg_co2_in_date_range(avg_start_date, avg_end_date, deseasonalize=False)

    # Last we need to fill in the "middleworld" between the tropopause and 380 K. The simplest way to do it is to
    # assume that at the tropopause the CO2 is equal to the lagged MLO/SAM record and interpolate linearly in theta
    # space between that and the first > 380 level.
    ow1 = np.argwhere(xx_overworld)[0]
    co2_entry_conc = co2_record.get_co2_for_dates(retrieval_date - co2_lag)

    co2_endpoints = [co2_entry_conc, prof_co2[ow1]]
    theta_endpoints = [tropopause_theta, prof_theta[ow1]]
    xx_middleworld = tropopause_theta < prof_theta < 380
    prof_co2[xx_middleworld] = np.interp(prof_theta[xx_middleworld], theta_endpoints, co2_endpoints)

    # TODO: add latency profile

    return prof_co2


def get_mean_age(theta, eq_lat, day_of_year, as_timedelta=False, clams_dat=dict()):

    if len(clams_dat) == 0:
        # Take advantage of mutable default arguments to cache the CLAMS data. The first time this function is called,
        # the dict will be empty, so the data will be loaded. The second time, since the dict will have been modified,
        # with all the data, we don't need to load it. This should hopefully speed up this part of the code.
        clams = ncdf.Dataset(clams_file, 'r')
        clams_dat['eqlat'] = clams.variables['lat'][:]
        clams_dat['theta'] = clams.variables['theta'][:]
        clams_dat['doy'] = clams.variables['doy'][:]
        clams_dat['age'] = clams.variables['age'][:]

        clams_dat['eqlat_grid'], clams_dat['theta_grid'] = np.meshgrid(clams_dat['eqlat'], clams_dat['theta'])
        if clams_dat['eqlat_grid'].shape != clams_dat['age'].shape[1:] or clams_dat['theta_grid'].shape != clams_dat['age'].shape[1:]:
            raise RuntimeError('Failed to create equivalent lat/theta grids the same shape as CLAMS age')

    idoy = np.argwhere(clams_dat['doy'] == day_of_year).item()
    age_interp = interp2d(clams_dat['eqlat_grid'], clams_dat['theta_grid'], clams_dat['age'][idoy, :, :])
    # By default, interp2d builds a grid from the query points. We don't want that, we just want the points for
    # (theta[i], eqlat[i]) which lie along the diagonal of the interpolated matrix.
    prof_ages = np.diag(age_interp(eq_lat, theta))
    if as_timedelta:
        # The CLAMS ages are in years, but relativedeltas don't accept fractional years. Instead, separate the whole
        # years and the fractional years. For simplicity, just assume 365 days per year.
        age_years = np.floor(prof_ages)
        age_fracs = np.mod(prof_ages, 1)
        prof_ages = [relativedelta(years=y, days=365*d) for y, d in zip(age_years, age_fracs)]

    return prof_ages


class CO2TropicsRecord(object):
    def __init__(self):
        self.co2_seasonal = self.get_mlo_smo_mean()
        # Deseasonalize the data by taking a 12 month rolling average. May replace with a more sophisticated approach
        # in the future.
        self.co2_trend = self.co2_seasonal.rolling(12, center=True).mean().dropna()

    @classmethod
    def read_insitu_co2(cls, fpath, fname):
        full_file_path = os.path.join(fpath, fname)
        with open(full_file_path, 'r') as f:
            hlines = f.readline().rstrip().split(': ')[1]

        df = pd.read_csv(full_file_path, skiprows=int(hlines), skipinitialspace=True,
                         delimiter=' ', header=None, names=['site', 'year', 'month', 'co2'])

        # set datetime index in df (requires 'day' column)
        df['day'] = 1
        df.set_index(pd.to_datetime(df[['year', 'month', 'day']]), inplace=True)

        return df

    @classmethod
    def get_mlo_smo_mean(cls):
        df_mlo = cls.read_insitu_co2(data_dir, 'ML_monthly_obs.txt')
        df_smo = cls.read_insitu_co2(data_dir, 'SMO_monthly_obs.txt')
        df_combined = pd.concat([df_mlo, df_smo], axis=1).dropna()
        df_combined['co2_mean'] = df_combined['co2'].mean(axis=1)
        df_combined.drop(['site', 'co2', 'year', 'month', 'day'], axis=1, inplace=True)

        return df_combined

    def get_co2_by_month(self, year, month, deseasonalize=False, limit_extrapolation_to=None):
        df = self.co2_trend if deseasonalize else self.co2_seasonal

        flag = 0
        day = 1
        fillval = np.nan
        target_date = pd.Timestamp(year, month, day, 0)
        if limit_extrapolation_to is None:
            # 100 years in the future should be sufficiently far as to be effectively no limit
            limit_extrapolation_to = pd.Timestamp.now() + relativedelta(years=100)
        llimit = min(df.index)

        if llimit < target_date <= limit_extrapolation_to:

            if (target_date > llimit) & (target_date < df.index[-1]):

                flag = 0
                print('Reading data from file...')
                val = df.loc[target_date]['co2_mean']

            else:
                flag = 1
                nyear = 6
                print('Date outside available time period... extrapolating!')

                # Need to find the most recent year that we have data for this month
                last_available_date = max(df.index)
                latest_date = pd.Timestamp(year, month, day) - relativedelta(years=1)
                while latest_date > last_available_date:
                    latest_date -= relativedelta(years=1)

                # Last data entry NOT within one year of desired date
                if target_date - df.index[-1] >= pd.Timedelta(days=2*365):
                    # TODO: check with Matt about this. May need to implement more general extrapolation for an
                    #  arbitrary number of years
                    raise NotImplementedError('Extrapolating CO2 values over more than two years not implemented')
                elif target_date - df.index[-1] >= pd.Timedelta(days=365):

                    # first extrapolate value for one year ago
                    prev_year = [y for y in range(2, nyear + 1)]
                    prev_date = [pd.Timestamp(year - item, month, day) for item in prev_year]
                    prev_co2 = df[df.index.isin(prev_date)]['co2_mean'].values
                    growth = np.diff(prev_co2).mean()
                    latest_date = pd.Timestamp(year, month, day) - relativedelta(years=prev_year[0])
                    val_latest = df.loc[latest_date]['co2_mean'] + growth

                    # update growth rate using latest extrapolated value
                    val_update = np.delete(np.append((df[df.index.isin(prev_date)]['co2_mean'].values), val_latest), 0)
                    growth_update = np.diff(val_update).mean()
                    val = val_latest + growth_update

                    # Last data entry within one year of desired date
                else:
                    prev_year = range(1, nyear)
                    prev_date = [pd.Timestamp(year - item, month, day) for item in prev_year]
                    growth = np.diff(df[df.index.isin(prev_date)]['co2_mean'].values).mean()
                    val = df.loc['%s-%s-%s' % (year - prev_year[0], month, day)]['co2_mean'] + growth

        elif target_date < df.index[0]:

            flag = 2
            print('No data available before {}! Setting fill value...'.format(df.index[0]))
            val = fillval

        elif target_date > limit_extrapolation_to:

            flag = 3
            print('Data too far in the future. Setting fill value...')
            val = fillval

        else:

            flag = 4
            print('Error!')
            val = fillval

        return val, flag

    def get_co2_for_dates(self, dates, deseasonalize=False):
        # Make inputs consistent: we expect dates to be a Pandas DatetimeIndex, but it may be a single timestamp or
        # datetime. For now, we will not allow collections of datetimes, such inputs must be converted to DatetimeIndex
        # instances before being passed in.
        if not isinstance(dates, pd.DatetimeIndex):
            try:
                timestamp_in = pd.Timestamp(dates)
            except (ValueError, TypeError):
                raise ValueError('dates must be a Pandas DatetimeIndex or an object convertible to a Pandas Timestamp. '
                                 'Objects of type {} are not supported'.format(type(dates).__name__))
            else:
                dates = pd.DatetimeIndex([timestamp_in])

        start_date = min(dates)
        end_date = max(dates)

        # Need to make sure we get data that bracket the start and end date, so set them the first days of month
        start_date_subset = mod_utils.start_of_month(start_date, out_type=pd.Timestamp)
        end_date_subset = mod_utils.start_of_month(end_date + relativedelta(months=1), out_type=pd.Timestamp)

        # First get just monthly data. freq='MS' gives us monthly data at the start of the month. Use the existing logic
        # to extrapolate the record for a given month if needed.
        monthly_idx = pd.date_range(start_date_subset, end_date_subset, freq='MS')
        monthly_df = pd.DataFrame(index=monthly_idx, columns=['co2_mean'], dtype=np.float)
        for timestamp in monthly_df.index:
            monthly_df['co2_mean'][timestamp], _ = self.get_co2_by_month(timestamp.year, timestamp.month, deseasonalize=deseasonalize)

        # Now we resample to the dates requested, making sure to keep the values at the start of each month on either
        # end of the record to ensure interpolation is successful
        sample_date_idx = dates.copy()
        sample_date_idx = sample_date_idx.append(monthly_idx)
        sample_date_idx = sample_date_idx.sort_values()  # is needed for successful interpolation
        df_resampled = monthly_df.reindex(sample_date_idx)

        # Verify we have non-NaN values for all monthly reference points
        if df_resampled['co2_mean'][monthly_idx].isna().any():
            raise RuntimeError('Failed to resample CO2 for date range {} to {}; first and/or last point is NA'
                               .format(start_date_subset, end_date_subset))

        df_resampled.interpolate(method='index', inplace=True)

        # Return the dataframe with just the originally requested dates
        return df_resampled.reindex(dates)

    def avg_co2_in_date_range(self, start_date, end_date, deseasonalize=False):
        if not isinstance(start_date, dt.date) or not isinstance(end_date, dt.date):
            raise TypeError('start_date and end_date must be datetime.date objects (cannot be datetime.datetime objects)')

        # In theory, different resolutions could be given but would need to be careful that the reindexing produced
        # values at the right times.
        resolution = dt.timedelta(days=1)

        avg_idx = pd.date_range(start=start_date, end=end_date, freq=resolution)
        df_resampled = self.get_co2_for_dates(avg_idx, deseasonalize=deseasonalize)
        return df_resampled['co2_mean'][avg_idx].mean()

    def get_co2_by_age(self, ref_date, age, deseasonalize=False):
        co2_dates = [ref_date - dt.timedelta(days=a*365.25) for a in age]
        return self.get_co2_for_dates(pd.DatetimeIndex(co2_dates), deseasonalize=deseasonalize)


def add_co2_trop_prior(prof_co2, obs_date, obs_lat, z_grid, z_trop, co2_record):
    # First get the ages of air for every grid point within the troposphere.
    xx_trop = z_grid <= z_trop
    air_age = mod_utils.age_of_air(obs_lat, z_grid[xx_trop], z_trop)

    # Now use that to look up the CO2 from the MLO/SMO record. This assumes that the age-of-air accounts for the fact
    # that the NH will generally precede the tropics and SH in CO2. This is not a great assumption currently, but may
    # be good enough.
    co2_df = co2_record.get_co2_by_age(obs_date, air_age, deseasonalize=True)
    prof_co2[xx_trop] = co2_df['co2_mean'].values

    # Finally, apply a parameterized seasonal cycle. This is better than using the seasonal cycle in the MLO/SMO data
    # because that is dominated by the NH cycle. This approach allows the seasonal cycle to vary in sign and intensity
    # with latitude.
    year_fraction = mod_utils.date_to_frac_year(obs_date)
    prof_co2[xx_trop] *= mod_utils.seasonal_cycle_factor(obs_lat, z_grid[xx_trop], z_trop, year_fraction, species='co2',
                                                         ref_lat=0.0)


def generate_tccon_prior(mod_file_data, obs_date, species='co2'):
    if isinstance(mod_file_data, str):
        mod_file_data = mod_utils.read_mod_file(mod_file_data)
    elif not isinstance(mod_file_data, dict):
        raise TypeError('mod_file_data must be a string (path pointing to a .mod file) or a dictionary')

    obs_lat = mod_file_data['constants']['obs_lat']

    z_met = mod_file_data['profile']['Height']
    theta_met = mod_file_data['profile']['PT']
    eq_lat_met = mod_file_data['profile']['EL']

    # We need the tropopause potential temperature. The GEOS FP-IT files give the temperature itself, and the pressure,
    # so we can calculate the potential temperature. Pressure needs to be in hPa, which it is by default.

    t_trop_met = mod_file_data['scalar']['TROPT']  # use the blended tropopause. TODO: reference why this is best?
    p_trop_met = mod_file_data['scalar']['TROPPB']
    theta_trop_met = mod_utils.calculate_potential_temperature(p_trop_met, t_trop_met)

    import pdb; pdb.set_trace()

    # The age-of-air calculation used for the tropospheric CO2 profile calculation needs the tropopause altitude.
    # Assume that potential temperature varies linearly with altitude to calculate that, use the potential temperature
    # of the tropopause to ensure consistency between the two parts of the profile.
    z_trop_met = mod_utils.mod_interpolation_new(theta_trop_met, theta_met, z_met, interp_mode='linear')

    species = species.lower()
    if species == 'co2':
        concentration_record = CO2TropicsRecord()
    else:
        raise ValueError('species "{}" not recognized'.format(species))

    # First we need to get the altitudes/theta levels that the prior will be defined on. We also need to get the blended
    # tropopause height from the GEOS met file. We will calculate the troposphere CO2 profile from a deseasonalized
    # average of the Mauna Loa and Samoa CO2 concentration using the existing GGG age-of-air parameterization assuming
    # a reference latitude of 0 deg. That will be used to set the base CO2 profile, which will then have a parameterized
    # seasonal cycle added on top of it.
    z_grid = np.arange(0., 71.)  # altitude levels 0 to 70 kilometers
    co2_prof = np.full_like(z_grid, np.nan)
    add_co2_trop_prior(co2_prof, obs_date, obs_lat, z_grid, z_trop_met, concentration_record)

    # Next we add the stratospheric profile, including interpolation between the tropopause and 380 K potential
    # temperature (the "middleworld").
    add_co2_strat_prior(co2_prof, obs_date, theta_met, eq_lat_met, theta_trop_met, concentration_record)


def _calculate_box_area(edge_lon, edge_lat):
    # https://stackoverflow.com/a/4683144
    lon1, lon2 = sorted(edge_lon)
    lat1, lat2 = sorted(edge_lat)

    latmid = np.mean(edge_lat)
    lonmid = np.mean(edge_lon)

    poly_lons = np.array([lon1, lon2, lon2, lon1, lon1])
    poly_lats = np.array([lat1, lat1, lat2, lat2, lat1])

    # May need special handling at poles
    eq_area_proj = pyproj.Proj('+proj=aea +lat_1={lat1} +lat_2={lat2} +lat_0={latmid} +lon_0={lonmid}'
                               .format(lat1=lat1, lat2=lat2, latmid=latmid, lonmid=lonmid))

    x, y = eq_area_proj(poly_lons, poly_lats)
    box_poly = {'type': 'Polygon', 'coordinates': [zip(x,y)]}
    return shape(box_poly).area
