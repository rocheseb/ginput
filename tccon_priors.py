from __future__ import print_function, division

import datetime as dt
from dateutil.relativedelta import relativedelta
import netCDF4 as ncdf
import numpy as np
import os
import pandas as pd
#import pyproj
from scipy.interpolate import interp2d, LinearNDInterpolator
#from shapely.geometry import shape

# TODO: move all into package and use a proper relative import
import mod_utils
import mod_constants as const

earth_radius = 6371e3  # meters


data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
clams_file = os.path.join(data_dir, 'clams_age_clim_scaled.nc')


class CO2RecordError(Exception):
    pass


def _is_pole(lat):
    return np.abs(np.abs(lat) - 90.0) < 1e-4


def _init_prof(profs, n_lev, n_profs=0):
    """
    Initialize arrays for various profiles.

    :param profs: input profiles or None.
    :param n_lev: the number of levels in the profiles.
    :type n_lev: int

    :param n_profs: how many profiles to make in the returned array. If 0, then an n_lev element vector is returned, if
     >0, and n_lev-by-n_profs array
    :type n_profs: int

    :return: the initialized array, either the same profiles as given, or a new array initialized with NaNs with the
     required shape if ``profs`` is ``None``.
    :rtype: :class:`numpy.ndarray`
    :raises ValueError: if the profiles were given (not ``None``) and have the wrong shape.
    """
    if n_profs < 0:
        raise ValueError('n_profs must be >= 0')
    if profs is None:
        size = (n_lev, n_profs) if n_profs > 0 else (n_lev,)
        return np.full(size, np.nan)
    else:
        target_shape = (n_lev,) if n_profs == 0 else (n_lev, n_profs)
        if profs.shape != target_shape:
            raise ValueError('Given profile do not have the expected shape! Expected: {}, actual: {}'.format(
                target_shape, profs.shape
            ))
        return profs


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
                        co2_lag=relativedelta(months=2), age_window_spread=0.3, profs_latency=None,
                        prof_aoa=None, prof_world_flag=None, prof_co2_date=None, prof_co2_date_width=None):
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

    :param co2_record: the Mauna Loa-Samoa CO2 record.
    :type co2_record: :class:`CO2TropicsRecord`

    :param co2_lag: the lag between the MLO/SAM record and the CO2 concentration at the tropopause.
    :type co2_lag: :class:`~dateutil.relativedelta.relativedelta` or :class:`datetime.timedelta`

    :param age_window_spread: a decimal value setting how wide a window the simplified "age spectrum" would cover. For
     s = ``age_window_spread`` and a = the age of air from CLAMS, then the CO2 age window will be :math:`a*(1-s)` to
     :math:`a*(1+s)`
    :type age_window_spread: float

    The following parameters are all optional; they are vectors that will be filled with the appropriate values in the
    stratosphere. The are also returned in the ancillary dictionary; if not given as inputs, they are initialized with
    NaNs. "nlev" below means the number of levels in the CO2 profile.

    :param profs_latency: nlev-by-3 array that will store how far forward in time the Mauna Loa/Samoa CO2 record had to
     be extrapolated, in years. The three columns will respectively contain the mean, min, and max latency.
    :param prof_aoa: nlev-element vector of ages of air, in years.
    :param prof_world_flag: nlev-element vector of ints which will indicate which levels are considered overworld and
     which middleworld. The values used for each are defined in :mod:`mod_constants`
    :param prof_co2_date: nlev-element vector that stores the date in the MLO/SMO record that the CO2 was taken from.
     Since most levels will have a window of dates, this is the middle of those windows. The dates are stored as a
     decimal year, e.g. 2016.5.
    :param prof_co2_date_width: nlev-element vector that stores the width (in years) of the age windows used to compute
     the CO2 concentrations.

    :return: the updated CO2 profile and a dictionary of the ancillary profiles.
    """
    latency_keys = ('mean', 'min', 'max')
    n_lev = np.size(prof_co2)
    profs_latency = _init_prof(profs_latency, n_lev, 3)
    prof_aoa = _init_prof(prof_aoa, n_lev)
    prof_world_flag = _init_prof(prof_world_flag, n_lev)
    prof_co2_date = _init_prof(prof_co2_date, n_lev)
    prof_co2_date_width = _init_prof(prof_co2_date_width, n_lev)

    # Next we find the age of air in the stratosphere for points with theta > 380 K. We'll get all levels now and
    # restrict to >= 380 K later.
    xx_overworld = prof_theta >= 380
    prof_world_flag[xx_overworld] = const.overworld_flag
    retrieval_doy = int(np.round(mod_utils.date_to_frac_year(retrieval_date) * 365.25))
    age_of_air_years = get_clams_age(prof_theta, prof_eqlat, retrieval_doy, as_timedelta=False)
    prof_aoa[xx_overworld] = age_of_air_years[xx_overworld]
    age_of_air = np.array(mod_utils.frac_years_to_reldelta(age_of_air_years))

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
        avg_mid_date = retrieval_date - co2_lag - age_of_air[i]
        avg_start_date = avg_mid_date - age_window_spread * age_of_air[i]
        avg_end_date = avg_mid_date + age_window_spread * age_of_air[i]

        if avg_end_date > retrieval_date:
            raise RuntimeError('CO2 averaging window has an end date after the retrieval data. This physically should '
                               'not happen, since that would imply part of the stratosphere came from the future.')

        prof_co2[i], latency_i = co2_record.avg_co2_in_date_range(avg_start_date, avg_end_date, deseasonalize=False)
        for j, k in enumerate(latency_keys):
            profs_latency[i, j] = latency_i[k]
        prof_co2_date[i] = mod_utils.date_to_decimal_year(avg_mid_date)
        prof_co2_date_width[i] = 2 * age_window_spread * age_of_air_years[i]

    # Last we need to fill in the "middleworld" between the tropopause and 380 K. The simplest way to do it is to
    # assume that at the tropopause the CO2 is equal to the lagged MLO/SAM record and interpolate linearly in theta
    # space between that and the first > 380 level.
    ow1 = np.argwhere(xx_overworld)[0]
    co2_entry_conc = co2_record.get_co2_for_dates(retrieval_date - co2_lag)

    co2_endpoints = np.array([co2_entry_conc.item(), prof_co2[ow1].item()])
    theta_endpoints = np.array([tropopause_theta, prof_theta[ow1].item()])
    xx_middleworld = (tropopause_theta < prof_theta) & (prof_theta < 380.0)
    prof_co2[xx_middleworld] = np.interp(prof_theta[xx_middleworld], theta_endpoints, co2_endpoints)
    prof_world_flag[xx_middleworld] = const.middleworld_flag

    # TODO: add latency profile

    return prof_co2, {'latency': profs_latency, 'age_of_air': prof_aoa, 'stratum': prof_world_flag,
                      'co2_date': prof_co2_date, 'co2_date_width': prof_co2_date_width}


def get_clams_age(theta, eq_lat, day_of_year, as_timedelta=False, clams_dat=dict()):
    """
    Get the age of air predicted by the CLAMS model for points defined by potential temperature and equivalent latitude.

    :param theta: a vector of potential temperatures, must be the same length as ``eq_lat``
    :type theta: :class:`numpy.ndarray`

    :param eq_lat: a vector of equivalent latitudes, must be the same length as ``theta``
    :type eq_lat: :class:`numpy.ndarray`

    :param day_of_year: which day of the year (e.g. Feb 1 = 32) to look up the age for
    :type day_of_year: int

    :param as_timedelta: set this to ``True`` to return the ages as :class:`relativedelta` instances. When ``False``
     (default) just returned in fractional years.
    :type as_timedelta: bool

    :param clams_dat: a dictionary containing the CLAMS data with keys 'eqlat' (l-element vector), 'theta' (m-element
     vector), 'doy' (n-element vector), and 'age' (l-by-m-by-n array). This can be passed manually if you want to use a
     custom map of age of air vs. equivalent latitude and theta, but by default will be read in from the CLAMS file
     provided by Arlyn Andrews and cached.
    :type clams_dat: dict

    :return: a vector of ages the same length as ``theta`` and ``eq_lat``. The contents of the vector depend on the
     value of ``as_timedelta``.
    :rtype: :class:`numpy.ndarray`
    """
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

    el_grid, th_grid = np.meshgrid(clams_dat['eqlat'], clams_dat['theta'])
    clams_points = np.array([[el, th] for el, th in zip(el_grid.flat, th_grid.flat)])

    # interp2d does not behave well here; it interpolates to points outside the range of eqlat/theta and gives a much
    # noiser result.
    age_interp = LinearNDInterpolator(clams_points, clams_dat['age'][idoy, :, :].flatten())
    prof_ages = np.array([age_interp(el, th).item() for el, th in zip(eq_lat, theta)])

    # For simplicity, we're just going to clamp ages for theta > max theta in CLAMS to the age given at the top of the
    # profile. In theory, this shouldn't matter too much since (a) CLAMS seems to approach an asymptote at the top of
    # the stratosphere and (b) there's just not that much mass up there.
    last_age = np.max(np.argwhere(~np.isnan(prof_ages)))
    if last_age < prof_ages.size:
        prof_ages[last_age+1:] = prof_ages[last_age]

    if as_timedelta:
        # The CLAMS ages are in years, but relativedeltas don't accept fractional years. Instead, separate the whole
        # years and the fractional years.
        prof_ages = np.array(mod_utils.frac_years_to_reldelta(prof_ages))

    return prof_ages


class CO2TropicsRecord(object):
    """
    This class stored the Mauna Loa/Samoa average CO2 record and provides methods to sample it.

    No arguments required for initialization.
    """
    months_avg_for_trend = 12

    def __init__(self):
        self.co2_seasonal = self.get_mlo_smo_mean()
        # Deseasonalize the data by taking a 12 month rolling average. May replace with a more sophisticated approach
        # in the future.
        self.co2_trend = self.co2_seasonal.rolling(self.months_avg_for_trend, center=True).mean().dropna()

    @classmethod
    def read_insitu_co2(cls, fpath, fname):
        """
        Read a CO2 record file. Assumes that the file is of monthly average CO2.

        :param fpath: the path to the directory containing the file.
        :type fpath: str

        :param fname: the name of the file
        :type fname: str

        :return: a data frame containing the monthly CO2 data along with the site, year, month, and day. The index will
         be a timestamp of the measurment time.
        :rtype: :class:`pandas.DataFrame`
        """
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
        """
        Generate the Mauna Loa/Samoa mean CO2 record from the files stored in this repository.

        Reads in the :file:`data/ML_monthly_obs.txt` and :file:`data/SMO_monthly_obs.txt` files included in this
        repository, averages them, and fills in any missing months by interpolation.

        :return: the data frame containing the mean CO2 ('co2_mean') and a flag ('interp_flag') set to 1 for any months
         that had to be interpolated. Index by timestamp.
        :rtype: :class:`pandas.DataFrame`
        """
        df_mlo = cls.read_insitu_co2(data_dir, 'ML_monthly_obs.txt')
        df_smo = cls.read_insitu_co2(data_dir, 'SMO_monthly_obs.txt')
        df_combined = pd.concat([df_mlo, df_smo], axis=1).dropna()
        df_combined['co2_mean'] = df_combined['co2'].mean(axis=1)
        df_combined.drop(['site', 'co2', 'year', 'month', 'day'], axis=1, inplace=True)

        # Fill in any missing months. Add a flag so we can keep track of whether they've had to be interpolated or
        # not. Having a consistent monthly frequency makes the rest of the code easier - we can just always assume that
        # there will be a value at the beginning of every month.
        n_months = df_combined.index.size
        df_combined = df_combined.assign(interp_flag=np.zeros((n_months,), dtype=np.int))
        all_months = pd.date_range(min(df_combined.index), max(df_combined.index), freq='MS')
        df_combined = df_combined.reindex(all_months)
        # set the interpolation flag
        missing = pd.isna(df_combined['interp_flag'])
        df_combined.loc[missing, 'interp_flag'] = 1
        # Now only co2_mean should have missing values
        df_combined.interpolate(method='index', inplace=True)

        return df_combined

    def get_co2_by_month(self, year, month, deseasonalize=False, limit_extrapolation_to=None):
        """
        Get CO2 for a specific month, extrapolating if necessary.

        :param year: what year to query
        :type year: int

        :param month: what month to query
        :type year: int

        :param deseasonalize: set to ``True`` to use the CO2 record with the seasonal record smoothed out. Default is
         ``False``, which keeps the seasonal record.
        :type deseasonalize: bool

        :param limit_extrapolation_to: a date beyond which not to extrapolate
        :type limit_extrapolation_to: a datetime-like object

        :return: the CO2 value for this month, and a dictionary with keys "flag" and "latency". "latency" will be the
         number of years that the CO2 value had to be extrapolated. Flag will be one of:

         * 0 = data read directly from record
         * 1 = data had to be extrapolated
         * 2 = requested date was before the start of the record
         * 3 = requested date was after the date set by ``limit_extrapolation_to``
         * 4 = unanticipated error occured.

        :rtype: float, dict
        """
        df = self.co2_trend if deseasonalize else self.co2_seasonal

        flag = 0
        years_extrap = 0
        day = 1
        fillval = np.nan
        target_date = pd.Timestamp(year, month, day)
        if limit_extrapolation_to is None:
            # 100 years in the future should be sufficiently far as to be effectively no limit
            limit_extrapolation_to = pd.Timestamp.now() + relativedelta(years=100)
        llimit = min(df.index)

        if llimit < target_date <= limit_extrapolation_to:

            if (target_date > llimit) & (target_date < df.index[-1]):

                flag = 0
                #print('Reading data from file...')
                val = df.loc[target_date]['co2_mean']

            else:
                flag = 1
                nyear = 5
                #print('Date outside available time period... extrapolating!')

                # Need to find the most recent year that we have data for this month
                last_available_date = max(df.index)
                latest_date = target_date
                while latest_date > last_available_date:
                    latest_date -= relativedelta(years=1)
                    years_extrap += 1

                # TODO: ask Matt to revalidate this
                # Get the most recent nyears CO2 concentrations for this month in the record. Set the initial CO2 value
                # to the last of those.
                prev_year = [y for y in range(years_extrap, nyear + years_extrap)]
                prev_date = [pd.Timestamp(year - item, month, day) for item in prev_year]
                prev_co2 = df[df.index.isin(prev_date)]['co2_mean'].values

                val = df.loc[latest_date]['co2_mean']
                for start_yr in range(years_extrap):
                    # For each year we need to extrapolate, calculate the growth rate as the average of the growth
                    # rate over five years. This should help smooth out any El Nino effects, which would tend to
                    # cause unusual growth rates.
                    growth = np.diff(prev_co2).mean()
                    val += growth

                    # Now that we have the extrapolated value, update the last 5 CO2 values to include it and remove
                    # the earliest one so that we have update CO2 values for the next time through the loop.
                    prev_co2 = np.append(prev_co2, val)
                    prev_co2 = np.delete(prev_co2, 0)

        elif target_date < df.index[0]:

            flag = 2
            #print('No data available before {}! Setting fill value...'.format(df.index[0]))
            val = fillval

        elif target_date > limit_extrapolation_to:

            flag = 3
            #print('Data too far in the future. Setting fill value...')
            val = fillval

        else:

            flag = 4
            #print('Error!')
            val = fillval

        return val, {'flag': flag, 'latency': years_extrap}

    def get_co2_for_dates(self, dates, deseasonalize=False, as_dataframe=False):
        """
        Get CO2 for one or more dates.

        This method will lookup CO2 for a specific date, interpolating between the monthly values as necessary.

        :param dates: the date or dates to get CO2 for. If giving a single date, it may be any time that can be
         converted to a Pandas :class:`~pandas.Timestamp`. If giving a series of dates, it must be a
         :class:`pandas.DatetimeIndex`.

        :param deseasonalize: whether to draw CO2 data from the trend only (``True``) or the seasonal cycle (``False``).
        :type deseasonalize: bool

        :param as_dataframe: whether to return the CO2 data as a dataframe (``True``) or numpy array (``False``)
        :type as_dataframe: bool

        :return: the CO2 data for the requested date(s), as a numpy vector or data frame. The data frame will also
         include the latency (how many years the CO2 had to be extrapolated).
        """
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
        monthly_df = pd.DataFrame(index=monthly_idx, columns=['co2_mean', 'latency'], dtype=np.float)
        for timestamp in monthly_df.index:
            monthly_df['co2_mean'][timestamp], info_dict = self.get_co2_by_month(timestamp.year, timestamp.month, deseasonalize=deseasonalize)
            monthly_df['latency'][timestamp] = info_dict['latency']

        # Now we resample to the dates requested, making sure to keep the values at the start of each month on either
        # end of the record to ensure interpolation is successful
        sample_date_idx = dates.copy()
        sample_date_idx = sample_date_idx.append(monthly_idx)
        sample_date_idx = sample_date_idx.sort_values()  # is needed for successful interpolation
        sample_date_idx = pd.unique(sample_date_idx)  # deal with the possibility that one of the requested dates was a month start
        df_resampled = monthly_df.reindex(sample_date_idx)

        # Verify we have non-NaN values for all monthly reference points
        if df_resampled['co2_mean'][monthly_idx].isna().any():
            raise RuntimeError('Failed to resample CO2 for date range {} to {}; first and/or last point is NA'
                               .format(start_date_subset, end_date_subset))

        df_resampled.interpolate(method='index', inplace=True)

        # Return with just the originally requested dates
        df_resampled = df_resampled.reindex(dates)
        if as_dataframe:
            return df_resampled
        else:
            return df_resampled['co2_mean'].values

    def avg_co2_in_date_range(self, start_date, end_date, deseasonalize=False):
        """
        Average the MLO/SMO record between the given dates

        :param start_date: the first date in the averaging period
        :type start_date: datetime-like object

        :param end_date: the last date in the averaging period
        :type end_date: datetime-like object

        :param deseasonalize: whether to draw CO2 data from the trend only (``True``) or the seasonal cycle (``False``).
        :type deseasonalize: bool

        :return: the average CO2 and a dictionary specifying the mean, minimum, and maximum latency (number of years the
         CO2 had to be extrapolated)
        :rtype: float, dict
        """
        if not isinstance(start_date, dt.date) or not isinstance(end_date, dt.date):
            raise TypeError('start_date and end_date must be datetime.date objects (cannot be datetime.datetime objects)')

        # In theory, different resolutions could be given but would need to be careful that the reindexing produced
        # values at the right times.
        resolution = dt.timedelta(days=1)

        avg_idx = pd.date_range(start=start_date, end=end_date, freq=resolution)
        df_resampled = self.get_co2_for_dates(avg_idx, deseasonalize=deseasonalize, as_dataframe=True)

        mean_co2 = df_resampled['co2_mean'][avg_idx].mean()
        latency = dict()
        latency['mean'] = df_resampled['latency'][avg_idx].mean()
        latency['min'] = df_resampled['latency'][avg_idx].min()
        latency['max'] = df_resampled['latency'][avg_idx].max()
        return mean_co2, latency

    def get_co2_by_age(self, ref_date, age, deseasonalize=False, as_dataframe=False):
        """
        Get CO2 for one or more times by specifying a reference date and age.

        This called :meth:`get_co2_for_dates` internally, so the CO2 is interpolated to the specific day just as that
        method does.

        :param ref_date: the date that the ages are relative to.
        :type ref_date: datetime-like object.

        :param age: the number of years before the reference date to get CO2 from. May be a non-whole number.
        :type age: float or sequence of floats

        :param deseasonalize: whether to draw CO2 data from the trend only (``True``) or the seasonal cycle (``False``).
        :type deseasonalize: bool

        :param as_dataframe: whether to return the CO2 data as a dataframe (``True``) or numpy array (``False``)
        :type as_dataframe: bool

        :return: the CO2 data for the requested date(s), as a numpy vector or data frame. The data frame will also
         include the latency (how many years the CO2 had to be extrapolated).
        """
        co2_dates = [ref_date - dt.timedelta(days=a*365.25) for a in age]
        return self.get_co2_for_dates(pd.DatetimeIndex(co2_dates), deseasonalize=deseasonalize,
                                      as_dataframe=as_dataframe)


def get_trop_eq_lat(prof_theta, p_levels, obs_lat, obs_date, theta_wt=1.0, lat_wt=1.0, dtheta_cutoff=0.25, theta_v_lat=dict()):
    def read_pres_range(nc_handle):
        range_str = nc_handle.theta_range  # it says theta range, its really the pressures theta is averaged over
        range_values = [float(s) for s in range_str.split('-')]
        if len(range_values) == 1:
            range_values *= 2

        return min(range_values), max(range_values)

    def theta_lat_cost(dtheta, dlat):
        return dtheta*theta_wt + dlat*lat_wt

    def find_closest_theta(theta, lat, obs_theta):
        # Find which index our obs_lat is closest to
        start = np.argmin(np.abs(lat - obs_lat))

        # Find the locations both north and south of the observation lat that have the smallest difference in theta
        theta_diff = np.abs(theta - obs_theta)
        south_min_ind = np.argmin(theta_diff[:start])
        south_dtheta = theta_diff[south_min_ind]
        south_dlat = np.abs(lat[south_min_ind] - obs_lat)
        north_min_ind = np.argmin(theta_diff[start:]) + start
        north_dtheta = theta_diff[north_min_ind]
        north_dlat = np.abs(lat[north_min_ind] - obs_lat)

        # In most cases, one or the other should have a much closer match. However, if both are similarly good, we need
        # a way to break the tie. What we want is to pick the one that is closer geographically. To do that, we'll use
        # basically a simple cost function that adds the difference in theta and latitude together. Eyeballing the plots
        # of theta vs. latitude from the above file, the typical gradient in the NH is between 0.5 and 1 K/deg. To me
        # that says that we can weight theta and latitude equally in the cost function.
        if np.abs(south_dtheta - north_dtheta) > dtheta_cutoff:
            if south_dtheta < north_dtheta:
                return lat[south_min_ind]
            else:
                return lat[north_min_ind]
        else:
            if theta_lat_cost(south_dtheta, south_dlat) < theta_lat_cost(north_dtheta, north_dlat):
                return lat[south_min_ind]
            else:
                return lat[north_min_ind]

    if len(theta_v_lat) == 0:
        theta_v_lat_file = os.path.join(const.data_dir, 'GEOS_FPIT_lat_vs_theta_2018_500-700hPa.nc')
        with ncdf.Dataset(theta_v_lat_file, 'r') as nch:
            theta_v_lat['theta'] = nch.variables['theta_mean'][:].squeeze()
            theta_v_lat['lat'] = nch.variables['latitude_mean'][:].squeeze()
            theta_v_lat['times'] = nch.variables['times'][:]
            theta_v_lat['times_units'] = nch.variables['times'].units
            theta_v_lat['times_calendar'] = nch.variables['times'].calendar

            # Read the pressure range that we're using
            theta_v_lat['pres_range'] = read_pres_range(nch)

            # Append the first time slice (which will be the first two weeks of the year) to the end so that we can
            # intepolate past the last date, assuming that the changes are cyclical. At the same time, let's record the
            # year used in the dates
            new_time = ncdf.num2date(theta_v_lat['times'][0], theta_v_lat['times_units'], theta_v_lat['times_calendar'])
            theta_v_lat['year'] = year = new_time.year
            new_time = ncdf.date2num(new_time.replace(year=year+1), theta_v_lat['times_units'], theta_v_lat['times_calendar'])
            theta_v_lat['times'] = np.concatenate([theta_v_lat['times'], [new_time]], axis=0)
            for k in ('theta', 'lat'):
                theta_v_lat[k] = np.concatenate([theta_v_lat[k], theta_v_lat[k][0:1, :]], axis=0)

    # First we need to get the lat vs. theta curve for this particular date
    ntimes, nbins = theta_v_lat['theta'].shape
    this_datenum = ncdf.date2num(obs_date.replace(year=theta_v_lat['year']), theta_v_lat['times_units'], theta_v_lat['times_calendar'])
    this_theta_clim = np.full((nbins,), np.nan)
    this_lat_clim = np.full((nbins,), np.nan)
    for i in range(nbins):
        this_theta_clim[i] = np.interp(this_datenum, theta_v_lat['times'], theta_v_lat['theta'][:, i])
        this_lat_clim[i] = np.interp(this_datenum, theta_v_lat['times'], theta_v_lat['lat'][:, i])

    # Then we find the theta for this profile
    zz = (p_levels >= theta_v_lat['pres_range'][0]) & (p_levels <= theta_v_lat['pres_range'][1])
    midtrop_theta = np.mean(prof_theta[zz])

    # Last we find the part on the lookup curve that has the same mid-tropospheric theta as our profile. We have to be
    # careful because we will have the same theta in both the NH and SH. The way we'll handle this is to require that we
    # stay in the same hemisphere if we're in the extra tropics (|lat| > 30) and just find the closest latitude with the
    # same theta in the climatology in the tropics
    if obs_lat > 0:
        yy = this_lat_clim >= 0.0
    else:
        yy = this_lat_clim <= 0.0

    this_lat_clim = this_lat_clim[yy]
    this_theta_clim = this_theta_clim[yy]

    return find_closest_theta(this_theta_clim, this_lat_clim, midtrop_theta)


def add_co2_trop_prior(prof_co2, obs_date, obs_lat, z_grid, z_trop, co2_record, theta_grid=None, pres_grid=None,
                       ref_lat=45.0, use_theta_eqlat=True, profs_latency=None, prof_aoa=None, prof_world_flag=None,
                       prof_co2_date=None, prof_co2_date_width=None):
    """
    Add troposphere CO2 to the prior profile.

    :param prof_co2: the profile CO2 mixing ratios, in ppm. Will be modified in-place to add the stratospheric
     component.
    :type prof_co2: :class:`numpy.ndarray` (in ppm)

    :param obs_date: the UTC date of the retrieval.
    :type obs_date: :class:`datetime.datetime`

    :param obs_lat: the latitude of the retrieval (degrees, south is negative)
    :type obs_lat: float

    :param z_grid: the grid of altitudes (in kilometers) that the CO2 profile is on.
    :type z_grid: :class:`numpy.ndarray`

    :param z_trop: the altitude of the tropopause (in kilometers)
    :type z_trop: float

    :param co2_record: the Mauna Loa-Samoa CO2 record.
    :type co2_record: :class:`CO2TropicsRecord`

    The following parameters are all optional; they are vectors that will be filled with the appropriate values in the
    stratosphere. The are also returned in the ancillary dictionary; if not given as inputs, they are initialized with
    NaNs. "nlev" below means the number of levels in the CO2 profile.

    :param profs_latency: nlev-by-3 array that will store how far forward in time the Mauna Loa/Samoa CO2 record had to
     be extrapolated, in years. The three columns will respectively contain the mean, min, and max latency.
    :param prof_aoa: nlev-element vector of ages of air, in years.
    :param prof_world_flag: nlev-element vector of ints which will indicate which levels are considered overworld and
     which middleworld. The values used for each are defined in :mod:`mod_constants`
    :param prof_co2_date: nlev-element vector that stores the date in the MLO/SMO record that the CO2 was taken from.
     Since most levels will have a window of dates, this is the middle of those windows. The dates are stored as a
     decimal year, e.g. 2016.5.
    :param prof_co2_date_width: nlev-element vector that stores the width (in years) of the age windows used to compute
     the CO2 concentrations.

    :return: the updated CO2 profile and a dictionary of the ancillary profiles.
    """
    n_lev = np.size(z_grid)
    prof_co2 = _init_prof(prof_co2, n_lev)
    profs_latency = _init_prof(profs_latency, n_lev, 3)
    prof_aoa = _init_prof(prof_aoa, n_lev)
    prof_world_flag = _init_prof(prof_world_flag, n_lev)
    prof_co2_date = _init_prof(prof_co2_date, n_lev)
    prof_co2_date_width = _init_prof(prof_co2_date_width, n_lev)

    # First get the ages of air for every grid point within the troposphere. The formula that Geoff Toon developed for
    # age of air has some nice properties, namely it has about a 6 month interhemispheric lag time at the surface which
    # decreases as you go higher up in elevation. It was built around reference measurements in the NH though, so to
    # make it age relative to MLO/SMO, we subtract the age at the surface at the equator. This gives us negative age in
    # the NH, which is right, b/c the NH CO2 concentration should precede MLO/SMO.  The reference latitude in this
    # context should specify where CO2 is emitted from, hence the 45 N (middle of NH) default.
    if use_theta_eqlat:
        if theta_grid is None or pres_grid is None:
            raise TypeError('theta_grid and pres_grid must be given if use_theta_eqlat is True')
        obs_lat = get_trop_eq_lat(theta_grid, pres_grid, obs_lat, obs_date)

    xx_trop = z_grid <= z_trop
    obs_air_age = mod_utils.age_of_air(obs_lat, z_grid[xx_trop], z_trop, ref_lat=ref_lat)
    mlo_smo_air_age = mod_utils.age_of_air(0.0, np.array([0.01]), z_trop, ref_lat=ref_lat).item()
    air_age = obs_air_age - mlo_smo_air_age
    prof_aoa[xx_trop] = air_age
    prof_world_flag[xx_trop] = const.trop_flag

    co2_df = co2_record.get_co2_by_age(obs_date, air_age, deseasonalize=True, as_dataframe=True)
    prof_co2[xx_trop] = co2_df['co2_mean'].values
    # Must reshape the 1D latency vector into an n-by-1 matrix to broadcast successfully
    profs_latency[xx_trop, :] = co2_df['latency'].values.reshape(-1, 1)
    # Record the date that the CO2 was taken from as a year with a fraction
    co2_dates = np.array([mod_utils.date_to_decimal_year(v) for v in co2_df.index])
    prof_co2_date[xx_trop] = co2_dates
    # The width for the age is defined as what time window we average over to detrend
    prof_co2_date_width[xx_trop] = co2_record.months_avg_for_trend / 12.0

    # Finally, apply a parameterized seasonal cycle. This is better than using the seasonal cycle in the MLO/SMO data
    # because that is dominated by the NH cycle. This approach allows the seasonal cycle to vary in sign and intensity
    # with latitude.
    year_fraction = mod_utils.date_to_frac_year(obs_date)
    prof_co2[xx_trop] *= mod_utils.seasonal_cycle_factor(obs_lat, z_grid[xx_trop], z_trop, year_fraction, species='co2',
                                                         ref_lat=ref_lat)

    return prof_co2, {'co2_latency': profs_latency, 'co2_date': prof_co2_date, 'co2_date_width': prof_co2_date_width,
                      'age_of_air': prof_aoa, 'stratum': prof_world_flag, 'ref_lat': ref_lat, 'trop_lat': obs_lat}


def generate_tccon_prior(mod_file_data, obs_date, utc_offset, species='co2', site_abbrev='xx', use_geos_grid=True,
                         use_eqlat_strat=True, write_map=False):
    """
    Driver function to generate the TCCON prior profiles for a single observation.

    :param mod_file_data: data from a .mod file prepared by Mod Maker. May either be a path to a .mod file, or a
     dictionary from :func:`~mod_utils.read_mod_file`.
    :type mod_file_data: str or dict

    :param obs_date: the date of the observation
    :type obs_date: datetime-like object

    :param species: which species to generate the prior profile for. Currently only CO2 is implemented (case
     insensitive).
    :type species: str

    :param site_abbrev: the two-letter site abbreviation. Currently only used in naming the output file.
    :type site_abbrev: str

    :param use_geos_grid: when ``True``, the native 42-level GEOS-FP pressure grid is used as the vertical grid for the
     CO2 profiles. Set to ``False`` to use a grid with 1 km altitude spacing
    :type use_geos_grid: bool

    :param write_map: set to ``False`` to disable writing the output pseudo .map file and just return the dictionary of
     profiles. Set to a path where to save the .map file in order to save it, e.g. set ``write_map='.'`` to save to the
     current directory.
    :type write_map: bool or str

    :return: a dictionary containing all the profiles (including many for debugging) and a dictionary containing the
     units of the values in each profile.
    :rtype: dict, dict
    """
    if isinstance(mod_file_data, str):
        mod_file_data = mod_utils.read_mod_file(mod_file_data)
    elif not isinstance(mod_file_data, dict):
        raise TypeError('mod_file_data must be a string (path pointing to a .mod file) or a dictionary')

    if write_map and not isinstance(write_map, str):
        raise TypeError('If write_map is truthy, then it must be a string')

    obs_lat = mod_file_data['constants']['obs_lat']
    # Make the UTC date a datetime object that is rounded to a date (hour/minute/etc = 0)
    obs_utc_date = dt.datetime.combine((obs_date - utc_offset).date(), dt.time())

    z_met = mod_file_data['profile']['Height']
    theta_met = mod_file_data['profile']['PT']
    eq_lat_met = mod_file_data['profile']['EL'] if use_eqlat_strat else np.full_like(z_met, obs_lat)

    # We need the tropopause potential temperature. The GEOS FP-IT files give the temperature itself, and the pressure,
    # so we can calculate the potential temperature. Pressure needs to be in hPa, which it is by default.

    t_trop_met = mod_file_data['scalar']['TROPT']  # use the blended tropopause. TODO: reference why this is best?
    p_trop_met = mod_file_data['scalar']['TROPPB']
    theta_trop_met = mod_utils.calculate_potential_temperature(p_trop_met, t_trop_met)

    # The age-of-air calculation used for the tropospheric CO2 profile calculation needs the tropopause altitude.
    # Assume that potential temperature varies linearly with altitude to calculate that, use the potential temperature
    # of the tropopause to ensure consistency between the two parts of the profile.
    z_trop_met = mod_utils.interp_to_tropopause_height(theta_met, z_met, theta_trop_met)

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

    if use_geos_grid:
        z_prof = z_met
        theta_prof = theta_met
        eq_lat_prof = eq_lat_met
        t_prof = mod_file_data['profile']['Temperature']
        p_prof = mod_file_data['profile']['Pressure']
    else:
        # z_prof = np.arange(0., 71.)  # altitude levels 0 to 70 kilometers
        z_prof = np.arange(0., 65.)
        theta_prof = mod_utils.mod_interpolation_new(z_prof, z_met, theta_met, interp_mode='linear')
        # Not sure what the theoretical relationship between equivalent latitude and altitude is, and plotting it is too
        # bouncy to tell, so just going to assume linear for now.
        eq_lat_prof = mod_utils.mod_interpolation_new(z_prof, z_met, eq_lat_met, interp_mode='linear')
        # For the map file we'll also want regular temperature and pressure on the grid
        t_prof = mod_utils.mod_interpolation_new(z_prof, z_met, mod_file_data['profile']['Temperature'],
                                                 interp_mode='linear')
        p_prof = mod_utils.mod_interpolation_new(z_prof, z_met, mod_file_data['profile']['Pressure'],
                                                 interp_mode='lin-log')

    n_lev = np.size(z_prof)
    co2_prof = np.full_like(z_prof, np.nan)
    co2_date_prof = np.full_like(z_prof, np.nan)
    co2_date_width_prof = np.full_like(z_prof, np.nan)
    latency_profs = np.full((n_lev, 3), np.nan)
    stratum_flag = np.full((n_lev,), -1)

    _, ancillary_trop = add_co2_trop_prior(co2_prof, obs_utc_date, obs_lat, z_prof, z_trop_met,  concentration_record,
                                           pres_grid=p_prof, theta_grid=theta_prof,
                                           profs_latency=latency_profs, prof_world_flag=stratum_flag,
                                           prof_co2_date=co2_date_prof, prof_co2_date_width=co2_date_width_prof)
    aoa_prof_trop = ancillary_trop['age_of_air']
    trop_ref_lat = ancillary_trop['ref_lat']
    trop_eqlat = ancillary_trop['trop_lat']

    # Next we add the stratospheric profile, including interpolation between the tropopause and 380 K potential
    # temperature (the "middleworld").
    _, ancillary_strat = add_co2_strat_prior(co2_prof, obs_utc_date, theta_prof, eq_lat_prof, theta_trop_met,
                                             concentration_record, profs_latency=latency_profs,
                                             prof_world_flag=stratum_flag, prof_co2_date=co2_date_prof,
                                             prof_co2_date_width=co2_date_width_prof)
    aoa_prof_strat = ancillary_strat['age_of_air']

    map_dict = {'Height': z_prof, 'Temp': t_prof, 'Pressure': p_prof, 'PT': theta_prof, 'EL': eq_lat_prof,
                'co2': co2_prof, 'mean_co2_latency': latency_profs[:, 0], 'min_co2_latency': latency_profs[:, 1],
                'max_co2_latency': latency_profs[:, 2], 'trop_age_of_air': aoa_prof_trop,
                'strat_age_of_air': aoa_prof_strat, 'atm_stratum': stratum_flag, 'co2_date': co2_date_prof,
                'co2_date_width': co2_date_width_prof}
    units_dict = {'Height': 'km', 'Temp': 'K', 'Pressure': 'hPa', 'PT': 'K', 'EL': 'degrees', 'co2': 'ppm',
                  'mean_co2_latency': 'yr', 'min_co2_latency': 'yr', 'max_co2_latency': 'yr', 'trop_age_of_air': 'yr',
                  'strat_age_of_air': 'yr', 'atm_stratum': 'flag', 'co2_date': 'yr', 'co2_date_width': 'yr'}
    var_order = ('Height', 'Temp', 'Pressure', 'PT', 'EL', 'co2', 'mean_co2_latency', 'min_co2_latency',
                 'max_co2_latency', 'trop_age_of_air', 'strat_age_of_air', 'atm_stratum', 'co2_date', 'co2_date_width')
    if write_map:
        map_dir = write_map if isinstance(write_map, str) else '.'
        map_name = os.path.join(map_dir, '{}{}_{}.map'.format(site_abbrev, mod_utils.format_lat(obs_lat), obs_date.strftime('%Y%m%d_%H%M')))
        mod_utils.write_map_file(map_name, obs_lat, trop_eqlat, trop_ref_lat, z_trop_met, use_eqlat_strat, map_dict, units_dict, var_order=var_order)

    return map_dict, units_dict


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
