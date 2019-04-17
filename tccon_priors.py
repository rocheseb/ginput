"""
Main module for generating TCCON trace gas priors.

This module is the main driver to construct priors for CO2, CO, CH4, N2O, and HF for the TCCON retrieval. Broadly, each
gas follows a similar scheme:

* In the troposphere, the historical record for the gas is obtained from NOAA flask observations at Mauna Loa and Samoa
  (MLO/SMO). This record is deseasonalized by taking a 12 month running mean of the data. The age-of-air in the
  observation profile is calculated using a parameterization developed empirically from various in situ measurements for
  previous versions of the GGG package. That age is then used to determine what date in the MLO/SMO should be looked up.
  A parameterized seasonal cycle (again, developed for previous versions of GGG from in situ observations) is applied.
  We use the parameterized seasonal cycle, rather than the real seasonal cycle in the MLO/SMO record, because the latter
  will not capture any latitudinal dependence.
* In the stratospheric overworld (potential temperature > 380 K), an age of air from CLAMS (Chemical Lagrangian Model of
  the Stratosphere), looked up based on a potential vorticity-based equivalent latitude and potential temperature, is
  used to determine what date to look up concentration from the MLO/SMO record. In this case, the MLO/SMO record with
  seasonality is used, as we assume that all air enters the stratosphere in the tropics, and will include the tropical
  seasonal cycle. This age is convolved with an age spectrum to account for mixing over time.
* In the middle world (above the tropopause & theta < 380 K), the profiles are interpolate with respect to theta between
  the tropopause and the first overworld level.

The stratospheric approach was developed by Arlyn Andrews, based on her research in Andrews et al. 2001 (JGR, 106 [D23],
pp. 32295-32314).
"""

from __future__ import print_function, division

from abc import abstractmethod, ABCMeta
from contextlib import closing
import ctypes
import datetime as dt
from dateutil.relativedelta import relativedelta
from glob import glob
import multiprocessing as mp
import netCDF4 as ncdf
import numpy as np
import os
import pandas as pd
import re
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize_scalar

# TODO: move all into package and use a proper relative import
import mod_utils
import ioutils
import mod_constants as const


_data_dir = const.data_dir
_clams_file = os.path.join(_data_dir, 'clams_age_clim_scaled.nc')


###########################################
# FUNCTIONS SUPPORTING PRIORS CALCUALTION #
###########################################

class GasRecordError(Exception):
    """
    Base error for problems in the CO2 record
    """
    pass


class GasRecordInputMissingError(GasRecordError):
    """
    Error to use when cannot find the necessary input files for a trace gas record
    """
    pass


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


class TraceGasTropicsRecord(object):
    """
    This class stores the Mauna Loa/Samoa average CO2 record and provides methods to sample it.

    No arguments required for initialization.
    """

    # these should be overridden in subclasses to specify the name and unit of the gas. The name must be recognized by
    # the seasonal cycle function
    gas_name = ''
    gas_unit = ''

    months_avg_for_trend = 12
    age_spec_regions = ('tropics', 'midlat', 'vortex')
    _age_spectrum_data_dir = os.path.join(_data_dir, 'age_spectra')
    _age_spectrum_time_file = os.path.join(_age_spectrum_data_dir, 'time.txt')
    _default_sbc_lag = relativedelta(months=2)

    def __init__(self, first_date=None, last_date=None, lag=None):
        # For the stratosphere data, since the age spectra are defined over a 30 year window, we need to make sure
        # we have values back to slightly more than 30 years before the first TCCON data. Assuming that's around 2004,
        # a default age of 2000 - 30 = 1970 should be good.
        age_spectra_length = relativedelta(years=30)

        if first_date is None:
            first_date = dt.datetime(2000, 1, 1) - age_spectra_length
        else:
            first_date -= age_spectra_length

        if last_date is None:
            # By default, we want to extrapolate to 2 years out from today; the max negative age-of-air in the
            # troposphere should be about 6 months at most, but we need to allow some room for the rolling average to
            # get the trend.
            last_date = dt.datetime.today() + relativedelta(years=2)

        if lag is None:
            self.sbc_lag = self._default_sbc_lag
        else:
            self.sbc_lag = lag

        self.conc_seasonal = self.get_mlo_smo_mean(first_date, last_date)

        # Deseasonalize the data by taking a 12 month rolling average. Only do that on the dmf_mean field,
        # leave the latency
        self.conc_trend = self.conc_seasonal.rolling(self.months_avg_for_trend, center=True).mean().dropna().drop('interp_flag', axis=1)

        self.conc_strat = self._calc_age_spec_gas(self.conc_seasonal, lag=self.sbc_lag)

    @classmethod
    def _get_agespec_files(cls, region):
        def _file_name_helper(reg, prefix):
            base_name = '{}.{}.txt'.format(prefix, reg)
            full_name = os.path.join(cls._age_spectrum_data_dir, base_name)
            if not os.path.isfile(full_name):
                agespec_files = glob(os.path.join(cls._age_spectrum_data_dir, '{}.*.txt'.format(prefix)))
                re_pattern = r'(?<={}\.)\w+(?=\.txt)'.format(prefix)
                agespec_regions = [re.search(re_pattern, os.path.basename(f)) for f in agespec_files]
                raise GasRecordInputMissingError('Cannot find an {pre} file for the region "{reg}". Available regions '
                                                 'are: {avail}'.format(pre=prefix, reg=region,
                                                                       avail=', '.join(agespec_regions)))

            return full_name

        age_file = _file_name_helper(region, 'age')
        agespec_file = _file_name_helper(region, 'agespec')

        return age_file, agespec_file

    @classmethod
    def _get_frac_remaining_by_age(cls, ages):
        """
        Get the fraction of a gas remaining for a given vector of ages.

        The default is to assume no loss, and so 1 will be returned for every age. Subclasses
        may override this method to calculate more complicated relationships between age and fraction remaining.

        :param ages: the vector of ages to calculate the fraction of the gas remaining for
        :type ages: :class:`numpy.ndarray` or float

        :return: a data frame indexed by age with one column, "fraction" containing the fraction remaining.
        :rtype: :class:`pandas.DataFrame`
        """
        return pd.DataFrame(data=np.ones_like(ages, dtype=np.float), columns=['fraction'], index=ages)

    @classmethod
    def _load_age_spectrum_data(cls, region, normalize_spectra=True):
        def _load_helper(filename):
            return pd.read_csv(filename, header=None, sep=' ')

        time = _load_helper(cls._age_spectrum_time_file)
        delt = np.mean(np.diff(time.values, axis=0))
        age_file, agespec_file = cls._get_agespec_files(region)
        age = _load_helper(age_file)
        spectra = _load_helper(agespec_file)
        if normalize_spectra:
            # Ensure that the integral of each age spectrum is 1. Assumes that time has a consistent spacing between
            # adjacent points, then basically does a midpoint rule integration
            for i in range(spectra.shape[0]):
                spec = spectra.iloc[i, :]
                spectra.iloc[i, :] = (delt * spec) / (np.nansum(spec) * delt)
        return time, delt, age, spectra

    @classmethod
    def read_insitu_gas(cls, fpath, fname):
        """
        Read a trace gas record file. Assumes that the file is of monthly average concentrations.

        :param fpath: the path to the directory containing the file.
        :type fpath: str

        :param fname: the name of the file
        :type fname: str

        :return: a data frame containing the monthly trace gas data along with the site, year, month, and day. The index will
         be a timestamp of the measurment time.
        :rtype: :class:`pandas.DataFrame`
        """
        full_file_path = os.path.join(fpath, fname)
        with open(full_file_path, 'r') as f:
            hlines = f.readline().rstrip().split(': ')[1]

        df = pd.read_csv(full_file_path, skiprows=int(hlines), skipinitialspace=True,
                         delimiter=' ', header=None, names=['site', 'year', 'month', cls.gas_name])

        # set datetime index in df (requires 'day' column)
        df['day'] = 1
        df.set_index(pd.to_datetime(df[['year', 'month', 'day']]), inplace=True)

        return df

    @classmethod
    def get_mlo_smo_mean(cls, first_date, last_date):
        """
        Generate the Mauna Loa/Samoa mean trace gas record from the files stored in this repository.

        Reads in the :file:`data/ML_monthly_obs.txt` and :file:`data/SMO_monthly_obs.txt` files included in this
        repository, averages them, and fills in any missing months by interpolation.

        :return: the data frame containing the mean trace gas concentration ('dmf_mean') and a flag ('interp_flag') set
         to 1 for any months that had to be interpolated. Index by timestamp.
        :rtype: :class:`pandas.DataFrame`
        """
        df_mlo = cls.read_insitu_gas(_data_dir, 'ML_monthly_obs_%s.txt'%cls.gas_name)
        df_smo = cls.read_insitu_gas(_data_dir, 'SMO_monthly_obs_%s.txt'%cls.gas_name)
        df_combined = pd.concat([df_mlo, df_smo], axis=1).dropna()
        df_combined['dmf_mean'] = df_combined[cls.gas_name].mean(axis=1)
        df_combined.drop(['site', cls.gas_name, 'year', 'month', 'day'], axis=1, inplace=True)

        # Fill in any missing months. Add a flag so we can keep track of whether they've had to be interpolated or
        # not. Having a consistent monthly frequency makes the rest of the code easier - we can just always assume that
        # there will be a value at the beginning of every month. Also track latency in the data frame.

        # Make sure that first_date and last_date are at the start of a month. If not, go to the start of month that
        # makes sure we cover the requested time period
        if first_date.day != 1:
            first_date = mod_utils.start_of_month(first_date)

        if last_date.day != 1:
            last_date = mod_utils.start_of_month(last_date) + relativedelta(months=1)
        all_months = pd.date_range(first_date, last_date, freq='MS')
        n_months = all_months.size

        df_combined = df_combined.reindex(all_months)
        df_combined = df_combined.assign(interp_flag=np.zeros((n_months,), dtype=np.int),
                                         latency=np.zeros((n_months,), dtype=np.int))

        # set the interpolation flag
        missing = pd.isna(df_combined['dmf_mean'])
        # filling the internal missing NaNs first, mark them as interpolated, then extrapolate
        df_combined.interpolate(method='index', inplace=True, limit_area='inside')
        interpolated = missing.values & ~pd.isna(df_combined['dmf_mean'])
        extrapolated = missing.values & pd.isna(df_combined['dmf_mean'])

        df_combined.loc[interpolated, 'interp_flag'] = 1
        df_combined.loc[extrapolated, 'interp_flag'] = 2

        # Almost there - handle the extrapolation. Only these values will have latency
        ex_inds = extrapolated.to_numpy().nonzero()[0]
        for i in ex_inds:
            timestamp = df_combined.index[i]
            df_combined.loc[timestamp, 'dmf_mean'], quality_dict = cls._calc_monthly_gas(df_combined[~extrapolated], timestamp.year, timestamp.month)
            df_combined.loc[timestamp, 'latency'] = quality_dict['latency']

        # Do any last post-processing to handle any problems that arise during extrapolation. We'll verify that the
        # indices remain unchanged by the processing and the original columns remain in the data frame (though
        # additional columns may be added)
        orig_index = df_combined.index
        orig_columns = df_combined.columns
        df_combined = cls._extrap_post_proc_hook(df_combined)

        if not (df_combined.index == orig_index).all():
            raise RuntimeError('The data frame returned from the extrapolation post processing has different indices '
                               'than it did before the processing')
        if any(c not in df_combined.columns for c in orig_columns):
            raise RuntimeError('One or more columns are missing from the data frame returned by the extrapolation post '
                               'processing')

        return df_combined

    @classmethod
    def _extrap_post_proc_hook(cls, gas_df):
        """
        Method to handle any necessary post processing of trace gas trends after extrapolated to the full time required

        The default behavior is to replace all dates extrapolated backwards with a smoothed version. This avoids issues
        when the extrapolation causes the seasonal cycle to become larger and larger as it goes back in time.

        This method should be overridden in subclasses if more careful treatment is required.

        :param gas_df: the data frame indexed by date giving the gas concentration (column "dmf_mean"), interpolation
         flag ("interp_flag") and latency ("latency").
        :type gas_df: :class:`pandas.DataFrame`

        :return: the data frame with values adjusted. Will have the same indices and columns. Note: subclass override
         methods must be sure to return the data frame with the same indices and the original columns. New columns may
         be added, but none of the original columns can be removed.
        :rtype: :class:`pandas.DataFrame`
        """

        # Testing showed that N2O and CH4 at least were very succeptible to an ever increasing seasonal cycle amplitude
        # as the extrapolation backward in time got further and further. This is the simplest fix, a 12 month average
        # wipes out the seasonal cycle, and since the central tendency didn't get messed up, this result is at least
        # reasonable.
        #
        # We allow Pandas to fill in values at the beginning (limit_direction='backward') with the nearest value to keep
        # the data frame defined over the full date range. Yes this will flatten the trend out in those first six
        # months, but since those first six months contribute very little to the age-spectrum average concentration for
        # any time period relevant to TCCON, that's not going to affect the priors much. A proper linear extrapolation
        # would be better long-term.
        #
        # Alternately, data from sites like
        # https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-4
        # that extend back further could be used to at least get the shape of the trends. If early (pre 1990) data
        # becomes crucial, that is probably the best approach.
        smoothed_conc = gas_df.dmf_mean.rolling(window=12, center=True).mean().interpolate(method='index', limit_direction='backward')
        xx = gas_df.latency < 0
        gas_df.loc[xx, 'dmf_mean'] = smoothed_conc
        return gas_df

    @staticmethod
    def _calc_monthly_gas(df, year, month, limit_extrapolation_to=None):
        """
        Get gas concentration for a specific month, extrapolating if necessary.

        :param year: what year to query
        :type year: int

        :param month: what month to query
        :type year: int

        :param deseasonalize: set to ``True`` to use the concentration record with the seasonal record smoothed out.
         Default is ``False``, which keeps the seasonal record.
        :type deseasonalize: bool

        :param limit_extrapolation_to: a date beyond which not to extrapolate
        :type limit_extrapolation_to: a datetime-like object

        :return: the concentration value for this month, and a dictionary with keys "flag" and "latency". "latency" will
         be the number of years that the concentration value had to be extrapolated. Flag will be one of:

         * 0 = data read directly from record
         * 1 = data had to be extrapolated
         * 2 = requested date was before the start of the record
         * 3 = requested date was after the date set by ``limit_extrapolation_to``
         * 4 = unanticipated error occured.

        :rtype: float, dict
        """

        years_extrap = 0
        day = 1
        fillval = np.nan
        target_date = pd.Timestamp(year, month, day)

        if limit_extrapolation_to is None:
            # 100 years should be sufficiently far as to be effectively no limit
            limit_extrapolation_to = relativedelta(years=100)

        first_available_date = df.index.min()
        last_available_date = df.index.max()
        first_allowed_date = first_available_date - limit_extrapolation_to
        last_allowed_date = last_available_date + limit_extrapolation_to

        if target_date < first_allowed_date or target_date > last_allowed_date:
            flag = 2
            val = fillval
        elif (target_date >= first_available_date) & (target_date <= last_available_date):

            flag = 0
            #print('Reading data from file...')
            # As of pandas version 0.24.1, df.dmf_mean[target_date] was ~10x faster than
            # df.loc[target_date]['dmf_mean']
            val = df.dmf_mean[target_date]

        else:
            if target_date < first_available_date:
                sign = -1
            else:
                sign = 1
            flag = 1
            nyear = 5 * sign
            #print('Date outside available time period... extrapolating!')

            # Need to find the most recent year that we have data for this month

            latest_date = target_date
            while latest_date > last_available_date or latest_date < first_available_date:
                latest_date -= relativedelta(years=1*sign)
                years_extrap += 1*sign

            # Get the most nearest nyears concentrations for this month in the record. Set the initial value
            # to the last of those.
            prev_year = [y for y in reversed(range(years_extrap, nyear + years_extrap, sign))]

            prev_date = [pd.Timestamp(year - item, month, day) for item in prev_year]
            prev_gas = df.loc[prev_date, 'dmf_mean'].values

            val = df.loc[latest_date]['dmf_mean']
            for start_yr in range(0, years_extrap, sign):
                # For each year we need to extrapolate, calculate the growth rate as the average of the growth
                # rate over five years. This should help smooth out any El Nino effects, which would tend to
                # cause unusual growth rates.
                growth = np.diff(prev_gas).mean()
                val += growth

                # Now that we have the extrapolated value, update the last 5 concentration values to include it and
                # remove the earliest one so that we have updated concentration values for the next time through the
                # loop.
                prev_gas = np.append(prev_gas, val)
                prev_gas = np.delete(prev_gas, 0)

        return val, {'flag': flag, 'latency': years_extrap}

    @classmethod
    def _calc_age_spec_gas(cls, df, lag, requested_dates=None):
        def index_to_dec_year(dframe):
            return [mod_utils.date_to_decimal_year(d) for d in dframe.index]

        def dec_year_to_dtindex(dec_yr, force_first_of_month=False):
            date_times = mod_utils.decimal_year_to_date(dec_yr)
            if force_first_of_month:
                # Get the start of the month on either side of this date. Figure out which one is closer, and set it to
                # that.
                for i, dtime in enumerate(date_times):
                    som = mod_utils.start_of_month(dtime, out_type=dt.datetime)
                    nearby_months = np.array([som, som+relativedelta(months=1)])
                    i_time = np.argmin(np.abs(nearby_months - dtime))
                    date_times[i] = nearby_months[i_time]

            return pd.DatetimeIndex(date_times)

        gas_conc = dict()

        # Apply the requested lag by adding it to the dates that make up the index of the input data frame. Adding it
        # means that, e.g. 2018-03-01 will actually point to data from 2018-01-01, which is what we want. We're lagging
        # the data because the stratospheric boundary condition should account for the fact that it takes time for air
        # to get from the tropical surface (where MLO/SMO measure) and into the stratosphere.
        lagged_index = pd.DatetimeIndex(d + lag for d in df.index)
        df_lagged = df.set_index(lagged_index)

        # We'll need both the date as decimal years and timestamps for different parts of this code, so make those
        # additional columns in the data frame. We'll switch between them for the index as needed
        df_lagged = df_lagged.assign(timestamp=df_lagged.index, dec_year=index_to_dec_year)

        # By default, we'll assume that we want the output to be on the same dates as the input. With the lag that will
        # mean that some extra points near the beginning are NaNs, but that is expected and okay. We're more limited by
        # how far back in time the
        out_dates = df.index if requested_dates is None else requested_dates

        for region in cls.age_spec_regions:
            time, delt, age, spectra = cls._load_age_spectrum_data(region, normalize_spectra=True)

            # Get the fraction remaining on the same ages as the age spectra are defined
            fgas = cls._get_frac_remaining_by_age(time.values.squeeze())

            # Add a zero age to the beginning of age
            age = np.concatenate([[0], age.to_numpy().squeeze()])
            out_df = pd.DataFrame(index=out_dates, columns=age)
            # Put the lagged record, without any age spectrum applied, as the zero age data
            out_df.iloc[:, 0] = df_lagged['dmf_mean']

            for i in range(spectra.shape[0]):
                # The first step is to put the trace gas record on the same time resolution as the age spectra. This is
                # necessary for the convolution to work. Note that the age spectra aren't assigned to any specific date,
                # we just need the adjacent points in the age spectra and gas record to have the same spacing in time.
                max_dec_year = mod_utils.date_to_decimal_year(df_lagged.index.max())
                # 1950 is the year Arlyn Andrews used in her code. That will cause some NaNs at the beginning of the
                # record before our gas records start, but that's fine.
                new_index = np.arange(1950.0, max_dec_year, delt)
                # To handle the reindexing properly, we need to keep the original indices in until we handle the
                # interpolation to the new values. For this part we need to use the decimal years as the index
                tmp_index = np.unique(np.concatenate([df_lagged['dec_year'], new_index]))
                df_asi = df_lagged.set_index('dec_year', drop=False).reindex(tmp_index).interpolate(method='index').reindex(new_index)

                # Now we can do the convolution. Note: in Arlyn's original R code, she had to flip the age spectrum to
                # act as the convolution kernel, but testing showed that in order to get the same answer using numpy's
                # convolution function we had to leave the spectrum unflipped.
                #
                # This is because the numpy convolution operation acts to flip the kernel internally. It is defined as
                #
                # (a * v)[n] = \sum_{m=-\infty}^{\infty} a[m]v[n-m]
                #
                # Note that v is indexed with n-m. This has the effect of reversing the kernel; for n=10, a[11] gets
                # multiplied by v[9], a[12] by v[8] and so on. R's convolve function uses a different indexing pattern
                # that does not reverse the kernel.
                #
                # We want the kernel reversed because the trace gas records are defined from old to new, while the age
                # spectra are from new to old. Therefore, we need to reverse the spectra before convolving to actually
                # put both in the same direction.
                #
                # We also multiply the age spectra by the fraction of gas remaining at a given age. For something like
                # CO2 that is not lost in the stratosphere, this will just be 1s, but for N2O, CH4, etc. that have loss
                # then as we get to older air we also need to reduce MLO/SMO average to the fraction remaining to
                # accurately represent the concentration in that air mass.
                conv_result = np.convolve(df_asi['dmf_mean'].values.squeeze(),
                                          spectra.iloc[i, :].values * fgas.values.squeeze(),
                                          mode='valid')
                conv_dates = new_index[(spectra.shape[1] - 1):]
                conv_dates = dec_year_to_dtindex(conv_dates, force_first_of_month=False)
                
                # Finally we put the age-convolved gas concentration back onto the dates of the input dataframe, unless
                # alternate dates were specified.
                conv_df = pd.DataFrame(conv_result, index=conv_dates)
                tmp_index = np.unique(np.concatenate([out_dates, conv_dates]))
                this_out_df = conv_df.reindex(tmp_index).interpolate(method='index').reindex(out_dates)

                # And store this result in the output data frame, remembering that we added an extra row at the
                # beginning for zero age air
                out_df.iloc[:, i+1] = this_out_df.reindex(out_df.index).values

            gas_conc[region] = out_df

        return gas_conc

    def get_strat_gas(self, date, ages, eqlat, as_dataframe=False):
        """
        Get stratospheric gas concentration for a given profile

        :param date: the UTC date of the observation
        :type date: datetime-like

        :param ages: the age or ages of air (in years) to get concentration for. Must be the same shape as ``eqlat``.
        :type ages: float or :class:`numpy.ndarray`

        :param eqlat: the equivalent latitude or eq. lat profile to get concentration for. Must be the same shape as
         ``ages``.
        :type eqlat: float or :class:`numpy.ndarray`

        :param as_dataframe: if ``True``, the gas concentration will be returned as a data frame. If ``False``, it will
         be returned as an array if ``ages`` and ``eqlat`` were arrays or a float if they were floats.
        :type as_dataframe: bool

        :return: the gas concentration as a data frame, numpy array, or scalar, depending on ``as_dataframe`` and the
         input types. Also returns None, a placeholder for future information about profile latency, etc.
        :rtype: float, :class:`numpy.ndarray`, or :class:`pandas.DataFrame`
        """

        if isinstance(ages, float):
            return_scalar = True
            # make sure that we get a 1D array when we turn this into a numpy array
            ages = [ages]
        else:
            return_scalar = False

        if isinstance(eqlat, float):
            eqlat = [eqlat]

        ages = np.array(ages)
        eqlat = np.array(eqlat)

        if ages.shape != eqlat.shape:
            raise ValueError('Both ages and eqlat must be the same shape')
        elif ages.ndim != 1 or eqlat.ndim != 1:
            raise ValueError('Both ages and eqlat expected to be 1D arrays, if given as arrays')

        # Get the concentrations for the given ages and equivalent latitudes for each region (tropics, midlat, and
        # vortex). We'll stitch them together after.
        gas_by_region = dict()

        # For each region, interpolate to the date and ages we need by creating a new data frame that has just the
        # entries for the months bracketing the obs date plus the actual obs date and obs ages.
        prev_month = mod_utils.start_of_month(date, pd.Timestamp)
        next_month = prev_month + relativedelta(months=1)
        date_timestamp = pd.Timestamp(date)
        df_dates = pd.unique(pd.DatetimeIndex([prev_month, date_timestamp, next_month]))
        ages_index = pd.Float64Index(ages)

        for region in self.age_spec_regions:
            df_ages = np.unique(np.concatenate([self.conc_strat[region].columns, ages_index]))
            region_df = self.conc_strat[region].reindex(df_dates, axis=0).reindex(df_ages, axis=1)

            # Interpolate to fill in the ages first, then the date. Do it this way because if an age has no data in
            # the surrounding months, we can't interpolate to the desired date
            region_df.interpolate(method='index', axis=0, limit_area='inside', inplace=True)
            region_df.interpolate(method='index', axis=1, limit_area='inside', inplace=True)

            region_df = region_df.reindex(pd.DatetimeIndex([date_timestamp]), axis=0).reindex(ages_index, axis=1)
            gas_by_region[region] = region_df.T  # transpose to put the ages as the index - makes more sense now that there's only one date

        gas_conc = gas_by_region['midlat']
        xx_tropics = np.abs(eqlat) < 20.0
        doy = mod_utils.day_of_year(date)
        if 140 < doy < 245:
            xx_vortex = (eqlat < -55.0) & (ages > 3.25)
        elif doy > 275 or doy < 60:
            xx_vortex = (eqlat > 55.0) & (ages > 3.25)
        else:
            xx_vortex = np.zeros_like(xx_tropics)

        gas_conc[xx_tropics] = gas_by_region['tropics'][xx_tropics]
        gas_conc[xx_vortex] = gas_by_region['vortex'][xx_vortex]

        if as_dataframe:
            return gas_conc, None
        elif not return_scalar:
            return gas_conc.to_numpy().squeeze(), None
        else:
            return gas_conc.to_numpy().item(), None

    def get_gas_for_dates(self, dates, deseasonalize=False, as_dataframe=False):
        """
        Get trace gas concentrations for one or more dates.

        This method will lookup concentrations for a specific date or dates, interpolating between the monthly values as
        necessary.

        :param dates: the date or dates to get concentrations for. If giving a single date, it may be any time that can
         be converted to a Pandas :class:`~pandas.Timestamp`. If giving a series of dates, it must be a
         :class:`pandas.DatetimeIndex`.

        :param deseasonalize: whether to draw concentrations data from the trend only (``True``) or the seasonal cycle
         (``False``).
        :type deseasonalize: bool

        :param as_dataframe: whether to return the concentrations data as a dataframe (``True``) or numpy array
         (``False``)
        :type as_dataframe: bool

        :return: the concentration data for the requested date(s), as a numpy vector or data frame. The data frame will
         also include the latency (how many years the concentrations had to be extrapolated).
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

        start_date = dates.min()
        end_date = dates.max()

        # Need to make sure we get data that bracket the start and end date, so set them the first days of month
        start_date_subset = mod_utils.start_of_month(start_date, out_type=pd.Timestamp)
        end_date_subset = mod_utils.start_of_month(end_date + relativedelta(months=1), out_type=pd.Timestamp)

        # First get just monthly data. freq='MS' gives us monthly data at the start of the month. Use the existing logic
        # to extrapolate the record for a given month if needed.
        monthly_idx = pd.date_range(start_date_subset, end_date_subset, freq='MS')
        monthly_df = pd.DataFrame(index=monthly_idx, columns=['dmf_mean', 'latency'], dtype=np.float)
        for timestamp in monthly_df.index:
            monthly_df.dmf_mean[timestamp], info_dict = self.get_gas_by_month(timestamp.year, timestamp.month, deseasonalize=deseasonalize)
            monthly_df.latency[timestamp] = info_dict['latency']

        # Now we resample to the dates requested, making sure to keep the values at the start of each month on either
        # end of the record to ensure interpolation is successful
        sample_date_idx = dates.copy()
        sample_date_idx = sample_date_idx.append(monthly_idx)
        sample_date_idx = sample_date_idx.sort_values()  # is needed for successful interpolation
        sample_date_idx = pd.unique(sample_date_idx)  # deal with the possibility that one of the requested dates was a month start
        df_resampled = monthly_df.reindex(sample_date_idx)

        # Verify we have non-NaN values for all monthly reference points
        if df_resampled['dmf_mean'][monthly_idx].isna().any():
            raise RuntimeError('Failed to resample concentrations for date range {} to {}; first and/or last point is NA'
                               .format(start_date_subset, end_date_subset))

        df_resampled.interpolate(method='index', inplace=True)

        # Return with just the originally requested dates
        df_resampled = df_resampled.reindex(dates)
        if as_dataframe:
            return df_resampled
        else:
            return df_resampled['dmf_mean'].values

    def avg_gas_in_date_range(self, start_date, end_date, deseasonalize=False):
        """
        Average the MLO/SMO record between the given dates

        :param start_date: the first date in the averaging period
        :type start_date: datetime-like object

        :param end_date: the last date in the averaging period
        :type end_date: datetime-like object

        :param deseasonalize: whether to draw concentration data from the trend only (``True``) or the seasonal cycle
         (``False``).
        :type deseasonalize: bool

        :return: the average concentration and a dictionary specifying the mean, minimum, and maximum latency
         (number of years the concentrations had to be extrapolated)
        :rtype: float, dict
        """
        if not isinstance(start_date, dt.date) or not isinstance(end_date, dt.date):
            raise TypeError('start_date and end_date must be datetime.date objects (cannot be datetime.datetime objects)')

        # In theory, different resolutions could be given but would need to be careful that the reindexing produced
        # values at the right times.
        resolution = dt.timedelta(days=1)

        avg_idx = pd.date_range(start=start_date, end=end_date, freq=resolution)
        df_resampled = self.get_gas_for_dates(avg_idx, deseasonalize=deseasonalize, as_dataframe=True)

        mean_gas_conc = df_resampled['dmf_mean'][avg_idx].mean()
        latency = dict()
        latency['mean'] = df_resampled['latency'][avg_idx].mean()
        latency['min'] = df_resampled['latency'][avg_idx].min()
        latency['max'] = df_resampled['latency'][avg_idx].max()
        return mean_gas_conc, latency

    def get_gas_by_age(self, ref_date, age, deseasonalize=False, as_dataframe=False):
        """
        Get concentrations for one or more times by specifying a reference date and age.

        This called :meth:`get_gas_for_dates` internally, so the concentration is interpolated to the specific day just
        as that method does.

        :param ref_date: the date that the ages are relative to.
        :type ref_date: datetime-like object.

        :param age: the number of years before the reference date to get the concentration from. May be a non-whole
         number.
        :type age: float or sequence of floats

        :param deseasonalize: whether to draw concentration data from the trend only (``True``) or the seasonal cycle
         (``False``).
        :type deseasonalize: bool

        :param as_dataframe: whether to return the concentration data as a dataframe (``True``) or numpy array
         (``False``)
        :type as_dataframe: bool

        :return: the concentration data for the requested date(s), as a numpy vector or data frame. The data frame will
         also include the latency (how many years the concentrations had to be extrapolated).
        """
        gas_dates = [ref_date - dt.timedelta(days=a*365.25) for a in age]
        return self.get_gas_for_dates(pd.DatetimeIndex(gas_dates), deseasonalize=deseasonalize,
                                      as_dataframe=as_dataframe)

    def get_gas_by_month(self, year, month, deseasonalize=False):
        """
        Get the trace gas concentration for a specific month

        :param year: the date's year
        :type year: int

        :param month: the date's month
        :type month: int

        :param deseasonalize: whether to draw concentration data from the trend only (``True``) or the seasonal cycle
         (``False``).
        :type deseasonalize: bool

        :return: the gas concentration and a dictionary with additional information (e.g. the latency, that is, how far
         the concentrations had to be extrapolated).
        :rtype: float, dict
        """
        df = self.conc_trend if deseasonalize else self.conc_seasonal
        ts = pd.Timestamp(year, month, 1)
        info_dict = {'latency': df.latency[ts]}
        return df.dmf_mean[ts], info_dict


class CO2TropicsRecord(TraceGasTropicsRecord):
    gas_name = 'co2'
    gas_unit = 'ppm'


class N2OTropicsRecord(TraceGasTropicsRecord):
    gas_name = 'n2o'
    gas_unit = 'ppb'

    @classmethod
    def _get_frac_remaining_by_age(cls, ages):
        fracs = [cls.calc_fn2o_from_age(a) for a in ages]
        return pd.DataFrame(data=fracs, columns=['fraction'], index=ages)

    @staticmethod
    def calc_fn2o_from_age(age):
        def age_fxn(n2o):
            # This equation relates N2O to age in Fig. 3 of Andrews et al. (JGR, 106, pp. 32295-32314). This was derived
            # from the ATLAS spectrometer on board the ER-2 aircraft flying in the 1990s. (In the paper, age is called
            # "CO2 lag".)
            return 0.0581 * (313.0 - n2o) - 2.54e-4 * (313 - n2o) ** 2.0 + 4.41e-7 * (313.0 - n2o) ** 3.0

        # Rather than try to invert this cubic function to get N2O in terms of age, we solve numerically for what value
        # of N2O gives the desired age. We apply some bounds to the minimization because this is about the range of
        # concentrations measured (technically that should stop around 50 ppbv on the lower bound, but we may need to go
        # a little past that to get to the oldest air).
        n2o_at_age = minimize_scalar(lambda x: abs(age_fxn(x) - age), method='bounded', bounds=(0.0, 320.0))
        n2o_at_age = n2o_at_age.x  # extract the actual concentration from the minimization result

        # From the equation, 313 ppb N2O is clearly the age 0 concentration. We assume that that is the stratospheric
        # boundary condition, and so the fraction of N2O remaining is the concentration divided by 313.
        return n2o_at_age / 313.0

    
class CH4TropicsRecord(TraceGasTropicsRecord):
    gas_name = 'ch4'
    gas_unit = 'ppb'


# Make the list of available gases' records
_gas_records = {r.gas_name: r for r in [CO2TropicsRecord, N2OTropicsRecord, CH4TropicsRecord]}


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
        clams = ncdf.Dataset(_clams_file, 'r')
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


def get_trop_eq_lat(prof_theta, p_levels, obs_lat, obs_date, theta_wt=1.0, lat_wt=1.0, dtheta_cutoff=0.25,
                    _theta_v_lat=dict()):
    """
    Compute the tropospheric equivalent latitude for an observation based on its mid-tropospheric potential temperature

    The rationale for using this approach is described in the module help for backend_analysis/geos_theta_lat.py. This
    function relies on a climatology created by that module, which should contain the zonal mean relationship between
    mid-tropospheric potential temperature and latitude at 2 week intervals.

    This function finds the equivalent latitude for an observation by looking for the point in the same hemisphere that
    has the closest mid-tropospheric potential temperature in the climatology as does the profile given as input.
    Exactly what is defined as mid-troposphere is set by the pressure range in the climatology file, currently it is
    700-500 hPa.

    This function checks both north and south of the observation latitude for the climatology latitude with the closest
    potential temperature. As long as one is sufficiently closer to the observation's potential temperature, that one
    is chosen directly. If the two are within the limit set by ``dtheta_cutoff``, then a more careful check is
    necessary. The limit is defined as:

    .. math::

       |(el_s - l) - (el_n - l)| < d\theta

    where :math:`el_s` and :math:`el_n` are the southern and northern latitudes in the climatology with the closest
    potential temperature to the observations, :math:`l` is the observation latitude, and :math:`d\theta` is
    ``dtheta_cutoff``.  If this condition is met, then rather than just choosing whichever one has the closer
    potential temperature, the algorithm uses a cost function:

    .. math:

       |w_t * d\theta| + |w_l * dl|

    where :math:`w_t` and :math:`w_l` are the weights for potential temperature (``theta_wt``) and latitude (``lat_wt``)
    respectively, and :math:`d\theta` and :math:`dl` are the difference in potential temperature and latitude,
    respectively, between the observation and the point chosen on the climatology curve.

    The goal of this approach is to deal with two cases:

    1. when the theta vs. lat curve from the climatology is monotonically increasing or decreasing
    2. when the curve has a minimum or maximum

    For #1, consider a case where theta decreases with latitude, and the observation's theta is greater than the
    climatological theta for that latitude. Then going south will match the theta much better, so the cutoff condition
    is not met, and we automatically choose the southern point.

    For #2, consider again a case where the observation's theta is greater that climatological theta for that latitude,
    but now the climatological curve has a minimum just north of the observation. In that case, we may find two equally
    good matches for the observation's theta, so, in the absence of other information, we choose the nearer one. This is
    admittedly a simplification - it is entirely possible that the actual synoptic transport carried air from the
    further position, but without a second tracer to differentiate that in the meteorology data, or information on
    prevailing north/south transport for a given lat/lon, the best assumption is to favor shorter transport.

    :param prof_theta: the profile of potential temperature values associated with this observation
    :type prof_theta: :class:`numpy.ndarray`

    :param p_levels: the profile of pressure levels that ``prof_theta`` is defined on
    :type p_levels: :class:`numpy.ndarray`

    :param obs_lat: the geographic latitude of the observation
    :type obs_lat: float

    :param obs_date: the date of the observation
    :type obs_date: datetime-lik

    :param theta_wt: a weight to use when deciding between two different latitudes with similar theta values. Increasing
     this relative to ``lat_wt`` will increase the cost for choosing the point with a greater difference in potential
     temperature.
    :type theta_wt: float

    :param lat_wt: similar to ``theta_wt``, but increasing this prefers the point closer in latitude.
    :type lat_wt: float

    :param dtheta_cutoff: how close the two (north and south) differences between the climatology and observed
     mid-troposphere potential temperature have to be to take into account which one is closer. See above.
    :type dtheta_cutoff: float

    :param _theta_v_lat: not intended to pass in; this is a dictionary that will be given the values read in from the
     climatology file to cache them for future function calls.

    :return: the equivalent latitude derived from mid-tropospheric potential temperature
    :rtype: float
    """
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

    if len(_theta_v_lat) == 0:
        theta_v_lat_file = os.path.join(const.data_dir, 'GEOS_FPIT_lat_vs_theta_2018_500-700hPa.nc')
        with ncdf.Dataset(theta_v_lat_file, 'r') as nch:
            _theta_v_lat['theta'] = nch.variables['theta_mean'][:].squeeze()
            _theta_v_lat['lat'] = nch.variables['latitude_mean'][:].squeeze()
            _theta_v_lat['times'] = nch.variables['times'][:]
            _theta_v_lat['times_units'] = nch.variables['times'].units
            _theta_v_lat['times_calendar'] = nch.variables['times'].calendar

            # Read the pressure range that we're using
            _theta_v_lat['pres_range'] = read_pres_range(nch)

            # Append the first time slice (which will be the first two weeks of the year) to the end so that we can
            # intepolate past the last date, assuming that the changes are cyclical. At the same time, let's record the
            # year used in the dates
            new_time = ncdf.num2date(_theta_v_lat['times'][0], _theta_v_lat['times_units'], _theta_v_lat['times_calendar'])
            _theta_v_lat['year'] = year = new_time.year
            new_time = ncdf.date2num(new_time.replace(year=year+1), _theta_v_lat['times_units'], _theta_v_lat['times_calendar'])
            _theta_v_lat['times'] = np.concatenate([_theta_v_lat['times'], [new_time]], axis=0)
            for k in ('theta', 'lat'):
                _theta_v_lat[k] = np.concatenate([_theta_v_lat[k], _theta_v_lat[k][0:1, :]], axis=0)

    # First we need to get the lat vs. theta curve for this particular date
    ntimes, nbins = _theta_v_lat['theta'].shape
    this_datenum = ncdf.date2num(obs_date.replace(year=_theta_v_lat['year']), _theta_v_lat['times_units'], _theta_v_lat['times_calendar'])
    this_theta_clim = np.full((nbins,), np.nan)
    this_lat_clim = np.full((nbins,), np.nan)
    for i in range(nbins):
        this_theta_clim[i] = np.interp(this_datenum, _theta_v_lat['times'], _theta_v_lat['theta'][:, i])
        this_lat_clim[i] = np.interp(this_datenum, _theta_v_lat['times'], _theta_v_lat['lat'][:, i])

    # Then we find the theta for this profile
    zz = (p_levels >= _theta_v_lat['pres_range'][0]) & (p_levels <= _theta_v_lat['pres_range'][1])
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


def adjust_zgrid(z_grid, z_trop, z_obs):
    
    idx_min = abs(z_grid - z_obs).argmin()
    z_min = z_grid[idx_min]
    dz = z_obs - z_min
    
    z_blend = z_obs+(z_trop-z_obs)/2.     
    idx_blend = abs(z_grid - z_blend).argmin()
    
    z_pbl = z_grid[0:idx_blend]
    z_ftrop = z_grid[idx_blend::]
  
    for i in range(idx_blend-1, idx_min-1, -1):
        factor = float(idx_blend - i)/float(idx_blend - idx_min)
        z_pbl[i] = (z_grid[i]+dz*factor**2)
        
    z_pbl=np.where(z_pbl<z_obs, 0, z_pbl)
    z_grid = np.hstack((z_pbl, z_ftrop))
    
    return z_grid


#########################
# MAIN PRIORS FUNCTIONS #
#########################

def add_trop_prior(prof_gas, obs_date, obs_lat, z_grid, z_obs, z_trop, gas_record, theta_grid=None, pres_grid=None,
                   ref_lat=45.0, use_theta_eqlat=True, profs_latency=None, prof_aoa=None, prof_world_flag=None,
                   prof_gas_date=None, prof_gas_date_width=None):
    """
    Add troposphere CO2 to the prior profile.

    :param prof_gas: the profile trace gas mixing ratios. Will be modified in-place to add the stratospheric
     component.
    :type prof_gas: :class:`numpy.ndarray`

    :param obs_date: the UTC date of the retrieval.
    :type obs_date: :class:`datetime.datetime`

    :param obs_lat: the latitude of the retrieval (degrees, south is negative)
    :type obs_lat: float

    :param z_grid: the grid of altitudes (in kilometers) that the CO2 profile is on.
    :type z_grid: :class:`numpy.ndarray`

    :param z_trop: the altitude of the tropopause (in kilometers)
    :type z_trop: float

    :param gas_record: the Mauna Loa-Samoa record for the desired gas.
    :type gas_record: :class:`TraceGasTropicsRecord`

    :param theta_grid: potential temperature on the same levels as ``z_grid``. Only needed is ``use_theta_eqlat`` is set
     to ``True``.
    :type theta_grid: :class:`numpy.ndarray`

    :param pres_grid: pressure levels for the same levels as ``z_grid``. Only needed is ``use_theta_eqlat`` is set
     to ``True``.
    :type theta_grid: :class:`numpy.ndarray`

    :param ref_lat: the reference latitude for age of air. Effectively sets where the age begins, i.e where the
     emissions are.
    :type ref_lat: float.

    :param use_theta_eqlat: set to ``True`` to use an equivalent latitude derive from the mid-tropospheric potential
     temperature as the latitude in the age of air and seasonal cycle calculations. This helps correct overly curved
     profiles at sites near the tropics that sometimes have more tropical-like profiles depending on synoptic scale
     transport. If this is ``False``, then ``obs_lat`` is used directly.
    :type use_theta_eqlat: bool

    The following parameters are all optional; they are vectors that will be filled with the appropriate values in the
    stratosphere. The are also returned in the ancillary dictionary; if not given as inputs, they are initialized with
    NaNs. "nlev" below means the number of levels in the CO2 profile.

    :param profs_latency: nlev-by-3 array that will store how far forward in time the Mauna Loa/Samoa CO2 record had to
     be extrapolated, in years. The three columns will respectively contain the mean, min, and max latency.

    :param prof_aoa: nlev-element vector of ages of air, in years.

    :param prof_world_flag: nlev-element vector of ints which will indicate which levels are considered overworld and
     which middleworld. The values used for each are defined in :mod:`mod_constants`

    :param prof_gas_date: nlev-element vector that stores the date in the MLO/SMO record that the gas was taken from.
     Since most levels will have a window of dates, this is the middle of those windows. The dates are stored as a
     decimal year, e.g. 2016.5.

    :param prof_gas_date_width: nlev-element vector that stores the width (in years) of the age windows used to compute
     the gas concentrations.

    :return: the updated CO2 profile and a dictionary of the ancillary profiles.
    """

    z_grid = adjust_zgrid(z_grid, z_trop, z_obs)
    n_lev = np.size(z_grid)
    prof_gas = _init_prof(prof_gas, n_lev)
    profs_latency = _init_prof(profs_latency, n_lev, 3)
    prof_aoa = _init_prof(prof_aoa, n_lev)
    prof_world_flag = _init_prof(prof_world_flag, n_lev)
    prof_gas_date = _init_prof(prof_gas_date, n_lev)
    prof_gas_date_width = _init_prof(prof_gas_date_width, n_lev)

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

    gas_df = gas_record.get_gas_by_age(obs_date, air_age, deseasonalize=True, as_dataframe=True)
    prof_gas[xx_trop] = gas_df['dmf_mean'].values
    # Must reshape the 1D latency vector into an n-by-1 matrix to broadcast successfully
    profs_latency[xx_trop, :] = gas_df['latency'].values.reshape(-1, 1)
    # Record the date that the CO2 was taken from as a year with a fraction
    gas_dates = np.array([mod_utils.date_to_decimal_year(v) for v in gas_df.index])
    prof_gas_date[xx_trop] = gas_dates
    # The width for the age is defined as what time window we average over to detrend
    prof_gas_date_width[xx_trop] = gas_record.months_avg_for_trend / 12.0

    # Finally, apply a parameterized seasonal cycle. This is better than using the seasonal cycle in the MLO/SMO data
    # because that is dominated by the NH cycle. This approach allows the seasonal cycle to vary in sign and intensity
    # with latitude.
    year_fraction = mod_utils.date_to_frac_year(obs_date)
    prof_gas[xx_trop] *= mod_utils.seasonal_cycle_factor(obs_lat, z_grid[xx_trop], z_trop, year_fraction,
                                                         species=gas_record.gas_name, ref_lat=ref_lat)

    return prof_gas, {'co2_latency': profs_latency, 'co2_date': prof_gas_date, 'co2_date_width': prof_gas_date_width,
                      'age_of_air': prof_aoa, 'stratum': prof_world_flag, 'ref_lat': ref_lat, 'trop_lat': obs_lat}


def add_strat_prior(prof_gas, retrieval_date, prof_theta, prof_eqlat, tropopause_theta, gas_record,
                    profs_latency=None, prof_aoa=None, prof_world_flag=None):
    """
    Add the stratospheric trace gas to a TCCON prior profile

    :param prof_gas: the profile trace gase mixing ratios. Will be modified in-place to add the stratospheric
     component.
    :type prof_gas: :class:`numpy.ndarray` (in ppm)

    :param retrieval_date: the UTC date of the retrieval.
    :type retrieval_date: :class:`datetime.datetime`

    :param prof_theta: the theta (potential temperature) coordinates for the TCCON profile.
    :type prof_theta: :class:`numpy.ndarray` (in K)

    :param prof_eqlat: the equivalent latitude coordinates for the TCCON profile.
    :type prof_eqlat: :class:`numpy.ndarray` (in degrees)

    :param tropopause_theta: the potential temperature at the tropopause, according to the input meteorology.
    :type tropopause_theta: float (in K)

    :param gas_record: the Mauna Loa-Samoa CO2 record.
    :type gas_record: :class:`TraceGasTropicsRecord`

    :param lag: the lag between the MLO/SAM record and the CO2 concentration at the tropopause.
    :type lag: :class:`~dateutil.relativedelta.relativedelta` or :class:`datetime.timedelta`

    The following parameters are all optional; they are vectors that will be filled with the appropriate values in the
    stratosphere. The are also returned in the ancillary dictionary; if not given as inputs, they are initialized with
    NaNs. "nlev" below means the number of levels in the CO2 profile.

    :param profs_latency: nlev-by-3 array that will store how far forward in time the Mauna Loa/Samoa CO2 record had to
     be extrapolated, in years. The three columns will respectively contain the mean, min, and max latency.
    :param prof_aoa: nlev-element vector of ages of air, in years.
    :param prof_world_flag: nlev-element vector of ints which will indicate which levels are considered overworld and
     which middleworld. The values used for each are defined in :mod:`mod_constants`

    :return: the updated CO2 profile and a dictionary of the ancillary profiles.
    """

    n_lev = np.size(prof_gas)
    profs_latency = _init_prof(profs_latency, n_lev, 3)
    prof_aoa = _init_prof(prof_aoa, n_lev)
    prof_world_flag = _init_prof(prof_world_flag, n_lev)

    # Next we find the age of air in the stratosphere for points with theta > 380 K. We'll get all levels now and
    # restrict to >= 380 K later.
    xx_overworld = prof_theta >= 380
    prof_world_flag[xx_overworld] = const.overworld_flag
    retrieval_doy = int(np.round(mod_utils.date_to_frac_year(retrieval_date) * 365.25))
    age_of_air_years = get_clams_age(prof_theta, prof_eqlat, retrieval_doy, as_timedelta=False)
    prof_aoa[xx_overworld] = age_of_air_years[xx_overworld]

    # Now, assuming that the CLAMS age is the mean age of the stratospheric air and that we can assume the CO2 has
    # not changed since it entered the stratosphere, we look up the CO2 at the boundary condition. Assume that the
    # CO2 record has daily points, so we just want the date (not datetime). We look up the CO2 using a method on the
    # record specifically designed for stratospheric CO2 that already incorporates the two month lag and the age
    # spectra.

    prof_gas[xx_overworld], _ = gas_record.get_strat_gas(retrieval_date, age_of_air_years[xx_overworld], prof_eqlat[xx_overworld])
    # TODO: decide how to calculate the latency for the CO2 profiles now that age spectra are used. Options:
    #   1. Convolve the latency as well
    #   2. Give the latency just for the retrieval date (possible with the two-month lag)
    #   3. Give the maximum latency

    # Last we need to fill in the "middleworld" between the tropopause and 380 K. The simplest way to do it is to
    # assume that at the tropopause the CO2 is equal to the lagged MLO/SAM record and interpolate linearly in theta
    # space between that and the first > 380 level.
    ow1 = np.argwhere(xx_overworld)[0]

    # For consistency, assume that the entry level concentration that serves as the lower limit for the interpolation
    # has the same lag as we've been using to calculate the stratospheric concentrations.
    gas_entry_conc = gas_record.get_gas_for_dates(retrieval_date - gas_record.sbc_lag)

    gas_endpoints = np.array([gas_entry_conc.item(), prof_gas[ow1].item()])
    theta_endpoints = np.array([tropopause_theta, prof_theta[ow1].item()])
    xx_middleworld = (tropopause_theta < prof_theta) & (prof_theta < 380.0)
    prof_gas[xx_middleworld] = np.interp(prof_theta[xx_middleworld], theta_endpoints, gas_endpoints)
    prof_world_flag[xx_middleworld] = const.middleworld_flag

    return prof_gas, {'latency': profs_latency, 'age_of_air': prof_aoa, 'stratum': prof_world_flag}


def generate_single_tccon_prior(mod_file_data, obs_date, utc_offset, concentration_record, site_abbrev='xx',
                                use_geos_grid=True, use_eqlat_trop=True, use_eqlat_strat=True, write_map=False):
    """
    Driver function to generate the TCCON prior profiles for a single observation.

    :param mod_file_data: data from a .mod file prepared by Mod Maker. May either be a path to a .mod file, or a
     dictionary from :func:`~mod_utils.read_mod_file`.
    :type mod_file_data: str or dict

    :param obs_date: the date of the observation
    :type obs_date: datetime-like object

    :param utc_offset: a timedelta giving the difference between the ``obs_date`` and UTC time. For example, if the
     ``obs_date`` was given in US Pacific Standard Time, this should be ``timedelta(hours=-8). This is used to correct
     the date to UTC to ensure the CO2 from the right time is used.

    :param concentration_record: which species to generate the prior profile for. Must be the proper subclass of
     TraceGasTropicsRecord for the given species. The latter is useful if you are making multiple calls to this
     function, as it removes the need to instantiate the record during each call
    :type concentration_record: str or :class:`TraceGasTropicsRecord`

    :param site_abbrev: the two-letter site abbreviation. Currently only used in naming the output file.
    :type site_abbrev: str

    :param use_geos_grid: when ``True``, the native 42-level GEOS-FP pressure grid is used as the vertical grid for the
     CO2 profiles. Set to ``False`` to use a grid with 1 km altitude spacing
    :type use_geos_grid: bool

    :param use_eqlat_trop: when ``True``, the latitude used for age-of-air and seasonal cycle calculations is calculate
     based on the climatology of latitude vs. mid-tropospheric potential temperature. When ``False``, the geographic
     latitude of the observation is used.
    :type use_eqlat_trop: bool

    :param use_eqlat_strat: when ``True``, the stratosphere profiles use equivalent latitude that must be given in the
     mod data (requires the variable "EL" in the dictionary/mod file). Setting this to ``False`` uses the geographic
     latitude of the observation instead. This allows you to skip the (fairly processor intensive) equivalent latitude
     calculation when preparing the .mod files, but can lead to ~2% differences in CO2 near the tropopause (in March).
    :type use_eqlat_strat: bool

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

    z_surf = mod_file_data['scalar']['Height']
    z_met = mod_file_data['profile']['Height']
    theta_met = mod_file_data['profile']['PT']
    eq_lat_met = mod_file_data['profile']['EL'] if use_eqlat_strat else np.full_like(z_met, obs_lat)

    # We need the tropopause potential temperature. The GEOS FP-IT files give the temperature itself, and the pressure,
    # so we can calculate the potential temperature. Pressure needs to be in hPa, which it is by default.

    t_trop_met = mod_file_data['scalar']['TROPT']  # use the blended tropopause. TODO: reference why this is best?
    p_trop_met = mod_file_data['scalar']['TROPPB']
    theta_trop_met = mod_utils.calculate_potential_temperature(p_trop_met, t_trop_met)

    # The age-of-air calculation used for the tropospheric trace gas profile calculation needs the tropopause altitude.
    # Assume that potential temperature varies linearly with altitude to calculate that, use the potential temperature
    # of the tropopause to ensure consistency between the two parts of the profile.
    z_trop_met = mod_utils.interp_to_tropopause_height(theta_met, z_met, theta_trop_met)
    if z_trop_met < np.nanmin(z_met):
        raise RuntimeError('Tropopause altitude calculated to be below the bottom of the profile. Something has '
                           'gone horribly wrong.')

    if not isinstance(concentration_record, TraceGasTropicsRecord):
        raise TypeError('concentration_record must be a subclass instance of TraceGasTropicsRecord')
    elif concentration_record.gas_name == '':
        raise TypeError('concentration_record must be a specific subclass instance of TraceGasTropicsRecord that '
                        'has a non-empty gas_name attribute; it cannot be an instance of TraceGasTropicsRecord itself.')

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
    gas_prof = np.full_like(z_prof, np.nan)
    gas_date_prof = np.full_like(z_prof, np.nan)
    gas_date_width_prof = np.full_like(z_prof, np.nan)
    latency_profs = np.full((n_lev, 3), np.nan)
    stratum_flag = np.full((n_lev,), -1)

    _, ancillary_trop = add_trop_prior(gas_prof, obs_utc_date, obs_lat, z_prof, z_surf, z_trop_met, concentration_record,
                                       pres_grid=p_prof, theta_grid=theta_prof, use_theta_eqlat=use_eqlat_trop,
                                       profs_latency=latency_profs, prof_world_flag=stratum_flag,
                                       prof_gas_date=gas_date_prof, prof_gas_date_width=gas_date_width_prof)
    aoa_prof_trop = ancillary_trop['age_of_air']
    trop_ref_lat = ancillary_trop['ref_lat']
    trop_eqlat = ancillary_trop['trop_lat']

    # Next we add the stratospheric profile, including interpolation between the tropopause and 380 K potential
    # temperature (the "middleworld").
    _, ancillary_strat = add_strat_prior(gas_prof, obs_utc_date, theta_prof, eq_lat_prof, theta_trop_met,
                                         concentration_record, profs_latency=latency_profs,
                                         prof_world_flag=stratum_flag)
    aoa_prof_strat = ancillary_strat['age_of_air']

    # Finally prepare the output, writing a .map file if needed.
    gas_name = concentration_record.gas_name
    gas_unit = concentration_record.gas_unit
    map_dict = {'Height': z_prof, 'Temp': t_prof, 'Pressure': p_prof, 'PT': theta_prof, 'EL': eq_lat_prof,
                gas_name: gas_prof, 'mean_latency': latency_profs[:, 0], 'min_latency': latency_profs[:, 1],
                'max_latency': latency_profs[:, 2], 'trop_age_of_air': aoa_prof_trop,
                'strat_age_of_air': aoa_prof_strat, 'atm_stratum': stratum_flag, 'gas_date': gas_date_prof,
                'gas_date_width': gas_date_width_prof}
    units_dict = {'Height': 'km', 'Temp': 'K', 'Pressure': 'hPa', 'PT': 'K', 'EL': 'degrees', gas_name: gas_unit,
                  'mean_latency': 'yr', 'min_latency': 'yr', 'max_latency': 'yr', 'trop_age_of_air': 'yr',
                  'strat_age_of_air': 'yr', 'atm_stratum': 'flag', 'gas_date': 'yr', 'gas_date_width': 'yr'}
    var_order = ('Height', 'Temp', 'Pressure', 'PT', 'EL', gas_name, 'mean_latency', 'min_latency',
                 'max_latency', 'trop_age_of_air', 'strat_age_of_air', 'atm_stratum', 'gas_date', 'gas_date_width')
    map_constants = {'site_lat': obs_lat, 'trop_eqlat': trop_eqlat, 'prof_ref_lat': trop_ref_lat, 'surface_alt': z_surf,
                     'tropopause_alt': z_trop_met, 'strat_used_eqlat': use_eqlat_strat}
    if write_map:
        map_dir = write_map if isinstance(write_map, str) else '.'
        map_name = os.path.join(map_dir, mod_utils.map_file_name(site_abbrev, obs_lat, obs_date))
        mod_utils.write_map_file(map_name, obs_lat, trop_eqlat, trop_ref_lat, z_surf, z_trop_met, use_eqlat_strat,
                                 map_dict, units_dict, var_order=var_order)

    return map_dict, units_dict, map_constants


def generate_tccon_priors_driver(mod_data, obs_dates, utc_offsets, species, site_abbrevs='xx', write_maps=False,
                                 **prior_kwargs):

    num_profiles = max(np.size(inpt) for inpt in [mod_data, obs_dates, utc_offsets, site_abbrevs])

    def check_input(inpt, name, allowed_types):
        type_err_msg = '{} must be either a collect or single instance of one of the types: {}'.format(
            name, ', '.join(t.__name__ for t in allowed_types)
        )
        if np.ndim(inpt) > 1:
            raise ValueError('{} must be 1-dimensional'.format(name))
        elif np.ndim(inpt) == 1:
            if np.size(inpt) != num_profiles:
                raise ValueError('{} must either be a scalar or 1D with the same number of elements ({}) as '
                                 'mod_file_data, obs_date, utc_offset, and site_abbrevs'.format(name, num_profiles))
            if not isinstance(inpt[0], allowed_types):
                raise TypeError(type_err_msg)
        elif np.ndim(inpt) == 0:
            if not isinstance(inpt, allowed_types):
                raise TypeError(type_err_msg)
            return [inpt] * num_profiles
        else:
            return inpt

    # Input checking. Make sure these are the right type and either the same size as each other or a single value. In
    # the latter case, replicate it. These will have one
    mod_data = check_input(mod_data, 'mod_data', (str, dict))
    obs_dates = check_input(obs_dates, 'obs_dates', (dt.datetime, pd.Timestamp))
    utc_offsets = check_input(utc_offsets, 'utc_offsets', (dt.timedelta, pd.Timedelta))
    site_abbrevs = check_input(site_abbrevs, 'site_abbrevs', (str,))

    # species will each be generated for every site.
    if isinstance(species, (str, TraceGasTropicsRecord)):
        species = [species]

    # if given species names, convert to the actual records.
    species = [_gas_records[s]() if isinstance(s, str) else s for s in species]

    if write_maps:
        maps_dir = write_maps
        write_maps = True
    else:
        maps_dir = ''

    # MAIN LOOP #
    # Loop over the requested profiles, creating a prior for each gas requested. Check that the other variables are all
    # the same for each gas, then combine them to make a single .map file or dict for each profile
    ancillary_variables = ('Height', 'Temp', 'Pressure', 'PT', 'EL')
    for iprofile in range(num_profiles):
        var_order = list(ancillary_variables)
        for ispecie, specie_record in enumerate(species):
            var_order.append(specie_record.gas_name)
            specie_profile, specie_units, specie_constants = \
                generate_single_tccon_prior(mod_data[iprofile], obs_dates[iprofile], utc_offsets[iprofile],
                                            specie_record, site_abbrev=site_abbrevs[iprofile], write_map=False,
                                            **prior_kwargs)

            if ispecie == 0:
                profile_dict = specie_profile
                units_dict = specie_units
                map_constants = specie_constants
            else:
                for ancvar in ancillary_variables:
                    if not np.allclose(specie_profile[ancvar], profile_dict[ancvar], equal_nan=True):
                        raise RuntimeError('Got different vectors for {} for difference species'.format(ancvar))

                # All good? Add the current specie concentration to the dicts
                gas_name = specie_record.gas_name
                profile_dict[gas_name] = specie_profile[gas_name]
                units_dict[gas_name] = specie_units[gas_name]

        # Write the combined .map file for all the requested species
        if write_maps:
            map_name = os.path.join(maps_dir, mod_utils.map_file_name(site_abbrevs[iprofile], ))
            mod_utils.write_map_file(map_name, variables=profile_dict, units=units_dict, var_order=var_order,
                                     **map_constants)


###########################################
# FUNCTIONS FOR GENERATING GRIDDED PRIORS #
###########################################

def _tonumpyarray(mparr, shape):
    return np.frombuffer(mparr).reshape(shape)


def prior_par_wrapper(i):
    mem_arr = shared_info_dict['shared_array']
    co2 = np.frombuffer(mem_arr)
    ntimes = shared_info_dict['ntimes']
    nlev = shared_info_dict['nlev']
    nlon = shared_info_dict['nlon']
    nlat = shared_info_dict['nlat']

    co2 = co2.reshape(ntimes, nlev, nlat, nlon)
    prior_wrapper(i, co2, **shared_info_dict)


def prior_wrapper(i, my_co2, ntimes, nlat, nlon, geos_prof_data, geos_surf_data, geos_dates, prior_kwargs, **kwargs):
    gc2mod_name_map = {'T': 'Temperature', 'H': 'Height'}
    itime, ilat, ilon = np.unravel_index(i, (ntimes, nlat, nlon))
    print('Generating prior at t = {t}, y = {y}, x = {x}'.format(t=itime, y=ilat, x=ilon))
    mod_dict = dict()
    mod_dict['profile'] = {'Pressure': geos_prof_data['lev']}
    for var_name, var_arr in geos_prof_data.items():
        mod_name = gc2mod_name_map[var_name] if var_name in gc2mod_name_map else var_name
        if np.ndim(var_arr) > 1:
            mod_dict['profile'][mod_name] = var_arr[itime, :, ilat, ilon]

    mod_dict['scalar'] = dict()
    for var_name, var_arr in geos_surf_data.items():
        mod_name = gc2mod_name_map[var_name] if var_name in gc2mod_name_map else var_name
        if np.ndim(var_arr) > 1:
            mod_dict['scalar'][mod_name] = var_arr[itime, ilat, ilon]

    mod_dict['constants'] = {'obs_lat': geos_prof_data['lat'][ilat]}

    map_dict, _ = generate_single_tccon_prior(mod_dict, geos_dates[itime], dt.timedelta(hours=0), **prior_kwargs)
    my_co2[itime, :, ilat, ilon] = map_dict['co2']


def save_gridded_priors(save_name, co2, geos_prof_data, geos_dates):
    with ncdf.Dataset(save_name, 'w') as nch:
        londim = ioutils.make_ncdim_helper(nch, 'lon', geos_prof_data['lon'], units='degrees_east')
        latdim = ioutils.make_ncdim_helper(nch, 'lat', geos_prof_data['lat'], units='degrees_north')
        levdim = ioutils.make_ncdim_helper(nch, 'lev', geos_prof_data['lev'], units='hPa')
        timedim, _ = ioutils.make_nctimedim_helper(nch, 'time', geos_dates)
        ioutils.make_ncvar_helper(nch, 'co2', co2, (timedim, levdim, latdim, londim),
                                  units='dry air mole fraction * 10^-6',
                                  description='CO2 profiles calculated using the TCCON prior algorithm')
        ioutils.make_ncvar_helper(nch, 'eqlat', geos_prof_data['EL'], (timedim, levdim, latdim, londim),
                                  units='degrees_north',
                                  description='Equivalent latitude calculated from Ertels Potential Vorticity')


def generate_gridded_co2_priors(start_date, end_date, geos_path, save_name=None, met_type='geos-fpit', use_eqlat_strat=True,
                                n_procs=0, **prior_kwargs):
    run_parallel = n_procs > 0
    # First, read in the met data, calculating equivalent latitude if necessary

    prof_vars = ['T', 'H'] + (['EPV'] if use_eqlat_strat else [])
    surf_vars = ['TROPPB', 'TROPT', 'PS']
    if met_type.startswith('geos'):
        geos_product = met_type.split('-')[1]
        geos_prof_data, geos_surf_data, geos_dates = mod_utils.read_geos_files(
            start_date, end_date, geos_path, prof_vars, surf_vars, product=geos_product, concatenate_arrays=True,
            set_mask_to_nan=True
        )
        # Height in GEOS is in meters, want kilometers
        geos_prof_data['H'] *= 0.001
        # Tropopause pressure in Pa, want hPa
        geos_surf_data['TROPPB'] *= 0.01
    else:
        raise ValueError('met_type "{}" is not recognized'.format(met_type))

    # Second, calculate derived quantities (theta and, if necessary, equivalent latitude)
    geos_prof_data['PT'] = mod_utils.calculate_model_potential_temperature(geos_prof_data['T'],
                                                                           pres_levels=geos_prof_data['lev'])

    if use_eqlat_strat:
        area = mod_utils.calculate_area(geos_prof_data['lat'], geos_prof_data['lon'])
        geos_prof_data['EL'] = mod_utils.calculate_eq_lat_on_grid(geos_prof_data['EPV']*1e6, geos_prof_data['PT'], area)
    else:
        geos_prof_data['EL'] = np.full_like(geos_prof_data['PT'], np.nan)

    # Third, pass each column to as a mod file-like dictionary, passing it to generate_single_tccon_prior, and storing the
    # result in a CO2 array.
    ntimes, nlev, nlat, nlon = geos_prof_data['H'].shape
    prior_kwargs.update({'use_eqlat_strat': use_eqlat_strat, 'write_map': False, 'use_geos_grid': True})

    co2_size = geos_prof_data['H'].size
    co2_shape = geos_prof_data['H'].shape
    n_columns = ntimes * nlat * nlon

    # Must contain all the keywords args required by prior_wrapper except i and my_co2.
    shared_info = {'ntimes': ntimes, 'nlev': nlev, 'nlat': nlat, 'nlon': nlon, 'geos_prof_data': geos_prof_data,
                   'geos_surf_data': geos_surf_data, 'geos_dates': geos_dates, 'prior_kwargs': prior_kwargs}

    flat_indices = np.arange(n_columns)

    if run_parallel:
        # Parallelization heavily inspired by https://stackoverflow.com/a/7908612
        def par_init(shared_info_dict_):
            # This may seem backward (assigning the input of the function to the global variable), but the way that
            # multiprocessing works is that each worker will essentially execute this file itself to get all the
            # function definitions and such, but it does not get the state of the variables from the parent process.
            # Essentially its a completely new python process.
            #
            # Therefore, to pass variables from the parent process to the workers, this function gets called on each of
            # the workers, receiving the info dict from the parent. It then assigns it as a global variable so that the
            # workers can access it inside its assigned function (here prior_par_wrapper).
            #
            # This is probably wasteful of memory because I suspect that the entirety of shared_info_dict gets
            # duplicated by each worker. It would be more efficient to use shared arrays for all of the GEOS data, but
            # unless memory usage is a problem, this approach is *far* simpler.
            global shared_info_dict
            shared_info_dict = shared_info_dict_

        # To get the data back from the workers, create a shared array in memory. We need to convert it back to a shaped
        # numpy array to do anything with it. We can use a RawArray because we don't need to lock the variable (to
        # prevent multiple processes from accessing elements at the same time) because each worker has its own slice of
        # the co2 array to operate on, and there should be no conflicts.
        shared_array = mp.RawArray(ctypes.c_double, co2_size)
        shared_info['shared_array'] = shared_array
        co2 = _tonumpyarray(shared_array, co2_shape)
        co2[:] = np.nan

        # TODO: add kill message? https://stackoverflow.com/questions/29571671/basic-multiprocessing-with-while-loop
        with closing(mp.Pool(processes=n_procs, initializer=par_init, initargs=(shared_info,))) as p:
            p.map_async(prior_par_wrapper, flat_indices)

        p.join()
    else:
        co2 = np.full(co2_shape, np.nan)
        pbar = mod_utils.ProgressBar(n_columns, prefix='Calculating CO2 priors', style='counter')
        for i in flat_indices:
            pbar.print_bar(i)
            prior_wrapper(i, co2, **shared_info)

    if save_name is not None:
        save_gridded_priors(save_name, co2, geos_prof_data, geos_dates)

    return co2
