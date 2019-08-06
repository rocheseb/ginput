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
* In the stratosphere, concentrations are calculated as the convolution of an age spectrum with the two month-lagged
  MLO/SMO record. That is, for different mean ages of air, there are defined spectra that provide the contribution of
  different ages to that air parcel. These can also be thought of as a probability distribution of age of air in a
  given parcel. The trace gas concentration for a given age and date is the product of the age spectrum and the trace
  gas record. A. Andrews derived different age spectra for tropics, midlatitudes, and polar vortex, therefore each
  level of the profile must be classified into one of these three regions by latitude, day of year, and age.
* In the middle world (above the tropopause & theta < 380 K), the profiles are interpolate with respect to theta between
  the tropopause and the first overworld level.

The stratospheric approach was developed by Arlyn Andrews, based on her research in Andrews et al. 2001 (JGR, 106 [D23],
pp. 32295-32314). For gases other than CO2, chemical loss or production in the stratosphere must be accounted for.

* For N2O, the relationship of N2O vs. age of air from Andrews et al. (2001) was recast as fraction of N2O remaining vs.
  age, which allows us to use the MLO/SMO record directly rather than apply a growth factor. That is, in A. Andrews
  code, she uses a function to calculate the N2O concentration based on age from a relationship derived in the 1990s,
  then adds a growth factor to account for increase in N2O concentration since the 1990s. We instead use MLO/SMO to
  get the stratospheric boundary condition, then multiply by the fraction remaining vs. age to get the actual
  concentration. This allows us to use the MLO/SMO record to get the growth, rather than having to calculate a growth
  rate separately.
* For CH4, we use ACE-FTS data to derive a CH4-N2O relationship. Again, this is in terms of fraction remaining for both
  CH4 and N2O.  Therefore for a given age, we find the fraction of N2O remaining and then use the ACE-FTS relationship
  to convert that to a fraction of CH4 remaining. Since the F(CH4):F(N2O) relationship varies with potential
  temperature, the CH4 lookup table includes a theta dependence.
* For HF, we use relationships between HF and CH4 concentration derived in Saad et al. (2014,
  doi: 10.5194/amt-7-2907-2014) and Washenfelder et al. (2003, doi: 10.1029/2003GL017969). These papers derive a slope
  of CH4 concentration vs. HF concentration. Saad et al. find variations in the slope with latitude; for consistency
  with the other gases, we use the tropics/midlatitudes/polar vortex regions rather than the latitude bins defined in
  Saad et al. We rederive our own slopes during the ACE-FTS era (from ~2004) for these three bins, then prepend the
  slopes back to ~1977 from Washenfelder et al. and fit the slope vs. time with an exponential. The HF code is set up
  to use the ACE-FTS derived slopes directly in years where they are available and slopes derived from the exponential
  fit outside the ACE window. The HF concentrations are calculated from CH4 concentrations assuming a linear
  relationship where the y-intercept is assumed to be the normal two month-lagged MLO/SMO stratospheric boundary
  condition and the slope is that precomputed from the ACE-FTS data or Washenfelder et al. Unlike N2O and CH4, fraction
  remaining was not used since the Washenfelder et al. results make clear the CH4/HF relationship changes substantially
  with time and we do not have ACE-FTS data to derive the F(CH4)/F(HF) relationship before 2004, after which the
  change in slope vs. time is essentially flat.

:func:`generate_single_tccon_prior` is the primary function to use to create a single prior. It is used for all gases
"""

from __future__ import print_function, division

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
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
import xarray as xr

from ..common_utils import mod_utils, ioutils, mod_constants as const
from ..common_utils.ggg_logging import logger

# _code_dep_modules should list any imported modules that you want to check if they've changed when decided whether to
# recalculate the strat LUTs. This module will be added on its own after. _code_dep_files should always be generated
# from _code_dep_modules; the latter is what is actually used to calculate the dependencies.
_code_dep_modules = (mod_utils, ioutils, const)
_code_dep_files = {f.__name__.split('.')[-1] + '_sha1': os.path.abspath(f.__file__) for f in _code_dep_modules}
# Add this module. Make sure to avoid storing the name as "__main__" so just always use the file name minus the
# extension
_code_dep_files[os.path.splitext(os.path.basename(__file__))[0]] = os.path.abspath(__file__)

_data_dir = const.data_dir
_clams_file = os.path.join(_data_dir, 'clams_age_clim_scaled.nc')
_excess_co_file = os.path.join(_data_dir, 'meso_co_lut.nc')

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


class GasRecordInconsistentDimsError(GasRecordError):
    """
    Error when arrays in a gas record have different dimensions and are not supposed to
    """
    pass


class GasRecordInputVerificationError(GasRecordError):
    """
    Error when the input could not be verified (i.e. if hashes don't match)
    """
    pass


class GasRecordExtrapolationError(GasRecordError):
    """
    Error to raise if there's a problem with the extrapolation of the MLO/SMO records.
    """
    pass


class GasRecordDateError(GasRecordError):
    """
    Error to raise for any issues with dates in the gas records
    """
    pass


class GGGPathError(Exception):
    pass


def _init_prof(profs, n_lev, n_profs=0, fill_val=np.nan):
    """
    Initialize arrays for various profiles.

    :param profs: input profiles or None.
    :param n_lev: the number of levels in the profiles.
    :type n_lev: int

    :param n_profs: how many profiles to make in the returned array. If 0, then an n_lev element vector is returned, if
     >0, and n_lev-by-n_profs array
    :type n_profs: int

    :param fill_val: what value to initialize the array to if necessary. Implicitly sets the data type.
    :type fill_val: Any

    :return: the initialized array, either the same profiles as given, or a new array initialized with NaNs with the
     required shape if ``profs`` is ``None``.
    :rtype: :class:`numpy.ndarray`
    :raises ValueError: if the profiles were given (not ``None``) and have the wrong shape.
    """
    if n_profs < 0:
        raise ValueError('n_profs must be >= 0')
    if profs is None:
        size = (n_lev, n_profs) if n_profs > 0 else (n_lev,)
        return np.full(size, fill_val)
    else:
        target_shape = (n_lev,) if n_profs == 0 else (n_lev, n_profs)
        if profs.shape != target_shape:
            raise ValueError('Given profile do not have the expected shape! Expected: {}, actual: {}'.format(
                target_shape, profs.shape
            ))
        return profs


class TraceGasRecord(object):
    # these should be overridden in subclasses to specify the name and unit of the gas. The name will be used by the
    # seasonal cycle function to determine if it uses the CO2 parameterization or the default one, and the seasonal
    # cycle coefficient scales the seasonal cycle. If the coefficient is None, the seasonal cycle function will raise
    # an error.
    _gas_name = ''
    _gas_unit = ''
    _gas_seas_cyc_coeff = None
    _gas_sec_trend = None
    _gas_lat_grad = None

    @property
    def gas_name(self):
        return self._gas_name

    @property
    def gas_unit(self):
        return self._gas_unit
    @property
    def gas_seas_cyc_coeff(self):
        return self._gas_seas_cyc_coeff

    @property
    def gas_sec_trend(self):
        return self._gas_sec_trend

    @property
    def gas_lat_grad(self):
        return self._gas_lat_grad

    @abstractmethod
    def add_trop_prior(self, prof_gas, obs_date, obs_lat, mod_data, **kwargs):
        pass

    @abstractmethod
    def add_strat_prior(self, prof_gas, retrieval_date, mod_data, **kwargs):
        pass

    @abstractmethod
    def add_extra_column(self, prof_gas, retrieval_date, mod_data, **kwargs):
        """
        Add a representation of out-of-range column density to the prior

        This method is intended for use to handle cases like CO, which has a fairly large mesospheric column that can't
        be directly represented in the prior. It will add extra concentration to one or more levels in the prior such
        that, when integrated, the extra column density is accounted for. Ideally, this should be called after
        interpolating to the final levels that the priors will be used on in GGG so that the column can be reproduced
        exactly by the integration.

        :param prof_gas: the profile to modify. Modified in-place
        :type prof_gas: :class:`numpy.ndarray`

        :param retrieval_date: the date/time of the prior profile (unused, present for consistency)

        :param mod_data: the dictionary of model data read in from the .mod file.
        :type mod_data: dict

        :param kwargs: unused, swallows extra keyword arguments.

        :return: the modified gas profile and a dictionary on ancillary information (currently empty).
        """
        pass


class MloSmoTraceGasRecord(TraceGasRecord):
    """
    This class stores the Mauna Loa/Samoa average CO2 record and provides methods to sample it.

    Initialization arguments:

    :param first_date: optional, the first date required in the concentration records. The actual first date will be
     before this, as the age spectra calculation require ~30 years of data preceding each date, therefore the simple
     time series records will be extended to ``first_date`` - 30 years. If not given, then 1 Jan 2000 is assumed
     (meaning the actual first date will be 1 Jan 1970). The date will always be moved to the first of a month.
    :type first_date: datetime-like

    :param last_date: optional, the last date required in the concentration records. The date will always be moved to
     the first of a month. If omitted, a date two years from today is used. Unlike ``first_date``, there is no
     modification to account for the needs of the age spectra.
    :type last_date: datetime-like

    :param lag: optional, the lag between Mauna Loa/Samoa measurements and the stratospheric boundary condition. Default
     is two months, i.e. the stratospheric boundary condition for a given date is assumed to be that measured at MLO/SMO
     two months previously.
    :type lag: timedelta-like

    :param mlo_file: optional, the path to the Mauna Loa flask data file. Must be formatted as a NOAA monthly flask data
     file, where the first line is "# f_header_lines: n" (n being the number of header lines) and the data being
     organized in four columns (space separated): site, year, month, value.
    :type mlo_file: str

    :param smo_file: optional, the path to the Samoa flask data file. Same format as the MLO file required.
    :type smo_file: str

    :param recalculate_strat_lut: optional, set to ``True`` to force the stratospheric concentrations look up table
     to be recalculated or ``False`` to always use the existing lookup table if it exists. Default is ``None``, which
     will check if any of the files that the LUT depends on have changed, and recalculate it if so.
    :type recalculate_strat_lut: bool or None

    :param save_strat: optional, set to ``False`` to avoid saving the stratospheric concentration lookup if it is
     recalculated. Default is ``None``, which will save the LUT if recalculated unless it was recalculated to cover the
     time frame requested. This option has no effect if the stratospheric lookup table is read from the netCDF file.
    :type save_strat: bool or None
    """

    # The lifetime is used to account for chemical loss between emission and the prior location. Setting to infinity
    # will effectively disable this correction, since e^(-x/infinity) = 1.0.
    gas_trop_lifetime_yrs = np.inf
    # The latitudinal corrections apply to the troposphere only. Give the northern and southern hemisphere correction
    # as a dictionary with values in ppx/degree, the unit "ppx" will be ppm, ppb, etc., whatever the base record is in.
    gas_lat_correction = {'nh': 0.0, 'sh': 0.0}

    months_avg_for_trend = 12
    age_spec_regions = ('tropics', 'midlat', 'vortex')
    _age_spectrum_data_dir = os.path.join(_data_dir, 'age_spectra')
    _age_spectrum_time_file = os.path.join(_age_spectrum_data_dir, 'time.txt')
    _default_sbc_lag = relativedelta(months=2)

    # This sets the maximum degree of the polynomial used to fit and extend the MLO/SMO trends. Set to 1 to use linear,
    # 2 for quadratic, etc. One special case: 'exp' will use an exponential, rather than polynomial, fit.
    _max_trend_poly_deg = 2
    _nyears_for_extrap_avg = 10
    _max_safe_extrap_forward = relativedelta(years=5)

    # coordinate to use for the stratospheric look up table along the theta dimension when there is no theta dependence
    _no_theta_coord = np.array([0.])

    # Sets an assumption for how long the age spectra go back. Currently (2019-06-27) only used to modify the start date
    # to ensure that there is strat data over the requested time.
    _age_spectra_length = relativedelta(years=30)

    @classmethod
    def get_strat_lut_file(cls):
        # This needed to be a class method so that the _load_strat_arrays() could be a classmethod. Unfortunately,
        # classproperties aren't a thing yet, so this remains a regular function.
        return os.path.join(_data_dir, '{}_strat_lut.nc'.format(cls._gas_name))

    @property
    def strat_has_theta_dep(self):
        no_dep = [np.all(reg_arr.theta == self._no_theta_coord).item() for reg_arr in self.conc_strat.values()]
        if all(no_dep):
            return False
        elif not any(no_dep):
            return True
        else:
            raise GasRecordInconsistentDimsError('Some regions have a theta dependence in their stratospheric lookup '
                                                 'table, some do not. This is not expected.')

    @property
    def first_record_date(self):
        return self._first_record_date(self.conc_seasonal)

    @property
    def last_record_date(self):
        # interp_flag 0 = read directly, 1 = interpolated, 2 = extrapolated. We want the latest date that wasn't
        # extrapolated.
        return self._last_record_date(self.conc_seasonal)

    def __init__(self, first_date=None, last_date=None, lag=None, mlo_file=None, smo_file=None,
                 strat_age_scale=1.0, recalculate_strat_lut=None, save_strat=None):
        has_custom_dates = first_date is not None or last_date is not None
        first_date, last_date, self.sbc_lag, mlo_file, smo_file = self._init_helper(first_date, last_date, lag, mlo_file, smo_file)
        self.mlo_file = mlo_file
        self.smo_file = smo_file
        self.strat_age_scale = strat_age_scale
        self.conc_seasonal = self.get_mlo_smo_mean(mlo_file, smo_file, first_date, last_date)

        # Deseasonalize the data by taking a 12 month rolling average. Only do that on the dmf_mean field,
        # leave the latency
        self.conc_trend = self.conc_seasonal.rolling(self.months_avg_for_trend, center=True).mean().dropna().drop('interp_flag', axis=1)

        # For the stratosphere, we need a lookup table that contains concentrations for given dates and ages. (Some
        # species may depend on additional variables, such as potential temperature.) This calculation involves
        # convolving the age spectra with the concentration record, so can be quite time consuming. To speed things up,
        # we usually load this table from a netCDF file, but if the prior code or the MLO/SMO files update, we'll need
        # to regenerate that lookup table and save it again.
        if not os.path.isfile(self.get_strat_lut_file()):
            # File does not exist, would have to recalculate no matter what the request was.
            recalculate_strat_lut = True
            logger.important('Strat LUT file ({}) does not exist, must recompute'.format(self.get_strat_lut_file()))
        elif recalculate_strat_lut is None:
            # Determine if the dependencies have changed, if so, recalculate.
            recalculate_strat_lut = self._have_strat_array_deps_changed()
            if recalculate_strat_lut:
                logger.important('Strat LUT dependencies have changed, recalculating')
            else:
                logger.important('Strat LUT dependencies unchanged; loading previous table')
        elif recalculate_strat_lut:
            # If told to force recalculation, do so
            logger.important('Recalculating strat LUT as requested')
        else:
            # recalculate_strat_lut must have been false to get here, therefore we should not recalculate
            # the LUT
            logger.important('Using existing strat LUT as requested')

        if not recalculate_strat_lut:
            # If we're loading the strat lookup table, we need to pass the MLO and SMO files to check their SHA1 hashes
            # against those stored in the netCDF file to ensure the MLO/SMO files are the same ones that were used to
            # generate the strat table stored in the netCDF file.
            logger.info('Loading {} strat LUT'.format(self.gas_name))
            self.conc_strat = self._load_strat_arrays()
            if not self._check_strat_dates(self.conc_strat, first_date, last_date):
                logger.important('Cached strat LUT does not span enough time to cover the requested dates, must '
                                 'recalculate.')
                recalculate_strat_lut = True
                if has_custom_dates:
                    # If custom dates were specified, then only save the strat LUT if the user explicitly requested it
                    save_strat = False if save_strat is None else save_strat
                else:
                    # If custom dates were not specified, then resave the LUT because it probably means we had to extend
                    # into a new month
                    save_strat = True

        if save_strat is None:
            save_strat = True

        # May enter this if recalculation required by user, dependencies, or dates.
        if recalculate_strat_lut:
            logger.info('Calculating {} strat LUT'.format(self.gas_name))
            self.conc_strat = self._calc_age_spec_gas(self.conc_seasonal, lag=self.sbc_lag)
            if save_strat:
                try:
                    self._save_strat_arrays()
                except PermissionError:
                    logger.important('Could not save strat LUT file due to permission error. This just means it will '
                                     'need recalculated the next time this record is loaded')
                else:

                    logger.important('Saved {} strat LUT file as "{}"'.format(self.gas_name, os.path.abspath(self.get_strat_lut_file())))

    @classmethod
    def _init_helper(cls, first_date, last_date, lag, mlo_file, smo_file):
        # For the stratosphere data, since the age spectra are defined over a 30 year window, we need to make sure
        # we have values back to slightly more than 30 years before the first TCCON data. Assuming that's around 2004,
        # a default age of 2000 - 30 = 1970 should be good.

        if first_date is None:
            first_date = dt.datetime(2000, 1, 1) - cls._age_spectra_length
        else:
            first_date -= cls._age_spectra_length

        if last_date is None:
            # By default, we want to extrapolate to 2 years out from today; the max negative age-of-air in the
            # troposphere should be about 6 months at most, but we need to allow some room for the rolling average to
            # get the trend. We need to make sure that we extend the record by whole months so go two years, one month
            # into the future. E.g. if in July 2019, then end date will be Aug 2021. This helps with caching the strat
            # LUT.
            last_date = mod_utils.start_of_month(dt.datetime.today()) + relativedelta(years=2, months=1)

        if lag is None:
            sbc_lag = cls._default_sbc_lag
        else:
            sbc_lag = lag

        files_none = [mlo_file is None, smo_file is None]
        if all(files_none):
            mlo_file = os.path.join(_data_dir, 'ML_monthly_obs_{}.txt'.format(cls._gas_name))
            smo_file = os.path.join(_data_dir, 'SMO_monthly_obs_{}.txt'.format(cls._gas_name))
        elif any(files_none):
            raise TypeError('Must give both MLO and SMO files or neither')

        return first_date, last_date, sbc_lag, mlo_file, smo_file

    @classmethod
    def _check_strat_dates(cls, strat_lut, first_date, last_date):
        strat_first_date = strat_lut['tropics'].coords['date'].min()
        strat_last_date = strat_lut['tropics'].coords['date'].max()

        for region, lut in strat_lut.items():
            if lut.coords['date'].min() != strat_first_date or lut.coords['date'].max() != strat_last_date:
                raise GasRecordDateError('"{}" strat LUT has different start/end dates than "tropics" LUT'.format(region))

        if strat_first_date > np.datetime64(first_date) or strat_last_date < np.datetime64(last_date)   :
            return False
        else:
            return True

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
    def get_frac_remaining_by_age(cls, ages):
        """
        Get the fraction of a gas remaining for a given vector of ages.

        The default is to assume no loss, and so 1 will be returned for every age. Subclasses
        may override this method to calculate more complicated relationships between age and fraction remaining.

        :param ages: the vector of ages to calculate the fraction of the gas remaining for
        :type ages: :class:`numpy.ndarray` or float

        :return: a data frame indexed by age with one column, "fraction" containing the fraction remaining.
        :rtype: :class:`pandas.DataFrame`
        """
        n_ages = np.size(ages)
        arr = xr.DataArray(data=np.ones([n_ages, 1], dtype=np.float), coords=[('age', ages), ('theta', cls._no_theta_coord)])
        return arr

    def get_latency_by_date(self, dates):
        return self.calc_latency(dates, self.first_record_date, self.last_record_date)

    @staticmethod
    def calc_latency(dates, first_record_date, last_record_date):
        if np.ndim(dates) != 1:
            raise ValueError('dates must have exactly 1 dimension')

        earlier_timedeltas = dates - first_record_date
        later_timedeltas = dates - last_record_date
        earlier_latency = np.array([mod_utils.timedelta_to_frac_year(td) for td in earlier_timedeltas if td < dt.timedelta(0)])
        later_latency = np.array([mod_utils.timedelta_to_frac_year(td) for td in later_timedeltas if td > dt.timedelta(0)])

        latency = np.zeros_like(dates, dtype=np.float)
        latency[dates < first_record_date] = earlier_latency
        latency[dates > last_record_date] = later_latency
        return latency

    @staticmethod
    def _first_record_date(df):
        xx = df.interp_flag < 2
        return df[xx].index.min()

    @staticmethod
    def _last_record_date(df):
        xx = df.interp_flag < 2
        return df[xx].index.max()

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
    def read_insitu_gas(cls, full_file_path):
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

        with open(full_file_path, 'r') as f:
            hlines = f.readline().rstrip().split(': ')[1]

        df = pd.read_csv(full_file_path, skiprows=int(hlines), skipinitialspace=True,
                         delimiter=' ', header=None, names=['site', 'year', 'month', cls._gas_name])

        # set datetime index in df (requires 'day' column)
        df['day'] = 1
        df.set_index(pd.to_datetime(df[['year', 'month', 'day']]), inplace=True)

        return df

    @classmethod
    def get_mlo_smo_mean(cls, mlo_file, smo_file, first_date, last_date):
        """
        Generate the Mauna Loa/Samoa mean trace gas record from the files stored in this repository.

        Reads in the given Mauna Loa and Samoa record files, averages then, fills in missing values by interpolation,
        extrapolates as needed to provide the full record requested, and returns the result.

        :param mlo_file: the name (not the full path) of the Mauna Loa flask data file that is included in the repo
         data directory.
        :type mlo_file: str

        :param smo_file: the name (not the full path) of the Samoa flask data file that is included in the repo
         data directory.
        :type smo_file: str

        :param first_date: the earliest date to use in the record. Note that the actual first date will always be the
         first of the month for this date.
        :type first_date: datetime-like

        :param last_date: the latest date to include in the record. Note that if it is not the first of the month, then
         the actual latest date used would be the next first of the month to follow this date. I.e. if this is June
         15th, then July 1st would be used instead.
        :type last_date: datetime-like

        :return: the data frame containing the mean trace gas concentration ('dmf_mean'), a flag ('interp_flag') set
         to 1 for any months that had to be interpolated and 2 for months that had to be extrapolated, and the latency
         ('latency') in years that a concentration had to be extrapolated. Index by timestamp.

        :return: the data frame containing the mean trace gas concentration ('dmf_mean') and a flag ('interp_flag') set
         to 1 for any months that had to be interpolated. Index by timestamp.
        :rtype: :class:`pandas.DataFrame`
        """
        df_mlo = cls.read_insitu_gas(mlo_file)
        df_smo = cls.read_insitu_gas(smo_file)
        df_combined = pd.concat([df_mlo, df_smo], axis=1).dropna()
        df_combined = pd.DataFrame(df_combined[cls._gas_name].mean(axis=1), columns=['dmf_mean'])

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

        cls._extend_mlo_smo_mean(df_combined, 'forward')
        cls._extend_mlo_smo_mean(df_combined, 'backward')

        df_combined['latency'] = cls.calc_latency(df_combined.index, cls._first_record_date(df_combined), cls._last_record_date(df_combined))

        return df_combined

    def add_trop_prior(self, prof_gas, obs_date, obs_lat, mod_data, **kwargs):
        """
        Add the tropospheric component of the prior.

        See the help for :func:`add_trop_prior_standard` in this module. All the inputs and outputs are the same except
        that ``gas_record`` will be given this instance.
        """
        return add_trop_prior_standard(gas_record=self, prof_gas=prof_gas, obs_date=obs_date, obs_lat=obs_lat,
                                       mod_data=mod_data, **kwargs)

    def add_strat_prior(self, prof_gas, retrieval_date, mod_data, **kwargs):
        """
        Add the tropospheric component of the prior.

        See the help for :func:`add_strat_prior_standard` in this module. All the inputs and outputs are the same except
        that ``gas_record`` will be given this instance.
        """
        return add_strat_prior_standard(gas_record=self, prof_gas=prof_gas, retrieval_date=retrieval_date,
                                        mod_data=mod_data, **kwargs)

    def add_extra_column(self, prof_gas, retrieval_date, mod_data, **kwargs):

        return prof_gas, dict()

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

    @classmethod
    def _extend_mlo_smo_mean(cls, df, direction, nyears=None, fit_type=None):
        nyears = cls._nyears_for_extrap_avg if nyears is None else nyears

        xx = np.flatnonzero(df.interp_flag < 2)
        avg_period = relativedelta(years=nyears, months=-1)
        if direction == 'forward':
            last_date = df.index[xx[-1]]
            first_date = last_date - avg_period
            to_fill = slice(xx[-1]+1, None)
        elif direction == 'backward':
            first_date = df.index[xx[0]]
            last_date = first_date + avg_period
            to_fill = slice(None, xx[0])
        else:
            raise ValueError('direction must be "forward" or "backward"')

        # Find the first non-extrapolated points, get the next or previous n years, detrend, and average them
        dmf = df.dmf_mean[first_date:last_date]

        # Fit with a quadratic to allow for nonlinear increase
        x = dmf.index.to_julian_date().to_numpy()
        y = dmf.to_numpy()
        fit = cls._fit_gas_trend(x, y, fit_type=fit_type)

        avg_dmf_detrended = np.nanmean((y - fit(x)).reshape(-1, 12), axis=0)
        months = dmf.index.month.to_numpy().reshape(-1, 12)
        if not np.isclose(months - months[0, :], 0).all():
            raise RuntimeError('Something went wrong while extending the MLO/SMO record - the months array was not '
                               'reshaped to have the same month in every row of one column.')
        months = months[0, :]

        # Now, avg_dmf_detrended is the different in concentration from the quadratic fit. So, for every point to be
        # extrapolated, find out what the fit thinks the concentration should be then add the seasonal offset for that
        # month
        delta_dmf_by_month = pd.DataFrame({'dmf_mean': avg_dmf_detrended}, index=months)
        extrap_dates = df.index[to_fill]
        delta_dmf = delta_dmf_by_month.reindex(extrap_dates.month).to_numpy().squeeze()
        extrap_julian_dates = extrap_dates.to_julian_date().to_numpy()

        cls._check_extrap_fit(df, fit, extrap_julian_dates, direction)

        df.loc[to_fill, 'dmf_mean'] = fit(extrap_julian_dates) + delta_dmf

    @classmethod
    def _fit_gas_trend(cls, x, y, fit_type=None):
        """
        Creating a fitting function for the gas record.

        :param x: dates for the concentrations in y, expressed in a numeric format.  These are typically expressed in
         Julian days (i.e. days since noon, Jan 1 4713 BC, c.f. documentation for :func:`pandas.Timestamp.to_julian_date`)
         but this function can accept any numerical representation. Note that this function in subclasses *may* expect
         julian days in order to fit reliably.
        :type x: :class:`numpy.ndarray`

        :param y: concentrations corresponding to the dates in x.
        :type y: :class:`numpy.ndarray`

        :return:
        """
        fit_type = cls._max_trend_poly_deg if fit_type is None else fit_type
        if fit_type == 'exp':
            logger.debug('Using exponential fit to extrapolate {}'.format(cls._gas_name))
            fit = np.polynomial.polynomial.Polynomial.fit(x, np.log(y), 1, w=np.sqrt(y))
            return lambda t: np.exp(fit(t))

        else:
            logger.debug('Using order {} polynomial to extrapolate {}'.format(fit_type, cls._gas_name))
            fit = np.polynomial.polynomial.Polynomial.fit(x, y, deg=fit_type)
            return fit

    @classmethod
    def _check_extrap_fit(cls, df, fit, extrap_julian_dates, direction, fit_type=None):
        """
        Verify that any extrapolation of the MLO/SMO record was successful.

        This method will be called just before extrapolating values in _extend_mlo_smo_mean. It should be overridden in
        any child classes that want to verify the extrapolation. If there is a problem, the method should raise a
        :class:`GasRecordExtrapolationError`.

        The default implementation checks that the fit is positive for all extrapolated dates and issues a warning if
        the program is trying to implement farther into the future than is advisable.

        :param fit: the numpy polynomial fit that will be used to extend the record.
        :type fit: :class:`numpy.polynomial.polynomial.Polynomial`

        :param extrap_julian_dates: Julian dates that will be extrapolated to. Julian date in this case refers to the
         dates returned by the ``to_julian_date`` method on :class:`pandas.DatetimeIndex` objects.
        :type extrap_julian_dates: array-like

        :param direction: string indicating whether extrapolation is going "forward" or "backward"
        :type direction: str

        :return: None
        :raises GasRecordExtrapolationError: if it detects any problem with the extrapolation.
        """

        if np.any(fit(extrap_julian_dates) < 0):
            raise GasRecordExtrapolationError('Extrapolation for {} would result in negative concentrations'.format(cls._gas_name))

        xx = df.interp_flag < 2
        last_date = df.index[xx].max()
        last_safe_jdate = (last_date + cls._max_safe_extrap_forward).to_julian_date()
        if np.any(extrap_julian_dates > last_safe_jdate):
            logger.warning('Trying to extrapolate to a date more than {time} after the end of the MLO/SMO record '
                           '({end}). Likely okay, but consider updating the MLO/SMO {gas} record if possible.'
                           .format(time=mod_utils.relativedelta2string(cls._max_safe_extrap_forward), end=last_date,
                                   gas=cls._gas_name))

        if fit_type == 'exp':
            df_jdates = df.index.to_julian_date().to_numpy()
            if fit(df_jdates[0]) > fit(df_jdates[-1]):
                raise GasRecordExtrapolationError('{} exponential fit is not increasing'.format(cls._gas_name))

    @staticmethod
    def _make_lagged_df(df, lag):
        # Apply the requested lag by adding it to the dates that make up the index of the input data frame. Adding it
        # means that, e.g. 2018-03-01 will actually point to data from 2018-01-01, which is what we want. We're lagging
        # the data because the stratospheric boundary condition should account for the fact that it takes time for air
        # to get from the tropical surface (where MLO/SMO measure) and into the stratosphere.
        lagged_index = pd.DatetimeIndex(d + lag for d in df.index)
        return df.set_index(lagged_index)

    @staticmethod
    def _index_to_dec_year(dframe):
        return [mod_utils.date_to_decimal_year(d) for d in dframe.index]

    @staticmethod
    def _dec_year_to_dtindex(dec_yr, force_first_of_month=False):
        date_times = mod_utils.decimal_year_to_date(dec_yr)
        if force_first_of_month:
            # Get the start of the month on either side of this date. Figure out which one is closer, and set it to
            # that.
            for i, dtime in enumerate(date_times):
                som = mod_utils.start_of_month(dtime, out_type=dt.datetime)
                nearby_months = np.array([som, som + relativedelta(months=1)])
                i_time = np.argmin(np.abs(nearby_months - dtime))
                date_times[i] = nearby_months[i_time]

        return pd.DatetimeIndex(date_times)

    @classmethod
    def _calc_age_spec_gas(cls, df, lag, requested_dates=None):
        gas_conc = dict()

        df_lagged = cls._make_lagged_df(df, lag)

        # We'll need both the date as decimal years and timestamps for different parts of this code, so make those
        # additional columns in the data frame. We'll switch between them for the index as needed
        df_lagged = df_lagged.assign(timestamp=df_lagged.index, dec_year=cls._index_to_dec_year)

        # By default, we'll assume that we want the output to be on the same dates as the input. With the lag that will
        # mean that some extra points near the beginning are NaNs, but that is expected and okay. We're more limited by
        # how far back in time the
        out_dates = df.index if requested_dates is None else requested_dates

        for region in cls.age_spec_regions:
            time, delt, age, spectra = cls._load_age_spectrum_data(region, normalize_spectra=True)

            # Add a zero age to the beginning of age
            age = np.concatenate([[0], age.to_numpy().squeeze()])
            fgas = cls.get_frac_remaining_by_age(age)
            theta = fgas.theta

            n_dates = out_dates.size
            n_ages = age.size
            n_theta = theta.size

            out_array = xr.DataArray(np.full((n_dates, n_ages, n_theta), np.nan),
                                     coords=[('date', out_dates), ('age', age), ('theta', theta)])

            # Put the lagged record, without any age spectrum applied, as the zero age data, for all higher variables
            # using broadcasting
            out_array[:, 0] = df_lagged['dmf_mean'].reindex(out_dates).to_numpy().reshape(-1, 1)

            max_dec_year = mod_utils.date_to_decimal_year(df_lagged.index.max())
            # 1950 is the year Arlyn Andrews used in her code. That will cause some NaNs at the beginning of the
            # record before our gas records start, but that's fine.
            new_index = np.arange(1950.0, max_dec_year, delt)
            conv_dates = None
            old_shape = -1  # will be initialized properly the first time through the loop

            for ispec in range(spectra.shape[0]):
                for itheta in range(n_theta):
                    # The first step is to put the trace gas record on the same time resolution as the age spectra. This
                    # is necessary for the convolution to work. Note that the age spectra aren't assigned to any
                    # specific date, we just need the adjacent points in the age spectra and gas record to have the same
                    # spacing in time.
                    # To handle the reindexing properly, we need to keep the original indices in until we handle the
                    # interpolation to the new values. For this part we need to use the decimal years as the index
                    tmp_index = np.unique(np.concatenate([df_lagged['dec_year'], new_index]))
                    df_asi = df_lagged.set_index('dec_year', drop=False).reindex(tmp_index).interpolate(method='index').reindex(new_index)

                    # Now we can do the convolution. Note: in Arlyn's original R code, she had to flip the age spectrum
                    # to act as the convolution kernel, but testing showed that in order to get the same answer using
                    # numpy's convolution function we had to leave the spectrum unflipped.
                    #
                    # This is because the numpy convolution operation acts to flip the kernel internally. It is defined
                    # as
                    #
                    # (a * v)[n] = \sum_{m=-\infty}^{\infty} a[m]v[n-m]
                    #
                    # Note that v is indexed with n-m. This has the effect of reversing the kernel; for n=10, a[11] gets
                    # multiplied by v[9], a[12] by v[8] and so on. R's convolve function uses a different indexing
                    # pattern that does not reverse the kernel.
                    #
                    # We want the kernel reversed because the trace gas records are defined from old to new, while the
                    # age spectra are from new to old. Therefore, we need to reverse the spectra before convolving to
                    # actually put both in the same direction.
                    conv_result = np.convolve(df_asi['dmf_mean'].values.squeeze(), spectra.iloc[ispec, :].values,
                                              mode='valid')
                    if conv_dates is None:
                        conv_dates = new_index[(spectra.shape[1] - 1):]
                        # The call to dec_year_to_dtindex was ~80% of the time for this function when it was being
                        # called in every loop. There should be no reason to recalculate on every loop; the spectra
                        # should not be changing size and should therefore always give results on the same dates
                        conv_dates = cls._dec_year_to_dtindex(conv_dates, force_first_of_month=False)
                        old_shape = spectra.shape[1]
                    elif spectra.shape[1] != old_shape:
                        raise NotImplementedError('Different spectra have different lengths; this is not allowed as '
                                                  'currently implemented')

                    # Finally we put the age-convolved gas concentration back onto the dates of the input dataframe,
                    # unless alternate dates were specified.
                    conv_df = pd.DataFrame(conv_result, index=conv_dates)
                    tmp_index = np.unique(np.concatenate([out_dates, conv_dates]))
                    this_out_df = conv_df.reindex(tmp_index).interpolate(method='index').reindex(out_dates)

                    # And store this result in the output data frame, remembering that we added an extra row at the
                    # beginning for zero age air, and again using broadcasting.
                    #
                    # We also deal with adding in any chemical loss here because we determine chemical loss from ACE
                    # data with respect to the mean age of the air, therefore we need to lookup the fraction remaining
                    # for that mean age, rather than apply it in the same convolution as the age spectra.
                    out_array[:, ispec+1, itheta] = this_out_df.reindex(out_dates).to_numpy().squeeze() * fgas.isel(age=ispec, theta=itheta).item()

            gas_conc[region] = out_array

        return gas_conc

    def _save_strat_arrays(self):
        # We can't just merge the different region's stratospheric concentration DataArrays into a single dataset
        # because that required that the arrays have the dimensions with the same names be the same, and the age
        # coordinate is not. We'll need to convert the coordinates to region-specific names and save that dataset.
        # We probably also want to write some extra attributes to the netCDF file, at least the mercurial commit
        # of the code that created this file.
        save_dict = dict()
        for name, darray in self.conc_strat.items():
            new_coords = [(name + '_' + dim, coord) for dim, coord in darray.coords.items()]
            new_array = xr.DataArray(darray.data, coords=new_coords)
            save_dict[name] = new_array

        save_ds = xr.Dataset(save_dict)

        # Add some extra attributes - we want to record how this was created (esp. the commit hash) as well as the
        # SHA1 hashes of the MLO and SMO files so that we can verify that those haven't changed.
        save_ds.attrs['history'] = ioutils.make_creation_info(self.get_strat_lut_file())
        for att_name, file_path in self.list_strat_dependent_files().items():
            save_ds.attrs[att_name] = ioutils.make_dependent_file_hash(file_path)
        save_ds.to_netcdf(self.get_strat_lut_file())

    def _have_strat_array_deps_changed(self, dependent_files=None, lut_file=None):
        """
        Check if dependencies for the strat LUTs have changed.

        :param dependent_files: dictionary specifying which files need to be checked. Keys must be the root level
         attribute names in the LUT netCDF file that store the SHA1 hashes of the dependency files, values must be the
         paths to those files. If omitted, the dictionary returned by ``cls.list_strat_dependent_files()`` is used.
        :type dependent_files: dict

        :param lut_file: the LUT netCDF file. If omitted, the path returned by ``cls.get_strat_lut_file()`` is used.
        :type lut_file: str

        :return: ``True`` if the dependencies have changed (meaning a hash is different, the file doesn't exist, or one
         of the expected files is missing), ``False`` otherwise.
        :rtype: bool
        """
        def check_hash(file_path, hash):
            if file_path is None:
                return True
            else:
                return hash == ioutils.make_dependent_file_hash(file_path)

        dependent_files = self.list_strat_dependent_files() if dependent_files is None else dependent_files
        lut_file = self.get_strat_lut_file() if lut_file is None else lut_file

        with xr.open_dataset(lut_file) as ds:
            # First verify that the SHA1 hashes for the MLO and SMO match. If not, we should recalculate the strat array
            # rather than use one calculated with old MLO/SMO data
            for att_name, file_path in dependent_files.items():
                if att_name not in ds.attrs:
                    logger.important('{dep_file} not listed as an attribute in {lut_file}, assuming strat LUT needs '
                                     'regenerated'.format(dep_file=att_name, lut_file=lut_file))
                    return True
                if not check_hash(file_path, ds.attrs[att_name]):
                    logger.important('{dep_file} appears to have changed since the last time the {lut_file} was '
                                     'generated'.format(dep_file=att_name, lut_file=lut_file))
                    return True

            return False

    @classmethod
    def _load_strat_arrays(cls, lut_file=None):
        strat_dict = dict()
        if lut_file is None:
            lut_file = cls.get_strat_lut_file()

        with xr.open_dataset(lut_file) as ds:
            for name, darray in ds.items():
                new_coords = [(dim.split('_')[1], coord) for dim, coord in darray.coords.items()]
                strat_dict[name] = xr.DataArray(darray.data, coords=new_coords)

        return strat_dict

    def list_strat_dependent_files(self):
        """
        Return a dictionary describing the files that the stratospheric LUT depends on.

        This dictionary will have the keys be the attribute names to use in the LUT netCDF file and the values be the
        paths to the files that the LUT depends on. Each file's SHA1 hash will get stored in the netCDF file under
        the global attribute named by its key.

        For most trace gas records, this will be the Mauna Loa and Samoa flask data files. However, if certain
        trace gas records depend on other files, this method should be overridden to return the proper dictionary.

        :rtype: dict
        """
        file_dict = deepcopy(_code_dep_files)
        file_dict.update({'mlo_sha1': self.mlo_file, 'smo_sha1': self.smo_file})
        return file_dict

    def get_strat_gas(self, date, ages, eqlat, theta=None, as_dataframe=False):
        """
        Get stratospheric gas concentration for a given profile

        :param date: the UTC date of the observation
        :type date: datetime-like

        :param ages: the age or ages of air (in years) to get concentration for. Must be the same shape as ``eqlat``.
        :type ages: array-like

        :param eqlat: the equivalent latitude or eq. lat profile to get concentration for. Must be the same shape as
         ``ages``.
        :type eqlat: array-like

        :param theta: the potential temperature profile associated with the prior. Only required if the

        :param as_dataframe: if ``True``, the gas concentration will be returned as a data frame. If ``False``, it will
         be returned as an array if ``ages`` and ``eqlat`` were arrays or a float if they were floats.
        :type as_dataframe: bool

        :return: the gas concentration as a data frame, numpy array, or scalar, depending on ``as_dataframe`` and the
         input types. Also returns None, a placeholder for future information about profile latency, etc.
        :rtype: float, :class:`numpy.ndarray`, or :class:`pandas.DataFrame`
        """

        ages = np.array(ages) * self.strat_age_scale
        eqlat = np.array(eqlat)

        ancillary_dict = dict()

        # Calculate the latency
        age_rdeltas = mod_utils.frac_years_to_reldelta(ages)
        # We need to subtract the lag, because we're looking up against dates that have been already shifted forward
        # by that lag. That is, the age 0 air in the strat table for 1 Mar 2019 corresponds to the MLO/SMO record from
        # 1 Jan 2019.
        ancillary_dict['gas_record_dates'] = np.array([pd.Timestamp(date - a - self.sbc_lag) for a in age_rdeltas])
        ancillary_dict['latency'] = self.get_latency_by_date(ancillary_dict['gas_record_dates'])

        if theta is None:
            # make it an array just to make the input checking easier
            theta = np.full_like(ages, self._no_theta_coord.item())
        else:
            theta = np.array(theta)

        if ages.shape != eqlat.shape or ages.shape != theta.shape:
            raise ValueError('ages, eqlat, and theta must be the same shapes')
        elif ages.ndim != 1 or eqlat.ndim != 1 or theta.ndim != 1:
            raise ValueError('ages, eqlat, and (if given) theta expected to be 1D arrays or convertible to 1D arrays')

        # Get the concentrations for the given ages and equivalent latitudes for each region (tropics, midlat, and
        # vortex). We'll stitch them together after.
        gas_by_region = dict()

        # We only want to interpolate dimensions that an actual effect on the lookup table. So if the theta dimension
        # has length 1, we can't interpolate along that dimension. Put any vectors interpolating along to get the
        # interpolation to return a 1D vector rather than a ND array. See xarray docs on advanced interpolation
        # (http://xarray.pydata.org/en/stable/interpolation.html#advanced-interpolation). Use an ordered dictionary so
        # interpolation happens in the same order all the time.
        interp_dims = OrderedDict([('date', date), ('age', xr.DataArray(ages, dims='level'))])
        if self.strat_has_theta_dep:
            interp_dims['theta'] = xr.DataArray(theta, dims='level')

        for region in self.age_spec_regions:
            region_arr = self.conc_strat[region]
            # We need to extrapolate because there can be ages outside those defined in the strat table or thetas just
            # outside the bin center. We can't extrapolate in multiple dimensions, so we need to iterate over the dims
            # and do each separately. It's better to do that that to handle extrapolation at the end once we have the
            # profile because if we did that we lose the trend along each physical dimension. For example, if age in the
            # last bin is just out of the range in the strat table and has a big jump from the bin below, extrapolating
            # the CO2 profile would lose the decrease at the top because the profile below could be flat, while
            # extrapolating along the age dimension captures the fact that the age is actually lower at the top.
            tmp_arr = region_arr
            for dim_name, dim_coords in interp_dims.items():
                tmp_arr = tmp_arr.interp(method='linear', kwargs={'fill_value': 'extrapolate'}, **{dim_name: dim_coords})
            # needed to handle cases without theta dependence - removes theta dimension
            tmp_arr = tmp_arr.squeeze()

            # Interpolating each dimension in sequence like this leaves each dimension that had a non-scalar target for
            # interpolation, so we need to get the diagonal which is the actual profile.
            # Also we make diag_inds instead of using np.diag b/c the latter doesn't work in >2 dimensions.
            diag_inds = tuple([np.arange(tmp_arr.level.size)]*tmp_arr.ndim)
            gas_by_region[region] = tmp_arr[diag_inds]

        gas_conc = gas_by_region['midlat']
        doy = mod_utils.day_of_year(date) + 1  # most of the code from Arlyn Andrews assumes Jan 1 -> DOY = 1
        xx_tropics = mod_utils.is_tropics(eqlat, doy, ages)
        xx_vortex = mod_utils.is_vortex(eqlat, doy, ages)

        gas_conc[xx_tropics] = gas_by_region['tropics'][xx_tropics]
        gas_conc[xx_vortex] = gas_by_region['vortex'][xx_vortex]

        if as_dataframe:
            return gas_conc.to_dataframe(name='dmf_mean').drop(['theta', 'date'], axis=1), ancillary_dict
        else:
            return gas_conc.squeeze(), ancillary_dict

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


class MidlatTraceGasRecord(TraceGasRecord):
    _std_vmr_file = ''
    # Latitudes and widths of ITCZ in July and January (every 15 deg from 0 to 360)
    # First and last points (O and 360 deg) are repeated to simplify interpolation.
    # from wiki https://en.wikipedia.org/wiki/Intertropical_Convergence_Zone#/media/File:ITCZ_january-july.png
    # Assume peak Northward excursion of ITCZ is mid-July (idoy=198)
    # Assume peak Southward excursion of ITCZ is mid-Jan (idoy=15)
    # Copied from gsetup/calc_itcz.f in changeset 20597c7420c2
    _vlat_jul = np.array([16, 20, 24, 27, 29, 30, 30, 30, 29, 26, 21, 20, 19, 19, 19, 19, 18, 16, 11, 5, 1, 4, 8, 12, 16], dtype=np.float)
    _vlat_jan = np.array([3, 5, -8, -14, -12, -8, -5, -4, -5, -7, -10, -13, -14, -14, -10, -4, 0, 1, 0, -2, -6, -8, -6, -2, 3], dtype=np.float)
    _vwidth = np.array([11, 10, 9, 8, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 7, 8, 11, 13, 14, 14, 13, 12, 11], dtype=np.float)

    @property
    def gas_name(self):
        return self._this_gas_name

    @property
    def gas_unit(self):
        return self._this_gas_unit

    @property
    def gas_seas_cyc_coeff(self):
        return self._this_gas_seas_cyc_coeff

    @property
    def gas_sec_trend(self):
        return self._this_gas_sec_trend

    @property
    def gas_lat_grad(self):
        return self._this_gas_lat_grad

    @property
    def gas_base_prof(self):
        return self._base_profile

    def __init__(self, gas, vmr_file=None):
        if vmr_file is None:
            vmr_file = self._std_vmr_file

        vmr_info = mod_utils.read_vmr_file(vmr_file, as_dataframes=True, style='old')
        if gas.lower() not in vmr_info['profile'].columns:
            raise ValueError('Gas "{}" not found in the .vmr file {}'.format(gas, vmr_file))

        self._this_gas_name = gas
        gas = gas.lower()
        self._this_gas_unit = 'mol/mol'
        self._this_gas_seas_cyc_coeff = vmr_info['prior_info'].loc['Seasonal Cycle', gas]
        self._this_gas_sec_trend = vmr_info['prior_info'].loc['Secular Trends', gas]
        self._this_gas_lat_grad = vmr_info['prior_info'].loc['Latitude Gradient', gas]
        self._base_profile = vmr_info['profile'][['altitude', gas]]
        self._base_tropopause = vmr_info['scalar']['ztrop_vmr'].item()
        self._ref_lat = vmr_info['scalar']['lat_vmr'].item()
        self._ref_decimal_date = vmr_info['scalar']['date_vmr'].item()

    def add_trop_prior(self, prof_gas, obs_date, obs_lat, mod_data, use_theta_eqlat=True, **kwargs):
        obs_doy = mod_utils.day_of_year(obs_date)
        itcz_lat, itcz_width = self.calc_itcz(lon_obs=mod_data['file']['lon'], doy_obs=obs_doy)

        z = mod_data['profile']['Height']
        p = mod_data['profile']['Pressure']
        ptrop = mod_data['scalar']['TROPPB']
        ztrop = mod_utils.interp_tropopause_height_from_pressure(p_trop_met=ptrop, p_met=p, z_met=z)

        prof_gas[:] = self.resample_vmrs_at_effective_altitudes(z=z, itcz_lat=itcz_lat, itcz_width=itcz_width,
                                                                ztrop_mod=ztrop, obslat_mod=obs_lat)

        xx_trop = z < ztrop
        if use_theta_eqlat:
            trop_eqlat = get_trop_eq_lat(prof_theta=mod_data['profile']['PT'], p_levels=p, obs_lat=obs_lat, obs_date=obs_date)
        else:
            trop_eqlat = obs_lat

        trop_aoa = mod_utils.age_of_air(lat=trop_eqlat, z=z[xx_trop], ztrop=ztrop, ref_lat=self._ref_lat)

        prof_gas[xx_trop] = self.apply_lat_grad(prof_gas[xx_trop], trop_eqlat, z=z[xx_trop], ztrop_mod=ztrop)
        prof_gas[xx_trop] = self.apply_secular_trends(prof_gas[xx_trop], date_obs=obs_date, age_of_air=trop_aoa)
        prof_gas[xx_trop] *= mod_utils.seasonal_cycle_factor(lat=trop_eqlat, z=z[xx_trop], ztrop=ztrop,
                                                             fyr=mod_utils.date_to_frac_year(obs_date), species=self,
                                                             ref_lat=self._ref_lat)

        # TODO: add what ancillary data is available.
        return prof_gas, dict()

    def add_strat_prior(self, prof_gas, retrieval_date, mod_data, **kwargs):
        z = mod_data['profile']['Height']
        p = mod_data['profile']['Pressure']
        ptrop = mod_data['scalar']['TROPPB']
        ztrop = mod_utils.interp_tropopause_height_from_pressure(p_trop_met=ptrop, p_met=p, z_met=z)

        prof_theta = mod_data['profile']['PT']
        prof_eqlat = mod_data['profile']['EqL']
        retrieval_doy = int(mod_utils.clams_day_of_year(retrieval_date))

        xx_strat = z >= ztrop
        xx_middleworld = np.zeros(prof_theta.shape, dtype=np.bool_)
        age_of_air_years = get_clams_age(prof_theta, prof_eqlat, retrieval_doy, as_timedelta=False)
        xx_middleworld[xx_strat & np.isnan(age_of_air_years)] = True
        age_of_air_years = age_of_air_years[xx_strat]
        # Because the legacy approach to dealing with the gases may identify a level below 380 K as "stratosphere
        #first_age = np.flatnonzero(~np.isnan(age_of_air_years))[0]
        #xx_nans = slice(first_age)
        #age_of_air_years[xx_nans] = age_of_air_years[first_age]

        prof_gas[xx_strat] = self.apply_lat_grad(prof_gas[xx_strat], lat_obs=prof_eqlat[xx_strat], z=z[xx_strat],
                                                 ztrop_mod=ztrop)
        prof_gas[xx_strat] = self.apply_secular_trends(prof_gas[xx_strat], date_obs=retrieval_date,
                                                       age_of_air=age_of_air_years)
        prof_gas[xx_strat] *= mod_utils.seasonal_cycle_factor(lat=prof_eqlat[xx_strat], z=z[xx_strat], ztrop=ztrop,
                                                              fyr=mod_utils.date_to_frac_year(retrieval_date),
                                                              species=self, ref_lat=self._ref_lat)

        prof_gas[xx_middleworld] = np.interp(prof_theta[xx_middleworld], prof_theta[~xx_middleworld], prof_gas[~xx_middleworld])

        return prof_gas, dict()

    def add_extra_column(self, prof_gas, retrieval_date, mod_data, **kwargs):
        return prof_gas, dict()

    @classmethod
    def calc_itcz(cls, lon_obs, doy_obs):
        if lon_obs < 0:
            lon_obs += 360.0

        nlon = cls._vwidth.size - 1
        xlon = nlon/360.0 * lon_obs
        ilon = int(xlon)
        fr = xlon - ilon
        ilon = ilon % nlon  # don't have to change from 1-based in fortran to 0-based here b/c the fortran arrays were
                            # zero-indexed

        itcz_width = (1 - fr) * cls._vwidth[ilon] + fr * cls._vwidth[ilon + 1]

        janlat = (1 - fr) * cls._vlat_jan[ilon] + fr * cls._vlat_jan[ilon + 1]
        jullat = (1 - fr) * cls._vlat_jul[ilon] + fr * cls._vlat_jul[ilon + 1]
        itcz_lat = 0.5 * (jullat + janlat - (jullat-janlat) * np.cos(2*np.pi*(doy_obs-15)/mod_utils.days_per_year))

        return itcz_lat, itcz_width

    def resample_vmrs_at_effective_altitudes(self, z, itcz_lat, itcz_width, ztrop_mod, obslat_mod):
        ztrop_vmr = self._base_tropopause
        zeff = np.full_like(z, np.nan)
        # troposphere - just scale to tropopause
        xx_trop = z < ztrop_mod
        zeff[xx_trop] = z[xx_trop] * ztrop_vmr/ztrop_mod

        # stratosphere - stretch/compress only the bottom levels, also account for the location of the ITCZ
        zs = z[~xx_trop]
        zeff[~xx_trop] = zs + np.exp(-(zs - ztrop_mod)/10.0) * (ztrop_vmr - ztrop_mod -
                                                                3.5*ztrop_mod*(zs/ztrop_mod - 1)**2.0 *
                                                                np.exp(-((obslat_mod - itcz_lat)/(itcz_width+10))**4.0))

        zeff[zeff > z[-1]] = z[-1]

        # Basically what we're doing here is interpolating the gas in the .vmr file to the levels that the .mod file is
        # defined on, *but* instead of interpolating to them directly, we're calculating effective altitudes that
        # account for the difference in tropopause height between the .vmr file and the .mod file.
        return np.interp(zeff, self._base_profile['altitude'].to_numpy(), self._base_profile[self.gas_name.lower()].to_numpy())

    def apply_lat_grad(self, vmrin, lat_obs, z, ztrop_mod):
        xref = self.gas_lat_grad * (self._ref_lat/15.0)/np.sqrt(1+(self._ref_lat/15)**2)
        xobs = self.gas_lat_grad * (lat_obs/15.0)/np.sqrt(1+(lat_obs/15)**2)
        fr = 1.0 / (1.0 + (z / ztrop_mod)**2)
        return vmrin * (1 + fr*xobs)/(1 + fr*xref)

    def apply_secular_trends(self, vmrin, date_obs, age_of_air):
        tdiff = mod_utils.date_to_decimal_year(date_obs) - self._ref_decimal_date
        tdmaoa = tdiff - age_of_air
        vmrout = vmrin * (1 + self.gas_sec_trend * tdmaoa)

        name = self.gas_name.lower()
        if name == 'co2':
            vmrout *= (1.0 + (tdmaoa/155.0)**2)
        elif name == 'ch4':
            vmrout *= (1.004 - 0.024 * (tdmaoa + 2.5)/np.sqrt(25.0 + (tdmaoa+2.5)**2))
        elif name == 'hf':
            vmrout *= (1.0 + np.exp((-tdmaoa-16.0)/5.0))
        elif name == 'f113':
            vmrout *= (1.0 + np.exp((-tdmaoa-4.0)/9.0))

        return vmrout


class HFTropicsRecord(MloSmoTraceGasRecord):
    _gas_name = 'hf'
    _gas_unit = 'ppb'
    _gas_seas_cyc_coeff = 0.0

    ch4_hf_slopes_file = os.path.join(_data_dir, 'ch4_hf_slopes.nc')

    @classmethod
    def get_mlo_smo_mean(cls, mlo_file, smo_file, first_date, last_date):
        """
        Generate the Mauna Loa/Samoa mean trace gas record.

        For HF, there is no MLO/SMO record because it has no presence in the troposphere. Since this method is called
        by __init__ to set the seasonal cycle concentration, we override it to just create a data frame with the correct
        format but with concentrations of 0 for all times.

        :param mlo_file: unused, kept for consistency with other TraceGasTropicsRecord subclasses
        :type mlo_file: str

        :param smo_file: unused, kept for consistency with other TraceGasTropicsRecord subclasses
        :type smo_file: str

        :param first_date: the earliest date to use in the record. Note that the actual first date will always be the
         first of the month for this date.
        :type first_date: datetime-like

        :param last_date: the latest date to include in the record. Note that if it is not the first of the month, then
         the actual latest date used would be the next first of the month to follow this date. I.e. if this is June
         15th, then July 1st would be used instead.
        :type last_date: datetime-like

        :return: the data frame containing the mean trace gas concentration ('dmf_mean'), a flag ('interp_flag') set
         to 1 for any months that had to be interpolated and 2 for months that had to be extrapolated, and the latency
         ('latency') in years that a concentration had to be extrapolated. Index by timestamp.
        :rtype: :class:`pandas.DataFrame`
        """

        # HF has no tropospheric concentration. Therefore, all we need to do is to set the concentration to 0 for
        # all dates. This will mimic the structure of the other tropics records' data frames.
        if first_date.day != 1:
            first_date = mod_utils.start_of_month(first_date)

        if last_date.day != 1:
            last_date = mod_utils.start_of_month(last_date) + relativedelta(months=1)
        all_months = pd.date_range(first_date, last_date, freq='MS')
        n_months = all_months.size

        df_combined = pd.DataFrame(index=all_months, columns=['dmf_mean']).fillna(0.0)
        df_combined = df_combined.assign(interp_flag=np.zeros((n_months,), dtype=np.int),
                                         latency=np.zeros((n_months,), dtype=np.int))

        # Post processing is currently unnecessary for HF (since all concentrations are 0). However, we keep this call
        # in for consistency.
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

    def list_strat_dependent_files(self):
        # Need to add the code files here, but not any MLO/SMO files so we can't just call the super method
        dep_dict = deepcopy(_code_dep_files)
        dep_dict.update({'ch4_sha1': CH4TropicsRecord.get_strat_lut_file()})
        return dep_dict

    @classmethod
    def _load_ch4_hf_slopes(cls):
        with xr.open_dataset(cls.ch4_hf_slopes_file) as nch:
            bin_names = ncdf.chartostring(nch.variables['bin_names'][:].T.data)
            slopes = nch['ch4_hf_slopes']
            fit_params = nch['slope_fit_params']
        return list(bin_names), slopes, fit_params

    @classmethod
    def _calc_hf_from_ch4(cls, ch4_concs, ch4_record, year, ch4_hf_slopes, ch4_hf_fit_params, lag, use_ace_specific_slopes=False):
        # We need to calculate the strat. bdy. cond. for each data point in ch4_concs because it's important to have the
        # correct boundary condition since that is the intercept for the HF:CH4 slope. Originally I tried to use just an
        #
        sbc_ch4 = xr.DataArray(np.zeros_like(ch4_concs.data), coords=ch4_concs.coords)

        # Holy incompatible types: ch4_concs.date is a numpy datetime64 which can't be added to a relativedelta. It is
        # easiest to convert to a datetime index, but that *also* can't be added to a relativedelta, so we have to
        # convert it further to an array of standard Python datetime objects.
        ch4_date_as_pydt = pd.DatetimeIndex(ch4_concs.coords['date'].data).to_pydatetime()
        for age in ch4_concs.coords['age'].data:
            # Since the CH4 record seasonal dataframe isn't lagged, to get the strat. bdy. cond. for the right dates, we
            # need to subtract the age from the ch4_concs date coordinate (to get back to the date when air of that age
            # entered the stratosphere) and subtract the lag (to get back to when air entering the stratosphere was at
            # the tropical surface).
            ch4_sbc_dates = ch4_date_as_pydt - mod_utils.frac_years_to_reldelta(age) - lag
            ch4_sbc_dates = pd.DatetimeIndex(ch4_sbc_dates).unique()
            # Get the CH4 record on the required dates and broadcast into the date/theta slice for this age in the
            # sbc_ch4
            tmp_index = ch4_record.conc_seasonal.index.append(ch4_sbc_dates).unique()
            ch4_sbc_vec = ch4_record.conc_seasonal.reindex(tmp_index).interpolate(method='index').reindex(ch4_sbc_dates).dmf_mean
            sbc_ch4.loc[{'age': age}] = ch4_sbc_vec.to_numpy().reshape(-1, 1)

        if use_ace_specific_slopes and year in ch4_hf_slopes.coords['year']:
            slope = ch4_hf_slopes.sel(year=year)
        else:
            slope = mod_utils.hf_ch4_slope_fit(year, *ch4_hf_fit_params)

        # The slope is CH4 vs. HF. We want HF as a function of CH4 so
        #    CH4 - sbc = m * HF
        # => HF = (CH4 - sbc)/m

        return (ch4_concs - sbc_ch4) / slope

    @classmethod
    def _calc_age_spec_gas(cls, df, lag, requested_dates=None, ch4_record=None):
        gas_conc = dict()

        df_lagged = cls._make_lagged_df(df, lag)

        # We'll need both the date as decimal years and timestamps for different parts of this code, so make those
        # additional columns in the data frame. We'll switch between them for the index as needed
        df_lagged = df_lagged.assign(timestamp=df_lagged.index, dec_year=cls._index_to_dec_year)

        # By default, we'll assume that we want the output to be on the same dates as the input. With the lag that will
        # mean that some extra points near the beginning are NaNs, but that is expected and okay. We're more limited by
        # how far back in time the
        out_dates = df.index if requested_dates is None else requested_dates

        # TODO: check that requested dates match available dates in CH4 record?

        # HF concentrations will be tied to CH4 concentrations via the relationships described in:
        #   Saad et al. 2014, AMT, doi: 10.5194/amt-7-2907-2014
        #   Washenfelder et al. 2003, GRL, doi: 10.1029/2003GL017969
        if ch4_record is None:
            ch4_record = CH4TropicsRecord(first_date=out_dates.min() + cls._age_spectra_length, last_date=out_dates.max())

        bin_names, ch4_hf_slopes, ch4_hf_fit_params = cls._load_ch4_hf_slopes()

        for region in cls.age_spec_regions:
            iregion = bin_names.index(region)
            region_ch4_hf_slopes = ch4_hf_slopes.isel(latitude_bins=iregion)
            region_ch4_hf_fits_params = ch4_hf_fit_params.isel(latitude_bins=iregion)

            ch4_concentrations = ch4_record.conc_strat[region]
            hf_concentrations = xr.DataArray(np.full_like(ch4_concentrations, np.nan), coords=ch4_concentrations.coords)
            for year in np.unique(hf_concentrations.coords['date'].dt.year):
                xx_date = slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))
                ch4_subset = ch4_concentrations.sel(date=xx_date)
                hf_concentrations.loc[{'date': xx_date}] = cls._calc_hf_from_ch4(ch4_subset, ch4_record, year, region_ch4_hf_slopes, region_ch4_hf_fits_params, lag)

            # Occasionally the slope calculation will give negative values. Set them to 0
            # xarray does not support multidimensional boolean indexing, so we must work on the underlying
            # numpy array
            hf_concentrations.data[hf_concentrations.data < 0.0] = 0.0

            gas_conc[region] = hf_concentrations

        return gas_conc


class CO2TropicsRecord(MloSmoTraceGasRecord):
    _gas_name = 'co2'
    _gas_unit = 'ppm'
    _gas_seas_cyc_coeff = 0.007
    # the default of infinity is kept for GGG2019 as the priors code was delivered to JPL before the necessity of this
    # correction was realized.
    # gas_trop_lifetime_yrs = 200.0 # estimated from Box 6.1 of Ch 6 of the IPCC AR5 (p. 473).
    _max_trend_poly_deg = 'exp'


class N2OTropicsRecord(MloSmoTraceGasRecord):
    _gas_name = 'n2o'
    _gas_unit = 'ppb'
    _gas_seas_cyc_coeff = 0.0
    gas_trop_lifetime_yrs = 121.0  # from table 8.A.1 of IPCC AR5, Ch 8, p. 731

    _ace_fn2o_file = os.path.join(_data_dir, 'ace_fn2o_lut.nc')

    @classmethod
    def get_frac_remaining_by_age(cls, ages):
        def fill_nans(row_age, row):
            # For each theta bin, fill in internal NaNs, then fill external
            # NaNs assuming that the next youngest bin is the best representation
            # at the beginning and the next oldest at the end. np.interp does
            # this automatically so is actually easier than using the xarray interp
            # methods.
            nans = np.isnan(row)
            row[nans] = np.interp(row_age[nans], row_age[~nans], row[~nans])
            return row

        with xr.open_dataset(cls._ace_fn2o_file) as dset:
            fn2o_lut = dset['fn2o']

            # Yes, this is filling in a different direction than the CH4 method. There it makes sense to extend along
            # theta, because (a) we don't expect much data beyond the available theta range from ACE and (b) plotted
            # against theta, the curves are flat parabolas, so a constant extrapolation is reasonable. Here, it makes
            # sense to extrapolate along the age b/c the curves converge at higher theta, so choosing a neighboring
            # age bin's curve should be a good approximation.
            for i in range(fn2o_lut.theta.size):
                fn2o_lut[{'theta': i}] = fill_nans(fn2o_lut.age, fn2o_lut.isel(theta=i))

            # ages is assumed to be a simple numpy array. We'll extrapolate in order to get the very youngest and oldest
            # ages that might be just outside the bin centers. Just in case, fill in any NaNs along the theta dimension
            # (was necessary for CH4, might not be here).
            fn2o_lut = fn2o_lut.interp(age=ages, kwargs={'fill_value': 'extrapolate'}).interpolate_na('theta')

        return fn2o_lut

    def list_strat_dependent_files(self):
        dep_dict = super(N2OTropicsRecord, self).list_strat_dependent_files()
        dep_dict.update({'ace_fn2o_lut_sha1': self._ace_fn2o_file})
        return dep_dict

    
class CH4TropicsRecord(MloSmoTraceGasRecord):
    _gas_name = 'ch4'
    _gas_unit = 'ppb'
    _gas_seas_cyc_coeff = 0.012
    gas_trop_lifetime_yrs = 12.4  # from Table 8.A.1 of IPCC AR5, Ch 8, p. 731
    # Values determined from the binned boundary layer (p > 800 hPa) differences between priors and ATom/HIPPO quantum
    # cascade laser CH4, filtering out over-land data. The bias increased from ~0 at the equator to ~60 ppb at 80 deg N.
    gas_lat_correction = {'nh': 0.75, 'sh': 0.0}

    _nyears_for_extrap_avg = 5

    _fn2o_fch4_lut_file = os.path.join(_data_dir, 'n2o_ch4_acefts.nc')

    @classmethod
    def get_frac_remaining_by_age(cls, ages):
        def replace_end_nans(row):
            # Find the first and last non-NaN values
            not_nans = ~np.isnan(row)
            if np.sum(not_nans) == 0:
                # All NaNs - can't do anything.
                return row
            not_nans = np.flatnonzero(not_nans)
            first_ind = not_nans[0]
            last_ind = not_nans[-1]

            # Replace the beginning NaNs with the first value and the end NaNs with the last value
            row[:first_ind] = row[first_ind]
            row[last_ind+1:] = row[last_ind]
            return row

        # First get the fraction of N2O remaining for the given ages
        fn2o = N2OTropicsRecord.get_frac_remaining_by_age(ages).squeeze()

        # Then get the relationship between F(N2O) and F(CH4) derived from ACE-FTS data. This lookup table was created
        # using `backend_analysis/ace_fts_analysis.make_fch4_fn2o_lookup_table()`.
        with xr.open_dataset(cls._fn2o_fch4_lut_file) as dset:
            fch4_lut = dset['fch4']

            # Extrapolate out to all thetas before interpolating to F(N2O). If we don't do this first, then we'll lose
            # information at higher thetas. Say we need to interpolate to F(N2O) = 0.03 and the F(N2O) = 0.025 bin goes
            # out to theta = 3500, but the F(N2O) = 0.075 bin only goes to theta = 2500. Then F(N2O) = 0.03 will get
            # NaNs for theta > 2500 and lose any information from F(N2O) = 0.025 past theta = 2500, despite being closer
            # to the F(N2O) = 0.025 bin.
            for j in range(fch4_lut.shape[0]):
                fch4_lut[j, :] = replace_end_nans(fch4_lut[j, :])

            # Now that F(N2O) has both age and theta as axes, we need to deal with that. Unfortunately, we can't handle
            # the interpolation along both axes in one shot, so we need to iterate over the theta values that we want
            # the F(CH4) LUT to have, interpolate F(N2O) to those to get a vector with the same length as ages, then
            # interpolate each theta row of F(CH4) to the F(N2O) values for each age.
            fch4_lut_final = xr.DataArray(np.full([np.size(ages), np.size(fch4_lut.theta)], np.nan),
                                          coords=[ages, fch4_lut.theta], dims=('age', 'theta'))
            for itheta, this_theta in enumerate(fch4_lut.theta):
                this_fn2o = fn2o.interp(theta=this_theta, kwargs={'fill_value': 'extrapolate'})
                this_fch4 = fch4_lut[{'theta': itheta}]
                # Use constant value extrapolation past the edge of the FN2O values in the LUT. Doing this rather than
                # linear extrapolation prevents undershooting the F(CH4) at high theta.
                fill_values = (this_fch4[0], this_fch4[-1])
                fch4_lut_final[{'theta': itheta}] = this_fch4.interp(fn2o=this_fn2o, kwargs={'fill_value': fill_values})

            # Fill in NaNs along each theta line
            fch4_lut_final = fch4_lut_final.interpolate_na('theta')

        return fch4_lut_final


class CORecord(TraceGasRecord):
    _gas_name = 'co'
    _gas_unit = 'ppb'
    _gas_seas_cyc_coeff = 0.2

    def add_trop_prior(self, prof_gas, obs_date, obs_lat, mod_data, **kwargs):
        """
        Add tropospheric CO prior.

        This copies the CO profile from the model data into the gas profile. It handles all levels, including the
        stratosphere. The result is scaled from dry mole fraction to ppb.

        :param prof_gas: the CO profile array. Modified in place to include the GEOS CO.
        :type prof_gas: :class:`numpy.ndarray`

        :param obs_date: the date/time of the prior profile (unused, present for consistency)

        :param obs_lat: the latitude of the prior profile (unused, present for consistency)

        :param mod_data: the dictionary of model data read in from the .mod file.
        :type mod_data: dict

        :param kwargs: unused, swallows extra keyword arguments.

        :return: the modified gas profile and a dictionary on ancillary information (currently empty).
        """
        prof_gas[:] = mod_data['profile']['CO'] * 1e9
        return prof_gas, dict()

    def add_strat_prior(self, prof_gas, retrieval_date, mod_data, **kwargs):
        """
        Add the stratospheric CO prior.

        This assumes that the GEOS CO was already added by ``add_trop_prior`` and this method handles adding in the
        extra CO resulting from mesospheric descent.

        :param prof_gas: the CO profile array. Modified in place to include the GEOS CO.
        :type prof_gas: :class:`numpy.ndarray`

        :param retrieval_date: the date/time of the prior profile (unused, present for consistency)

        :param mod_data: the dictionary of model data read in from the .mod file.
        :type mod_data: dict

        :param kwargs: unused, swallows extra keyword arguments.

        :return: the modified gas profile and a dictionary on ancillary information (currently empty).
        """
        if np.any(np.isnan(prof_gas)):
            raise GasRecordError('Expected the CO profile to be all non-NaN values before adding the stratospheric '
                                 'modifications.')


        pres = mod_data['profile']['Pressure']
        theta = mod_data['profile']['PT']
        eqlat = mod_data['profile']['EqL']
        trop_pres = mod_data['scalar']['TROPPB']

        co_file = kwargs.pop('excess_co_lut', _excess_co_file)

        # prof_gas modified in-place
        modify_strat_co(base_co_profile=prof_gas, pres_profile=pres, pt_profile=theta, trop_pres=trop_pres,
                        eqlat_profile=eqlat, prof_date=retrieval_date, excess_co_lut=co_file)

        return prof_gas, dict()

    def add_extra_column(self, prof_gas, retrieval_date, mod_data, **kwargs):
        """

        :param prof_gas:
        :param retrieval_date:
        :param mod_data:
        :param kwargs:
        :return:
        """
        height = mod_data['profile']['Height']
        pres = mod_data['profile']['Pressure']
        temperature = mod_data['profile']['Temperature']
        eqlat = mod_data['profile']['EqL']

        extra_co_ppb = calculate_meso_co(alt_profile=height, eqlat_profile=eqlat, pres_profile=pres, temp_profile=temperature,
                                         prof_date=retrieval_date)
        prof_gas[-1] += extra_co_ppb
        return prof_gas, dict()

    @staticmethod
    def _calc_meso_co_dmf(z_grid, p_grid, t_grid, level_ind_to_add):

        meso_co_column = 0.0

        p_level = p_grid[level_ind_to_add]
        t_level = t_grid[level_ind_to_add]
        nair_level = mod_utils.number_density_air(p_level, t_level)
        vpath_level = mod_utils.effective_vertical_path(z_grid, p_grid, t_grid)[level_ind_to_add]
        vpath_level *= 1e5  # kilometers -> cm

        # The concentration at the level we need to add will be the CO column divided by the level depth, which is
        # given by the effective path length defined for that level by gfit. We then convert to the appropriate dry
        # mole fraction by dividing by the number density at that level.
        extra_co_dmf = meso_co_column / vpath_level / nair_level * 1e9
        return extra_co_dmf, level_ind_to_add


class H2ORecord(TraceGasRecord):
    _gas_name = 'h2o'
    _gas_unit = 'mol/mol'
    _gas_seas_cyc_coeff = None

    def add_trop_prior(self, prof_gas, obs_date, obs_lat, mod_data, **kwargs):
        prof_gas[:] = mod_data['profile']['H2O']
        return prof_gas, dict()

    def add_strat_prior(self, prof_gas, retrieval_date, mod_data, **kwargs):
        return prof_gas, dict()

    def add_extra_column(self, prof_gas, retrieval_date, mod_data, **kwargs):
        return prof_gas, dict()


class O3Record(TraceGasRecord):
    _gas_name = 'o3'
    _gas_unit = 'ppb'
    _gas_seas_cyc_coeff = None

    def add_trop_prior(self, prof_gas, obs_date, obs_lat, mod_data, **kwargs):
        o3 = mod_data['profile']['O3']
        # in kg/kg, need to convert using the mean molecular weight of air. We are NOT using the MMW from the .mod files
        # because that is wet and we want the .vmr files to be dry.

        # ratio molar masses. O3 = 3*O (16e-3 kg/mol)
        rmm = (3 * 16.0e-3) / const.mass_dry_air
        # convert to mol/mol. don't use /= because that changes the array in mod_data too.
        o3 = o3 / rmm
        # convert to ppb
        prof_gas[:] = o3 * 1e9
        return prof_gas, dict()

    def add_strat_prior(self, prof_gas, retrieval_date, mod_data, **kwargs):
        return prof_gas, dict()

    def add_extra_column(self, prof_gas, retrieval_date, mod_data, **kwargs):
        return prof_gas, dict()


# Make the list of available gases' records
gas_records = {r._gas_name: r for r in [CO2TropicsRecord, N2OTropicsRecord, CH4TropicsRecord, HFTropicsRecord,
                                       CORecord, O3Record, H2ORecord]}


def regenerate_gas_strat_lut_files():
    """
    Driver function to regenerate the stratospheric concentration lookup tables for all of the gas records.

    :return: None
    """
    for record in gas_records.values():
        record(force_strat_calculation=True, save_strat=True)


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
        with ncdf.Dataset(_clams_file, 'r') as clams:
            clams_dat['eqlat'] = clams.variables['lat'][:]
            clams_dat['theta'] = clams.variables['extended_theta'][:]
            clams_dat['doy'] = clams.variables['doy'][:]

            # The original CLAMS file provided by Arlyn only went up to 2000 K. At first we tried just using the top
            # for greater potential temperatures, but that led to too-great N2O values at those levels. We now
            # extrapolate using the three end points to calculate a slope of age vs. theta. This calculation takes some
            # time, so we've added the extended age to the CLAMS file using backend_analysis.clams.modify_clams_file().
            clams_dat['age'] = clams.variables['extended_age'][:]

            clams_dat['eqlat_grid'], clams_dat['theta_grid'] = np.meshgrid(clams_dat['eqlat'], clams_dat['theta'])
            if clams_dat['eqlat_grid'].shape != clams_dat['age'].shape[1:] or clams_dat['theta_grid'].shape != clams_dat['age'].shape[1:]:
                raise RuntimeError('Failed to create equivalent lat/theta grids the same shape as CLAMS age')

    idoy = np.argwhere(clams_dat['doy'] == day_of_year).item()

    el_grid, th_grid = np.meshgrid(clams_dat['eqlat'], clams_dat['theta'])
    clams_points = np.array([[el, th] for el, th in zip(el_grid.flat, th_grid.flat)])

    # interp2d does not behave well here; it interpolates to points outside the range of eqlat/theta and gives a much
    # noisier result.
    age_interp = LinearNDInterpolator(clams_points, clams_dat['age'][idoy, :, :].flatten())
    prof_ages = np.array([age_interp(el, th).item() for el, th in zip(eq_lat, theta)])

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
        south_min_ind = np.argmin(theta_diff[:start+1])
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
            lat_tmp = nch.variables['latitude_mean'][:].squeeze()
            # Sometimes lats near 0 get read in as very small non-zero numbers. This causes a 
            # problem later when we select all lats in one hemisphere since the equator needs
            # to be in both hemispheres for this to work
            lat_tmp[np.abs(lat_tmp) < 0.001] = 0.0
            _theta_v_lat['lat'] = lat_tmp
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
    z_grid = z_grid.copy()
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

def add_trop_prior_standard(prof_gas, obs_date, obs_lat, gas_record, mod_data, ref_lat=45.0, use_theta_eqlat=True,
                            profs_latency=None, prof_aoa=None, prof_world_flag=None, prof_gas_date=None):
    """
    Add troposphere concentration to the prior profile using the standard approach.

    :param prof_gas: the profile trace gas mixing ratios. Will be modified in-place to add the stratospheric
     component.
    :type prof_gas: :class:`numpy.ndarray`

    :param obs_date: the UTC date of the retrieval.
    :type obs_date: :class:`datetime.datetime`

    :param obs_lat: the latitude of the retrieval (degrees, south is negative)
    :type obs_lat: float

    :param gas_record: the Mauna Loa-Samoa record for the desired gas.
    :type gas_record: :class:`MloSmoTraceGasRecord`

    :param mod_data: the dictionary of .mod file data. Must have the tropo

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
     datetime object.

    :return: the updated CO2 profile and a dictionary of the ancillary profiles.
    """
    # Extract the necessary data from the .mod dict
    z_grid = mod_data['profile']['Height']
    z_obs = mod_data['scalar']['Height']
    theta_grid = mod_data['profile']['PT']
    pres_grid = mod_data['profile']['Pressure']
    z_trop = mod_utils.interp_tropopause_height_from_pressure(mod_data['scalar']['TROPPB'], pres_grid, z_grid)

    z_grid = adjust_zgrid(z_grid, z_trop, z_obs)
    n_lev = np.size(z_grid)
    prof_gas = _init_prof(prof_gas, n_lev)
    profs_latency = _init_prof(profs_latency, n_lev)
    prof_aoa = _init_prof(prof_aoa, n_lev)
    prof_world_flag = _init_prof(prof_world_flag, n_lev)
    prof_gas_date = _init_prof(prof_gas_date, n_lev, fill_val=None)

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

    # Apply a correction to account for chemical loss between the emission and MLO/SMO measurement or between the
    # prior location and MLO/SMO. For some gases we also apply a latitudinal correction.
    lifetime_adj = np.exp(-air_age / gas_record.gas_trop_lifetime_yrs)
    lat_correction_factor = gas_record.gas_lat_correction['nh'] if obs_lat >= 0 else gas_record.gas_lat_correction['sh']
    lat_correction = lat_correction_factor * obs_lat

    prof_gas[xx_trop] = gas_df['dmf_mean'].values * lifetime_adj + lat_correction
    # Must reshape the 1D latency vector into an n-by-1 matrix to broadcast successfully
    profs_latency[xx_trop] = gas_df['latency'].values
    # Record the date that the CO2 was taken from
    prof_gas_date[xx_trop] = gas_df.index

    # Finally, apply a parameterized seasonal cycle. This is better than using the seasonal cycle in the MLO/SMO data
    # because that is dominated by the NH cycle. This approach allows the seasonal cycle to vary in sign and intensity
    # with latitude.
    year_fraction = mod_utils.date_to_frac_year(obs_date)
    prof_gas[xx_trop] *= mod_utils.seasonal_cycle_factor(obs_lat, z_grid[xx_trop], z_trop, year_fraction,
                                                         species=gas_record, ref_lat=ref_lat)

    return prof_gas, {'co2_latency': profs_latency, 'co2_date': prof_gas_date, 'age_of_air': prof_aoa,
                      'stratum': prof_world_flag, 'ref_lat': ref_lat, 'trop_lat': obs_lat, 'tropopause_alt': z_trop}


def add_strat_prior_standard(prof_gas, retrieval_date, gas_record, mod_data,
                             profs_latency=None, prof_aoa=None, prof_world_flag=None, gas_record_dates=None):
    """
    Add the stratospheric trace gas to a TCCON prior profile using the standard approach.

    :param prof_gas: the profile trace gase mixing ratios. Will be modified in-place to add the stratospheric
     component.
    :type prof_gas: :class:`numpy.ndarray` (in ppm)

    :param retrieval_date: the UTC date of the retrieval.
    :type retrieval_date: :class:`datetime.datetime`

    :param gas_record: the Mauna Loa-Samoa CO2 record.
    :type gas_record: :class:`MloSmoTraceGasRecord`

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
    prof_theta = mod_data['profile']['PT']
    prof_eqlat = mod_data['profile']['EqL']
    prof_pres = mod_data['profile']['Pressure']
    tropopause_t = mod_data['scalar']['TROPT']  # use the blended tropopause. TODO: reference why this is best?
    tropopause_pres = mod_data['scalar']['TROPPB']
    tropopause_theta = mod_utils.calculate_potential_temperature(tropopause_pres, tropopause_t)

    n_lev = np.size(prof_gas)
    profs_latency = _init_prof(profs_latency, n_lev)
    prof_aoa = _init_prof(prof_aoa, n_lev)
    prof_world_flag = _init_prof(prof_world_flag, n_lev)
    gas_record_dates = _init_prof(gas_record_dates, n_lev, fill_val=None)

    # Next we find the age of air in the stratosphere for points with theta > 380 K. We'll get all levels now and
    # restrict to >= 380 K later.
    xx_overworld = mod_utils.is_overworld(prof_theta, prof_pres, tropopause_pres)
    prof_world_flag[xx_overworld] = const.overworld_flag
    # Need the +1 because Jan 1 will be frac_year = 0, but CLAMS expects 1 <= doy <= 366
    retrieval_doy = int(mod_utils.clams_day_of_year(retrieval_date))
    age_of_air_years = get_clams_age(prof_theta, prof_eqlat, retrieval_doy, as_timedelta=False)
    prof_aoa[xx_overworld] = age_of_air_years[xx_overworld]

    # Now, assuming that the CLAMS age is the mean age of the stratospheric air and that we can assume the CO2 has
    # not changed since it entered the stratosphere, we look up the CO2 at the boundary condition. Assume that the
    # CO2 record has daily points, so we just want the date (not datetime). We look up the CO2 using a method on the
    # record specifically designed for stratospheric CO2 that already incorporates the two month lag and the age
    # spectra.

    prof_gas[xx_overworld], strat_extra_info = gas_record.get_strat_gas(retrieval_date, age_of_air_years[xx_overworld],
                                                                        prof_eqlat[xx_overworld], prof_theta[xx_overworld])

    profs_latency[xx_overworld] = strat_extra_info['latency']
    gas_record_dates[xx_overworld] = strat_extra_info['gas_record_dates']
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

    return prof_gas, {'latency': profs_latency, 'gas_record_dates': gas_record_dates, 'age_of_air': prof_aoa,
                      'stratum': prof_world_flag}


def _load_co_lut(lut_file):
    with xr.open_dataset(lut_file) as ds:
        co_lut = ds['co_excess']

    # We'll trick this into doing periodic interpolation in day of year by repeating the first and last slices in that
    # dimension
    co_first_slice = co_lut.isel(doy=0)
    co_first_slice.coords['doy'] = co_first_slice.coords['doy'] + mod_utils.days_per_year
    co_last_slice = co_lut.isel(doy=-1)
    co_last_slice.coords['doy'] = co_last_slice.doy - mod_utils.days_per_year

    return xr.concat([co_last_slice, co_lut, co_first_slice], dim='doy')


def modify_strat_co(base_co_profile, pres_profile, eqlat_profile, pt_profile, trop_pres, prof_date,
                    model_transition_pressures=(30.0, 10.0), excess_co_lut=_excess_co_file, keep_orig_nans=False):
    """
    Takes the baseline GEOS CO profile and adds the mesospheric contribution to the stratosphere.

    The GEOS FP-IT product includes CO profiles, however it does not account for CO drawn down from the mesosphere into
    the stratosphere. This function adds an estimated contribution of mesospheric CO, derived from ACE-FTS data.

    :param base_co_profile: the baseline CO profile in ppb.
    :type base_co_profile: array-like

    :param pres_profile: the pressure levels that the CO profile is defined on.
    :type pres_profile: array-like

    :param eqlat_profile: the equivalent latitude profile, in degrees north.
    :type eqlat_profile: arrary-like

    :param pt_profile: the potential temperature profile on the same levels as the CO profile, in Kelvin.
    :type pt_profile: array-like

    :param trop_pres: the tropopause pressure, in the same units as the pressure profile.
    :type trop_pres: float

    :param prof_date: the date of the profile
    :type prof_date: datetime-like

    :param excess_co_lut: the path to the lookup table with the excess mesospheric CO. If not given, the standard table
     file included in the repo is used. This file is created with
     :func:`ggg_inputs.priors.backend_analysis.ace_fts_analysis.make_excess_co_lut`.
    :type excess_co_lut: str

    :return: the CO profile with the CO from mesospheric descent added.
    :rtype: array-like
    """
    if len(model_transition_pressures) != 2 or model_transition_pressures[0] < model_transition_pressures[1]:
        raise ValueError('model_transition_pressures must be a two element tuple with the second element less than '
                         'the first.')

    with xr.open_dataset(excess_co_lut) as ds:
        co_lut = ds['co'][:]

    # Let's first get the CMAM CO profile for the right day of year and latitude
    xx_overworld = mod_utils.is_overworld(pt_profile, pres_profile, trop_pres)
    base_overworld_co = base_co_profile[xx_overworld]
    level_ind = np.arange(np.size(base_co_profile))[xx_overworld]
    level_coords = [('idx', level_ind)]

    prof_doy = mod_utils.day_of_year(prof_date)

    pres_profile = xr.DataArray(pres_profile[xx_overworld], coords=level_coords)
    eqlat_profile = xr.DataArray(eqlat_profile[xx_overworld], coords=level_coords)
    doy_grid = xr.DataArray(np.full(eqlat_profile.shape, prof_doy), coords=level_coords)

    cmam_co_prof = co_lut.interp(doy=doy_grid, lat=eqlat_profile, plev=pres_profile).data

    # Rather than mess with calculating "excess" CO concentrations for the lookup table, we just averaged the CMAM model
    # and will transition between the GEOS CO and CMAM CO between the transition range pressures. I chose the default
    # of 30 and 10 hPa based on looking at comparisions of GEOS and ACE CO, generally it seems like 30 hPa is the
    # pressure where we first start seeing excess CO that GEOS doesn't capture.
    xx_trans = (pres_profile <= model_transition_pressures[0]) & (pres_profile >= model_transition_pressures[1])
    xx_trans = xx_trans.data
    prior_plog = np.log(pres_profile[xx_trans])
    trans_plog = [np.log(p) for p in model_transition_pressures]
    cmam_weights = (prior_plog - trans_plog[0])/(trans_plog[1] - trans_plog[0])
    base_overworld_co[xx_trans] = base_overworld_co[xx_trans] * (1 - cmam_weights) + cmam_co_prof[xx_trans] * cmam_weights

    # Above the transition regime, just replace GEOS with the CMAM model
    xx_cmam = pres_profile < model_transition_pressures[1]
    xx_cmam = xx_cmam.data
    base_overworld_co[xx_cmam] = cmam_co_prof[xx_cmam]
    orig_nans = np.isnan(base_co_profile)
    base_co_profile[xx_overworld] = base_overworld_co
    if keep_orig_nans:
        base_co_profile[orig_nans] = np.nan

    return base_co_profile


def calculate_meso_co(alt_profile, eqlat_profile, pres_profile, temp_profile, prof_date, excess_co_lut=_excess_co_file):
    top_alt = alt_profile[-1]
    top_eqlat = eqlat_profile[-1]
    top_pres = pres_profile[-1]
    top_temp = temp_profile[-1]

    with xr.open_dataset(excess_co_lut) as ds:
        co_nd = ds['co_nd'][:]
        co_nair = ds['nair'][:]
        co_alts = ds['altitude'][:]

    # In the LUT, nair is assumed to be the same for every profile (because pressure is) so we don't need to interpolate
    # anything before we calculate the effective vertical path we'll use to integrate the CO profiles
    vpath = mod_utils.effective_vertical_path(co_alts.data, nair=co_nair.data)
    plev = co_nd.plev
    prof_doy = mod_utils.day_of_year(prof_date)
    # Unlike the extra strat CO, we don't need the CO on the same levels as any existing profile, we want it on its
    # original levels, so we need to make the interpolation coordinates have the same vertical coordinates as the
    # CO table.
    eqlat_profile = xr.DataArray(np.full(plev.shape, top_eqlat), coords=[plev], dims=['plev'])
    doy_profile = xr.DataArray(np.full(plev.shape, prof_doy), coords=[plev], dims=['plev'])
    cmam_co_prof = co_nd.interp(lat=eqlat_profile, doy=doy_profile, plev=plev)

    xx_meso = co_alts > top_alt
    # vpath will be in kilometers, since co_alts is in kilometers. This will be a column density (molec. cm^-2) of the
    # total CO column above the top level of the profile.
    meso_co_col = np.sum(vpath[xx_meso] * 1e5 * cmam_co_prof[xx_meso])

    # Now we need the effective vertical path for the top level of the actual CO profile.
    top_prior_vpath = mod_utils.effective_vertical_path(z=alt_profile, p=pres_profile, t=temp_profile)[-1]

    # So we can calculate the number density and then the mixing ratio (convert to ppb) that would result from putting
    # the CO column above the top level into the top level with the effective path length that we calculated and an
    # ideal number density.
    meso_co_nd = meso_co_col / (top_prior_vpath * 1e5)
    meso_co_mix = meso_co_nd / mod_utils.number_density_air(p=top_pres, t=top_temp) * 1e9
    return meso_co_mix


def _setup_zgrid(zgrid):
    """
    Setup a fixed altitude grid
    :param zgrid:
    :return:
    """
    if isinstance(zgrid, str):
        int_info = mod_utils.read_integral_file(zgrid)
        return int_info['Height']
    elif not isinstance(zgrid, (type(None), np.ndarray, xr.DataArray)):
        raise TypeError('zgrid must be a string, None, numpy array, or xarray DataArray')
    else:
        return zgrid


def _datetime2float(dtarray):
    return np.array([np.nan if d is None else mod_utils.to_unix_time(d) for d in dtarray])


def _float2datetime(dtarray):
    return np.array([None if np.isnan(d) else mod_utils.from_unix_time(d, pd.Timestamp) for d in dtarray])


def interp_to_zgrid(profile_dict, zgrid):
    """
    Interpolate the output profile variables to a desired altitude grid.

    This mimics the behavior in `read_refvmrs` of the Fortran gsetup code. It interpolates the output variables linearly
    with respect to altitude. It will extrapolate linearly towards the surface, but will raise an error if the profile
    does not go high enough.

    :param profile_dict: a dictionary of profile variables. Must include the altitude levels as the key "Height". The
     keys "gas_record_dates" and "gas_date" are treated as dates for interpolation.
    :type profile_dict: dict

    :param zgrid: any specification of a fixed altitude grid accepted by :func:`_setup_zgrid`, i.e. ``None`` to do no
     interpolation, a path to an integral file, or an array directly specifying the altitude grid.
    :type zgrid: None, str, or :class:`numpy.ndarray`.

    :return: the profile dictionary with its variables interpolated/extrapolated to the new altitude grid. It will
     also have been modified in-place.
    :rtype: dict
    """
    # Specify pairs of functions to convert and unconvert the values of the input arrays. Each key must match a key in
    # the profile_dict, and the values must be two element tuples, where the first element is the function to use to
    # convert the values in the dict to ones that can be interpolated and the second does the reverse process.
    dt_converters = (_datetime2float, _float2datetime)
    converters = {'gas_record_dates': dt_converters,
                  'gas_date': dt_converters}

    zgrid = _setup_zgrid(zgrid)
    if zgrid is None:
        return profile_dict

    profile_z = profile_dict['Height'].copy()
    for k, v in profile_dict.items():
        if k in converters:
            v = converters[k][0](v)

        v = xr.DataArray(v, coords=[profile_z], dims=['z'])
        v = v.interp(z=zgrid, method='linear', kwargs={'fill_value': 'extrapolate'})

        if k in converters:
            v = converters[k][1](v.data)
        else:
            v = v.data
        profile_dict[k] = v

    return profile_dict


def generate_single_tccon_prior(mod_file_data, utc_offset, concentration_record, site_abbrev='xx', zgrid=None,
                                use_eqlat_trop=True, use_eqlat_strat=True, write_map=False):
    """
    Driver function to generate the TCCON prior profiles for a single observation.

    :param mod_file_data: data from a .mod file prepared by Mod Maker. May either be a path to a .mod file, or a
     dictionary from :func:`~mod_utils.read_mod_file`.
    :type mod_file_data: str or dict

    :param utc_offset: a timedelta giving the difference between the ``file_date`` and UTC time. For example, if the
     ``file_date`` was given in US Pacific Standard Time, this should be ``timedelta(hours=-8). This is used to correct
     the date to UTC to ensure the CO2 from the right time is used.

    :param concentration_record: which species to generate the prior profile for. Must be the proper subclass of
     TraceGasTropicsRecord for the given species. The latter is useful if you are making multiple calls to this
     function, as it removes the need to instantiate the record during each call
    :type concentration_record: str or :class:`MloSmoTraceGasRecord`

    :param site_abbrev: the two-letter site abbreviation. Currently only used in naming the output file.
    :type site_abbrev: str

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
    file_date = mod_file_data['file']['datetime']
    file_lat = mod_file_data['file']['lat']
    file_lon = mod_file_data['file']['lon']
    # Make the UTC date a datetime object that is rounded to a date (hour/minute/etc = 0)
    obs_utc_date = dt.datetime.combine((file_date - utc_offset).date(), dt.time())

    n_lev = np.size(mod_file_data['profile']['Height'])

    if not isinstance(concentration_record, TraceGasRecord):
        raise TypeError('concentration_record must be a subclass instance of TraceGasTropicsRecord')
    elif concentration_record.gas_name == '':
        raise TypeError('concentration_record must be a specific subclass instance of TraceGasTropicsRecord that '
                        'has a non-empty gas_name attribute; it cannot be an instance of TraceGasTropicsRecord itself.')

    # First we need to get the altitudes/theta levels that the prior will be defined on. We also need to get the blended
    # tropopause height from the GEOS met file. We will calculate the troposphere CO2 profile from a deseasonalized
    # average of the Mauna Loa and Samoa CO2 concentration using the existing GGG age-of-air parameterization assuming
    # a reference latitude of 0 deg. That will be used to set the base CO2 profile, which will then have a parameterized
    # seasonal cycle added on top of it.

    gas_prof = np.full((n_lev,), np.nan)
    gas_date_prof = np.full((n_lev,), None)
    latency_profs = np.full((n_lev,), np.nan)
    stratum_flag = np.full((n_lev,), -1)

    # gas_prof is modified in-place
    _, ancillary_trop = concentration_record.add_trop_prior(gas_prof, obs_utc_date, obs_lat, mod_file_data,
                                                            use_theta_eqlat=use_eqlat_trop, profs_latency=latency_profs,
                                                            prof_world_flag=stratum_flag, prof_gas_date=gas_date_prof)
    aoa_prof_trop = ancillary_trop['age_of_air'] if 'age_of_air' in ancillary_trop else np.full_like(gas_prof, np.nan)
    trop_ref_lat = ancillary_trop['ref_lat'] if 'ref_lat' in ancillary_trop else np.nan
    trop_eqlat = ancillary_trop['trop_lat'] if 'trop_lat' in ancillary_trop else np.nan
    z_trop_met = ancillary_trop['tropopause_alt'] if 'tropopause_alt' in ancillary_trop else np.nan

    # Next we add the stratospheric profile, including interpolation between the tropopause and 380 K potential
    # temperature (the "middleworld").
    _, ancillary_strat = concentration_record.add_strat_prior(
        gas_prof, obs_utc_date, mod_file_data, profs_latency=latency_profs, prof_world_flag=stratum_flag,
        gas_record_dates=gas_date_prof
    )
    aoa_prof_strat = ancillary_strat['age_of_air'] if 'age_of_air' in ancillary_trop else np.full_like(gas_prof, np.nan)

    # Combine the profile variables into a temporary dict, which we'll use to do the interpolation/extrapolation to the
    # fixed altitude levels if needed.
    gas_name = concentration_record.gas_name
    gas_unit = concentration_record.gas_unit
    map_dict = {'Height': mod_file_data['profile']['Height'],
                'Temp': mod_file_data['profile']['Temperature'],
                'Pressure': mod_file_data['profile']['Pressure'],
                'PT': mod_file_data['profile']['PT'],
                'EqL': mod_file_data['profile']['EqL'],
                gas_name: gas_prof,
                'mean_latency': latency_profs,
                'trop_age_of_air': aoa_prof_trop,
                'strat_age_of_air': aoa_prof_strat,
                'atm_stratum': stratum_flag,
                'gas_date': gas_date_prof,  # TODO: eliminate duplicate
                'gas_record_dates': gas_date_prof}

    map_dict = interp_to_zgrid(map_dict, zgrid)
    concentration_record.add_extra_column(map_dict[gas_name], retrieval_date=obs_utc_date, mod_data=mod_file_data)

    # Finally prepare the output, writing a .map file if needed.
    units_dict = {'Height': 'km',
                  'Temp': 'K',
                  'Pressure': 'hPa',
                  'PT': 'K',
                  'EqL': 'degrees',
                  gas_name: gas_unit,
                  'mean_latency': 'yr',
                  'trop_age_of_air': 'yr',
                  'strat_age_of_air': 'yr',
                  'atm_stratum': 'flag',
                  'gas_date': 'yr',
                  'gas_date_width': 'yr',
                  'gas_record_dates': 'UTC date'}

    map_constants = {'site_lon': file_lon,
                     'site_lat': file_lat,
                     'datetime': file_date,
                     'trop_eqlat': trop_eqlat,
                     'prof_ref_lat': trop_ref_lat,
                     'surface_alt': mod_file_data['scalar']['Height'],
                     'tropopause_alt': z_trop_met,
                     'strat_used_eqlat': use_eqlat_strat}

    var_order = ('Height', 'Temp', 'Pressure', 'PT', 'EqL', gas_name, 'mean_latency', 'trop_age_of_air',
                 'strat_age_of_air', 'atm_stratum', 'gas_date')

    converters = {'gas_date': mod_utils.date_to_decimal_year}
    if write_map:
        map_dir = write_map if isinstance(write_map, str) else '.'
        map_name = os.path.join(map_dir, mod_utils.map_file_name(site_abbrev, obs_lat, file_date))
        mod_utils.write_map_file(map_file=map_name, site_lat=obs_lat, trop_eqlat=trop_eqlat, prof_ref_lat=trop_ref_lat,
                                 surface_alt=mod_file_data['scalar']['Height'], tropopause_alt=z_trop_met,
                                 strat_used_eqlat=use_eqlat_strat, variables=map_dict, units=units_dict,
                                 var_order=var_order, converters=converters)

    return map_dict, units_dict, map_constants


def _get_std_vmr_file(std_vmr_file):
    """
    Get the path to the standard .vmr file

    Looks for ``$GGGPATH/vmrs/gnd/summer_35N.vmr``, unless a path to a .vmr file is given explicitly or a false-y value
    is given to skip that.

    :param std_vmr_file: the input given for the .vmr file
    :type std_vmr_file: None, str, or bool

    :return: the path to the .vmr file or the other input
    :raises GGGPathError: if ``$GGGPATH`` is not defined and it needs to find the standard file or it cannot find the
     standard file in the expected place.
    """
    if std_vmr_file is None:
        gggpath = os.getenv('GGGPATH')
        if gggpath is None:
            raise GGGPathError('GGGPATH environmental variable is not defined. Either define it, explicitly pass a '
                               'path to a .vmr file with northern midlat profiles for all gases, or pass False to '
                               'only write the primary gases to the .vmr file')
        std_vmr_file = os.path.join(gggpath, 'vmrs', 'gnd', 'summer_35N.vmr')
        if not os.path.isfile(std_vmr_file):
            raise GGGPathError('The standard .vmr file is not present in the expected location ({}). Your GGGPATH '
                               'environmental variable may be incorrect, or the structure of the GGG directory has '
                               'changed. Either correct your GGGPATH value, explicitly pass a path to a .vmr file with '
                               'northern midlat profiles for all gases, or pass False to only write the primary gases '
                               'to the .vmr file')
        return std_vmr_file
    else:
        return std_vmr_file


def generate_full_tccon_vmr_file(mod_data, utc_offsets, save_dir, std_vmr_file=None, site_abbrevs='xx',
                                 keep_latlon_prec=False):
    """
    Generate a .vmr file with all the gases required by TCCON (both retrieved and secondary).

    ``mod_data``, ``utc_offsets`` and ``site_abbrevs`` may be single values or collections. See
    :func:`generate_tccon_priors_driver` in this module for details.

    :param mod_data: a dictionary mimicking that from reading a .mod file or the path to a .mod file
    :type mod_data: dict or str

    :param utc_offsets: difference(s) between local time and UTC time for each site
    :type utc_offsets: :class:`datetime.timedelta` or list(:class:`datetime.timedelta`)

    :param save_dir: where to save the .vmr files
    :type save_dir: str

    :param std_vmr_file: a standard .vmr file that has profiles for all the gases needed by TCCON, as well as their
     seasonal cycles, latitudinal gradients, and secular trends. These profiles are assumed to be base profiles
     representative of one latitude/time that can be modified for other locations/times. If this is not given, then the
     code will try to look for ``$GGGPATH/vmrs/gnd/summer_35N.vmr``. If you do not have ``GGGPATH`` defined as an
     environmental variable, it will error. You may pass an explicit path to a .vmr file to override that, or ``False``
     to only write the primary gases to the .vmr file.
    :type std_vmr_file: None, str, or bool

    :param site_abbrevs: abbreviation or list of abbreviations for the sites the .vmr files are being written for.
    :type site_abbrevs: str or list(str)

    :param keep_latlon_prec: by default, latitude/longitude in the .vmr filenames is rounded to the nearest integer.
     Set this to ``True`` to keep 2 decimal places of precision.
    :type keep_latlon_prec: bool

    :return: none, writes .vmr files
    :raises GGGPathError: if ``$GGGPATH`` is not defined and it needs to find the standard file or it cannot find the
     standard file in the expected place.
    """
    std_vmr_file = _get_std_vmr_file(std_vmr_file)
    if std_vmr_file:
        std_vmr_gases = mod_utils.read_vmr_file(std_vmr_file, lowercase_names=False, style='old')
        std_vmr_gases = list(std_vmr_gases['profile'].keys())
        std_vmr_gases.remove('Altitude')
    else:
        std_vmr_gases = list(gas_records.keys())
    species = [gas_records[gas.lower()]() if gas.lower() in gas_records else MidlatTraceGasRecord(gas, vmr_file=std_vmr_file) for gas in std_vmr_gases]
    generate_tccon_priors_driver(mod_data=mod_data, utc_offsets=utc_offsets, species=species, site_abbrevs=site_abbrevs,
                                 write_vmrs=save_dir, keep_latlon_prec=keep_latlon_prec, gas_name_order=std_vmr_gases)


def generate_tccon_priors_driver(mod_data, utc_offsets, species, site_abbrevs='xx', write_maps=False,
                                 write_vmrs=False, gas_name_order=None, keep_latlon_prec=False, **prior_kwargs):
    """
    Generate multiple TCCON priors or a file containing multiple gas concentrations

    This function wraps :func:`generate_single_tccon_prior`  in order to generate priors for one or more gases for one
    or more sites. The inputs ``mod_data``, ``utc_offsets``, and ``site_abbrevs`` determine the number of
    sites. Each of these must be either a single instance of the correct type or a collection of those types. Any of
    them given as collections must have the same number of elements; those given as single instances will be used for
    all profiles.

    For example, say that you wanted to generate profiles for three days. ``mod_data`` and ``obs_dates`` would need to
    be lists of paths to .mod files and Python datetimes, respectively, but ``utc_offsets`` and ``site_abbrevs`` could
    be a single timedelta and string, respectively.

    ``species`` can likewise be a single instance or a collection, but in either case will be applied to all
    sites/times. This determines which species will have profiles generated.

    :param mod_data: input to :func:`generate_single_tccon_prior`, see that function.

    :param utc_offsets:  input to :func:`generate_single_tccon_prior`, see that function.

    :param species: either gas names as strings or instances of :class:`TraceGasTropicsRecord` that set up which gases'
     profiles are created. If given as a list, then all species given will be generated for each site/time.
    :type species: str, :class:`TraceGasTropicsRecord`, list(str), or list(:class:`TraceGasTropicsRecord`)

    :param site_abbrevs:  input to :func:`generate_single_tccon_prior`, see that function.

    :param write_maps: if ``False``, then .map files are not written. If truthy, then it must be a path to the directory
     where the .map files are to be written.

    :param write_vmrs: if ``False``, then .vmr files are not written. If truthy, then it must be a path to the directory
     where the .map files are to be written.

    :param gas_name_order: the order that the gases are to be written in in the output files. Currently only affects the
     .vmr files. See :func:`mod_utils.write_vmr_file` for more information.
    :type gas_name_order: list(str)

    :param keep_latlon_prec: if ``False``, then .vmr files written are named with lat/lon rounded to integers. If
     ``True``, then 2 decimal places are retained.
    :type keep_latlon_prec: bool

    :param prior_kwargs:
    :return:
    """
    num_profiles = max(np.size(inpt) for inpt in [mod_data, utc_offsets, site_abbrevs])

    def check_input(inpt, name, allowed_types):
        type_err_msg = '{} must be either a collection or single instance of one of the types: {}'.format(
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

        return inpt

    def parse_boollike_input(inpt):
        if inpt:
            return inpt, True
        else:
            return '', False

    def get_scale_factor(unit):
        unit = unit.lower()
        if unit == 'ppm':
            return 1e-6
        elif unit == 'ppb':
            return 1e-9
        elif unit == 'mol/mol':
            return 1.0
        else:
            raise ValueError('No conversion factor defined for "{}"'.format(unit))

    # Input checking. Make sure these are the right type and either the same size as each other or a single value. In
    # the latter case, replicate it. These will have one
    mod_data = check_input(mod_data, 'mod_data', (str, dict))
    utc_offsets = check_input(utc_offsets, 'utc_offsets', (dt.timedelta, pd.Timedelta))
    site_abbrevs = check_input(site_abbrevs, 'site_abbrevs', (str,))

    # species will each be generated for every site.
    if isinstance(species, (str, MloSmoTraceGasRecord)):
        species = [species]

    # if given species names, convert to the actual records.
    species = [gas_records[s]() if isinstance(s, str) else s for s in species]

    maps_dir, write_maps = parse_boollike_input(write_maps)
    vmrs_dir, write_vmrs = parse_boollike_input(write_vmrs)

    # MAIN LOOP #
    # Loop over the requested profiles, creating a prior for each gas requested. Check that the other variables are all
    # the same for each gas, then combine them to make a single .map file or dict for each profile
    ancillary_variables = ('Height', 'Temp', 'Pressure', 'PT', 'EqL')
    vmr_gases = dict()
    for iprofile in range(num_profiles):
        var_order = list(ancillary_variables)
        for ispecie, specie_record in enumerate(species):
            gas_name = specie_record.gas_name
            var_order.append(specie_record.gas_name)
            specie_profile, specie_units, specie_constants = \
                generate_single_tccon_prior(mod_data[iprofile], utc_offsets[iprofile],
                                            specie_record, site_abbrev=site_abbrevs[iprofile], write_map=False,
                                            **prior_kwargs)

            if ispecie == 0 or np.isnan(map_constants['tropopause_alt']):
                profile_dict = specie_profile
                units_dict = specie_units
                map_constants = specie_constants
            else:
                for ancvar in ancillary_variables:
                    if not np.allclose(specie_profile[ancvar], profile_dict[ancvar], equal_nan=True):
                        raise RuntimeError('Got different vectors for {} for difference species'.format(ancvar))

                # All good? Add the current specie concentration to the dicts
                profile_dict[gas_name] = specie_profile[gas_name]
                units_dict[gas_name] = specie_units[gas_name]

            # Record the profiles for the .vmr files, converted to dry mole fraction
            vmr_gases[gas_name] = specie_profile[gas_name] * get_scale_factor(specie_units[gas_name])

        # Write the combined .map file for all the requested species
        site_lat = map_constants['site_lat']
        site_lon = map_constants['site_lon']
        site_date = map_constants['datetime']
        if write_maps:
            map_name = os.path.join(maps_dir, mod_utils.map_file_name(site_abbrevs[iprofile], site_lat, site_date))
            mod_utils.write_map_file(map_name, variables=profile_dict, units=units_dict, var_order=var_order,
                                     **map_constants)
        if write_vmrs:
            vmr_name = mod_utils.vmr_file_name(obs_date=site_date, lon=site_lon, lat=site_lat,
                                               keep_latlon_prec=keep_latlon_prec)
            vmr_name = os.path.join(vmrs_dir, vmr_name)
            mod_utils.write_vmr_file(vmr_name, tropopause_alt=map_constants['tropopause_alt'],
                                     profile_date=site_date, profile_lat=site_lat,
                                     profile_alt=profile_dict['Height'], profile_gases=vmr_gases,
                                     gas_name_order=gas_name_order)

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
    mod_dict['file'] = {'datetime': geos_dates[itime], 'lat': geos_prof_data['lat'][ilat], 'lon': geos_prof_data['lon'][ilon]}

    map_dict, _ = generate_single_tccon_prior(mod_dict, dt.timedelta(hours=0), **prior_kwargs)
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
        ioutils.make_ncvar_helper(nch, 'eqlat', geos_prof_data['EqL'], (timedim, levdim, latdim, londim),
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
        geos_prof_data['EqL'] = mod_utils.calculate_eq_lat_on_grid(geos_prof_data['EPV']*1e6, geos_prof_data['PT'], area)
    else:
        geos_prof_data['EqL'] = np.full_like(geos_prof_data['PT'], np.nan)

    # Third, pass each column to as a mod file-like dictionary, passing it to generate_single_tccon_prior, and storing
    # the result in a CO2 array.
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
