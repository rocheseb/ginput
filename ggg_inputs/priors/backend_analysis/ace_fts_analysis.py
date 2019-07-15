from __future__ import print_function, division, absolute_import

from collections import OrderedDict
import datetime as dt
from dateutil.relativedelta import relativedelta
from itertools import repeat
from multiprocessing import Pool
import netCDF4 as ncdf
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import TukeyBiweight
import xarray as xr

from ...common_utils import mod_utils, ioutils, sat_utils
from ...common_utils.ggg_logging import logger
from .. import tccon_priors
from ...mod_maker import mod_maker as mm
from .backend_utils import read_ace_var, read_ace_date, read_ace_theta, read_ace_latlon

_mydir = os.path.abspath(os.path.dirname(__file__))
_tccon_top_alt = 65.0


def _bin_centers(bins):
    return bins[:-1] + np.diff(bins)/2


_std_age_bins = np.arange(0.0, 10.0, 0.25)
_std_age_bin_centers = _bin_centers(_std_age_bins)
_std_frac_bins = np.arange(0.0, 1.05, 0.05)
_std_frac_bin_centers = _bin_centers(_std_frac_bins)
_std_theta_bins = np.concatenate([np.arange(380, 1080, 50), np.arange(1080, 1680, 100), np.arange(1680, 3680, 200)])
_std_theta_bin_centers = _bin_centers(_std_theta_bins)


def make_fn2o_lookup_table(ace_age_file, ace_n2o_file, lut_save_file):
    logger.info('Loading ACE data')
    ace_alt, ace_fn2o, ace_theta, extra_info = calc_fraction_remaining_from_acefts(ace_n2o_file, 'N2O', bc_approach='fit')

    ace_dates = extra_info['dates']
    ace_doy = np.array([int(mod_utils.clams_day_of_year(date)) for date in ace_dates.flat]).reshape(ace_dates.shape)

    with ncdf.Dataset(ace_n2o_file, 'r') as nch:
        ace_lat = read_ace_var(nch, 'latitude', None)

    with ncdf.Dataset(ace_age_file, 'r') as nch:
        ace_age = nch.variables['age'][:].filled(np.nan)

    logger.info('Binning F(N2O) vs. age and theta')
    # Similar to the F(CH4) vs. F(N2O) approach, we ignore fill values, want only positive F(N2O) values, want to
    # avoid the polar vortex and want to stay within TCCON relevant altitudes.
    #
    # is_vortex expects everything to be the same size, so we need to tile lat and doy
    ace_lat = np.tile(ace_lat[:, np.newaxis], [1, ace_age.shape[1]])
    ace_doy = np.tile(ace_doy[:, np.newaxis], [1, ace_age.shape[1]])

    xx = ~np.isnan(ace_fn2o) & ~np.isnan(ace_age) & (ace_fn2o >= 0) & \
         ~mod_utils.is_vortex(ace_lat, ace_doy, ace_age) & (ace_alt[np.newaxis, :] < _tccon_top_alt)

    age_bins = _std_age_bins
    age_bin_centers = _bin_centers(age_bins)
    theta_bins = _std_theta_bins
    theta_bin_centers = _bin_centers(theta_bins)

    # We're not going to worry about outliers initially
    fn2o_means, fn2o_counts, fn2o_overall, theta_overall = _bin_z_vs_xy(ace_age, ace_theta, ace_fn2o, xx, age_bins, theta_bins)

    fn2o_means[fn2o_means > 1] = 1
    fn2o_overall[fn2o_overall > 1] = 1

    _save_fn2o_lut(lut_save_file, fn2o_means, fn2o_counts, fn2o_overall, theta_overall, age_bin_centers, age_bins,
                   theta_bin_centers, theta_bins, ace_n2o_file)


def make_fch4_fn2o_lookup_table(ace_age_file, ace_n2o_file, ace_ch4_file, lut_save_file):
    logger.info('Reading ACE data')
    ace_alt, ace_fn2o, ace_theta, extra_info = calc_fraction_remaining_from_acefts(ace_n2o_file, 'N2O', bc_approach='fit')
    ace_dates = extra_info['dates']
    ace_doy = np.array([int(mod_utils.clams_day_of_year(date)) for date in ace_dates.flat]).reshape(ace_dates.shape)

    _, ace_fch4, _, _ = calc_fraction_remaining_from_acefts(ace_ch4_file, 'CH4', bc_approach='fit')

    with ncdf.Dataset(ace_age_file, 'r') as nch:
        ace_age = nch.variables['age'][:].filled(np.nan)

    with ncdf.Dataset(ace_ch4_file, 'r') as nch:
        ace_ch4_raw = nch.variables['CH4'][:].filled(np.nan)
        ace_lat = nch.variables['latitude'][:].filled(np.nan)

    logger.info('Binning F(CH4) vs. F(N2O)')
    # We're looking for reliable stratospheric relationships. Therefore we limit to data where the concentration is
    # positive and definitely not tropospheric (CH4 < 2e-6 i.e. < 2000 ppb), not in the polar vortex (abs(lat) < 50)
    # and not in the mesosphere or upper stratosphere (alt < 40).
    ace_lat = np.tile(ace_lat[:, np.newaxis], [1, ace_age.shape[1]])
    ace_doy = np.tile(ace_doy[:, np.newaxis], [1, ace_age.shape[1]])
    xx = ~np.isnan(ace_fch4) & ~np.isnan(ace_fn2o) & (ace_fch4 >= 0) & (ace_fn2o >= 0) & (ace_ch4_raw < 2e-6) & \
         ~mod_utils.is_vortex(ace_lat, ace_doy, ace_age) & (ace_alt[np.newaxis, :] < _tccon_top_alt)

    # Define bins for F(N2O) and theta
    fn2o_bins = _std_frac_bins
    fn2o_bin_centers = _std_frac_bin_centers

    theta_bins = _std_theta_bins
    theta_bin_centers = _std_theta_bin_centers

    # Find outliers to avoid overly noisy relationships
    oo = _find_ch4_outliers(ace_fn2o, ace_fch4, fn2o_bins, xx)

    # Do the actual binning
    fch4_means, fch4_counts, fch4_overall, theta_overall = _bin_z_vs_xy(ace_fn2o, ace_theta, ace_fch4, xx & oo,
                                                                        fn2o_bins, theta_bins)

    logger.info('Saving F(CH4):F(N2O) file')
    _save_fch4_lut(lut_save_file, fch4_means, fch4_counts, fch4_overall, theta_overall,
                   fn2o_bin_centers, fn2o_bins, theta_bin_centers, theta_bins)


def make_hf_ch4_slopes(ace_ch4_file, ace_hf_file, washenfelder_supp_table_file, lut_save_file, ch4=None):
    if ch4 is None:
        logger.info('Instantiating CH4 record')
        ch4 = tccon_priors.CH4TropicsRecord()

    logger.info('Loading ACE data')
    with ncdf.Dataset(ace_ch4_file, 'r') as nch_ch4, ncdf.Dataset(ace_hf_file, 'r') as nch_hf:
        ace_ch4 = nch_ch4.variables['CH4'][:].filled(np.nan)
        ace_ch4_err = nch_ch4.variables['CH4_error'][:].filled(np.nan)
        ace_hf = nch_hf.variables['HF'][:].filled(np.nan)
        ace_hf_err = nch_hf.variables['HF_error'][:].filled(np.nan)

        ace_ch4_qual = nch_ch4.variables['quality_flag'][:].filled(9)
        ace_hf_qual = nch_ch4.variables['quality_flag'][:].filled(9)

        ace_lat = nch_ch4.variables['latitude'][:].filled(np.nan)
        ace_lat = np.tile(ace_lat.reshape(-1, 1), [1, ace_ch4.shape[1]])
        ace_alt = nch_ch4.variables['altitude'][:].filled(np.nan)
        ace_alt = np.tile(ace_alt, [ace_ch4.shape[0], 1])

        ace_year = nch_ch4.variables['year'][:].filled(np.nan)
        ace_dates = read_ace_date(nch_ch4)
        ace_doy = np.array([mod_utils.day_of_year(d) + 1 for d in ace_dates])
        ace_doy = ace_doy.reshape(-1, 1)
        ace_dates = np.tile(ace_dates.reshape(-1, 1), [1, 150])

        if 'age' not in nch_ch4.variables:
            add_clams_age_to_file(ace_ch4_file)
        ace_ages = nch_ch4.variables['age'][:].filled(np.nan)

    logger.info('Calculating CH4 vs. HF slopes')
    # We do the same filtering as for the F(N2O):F(CH4) relationship, plus we require that the CH4 and HF error is < 5%.
    # Also have to filter for excessively high HF concentrations, or it will mess up the tropics. Generally HF doesn't
    # exceed 3 ppb and the ultra high values were all > 200 ppb, so 10 should leave all reasonable data and exclude
    # unreasonable data (this occurred in tropics 2005)
    #
    # Fill values in v3 are -999 so remove values below -900
    xx = ~np.isnan(ace_ch4) & ~np.isnan(ace_ch4_err) & (ace_ch4_err / ace_ch4 < 0.05) & ~np.isnan(ace_hf) \
         & ~np.isnan(ace_hf_err) & (ace_hf_err / ace_hf < 0.05) & (ace_ch4 < -900.0) & (ace_ch4 <= 2e-6) \
         & (ace_hf < -900.0) & (ace_alt < _tccon_top_alt) & (ace_ch4_qual == 0) & (ace_hf_qual == 0) & (ace_hf < 10e-9)

    # Read in the early slopes from Washenfelder et al. 2003 (doi: 10.1029/2003GL017969), table S3.
    washenfelder_df = pd.read_csv(washenfelder_supp_table_file, header=2, sep=r'\s+')
    washenfelder_slopes = washenfelder_df.set_index('Date').b.astype(np.float)
    washenfelder_slopes.index = pd.DatetimeIndex(washenfelder_slopes.index)
    washenfelder_slopes.name = 'slope'

    # For each bin, get the slopes from the ACE-FTS data. Fit the data (including the Washenfelder slopes) to an
    # exponential.
    lat_bin_functions = OrderedDict([('tropics', mod_utils.is_tropics),
                                 ('midlat', mod_utils.is_midlat),
                                 ('vortex', mod_utils.is_vortex)])
    lat_bin_slopes = pd.DataFrame(index=_get_ace_date_range(ace_dates), columns=lat_bin_functions.keys())
    lat_bin_counts = lat_bin_slopes.copy()

    fit_params = np.zeros([len(lat_bin_functions), 4], dtype=np.float)

    for ibin, (bin_name, bin_fxn) in enumerate(lat_bin_functions.items()):
        for year_start_date in lat_bin_slopes.index:
            year = year_start_date.year
            logger.debug('Generating {} slopes for {}'.format(bin_name, year))
            lat_bin_slopes.loc[year_start_date, bin_name], lat_bin_counts.loc[year_start_date, bin_name] \
                = _bin_and_fit_hf_vs_ch4(ace_lat, ace_year, ace_dates, ace_doy, ace_ages, ace_ch4, ace_hf, ch4, xx,
                                         year, bin_fxn)

        full_series = pd.concat([washenfelder_slopes, lat_bin_slopes.loc[:, bin_name]])
        full_series = full_series.sort_index()
        full_series_year = full_series.index.year
        full_series = full_series.to_numpy().astype(np.float)

        # This was found to be a good initial guess for all latitude bins by trial and error. 3000.0 gets about the
        # right magnitude at year 0, 0.18 get about the right decay, using the last slope as the offset generally works
        # better than using the maximum (sometimes curve_fit has trouble moving this offset enough) and naturally we
        # want t0 to be the first year. Trying to allow curve_fit to choose its own initial conditions ends badly.
        p0 = [-3000.0, -0.18, full_series[-1], full_series_year[0]]
        fit = curve_fit(mod_utils.hf_ch4_slope_fit, full_series_year, full_series, p0=p0)
        fit_params[ibin, :] = fit[0]

    # for the netCDF file, we'll include both the fits and the actual slopes for the ACE-FTS era. We'll use the fit
    # to fill in values before the ACE era rather than the Washenfelder values directly because those values are
    # irregularly spaced in time
    first_year = washenfelder_slopes.index.min().year
    last_year = lat_bin_slopes.index.max().year
    full_dtindex = pd.date_range(start=dt.datetime(first_year, 1, 1), end=dt.datetime(last_year, 1, 1), freq='YS')

    logger.info('Saving CH4 vs. HF slopes')
    _save_hf_ch4_lut(lut_save_file, ace_ch4_file, ace_hf_file, lat_bin_functions, full_dtindex,
                     lat_bin_slopes, lat_bin_counts, fit_params)

    return lat_bin_slopes, lat_bin_counts


def _find_ch4_outliers(ace_fn2o, ace_fch4, fn2o_bins, good_data):

    # Identify outliers in each F(N2O) bin. We're not binning by theta yet; we just want to find outliers in the dataset
    # as a whole right now. I chose m=5 since 5 sigma is the standard threshold in physics, even through we're using
    # mean absolute deviation and not standard deviation, since the former is more robust.
    #
    # oo will be true for non-outlier values
    oo = np.zeros_like(ace_fch4, dtype=np.bool_)
    bin_inds = np.digitize(ace_fn2o, fn2o_bins) - 1
    for b_ind in np.unique(bin_inds):
        bb = good_data & (bin_inds == b_ind)
        ch4_sub = ace_fch4[bb]
        oo[bb] = ~mod_utils.isoutlier(ch4_sub, m=5)

    return oo


def _bin_z_vs_xy(x, y, z, good_data, x_bins, y_bins):
    n_x_bins = x_bins.size - 1
    n_y_bins = y_bins.size - 1
    x_means = np.full([n_x_bins, n_y_bins], np.nan)
    x_counts = np.zeros_like(x_means, dtype=np.int)

    for i, (tlow, thigh) in enumerate(zip(y_bins[:-1], y_bins[1:])):
        aa = good_data & (y >= tlow) & (y < thigh)
        x_sub = x[aa]
        z_sub = z[aa]
        bin_inds = np.digitize(x_sub, x_bins) - 1

        for j in range(n_x_bins):
            jj = bin_inds == j
            x_means[j, i] = np.nanmean(z_sub[jj])
            x_counts[j, i] = np.sum(jj)

    z_overall = np.full([n_x_bins], np.nan)
    y_overall = np.full([n_x_bins], np.nan)
    bin_inds = np.digitize(x, x_bins) - 1

    for j in range(n_x_bins):
        jj = (bin_inds == j) & good_data
        z_overall[j] = np.nanmean(z[jj])
        y_overall[j] = np.nanmean(y[jj])

    return x_means, x_counts, z_overall, y_overall


def _bin_and_fit_hf_vs_ch4(ace_lat, ace_year, ace_dates, ace_doy, ace_ages, ace_ch4, ace_hf, ch4, xx, year, bin_fxn):
    yy = bin_fxn(ace_lat, ace_doy, ace_ages) & (np.isclose(ace_year, year)).reshape(-1, 1)

    ace_lagged_dates = pd.DatetimeIndex([d - relativedelta(months=2) for d in ace_dates[xx & yy]])
    profile_ch4_sbc = ch4.get_gas_for_dates(ace_lagged_dates)

    x = ace_hf[xx & yy] * 1e9
    y = ace_ch4[xx & yy] * 1e9 - profile_ch4_sbc

    # We use the robust fitting method with Tukey's biweighting as in Saad et al. 2014. That paper used `robustfit` from
    # Matlab, RLM from statsmodels is the Python equivalent. We do not include an intercept because we are using the
    # CH4 stratospheric boundary condition as the intercept, and we've already subtracted that.
    slope = RLM(y, x, M=TukeyBiweight()).fit().params.item()

    return slope, np.sum(xx & yy)


def _get_year_avg_ch4_sbc(year, ch4):
    tt = (ch4.conc_seasonal.index >= dt.datetime(year, 1, 1)) & (ch4.conc_seasonal.index < dt.datetime(year + 1, 1, 1))
    return ch4.conc_seasonal.dmf_mean[tt].mean()


def _save_fn2o_lut(nc_filename, fn2o_means, fn2o_counts, fn2o_overall, theta_overall,
                   age_bin_centers, age_bin_edges, theta_bin_centers, theta_bin_edges, ace_n2o_file):

    with ncdf.Dataset(nc_filename, 'w') as nch:
        age_dim = ioutils.make_ncdim_helper(nch, 'age', age_bin_centers,
                                            description='Age of stratospheric air from the CLaMS model',
                                            units='years')
        ioutils.make_ncdim_helper(nch, 'age_bins', age_bin_edges,
                                  description='Edges of age bins used when binning the F(N2O) data',
                                  units='years')
        theta_dim = ioutils.make_ncdim_helper(nch, 'theta', theta_bin_centers,
                                              description='Potential temperature',
                                              units='K')
        ioutils.make_ncdim_helper(nch, 'theta_bins', theta_bin_edges,
                                  description='Edges of potential temperature bins used when binning the F(N2O) data',
                                  units='K')

        ioutils.make_ncvar_helper(nch, 'fn2o', fn2o_means, (age_dim, theta_dim),
                                  description='Mean value of F(N2O) (fraction of N2O remaining relative to the '
                                              'stratospheric boundary condition) in the age/theta bin',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'fn2o_counts', fn2o_counts, (age_dim, theta_dim),
                                  description='Number of F(N2O) observations in each age/theta bin',
                                  units='number')
        ioutils.make_ncvar_helper(nch, 'fn2o_overall', fn2o_overall, (age_dim,),
                                  description='Mean value of F(N2O) for a given age bin, not separated by theta',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'theta_overall', theta_overall, (age_dim,),
                                  description='Mean value of potential temperature in a given age bin',
                                  units='K')
        ioutils.add_dependent_file_hash(nch, 'ace_n2o_file_sha1', ace_n2o_file)


def _save_fch4_lut(nc_filename, fch4_means, fch4_counts, fch4_overall, theta_overall,
                   fn2o_bin_centers, fn2o_bin_edges, theta_bin_centers, theta_bin_edges):
    with ncdf.Dataset(nc_filename, 'w') as nch:
        fn2o_dim = ioutils.make_ncdim_helper(nch, 'fn2o', fn2o_bin_centers,
                                             description='Fraction of N2O remaining relative to the stratospheric boundary condition.',
                                             units='unitless')
        ioutils.make_ncdim_helper(nch, 'fn2o_bins', fn2o_bin_edges,
                                  description='Edges of F(N2O) bins used when binning the F(CH4) data',
                                  units='unitless')
        theta_dim = ioutils.make_ncdim_helper(nch, 'theta', theta_bin_centers,
                                              description='Potential temperature',
                                              units='K')
        ioutils.make_ncdim_helper(nch, 'theta_bins', theta_bin_edges,
                                  description='Edges of potential temperature bins used when binning the F(CH4) data',
                                  units='K')

        ioutils.make_ncvar_helper(nch, 'fch4', fch4_means, (fn2o_dim, theta_dim),
                                  description='Mean value of F(CH4) (fraction of CH4 remaining relative to the '
                                              'stratospheric boundary condition) in the F(N2O)/theta bin',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'fch4_counts', fch4_counts, (fn2o_dim, theta_dim),
                                  description='Number of CH4 observations in each F(N2O)/theta bin',
                                  units='number')
        ioutils.make_ncvar_helper(nch, 'fch4_overall', fch4_overall, (fn2o_dim,),
                                  description='Mean value of F(CH4) for a given F(N2O) bin, not separated by theta',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'theta_overall', theta_overall, (fn2o_dim,),
                                  description='Mean value of potential temperature in a given F(N2O) bin',
                                  units='K')


def _save_hf_ch4_lut(nc_filename, ace_ch4_file, ace_hf_file, lat_bin_edges, full_date_index, ace_slopes, ace_counts, slope_fit_params):
    with ncdf.Dataset(nc_filename, 'w') as nch:
        ioutils.add_creation_info(nch, creation_note='ace_fts_analysis.make_hf_ch4_slopes')
        ioutils.add_dependent_file_hash(nch, 'ace_ch4_file_sha1', ace_ch4_file)
        ioutils.add_dependent_file_hash(nch, 'ace_hf_file_sha1', ace_hf_file)

        nch.ace_ch4_file = ace_ch4_file
        nch.ace_hf_file = ace_hf_file

        ch4_hf_slopes = pd.DataFrame(index=full_date_index, columns=ace_slopes.columns)
        ch4_hf_count = pd.DataFrame(index=full_date_index, columns=ace_counts.columns).fillna(0)
        ch4_hf_source = pd.DataFrame(index=full_date_index, columns=ace_slopes.columns).fillna(1)
        # transpose so that the bins dim is second like the above dataframes
        bin_names_char_array = ncdf.stringtochar(np.array(lat_bin_edges.keys())).T
        slope_fit_params = slope_fit_params.T

        # For each latitude bin, make a pandas series for the full date index from the fit first, then replace available
        # years with the actual fit
        for ibin, bin_name in enumerate(lat_bin_edges.keys()):
            ch4_hf_slopes.loc[:, bin_name] = mod_utils.hf_ch4_slope_fit(ch4_hf_slopes.index.year, *slope_fit_params[:, ibin])
            ch4_hf_slopes.loc[ace_slopes.index, bin_name] = ace_slopes.loc[:, bin_name]
            ch4_hf_count.loc[ace_counts.index, bin_name] = ace_counts.loc[:, bin_name]
            ch4_hf_source.loc[ace_slopes.index, bin_name] = 0

        # Setup dimensions.
        nbins = len(lat_bin_edges)
        nchars = bin_names_char_array.shape[0]
        nparams = slope_fit_params.shape[0]
        year_dim = ioutils.make_ncdim_helper(nch, 'year', full_date_index.year.to_numpy())
        bins_dim = nch.createDimension('latitude_bins', size=nbins)
        char_dim = nch.createDimension('bin_name_length', size=nchars)
        fit_params_dim = nch.createDimension('exp_fit_params_length', size=nparams)

        # Save the variables
        ioutils.make_ncvar_helper(nch, 'ch4_hf_slopes', ch4_hf_slopes.to_numpy().astype(np.float), [year_dim, bins_dim],
                                  units='ppb CH4/ppb HF',
                                  description='Slope of stratospheric CH4 concentrations vs HF concentrations from '
                                              'ACE-FTS data for specific years and latitude bins')
        ioutils.make_ncvar_helper(nch, 'ace_counts', ch4_hf_count.to_numpy().astype(np.float), [year_dim, bins_dim],
                                  units='#',
                                  description='Number of ACE-FTS observations used to calculate this slope')
        ioutils.make_ncvar_helper(nch, 'ch4_hf_slopes_source', ch4_hf_source.to_numpy().astype(np.int), [year_dim, bins_dim],
                                  units='N/A',
                                  description='Flag indicating how the slope was computed. 0 means taken directly from '
                                              'fits of ACE-FTS CH4 vs. HF; 1 means sampled from the exponential fit '
                                              'of ACE-FTS + Washenfelder 03 slopes')
        ioutils.make_ncvar_helper(nch, 'slope_fit_params', slope_fit_params, [fit_params_dim, bins_dim],
                                  description='Fitting parameters a, b, c, and t0 for a function of form '
                                              '"a * exp(b*(t - t0)) + c" used to fit the temporal trend of CH4 vs. HF '
                                              'slopes, where t is the year (e.g. 2004)')
        ioutils.make_ncvar_helper(nch, 'bin_names', bin_names_char_array, [char_dim, bins_dim],
                                  description='The names used for the latitude bins')


def calc_fraction_remaining_from_acefts(nc_file, gas_name, bc_approach='per-profile'):
    """
    Calculate the fraction remaining of a gas in the stratosphere from an ACE-FTS netCDF file

    :param nc_file: the path to the ACE-FTS netCDF file
    :type nc_file: str

    :param gas_name: the variable name in the netCDF file that holds the gas concentrations
    :type gas_name: str

    :param gas_record: a subclass instance of :class:`TropicsTraceGasRecord` that provides the stratospheric boundary
     condition for the given gas.
    :type gas_record: :class:`tccon_priors.TropicsTraceGasRecord`

    :param tropopause_approach: how to find the tropopause. Options are:

        * 'wmo' - uses the WMO definition, looking for lapse rate < 2 K/km
        * 'theta' - finds the altitude at which potential temperature is 380 K

    :type tropopause_approach: str

    :return: the vector of altitudes that the ACE-FTS profiles are defined on and the array of fraction of gas
     remaining. The latter will be set to NaN in the troposphere.
    :rtype: :class:`numpy.ndarray` x2
    """
    bottom_pt = 360
    top_pt = 390

    def _get_bc_per_profile(dates, concentration, pt, lat):
        bc_conc = np.full([concentration.shape[0]], np.nan)
        for i, (conc_row, pt_row) in enumerate(zip(concentration, pt)):
            zz = (pt_row > bottom_pt) & (pt_row < top_pt)

            if zz.sum() > 0:
                bc_conc[i] = np.nanmean(conc_row[zz])

        return bc_conc

    def _get_bc_by_fit(dates, concentration, pt, lat):
        datenums = np.full_like(concentration, np.nan)
        datenums[:] = np.array([np.datetime64(d) for d in dates]).reshape(-1, 1)
        in_pt = (pt > 360) & (pt < 390)
        in_tropics = mod_utils.is_tropics(lat, None, None)
        in_both = in_pt & in_tropics.reshape(-1, 1)
        bc_concs = concentration[in_both]
        bc_datenums = datenums[in_both]
        not_outliers = ~mod_utils.isoutlier(bc_concs, m=5)
        not_nans = ~np.isnan(bc_datenums) & ~np.isnan(bc_concs)
        xx = not_outliers & not_nans
        fit = np.polynomial.polynomial.Polynomial.fit(bc_datenums[xx], bc_concs[xx], deg=2)
        return fit(datenums[:, 0])

    with ncdf.Dataset(nc_file, 'r') as nch:
        alt = read_ace_var(nch, 'altitude', qflags=None)
        latitude = read_ace_var(nch, 'latitude', qflags=None)
        ace_dates = read_ace_date(nch)
        qflags = read_ace_var(nch, 'quality_flag', qflags=None)

        gas_conc = read_ace_var(nch, gas_name, qflags=qflags)
        # ACE data appears to use -999 as fill value in the concentrations
        gas_conc[gas_conc < -900.0] = np.nan

        theta = read_ace_theta(nch, qflags)

    if bc_approach == 'per-profile':
        bc_fxn = _get_bc_per_profile
    elif bc_approach == 'fit':
        bc_fxn = _get_bc_by_fit
    else:
        raise ValueError('bc_approach must be "per-profile" or "fit"')

    bc_concentrations = bc_fxn(ace_dates, gas_conc, theta, latitude)
    fgas = gas_conc / bc_concentrations.reshape(-1, 1)
    fgas[theta < 380] = np.nan

    return alt, fgas, theta, {'dates': ace_dates, 'bc_concs': bc_concentrations, 'gas_concs': gas_conc}


def _get_ace_date_range(ace_dates, freq='YS'):
    min_date = np.min(ace_dates)
    if min_date < dt.datetime(min_date.year, 4, 1):
        start_year = min_date.year
    else:
        start_year = min_date.year + 1

    max_date = np.max(ace_dates)
    if max_date >= dt.datetime(max_date.year, 10, 1):
        end_year = max_date.year
    else:
        end_year = max_date.year - 1

    return pd.date_range(start=dt.datetime(start_year, 1, 1), end=dt.datetime(end_year, 1, 1), freq=freq)


def add_clams_age_to_file(ace_nc_file, save_nc_file=None):
    """
    Calculate CLaMS age for profiles in an ACE-FTS files. Save either to the given ACE file or a new file.

    Note: this function does not calculate equivalent latitude for the ACE profiles before computing the age, it simply
    uses the geographic latitude of the ACE profiles.

    :param ace_nc_file: the ACE-FTS netCDF file to compute ages for.
    :type ace_nc_file: str

    :param save_nc_file: optional, if given a path to a file, the ages will be saved to this file (which is overwritten)
     along with the altitude dimension. If not given (or ``None``), the ages are added to the file specified by
     ``ace_nc_file``.
    :type save_nc_file: str

    :return: None
    """
    with ncdf.Dataset(ace_nc_file, 'r') as nch:
        ace_lat = read_ace_var(nch, 'latitude', None)
        ace_alt = read_ace_var(nch, 'altitude', None)
        ace_dates = read_ace_date(nch)
        ace_doy = np.array([mod_utils.day_of_year(d) + 1 for d in ace_dates])
        ace_theta = read_ace_theta(nch, None)

    ace_ages = np.full_like(ace_theta, np.nan)
    pbar = mod_utils.ProgressBar(ace_theta.shape[0], prefix='Calculating ages for profile', style='counter')
    for iprof in range(ace_theta.shape[0]):
        pbar.print_bar(iprof)
        prof_theta = ace_theta[iprof]
        prof_lat = np.full_like(prof_theta, ace_lat[iprof])
        prof_doy = ace_doy[iprof]
        ace_ages[iprof, :] = tccon_priors.get_clams_age(prof_theta, prof_lat, prof_doy)

    pbar.finish()

    if save_nc_file is None:
        logger.info('Added CLaMS age to input file: {}'.format(ace_nc_file))
        with ncdf.Dataset(ace_nc_file, 'a') as nch:
            ioutils.make_ncvar_helper(nch, 'age', ace_ages, ['index', 'altitude'], units='years',
                                      description='Age from the CLaMS model along the ACE profiles')
    else:
        logger.info('Saving CLaMS age to new file: {}'.format(save_nc_file))
        with ncdf.Dataset(save_nc_file, 'w') as nch:
            index_dim = nch.createDimension('index', ace_ages.shape[0])
            alt_dim = ioutils.make_ncdim_helper(nch, 'altitude', ace_alt, long_name='altitude',
                                                description='Fixed altitude grid', units='km')
            ioutils.make_ncvar_helper(nch, 'age', ace_ages, [index_dim, alt_dim], units='years', long_name='clams_age',
                                      description='Age from the CLaMS model along the ACE profiles')


def generate_ace_age_file(ace_in_file, age_file_out, geos_path, use_geos_theta_for_age=False, nprocs=0):
    # This will need to group ACE profiles by which GEOS files they fall between, read the PV from those files,
    # interpolate to the ACE profile lat/lon/time, generate the EL interpolators, calculate the EL profiles, then
    # finally lookup age from CLAMS.
    ace_data = dict()
    with ncdf.Dataset(ace_in_file) as nh:
        ace_data['dates'] = read_ace_date(nh)
        ace_qflags = read_ace_var(nh, 'quality_flag', None)
        # This wraps lons outside [-180,180] and lats outside [-90,90] into
        # those ranges
        ace_data['lon'], ace_data['lat'] = read_ace_latlon(nh, clip=True)
        ace_data['theta'] = read_ace_theta(nh, qflags=ace_qflags)

    date_groups = _bin_ace_to_geos_times(ace_data['dates'])

    # Now we need to loop over the date groups and calculate the eq. lat. and age for each group (since each group
    # is the ace profiles in a 3 hour window bracketed by 2 GEOS files). There's usually only a few profiles per group,
    # so it makes more sense to parallelize over groups than profiles within groups.
    #
    # Even though it may be memory intensive, we're going to build the inputs for each call to _calc_el_age_for_ace
    # first then iterate so that we can use the same inputs for serial or parallel mode. We'll build as iterators to
    # limit the memory use as much as possible.
    group_inputs = dict()
    group_inputs['ace_dates'] = (ace_data['dates'][xx] for xx in date_groups.values())
    group_inputs['ace_lons'] = (ace_data['lon'][xx] for xx in date_groups.values())
    group_inputs['ace_lats'] = (ace_data['lat'][xx] for xx in date_groups.values())
    group_inputs['ace_theta'] = (ace_data['theta'][xx, :] for xx in date_groups.values())
    group_inputs['geos_dates'] = ((d, d+dt.timedelta(hours=3)) for d in date_groups.keys())
    group_inputs['geos_path'] = repeat(geos_path)
    group_inputs['use_geos_theta'] = repeat(use_geos_theta_for_age)

    input_order = ('ace_dates', 'ace_lons', 'ace_lats', 'ace_theta', 'geos_dates', 'geos_path', 'use_geos_theta')
    inputs_interator = zip(*[group_inputs[k] for k in input_order])

    if nprocs == 0:
        results = []
        for inputs in inputs_interator:
            results.append(_calc_el_age_for_ace(*inputs))
    else:
        with Pool(processes=nprocs) as pool:
            results = pool.starmap(_calc_el_age_for_ace, inputs_interator)

    # _calc_el_age_for_age returns a dictionary of nprofs-by-nlevels arrays, where nprofs is the number of profiles in
    # the group it was given. results is then a list of these dictionaries, so we need to go through those dictionaries
    # and put their subset of profiles in the right rows in the bigger array
    output_keys = ('EL', 'age')
    ace_outputs = {k: np.full_like(ace_data['theta'], np.nan) for k in output_keys}
    for i, inds in enumerate(date_groups.values()):
        for k in output_keys:
            ace_outputs[k][inds] = results[i][k]

    save_ace_age_file(age_file_out, ('longitude', 'latitude', 'orbit', 'year', 'month', 'day', 'hour', 'quality_flag', 'temperature', 'pressure'),
                      ace_in_file, ace_outputs['EL'], ace_outputs['age'])


def _calc_el_age_for_ace(ace_dates, ace_lon, ace_lat, ace_theta, geos_dates, geos_path, use_geos_theta_for_age=False):
    date_str = ', '.join(d.strftime('%Y-%m-%d %H:%M') for d in ace_dates)
    logger.info('Calculating eqlat and age for profiles on {}'.format(date_str))
    # Read in the GEOS data, calculating quantities as necessary #
    geos_vars = {'EPV': 1e6, 'T': 1.0}

    try:
        geos_files = [os.path.join(geos_path, 'Np', mod_utils._format_geosfp_name('fpit', 'Np', d)) for d in geos_dates]
        geos_data_on_std_times = []
        for i, f in enumerate(geos_files):
            geos_data_on_std_times.append(_interp_geos_vars_to_ace_lat_lon(f, geos_vars, ace_lon, ace_lat))

        with ncdf.Dataset(geos_files[0]) as nh:
            geos_pres = nh.variables['lev'][:]
        geos_pres = np.tile(geos_pres.reshape(1, -1), [ace_lon.size, 1])

        # Generate the eq. lat. intepolators and find the eq. lat. profiles on the GEOS levels #
        eqlat_interpolators = mm.equivalent_latitude_functions_from_geos_files(geos_files, geos_dates, muted=True)
        for i, geos_data in enumerate(geos_data_on_std_times):
            gdate = geos_dates[i]
            geos_data['PT'] = mod_utils.calculate_potential_temperature(geos_pres, geos_data['T'])
            # Calculate the equivalent latitude profiles.
            eqlat = np.full_like(geos_data['T'], np.nan)
            for j in range(ace_lon.size):
                eqlat[j, :] = mod_utils.get_eqlat_profile(eqlat_interpolators[gdate], geos_data['EPV'][j, :], geos_data['PT'][j, :])
            geos_data['EL'] = eqlat

        # Put the profiles onto the ACE times and levels #
        geos_data_at_ace_times = _interp_geos_vars_to_ace_times(ace_dates, geos_dates, geos_data_on_std_times)
        geos_data_on_ace_levels = dict()
        for varname, geos_data in geos_data_at_ace_times.items():
            ace_profiles = np.full_like(ace_theta, np.nan)
            for i, profile in enumerate(geos_data):
                # This is one case where I think we do not want to extrapolate because the ACE data spans much greater
                # vertical extent than the GEOS data, so it can't be reliable
                ace_profiles[i, :] = np.interp(ace_theta[i, :], geos_data_at_ace_times['PT'][i, :], geos_data[i, :],
                                               left=np.nan, right=np.nan)
            geos_data_on_ace_levels[varname] = ace_profiles

        # Use the equivalent latitude and potential temperature profiles to look up CLAMS ages #
        ace_age = np.full_like(ace_theta, np.nan)
        for i in range(ace_lon.size):
            if use_geos_theta_for_age:
                theta_prof = geos_data_on_ace_levels['PT'][i, :]
            else:
                theta_prof = ace_theta[i, :]

            eqlat_prof = geos_data_on_ace_levels['EL'][i, :]
            ace_doy = mod_utils.clams_day_of_year(ace_dates[i])

            ace_age[i, :] = tccon_priors.get_clams_age(theta_prof, eqlat_prof, ace_doy)

        geos_data_on_ace_levels['age'] = ace_age
        return geos_data_on_ace_levels
    except Exception as err:
        msg = 'Problem occurred in ACE profile group for dates {} (GEOS file date {}).'.format(date_str, geos_dates[0]) + err.args[0]
        raise err.__class__(msg)


def _bin_ace_to_geos_times(ace_dates):
    """
    Group ACE profiles by which GEOS FP files surround them

    :param ace_dates: an array of ACE profile datetimes
    :type ace_dates: array of datetimes

    :return: a dictionary where the keys will be Pandas timestamps representing the time of the GEOS file preceding the
     profiles in that group and the values are the indices of the profiles in that group as a numpy array.
    :rtype: dict
    """
    def to_3hr(d):
        return dt.datetime(d.year, d.month, d.day, to_3(d.hour))

    def to_3(v):
        return int(3*np.floor(v/3))

    first_time = ace_dates.min()
    last_time = ace_dates.max()

    first_time = to_3hr(first_time)
    last_time = to_3hr(last_time) + dt.timedelta(hours=3)

    # Create the bins as floats so we can use numpy.digitize
    ace_datenums = np.array([sat_utils.datetime2datenum(d) for d in ace_dates])
    bin_edges = pd.date_range(first_time, last_time, freq='3H')
    bin_edge_datenums = np.array([sat_utils.datetime2datenum(d) for d in bin_edges])

    inds = np.digitize(ace_datenums, bin_edge_datenums) - 1
    date_groups = OrderedDict()
    for i in np.unique(inds):
        date_groups[bin_edges[i]] = np.flatnonzero(inds == i)

    return date_groups


def _interp_geos_vars_to_ace_lat_lon(geos_file_path, geos_vars, ace_lon, ace_lat):
    def interp_prof_helper(data, glat, glon, alat, alon, interp_inds):
        profs_out = np.full([alat.size, data.shape[0]], np.nan)
        for i, data_layer in enumerate(data):
            # Reshape to nprofs-by-nlevels to be consistent with ACE
            profs_out[:, i] = mm.lat_lon_interp(data_layer, glat, glon, alat, alon, interp_inds)

        return profs_out

    geos_data = dict()
    geos_interp_inds = []
    with ncdf.Dataset(geos_file_path) as nh:
        geos_lon_half_res = 0.5 * float(nh.LongitudeResolution)
        geos_lat_half_res = 0.5 * float(nh.LatitudeResolution)

        for varname, scale in geos_vars.items():
            # convert to 3D - remove singleton time dimension
            geos_data[varname] = nh.variables[varname][0].filled(np.nan) * scale

        for lon, lat in zip(ace_lon, ace_lat):
            geos_interp_inds.append(mm.querry_indices(nh, lat, lon, geos_lat_half_res, geos_lon_half_res))

        geos_lat = nh.variables['lat'][:].filled(np.nan)
        geos_lon = nh.variables['lon'][:].filled(np.nan)

        for varname, array in geos_data.items():
            geos_data[varname] = interp_prof_helper(array, geos_lat, geos_lon, ace_lat, ace_lon, geos_interp_inds)

    return geos_data


def _interp_geos_vars_to_ace_times(ace_datetimes, geos_datetimes, geos_vars):
    interped_vars = dict()
    for ace_dt in ace_datetimes:
        weight = sat_utils.time_weight_from_datetime(ace_dt, geos_datetimes[0], geos_datetimes[1])
        for k in geos_vars[0].keys():
            interped_vars[k] = weight * geos_vars[0][k] + (1 - weight) * geos_vars[1][k]

    return interped_vars


def save_ace_age_file(ace_out_file, vars_to_copy, orig_ace_file, eqlat, age):
    def get_var_attrs(nh, varname):
        return {k: nh.variables[varname].getncattr(k) for k in nh.variables[varname].ncattrs()}

    with ncdf.Dataset(orig_ace_file, 'r') as nhread, ncdf.Dataset(ace_out_file, 'w') as nhwrite:
        # Copy all the dimensions
        for dimname in nhread.dimensions.keys():
            try:
                attributes = get_var_attrs(nhread, dimname)
                ioutils.make_ncdim_helper(nhwrite, dimname, nhread.variables[dimname][:], **attributes)
            except KeyError:
                # Sometimes we get a dimension with no corresponding variable
                nhwrite.createDimension(dimname, nhread.dimensions[dimname].size)

        # Copy all the requested variables
        for varname in vars_to_copy:
            attributes = get_var_attrs(nhread, varname)
            dims = nhread.variables[varname].dimensions
            ioutils.make_ncvar_helper(nhwrite, varname, nhread.variables[varname][:], dims, **attributes)

        # Add the new variables
        dims = ('index', 'altitude')
        ioutils.make_ncvar_helper(nhwrite, 'eqlat', eqlat, dims, long_name='equivalent_latitude', units='degrees_north',
                                  description='Equivalent latitude computed from potential vorticity')
        ioutils.make_ncvar_helper(nhwrite, 'age', age, dims, long_name='clams_age', units='years',
                                  description='Time in years since the air entered the stratosphere')

        ioutils.add_creation_info(nhwrite, creation_note='ace_fts_analysis.generate_ace_age_file')
        nhwrite.setncattr('template_ace_file', orig_ace_file)
        ioutils.add_dependent_file_hash(nhwrite, 'template_ace_sha1', orig_ace_file)


def strat_co(ch4, T, P, age, co_0, oh=2.5e5, gamma=3):
    R = 8.314 * 100 ** 3 / 100 / 6.626e23

    def ndens_air(T, P):
        return P / (R * T)

    def kCO_OH(nair):
        return 1.57e-13 + 3.54e-33 * nair

    def kCH4_OH(T):
        return 2.8e-14 * T ** 0.667 * np.exp(-1575 / T)

    ndens = ndens_air(T, P)
    age = age * 365.25 * 24 * 3600
    ss = gamma * kCH4_OH(T) * ch4 / kCO_OH(ndens)
    bc = co_0 * np.exp(-kCO_OH(ndens) * oh * age)
    return ss + bc


def make_excess_co_lut(save_file, ace_co_file, ace_ch4_file, ace_age_file, lat='eq', min_req_pts=10):
    R = 8.314 * 100 ** 3 / 100 / 6.626e23
    lat_bins = np.array([-90., -55., -20., 20., 55., 90.])
    lat_bin_centers = np.array([-55., -37.5, 0., 37.5, 55.])

    def ndens_air(T, P):
        return P / (R * T)

    def kCO_OH(nair):
        return 1.57e-13 + 3.54e-33 * nair

    def kCH4_OH(T):
        return 2.8e-14 * T ** 0.667 * np.exp(-1575 / T)

    def strat_co(ch4, T, P, age, co_0, oh=2.5e5, gamma=3):
        ndens = ndens_air(T, P)
        age = age * 365.25 * 24 * 3600
        ss = gamma * kCH4_OH(T) * ch4 / kCO_OH(ndens)
        bc = co_0 * np.exp(-kCO_OH(ndens) * oh * age)
        return ss + bc

    #################
    # Load ACE data #
    #################

    with ncdf.Dataset(ace_age_file) as ds:
        ace_dates = read_ace_date(ds)
        if lat == 'geog':
            ace_lat = read_ace_var(ds, 'latitude', None)
        elif lat == 'eq':
            ace_lat = read_ace_var(ds, 'eqlat', None)
            # 55 km up is just above where the excess CO start to become apparent, so it's a good level to base
            # equivalent latitude on.
            ace_lat = ace_lat[:, 55]
        else:
            raise ValueError('lat must be "geog" or "eq"')

        ace_alt = read_ace_var(ds, 'altitude', None)

        qflags = read_ace_var(ds, 'quality_flag', None)
        ace_t = read_ace_var(ds, 'temperature', qflags)
        ace_p = read_ace_var(ds, 'pressure', qflags) * 1013.25  # atm -> hPa
        ace_age = read_ace_var(ds, 'age', qflags)

    with ncdf.Dataset(ace_ch4_file) as ds:
        qflags = read_ace_var(ds, 'quality_flag', None)
        ace_ch4 = read_ace_var(ds, 'CH4', qflags)

    with ncdf.Dataset(ace_co_file) as ds:
        qflags = read_ace_var(ds, 'quality_flag', None)
        all_ace_co = read_ace_var(ds, 'CO', qflags)

    ace_dates = pd.DatetimeIndex(ace_dates)

    ########################################################
    # Calculate the excess CO for each latitude bin/season #
    ########################################################

    season_months = OrderedDict([('Jan', [1]),
                                 ('Feb', [2]),
                                 ('Mar', [3]),
                                 ('Apr', [4]),
                                 ('May', [5]),
                                 ('Jun', [6]),
                                 ('Jul', [7]),
                                 ('Aug', [8]),
                                 ('Sep', [9]),
                                 ('Oct', [10]),
                                 ('Nov', [11]),
                                 ('Dec', [12])])

    seasons = OrderedDict([(k, np.isin(ace_dates.month, v)) for k, v in season_months.items()])

    season_mid_doys = [(k, mod_utils.day_of_year(dt.datetime(2001, v[0], 15))) for k, v in season_months.items()]
    season_mid_doys = OrderedDict(season_mid_doys)

    co_calc = strat_co(ace_ch4, ace_t, ace_p, ace_age, 50.0e-9)
    xx_nans = np.isnan(all_ace_co) | np.isnan(co_calc)
    all_ace_co[xx_nans] = np.nan
    co_calc[xx_nans] = np.nan

    n_lev = np.size(ace_alt)
    n_seas = len(seasons)
    n_bins = len(lat_bins) - 1

    diff_array = np.full([n_lev, n_seas, n_bins], np.nan)
    pres_array = np.full([n_lev, n_seas, n_bins], np.nan)

    for iseas, xx_seas in enumerate(seasons.values()):
        for ibin in range(len(lat_bins)-1):
            bin_start = lat_bins[ibin]
            bin_stop = lat_bins[ibin+1]

            xx_lat = (ace_lat >= bin_start) & (ace_lat < bin_stop)
            xx_this = xx_seas & xx_lat

            this_calc_co = np.nanmean(co_calc[xx_this], axis=0)
            this_ace_co = np.nanmean(all_ace_co[xx_this], axis=0)
            these_counts = np.sum(~xx_nans[xx_this], axis=0)
            diff_array[:, iseas, ibin] = (this_ace_co - this_calc_co)*1e9
            diff_array[these_counts < min_req_pts, iseas, ibin] = np.nan
            pres_array[:, iseas, ibin] = np.nanmean(ace_p[xx_this], axis=0)
            pres_array[these_counts < min_req_pts, iseas, ibin] = np.nan

    _save_excess_co_lut(save_file=save_file, co_excess=diff_array, lat_bins=lat_bins, lat_bin_centers=lat_bin_centers,
                        pressure=pres_array, lat_type=lat, seasons=season_mid_doys, altitude=ace_alt,
                        ace_co_file=ace_co_file, ace_ch4_file=ace_ch4_file, ace_age_file=ace_age_file)


def _save_excess_co_lut(save_file, co_excess, pressure, lat_bins, lat_bin_centers, seasons, altitude, lat_type,
                        ace_co_file, ace_ch4_file, ace_age_file):
    #lat_bin_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    season_names = np.array(list(seasons.keys()))
    doys = np.array(list(seasons.values()))

    with ncdf.Dataset(save_file, 'w') as nh:
        alt_dim = ioutils.make_ncdim_helper(nh, 'altitude', altitude, unit='km')
        doy_dim = ioutils.make_ncdim_helper(nh, 'doy', doys, unit='day of year')
        lat_dim = ioutils.make_ncdim_helper(nh, 'lat_bin', lat_bin_centers, unit='degrees_north')

        ioutils.make_ncdim_helper(nh, 'lat_bin_edges', lat_bins)
        ioutils.make_ncvar_helper(nh, 'seasons', season_names, [doy_dim])

        ioutils.make_ncvar_helper(nh, 'co_excess', co_excess, [alt_dim, doy_dim, lat_dim],
                                  unit='ppb', description='Excess CO in ACE-FTS data beyond that predicted by a '
                                                          'kinetic model of the CO chemistry in the stratosphere.')
        ioutils.make_ncvar_helper(nh, 'pressure', pressure, [alt_dim, doy_dim, lat_dim],
                                  unit='hPa', description='Average pressure levels for the CO excess profiles')

        ioutils.add_dependent_file_hash(nh, 'ace_co_file', ace_co_file)
        ioutils.add_dependent_file_hash(nh, 'ace_ch4_file', ace_ch4_file)
        ioutils.add_dependent_file_hash(nh, 'ace_age_file', ace_age_file)
        ioutils.add_creation_info(nh, 'ace_fts_analysis.create_excess_co_lut')
        nh.setncattr('latitude_type', 'Binned by {} latitude'.format(lat_type))