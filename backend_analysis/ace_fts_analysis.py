from __future__ import print_function, division

from collections import OrderedDict
import datetime as dt
from dateutil.relativedelta import relativedelta
import netCDF4 as ncdf
import numpy as np
import pandas as pd
import os
import re
from scipy.optimize import curve_fit
import sys

_mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_mydir, '..'))
import mod_utils, ioutils, tccon_priors

_tccon_top_alt = 65.0

def make_fch4_fn2o_lookup_table(ace_n2o_file, ace_ch4_file, lut_save_file):

    n2o_record = tccon_priors.N2OTropicsRecord()
    ace_alt, ace_fn2o, ace_theta = calc_fraction_remaining_from_acefts(ace_n2o_file, 'N2O', n2o_record,
                                                                       tropopause_approach='theta')
    ch4_record = tccon_priors.CH4TropicsRecord()
    _, ace_fch4, _ = calc_fraction_remaining_from_acefts(ace_ch4_file, 'CH4', ch4_record,
                                                         tropopause_approach='theta')

    with ncdf.Dataset(ace_ch4_file, 'r') as nch:
        ace_ch4_raw = nch.variables['CH4'][:].filled(np.nan)
        ace_lat = nch.variables['latitude'][:].filled(np.nan)

    # We're looking for reliable stratospheric relationships. Therefore we limit to data where the concentration is
    # positive and definitely not tropospheric (CH4 < 2e-6 i.e. < 2000 ppb), not in the polar vortex (abs(lat) < 50)
    # and not in the mesosphere or upper stratosphere (alt < 40).
    xx = ~np.isnan(ace_fch4) & ~np.isnan(ace_fn2o) & (ace_fch4 >= 0) & (ace_fn2o >= 0) & (ace_ch4_raw < 2e-6) & \
         (np.abs(ace_lat[:, np.newaxis]) < 50) & (ace_alt[np.newaxis, :] < _tccon_top_alt)

    # Define bins for F(N2O) and theta
    fn2o_bins = np.arange(0.0, 1.05, 0.05)
    fn2o_bin_centers = fn2o_bins[:-1] + np.diff(fn2o_bins) / 2

    #theta_bins = np.concatenate([np.arange(380, 680, 20), np.arange(680, 1080, 50), np.arange(1080, 1680, 100)])
    #theta_bins = np.concatenate([np.arange(380, 1080, 50), np.arange(1080, 1680, 100)])
    theta_bins = np.concatenate([np.arange(380, 1080, 50), np.arange(1080, 1680, 100), np.arange(1680, 3680, 200)])
    theta_bin_centers = theta_bins[:-1] + np.diff(theta_bins) / 2

    # Find outliers to avoid overly noisy relationships
    oo = _find_ch4_outliers(ace_fn2o, ace_fch4, fn2o_bins, xx)

    # Do the actual binning
    fch4_means, fch4_counts, fch4_overall, theta_overall = _bin_fch4(ace_fn2o, ace_fch4, ace_theta, xx & oo,
                                                                     theta_bins, fn2o_bins)
    _save_fch4_lut(lut_save_file, fch4_means, fch4_counts, fch4_overall, theta_overall,
                   fn2o_bin_centers, fn2o_bins, theta_bin_centers, theta_bins)


def make_hf_ch4_slopes(ace_ch4_file, ace_hf_file, washenfelder_supp_table_file, lut_save_file, ch4=None):
    if ch4 is None:
        ch4 = tccon_priors.CH4TropicsRecord()

    with ncdf.Dataset(ace_ch4_file, 'r') as nch_ch4, ncdf.Dataset(ace_hf_file, 'r') as nch_hf:
        ace_ch4 = nch_ch4.variables['CH4'][:].filled(np.nan)
        ace_ch4_err = nch_ch4.variables['CH4_error'][:].filled(np.nan)
        ace_hf = nch_hf.variables['HF'][:].filled(np.nan)
        ace_hf_err = nch_hf.variables['HF_error'][:].filled(np.nan)
        ace_lat = nch_ch4.variables['latitude'][:].filled(np.nan)
        ace_ch4_qual = nch_ch4.variables['quality_flag'][:].filled(9)
        ace_hf_qual = nch_ch4.variables['quality_flag'][:].filled(9)

        ace_alt = nch_ch4.variables['altitude'][:].filled(np.nan)
        ace_alt = np.tile(ace_alt, [ace_ch4.shape[0], 1])

        ace_year = nch_ch4.variables['year'][:].filled(np.nan)
        ace_dates = read_ace_date(nch_ch4)
        ace_dates = np.tile(ace_dates.reshape(-1, 1), [1, 150])

    # We do the same filtering as for the F(N2O):F(CH4) relationship, plus we require that the CH4 and HF error is < 5%.
    xx = ~np.isnan(ace_ch4) & ~np.isnan(ace_ch4_err) & (ace_ch4_err / ace_ch4 < 0.05) & ~np.isnan(ace_hf) \
         & ~np.isnan(ace_hf_err) & (ace_hf_err / ace_hf < 0.05) & (ace_ch4 >= 0.0) & (ace_ch4 <= 2e-6) \
         & (ace_hf >= 0.0) & (ace_alt < _tccon_top_alt) & (ace_ch4_qual == 0) & (ace_hf_qual == 0)

    # Read in the early slopes from Washenfelder et al. 2003 (doi: 10.1029/2003GL017969), table S3.
    washenfelder_df = pd.read_csv(washenfelder_supp_table_file, header=2, sep=r'\s+')
    washenfelder_slopes = washenfelder_df.set_index('Date').b.astype(np.float)
    washenfelder_slopes.index = pd.DatetimeIndex(washenfelder_slopes.index)
    washenfelder_slopes.name = 'slope'

    # For each bin, get the slopes from the ACE-FTS data. Fit the data (including the Washenfelder slopes) to an
    # exponential.
    lat_bin_edges = OrderedDict([('Antarctic', (-90.0, -60.0)),
                                 ('SMidlats', (-60.0, -30.0)),
                                 ('Tropics', (-30.0, 30.0)),
                                 ('NMidlats', (30.0, 60.0)),
                                 ('Arctic', (60.0, 90.0))])
    lat_bin_slopes = pd.DataFrame(index=_get_ace_date_range(ace_dates), columns=lat_bin_edges.keys())

    fit_params = np.zeros([len(lat_bin_edges), 4], dtype=np.float)

    for ibin, (bin_name, lat_bin) in enumerate(lat_bin_edges.items()):
        for year_start_date in lat_bin_slopes.index:
            year = year_start_date.year
            print(bin_name, year)
            lat_bin_slopes.loc[year_start_date, bin_name] = _bin_and_fit_hf_vs_ch4(ace_lat, ace_year, ace_dates,
                                                                                   ace_ch4, ace_hf, ch4, xx, year,
                                                                                   lat_bin)

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

    import pdb; pdb.set_trace()
    _save_hf_ch4_lut(lut_save_file, lat_bin_edges, full_dtindex, lat_bin_slopes, fit_params)

    return lat_bin_slopes


def _find_ch4_outliers(ace_fn2o, ace_fch4, fn2o_bins, good_data):
    # from https://stackoverflow.com/a/16562028
    def isoutlier(data, m=2):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        return s >= m

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
        oo[bb] = ~isoutlier(ch4_sub, m=5)

    return oo


def _bin_fch4(ace_fn2o, ace_fch4, ace_theta, good_data, theta_bins, fn2o_bins):
    n_fn2o_bins = fn2o_bins.size - 1
    n_theta_bins = theta_bins.size - 1
    fch4_means = np.full([n_fn2o_bins, n_theta_bins], np.nan)
    fch4_counts = np.zeros_like(fch4_means, dtype=np.int)

    for i, (tlow, thigh) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
        zz = good_data & (ace_theta >= tlow) & (ace_theta < thigh)
        n2o_sub = ace_fn2o[zz]
        ch4_sub = ace_fch4[zz]
        bin_inds = np.digitize(n2o_sub, fn2o_bins) - 1

        for j in range(n_fn2o_bins):
            jj = bin_inds == j
            fch4_means[j, i] = np.nanmean(ch4_sub[jj])
            fch4_counts[j, i] = np.sum(jj)

    fch4_overall = np.full([n_fn2o_bins], np.nan)
    theta_overall = np.full([n_fn2o_bins], np.nan)
    bin_inds = np.digitize(ace_fn2o, fn2o_bins) - 1

    for j in range(n_fn2o_bins):
        jj = (bin_inds == j) & good_data
        fch4_overall[j] = np.nanmean(ace_fch4[jj])
        theta_overall[j] = np.nanmean(ace_theta[jj])

    return fch4_means, fch4_counts, fch4_overall, theta_overall


def _bin_and_fit_hf_vs_ch4(ace_lat, ace_year, ace_dates, ace_ch4, ace_hf, ch4, xx, year, lat_bin):
    yy = (ace_lat >= lat_bin[0]) & (ace_lat < lat_bin[1]) & (np.isclose(ace_year, year))
    yy = yy.reshape(-1, 1)

    ace_lagged_dates = pd.DatetimeIndex([d - relativedelta(months=2) for d in ace_dates[xx & yy]])
    profile_ch4_sbc = ch4.get_gas_for_dates(ace_lagged_dates)

    x = ace_hf[xx & yy] * 1e9
    y = ace_ch4[xx & yy] * 1e9 - profile_ch4_sbc

    # By default, numpy's polynomials are normalized to make fitting more reliable. convert() puts it back to the
    # original units.
    poly = np.polynomial.polynomial.Polynomial.fit(x, y, 1).convert()

    return poly.coef[1]  # returns the slope


def _get_year_avg_ch4_sbc(year, ch4):
    tt = (ch4.conc_seasonal.index >= dt.datetime(year, 1, 1)) & (ch4.conc_seasonal.index < dt.datetime(year + 1, 1, 1))
    return ch4.conc_seasonal.dmf_mean[tt].mean()


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


def _save_hf_ch4_lut(nc_filename, lat_bin_edges, full_date_index, ace_slopes, slope_fit_params):
    with ncdf.Dataset(nc_filename, 'w') as nch:

        ch4_hf_slopes = pd.DataFrame(index=full_date_index, columns=ace_slopes.columns)
        ch4_hf_source = pd.DataFrame(index=full_date_index, columns=ace_slopes.columns).fillna(1)
        lat_bin_edges_df = pd.DataFrame(index=pd.Int64Index([0, 1]), columns=ace_slopes.columns)
        # transpose so that the bins dim is second like the above dataframes
        bin_names_char_array = ncdf.stringtochar(np.array(lat_bin_edges.keys())).T
        slope_fit_params = slope_fit_params.T

        # For each latitude bin, make a pandas series for the full date index from the fit first, then replace available
        # years with the actual fit
        for ibin, (bin_name, bin_edges) in enumerate(lat_bin_edges.items()):
            lat_bin_edges_df.loc[:, bin_name] = bin_edges
            ch4_hf_slopes.loc[:, bin_name] = mod_utils.hf_ch4_slope_fit(ch4_hf_slopes.index.year, *slope_fit_params[:, ibin])
            ch4_hf_slopes.loc[ace_slopes.index, bin_name] = ace_slopes.loc[:, bin_name]
            ch4_hf_source.loc[ace_slopes.index, bin_name] = 0

        # Setup dimensions.
        nbins = len(lat_bin_edges)
        nchars = bin_names_char_array.shape[0]
        nparams = slope_fit_params.shape[0]
        year_dim = ioutils.make_ncdim_helper(nch, 'year', full_date_index.year.to_numpy())
        bins_dim = nch.createDimension('latitude_bins', size=nbins)
        bin_edges_dim = nch.createDimension('latitude_bin_edges', size=2)
        char_dim = nch.createDimension('bin_name_length', size=nchars)
        fit_params_dim = nch.createDimension('exp_fit_params_length', size=nparams)

        # Save the variables
        ioutils.make_ncvar_helper(nch, 'ch4_hf_slopes', ch4_hf_slopes.to_numpy().astype(np.float), [year_dim, bins_dim],
                                  units='ppb CH4/ppb HF',
                                  description='Slope of stratospheric CH4 concentrations vs HF concentrations from '
                                              'ACE-FTS data for specific years and latitude bins')
        ioutils.make_ncvar_helper(nch, 'ch4_hf_slopes_source', ch4_hf_source.to_numpy().astype(np.int), [year_dim, bins_dim],
                                  units='N/A',
                                  description='Flag indicating how the slope was computed. 0 means taken directly from '
                                              'fits of ACE-FTS CH4 vs. HF; 1 means sampled from the exponential fit '
                                              'of ACE-FTS + Washenfelder 03 slopes')
        ioutils.make_ncvar_helper(nch, 'slope_fit_params', slope_fit_params, [fit_params_dim, bins_dim],
                                  description='Fitting parameters a, b, c, and t0 for a function of form '
                                              '"a * exp(b*(t - t0)) + c" used to fit the temporal trend of CH4 vs. HF '
                                              'slopes, where t is the year (e.g. 2004)')
        ioutils.make_ncvar_helper(nch, 'bin_edges', lat_bin_edges_df.to_numpy().astype(np.float), [bin_edges_dim, bins_dim],
                                  units='degrees_north',
                                  description='The edges of the latitude bins')
        ioutils.make_ncvar_helper(nch, 'bin_names', bin_names_char_array, [char_dim, bins_dim],
                                  description='The names used for the latitude bins')


def calc_fraction_remaining_from_acefts(nc_file, gas_name, gas_record, tropopause_approach='theta'):
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
    def read_var(nc_handle, varname, qflags):
        data = nc_handle.variables[varname][:].filled(np.nan)
        if qflags is not None:
            data[qflags != 0] = np.nan
        return data

    tropopause_approach = tropopause_approach.lower()

    with ncdf.Dataset(nc_file, 'r') as nch:
        alt = read_var(nch, 'altitude', qflags=None)
        ace_dates = read_ace_date(nch)
        qflags = read_var(nch, 'quality_flag', qflags=None)

        gas_conc = read_var(nch, gas_name, qflags=qflags)
        # ACE data appears to use -999 as fill value in the concentrations
        gas_conc[gas_conc < -900.0] = np.nan

        temperature = read_var(nch, 'temperature', qflags=qflags)
        pressure = read_var(nch, 'pressure', qflags=qflags) * 1013.25  # Pressure given in atm, need hPa
        theta = mod_utils.calculate_potential_temperature(pressure, temperature)

    # Define a function to calculate the tropopause. Must be named `get_tropopause` and take one argument (the profile
    # index), and return the tropopause altitude as a scalar float.

    if tropopause_approach == 'wmo':
        def get_tropopause(prof_indx):
            return mod_utils.calc_wmo_tropopause(alt, temperature[prof_indx, :], raise_error=False)
    elif tropopause_approach == 'theta':
        def get_tropopause(prof_indx):
            theta_prof = theta[prof_indx, :]

            # Find all values above 380 K. If none exist (probably all NaNs) then we can't find the tropopause - return
            # NaN. Otherwise get the first level below 380 K.
            zz = np.flatnonzero(theta_prof > 380)
            if zz.size == 0:
                return np.nan
            else:
                zz = zz[0] - 1
            # Assuming that theta is increasing in the stratosphere, necessary for interpolation
            assert np.all(np.diff(theta_prof[zz:])) > 0, 'Theta is not monotonically increasing above 380 K'
            return np.interp(380.0, theta_prof[[zz, zz+1]], alt[[zz, zz+1]])
    else:
        raise ValueError('No tropopause calculation defined for tropopause_approach == "{}"'.format(tropopause_approach))

    # Need a conversion factor between the gas record (which may be in ppbv, ppmv, etc) and the ACE data (in mixing
    # ratio, no scaling).
    gas_unit = gas_record.gas_unit
    if re.match(r'ppmv?$', gas_unit):
        scale_factor = 1e-6
    elif re.match(r'ppbv?$', gas_unit):
        scale_factor = 1e-9
    else:
        raise NotImplementedError('No scale factor defined for SBC record unit "{}"'.format(gas_unit))

    # The gas variables are shaped (nprofiles) by (nlevels). Iterate over the profiles, calculate the tropopause for
    # each profile, and limit data to just data above the tropopause. Look up the MLO/SMO stratospheric boundary
    # condition for this date and normalize the profile to that.

    pbar = mod_utils.ProgressBar(gas_conc.shape[0], style='counter', prefix='Profile:')

    for iprof in range(gas_conc.shape[0]):
        pbar.print_bar(iprof)

        trop_alt = get_tropopause(iprof)
        if np.isnan(trop_alt):
            # Couldn't find tropopause, eliminate this profile from consideration
            gas_conc[iprof, :] = np.nan
            theta[iprof, :] = np.nan
            continue

        sbc_date = ace_dates[iprof] - gas_record.sbc_lag
        sbc_conc = gas_record.get_gas_for_dates(sbc_date).item() * scale_factor

        # The fraction remaining will be defined as relative to the tropopause concentration. In the troposphere, set it
        # to NaN because it doesn't make sense to define the fraction remaining there
        gas_conc[iprof, :] /= sbc_conc
        gas_conc[iprof, alt < trop_alt] = np.nan

    pbar.finish()

    return alt, gas_conc, theta


def read_ace_date(ace_nc_handle, out_type=dt.datetime):
    """
    Read datetimes from an ACE-FTS file

    :param ace_nc_handle: the handle to a netCDF4 dataset for an ACE-FTS file. (Must have variables year, month, day,
     and hour.)
    :type ace_nc_handle: :class:`netCDF4.Dataset`

    :param out_type: the type to return the dates as. May be any time that meets two criteria:

        1. Must be able to be called as ``out_type(year, month, day)`` where year, month, and day are integers to
           produce a datetime.
        2. Must be able to be added to a :class:`datetime.timedelta`

    :return: a numpy array of dates, as type ``out_type``.
    :rtype: :class:`numpy.ndarray`
    """
    ace_years = ace_nc_handle.variables['year'][:].filled(np.nan)
    ace_months = ace_nc_handle.variables['month'][:].filled(np.nan)
    ace_days = ace_nc_handle.variables['day'][:].filled(np.nan)

    ace_hours = ace_nc_handle.variables['hour'][:].filled(np.nan)
    ace_hours = ace_hours.astype(np.float64)  # timedelta demands a 64-bit float, can't be 32-bit

    dates = [out_type(y, m, d) + dt.timedelta(hours=h) for y, m, d, h in zip(ace_years, ace_months, ace_days, ace_hours)]
    return np.array(dates)


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