from __future__ import print_function, division

import datetime as dt
import netCDF4 as ncdf
import numpy as np
import os
import re
import sys

_mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_mydir, '..'))
import mod_utils, ioutils, tccon_priors


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
         (np.abs(ace_lat[:, np.newaxis]) < 50) & (ace_alt[np.newaxis, :] < 65)

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