from __future__ import print_function, division
import netCDF4 as ncdf
import numpy as np
import os
import pandas as pd
import sys

_mydir = os.path.dirname(__file__)
sys.path.append(os.path.join(_mydir, '..'))
import mod_utils


_theta_bin_edges = np.arange(220.0, 350.0, 2.0)
_theta_bin_centers = (_theta_bin_edges[:-1] + _theta_bin_edges[1:])/2.0


class GEOSError(Exception):
    pass


class GEOSDimensionError(GEOSError):
    pass


class GEOSCoordinateError(GEOSError):
    pass


class ProgressBar(object):
    def __init__(self, num_symbols, prefix='', suffix='', add_one=True, style='*'):
        if len(prefix) > 0 and not prefix.endswith(' '):
            prefix += ' '
        if len(suffix) > 0 and not suffix.startswith(' '):
            suffix = ' ' + suffix

        if style == '*':
            self._fmt_str = '{pre}[{{pstr:<{n}}}]{suf}'.format(pre=prefix, n=num_symbols, suf=suffix)
        elif style == 'counter':
            self._fmt_str = '{pre}{{i:>{l}}}/{n}{suf}'.format(pre=prefix, n=num_symbols, suf=suffix, l=len(str(num_symbols)))
        else:
            raise ValueError('style "{}" not recognized'.format(style))
        self._add_one = add_one

    def print_bar(self, i):
        if self._add_one:
            i += 1

        pstr = '*' * i
        pbar = self._fmt_str.format(pstr=pstr, i=i)
        sys.stdout.write('\r' + pbar)
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write('\n')
        sys.stdout.flush()


def bin_lat_vs_theta(lat, theta, pres_lev, target_pres=700, percentiles=np.arange(10.0, 100.0, 10.0), bin_var='theta'):
    if np.ndim(theta) < 3 or np.ndim(theta) > 4:
        raise ValueError('theta expected to be a 3- or 4- D array')
    elif np.ndim(lat) != 1:
        raise ValueError('lat expected to be a 1D array')
    elif np.ndim(pres_lev) != 1:
        raise ValueError('pres_lev expected to be a 1D array')

    # If theta is 3D, make it 4D with only length 1 in the first dimension
    if np.ndim(theta) == 3:
        theta = theta.reshape((1,) + theta.shape)
    elif theta.shape[1] != np.size(pres_lev) or theta.shape[2] != np.size(lat):
        raise ValueError('theta is expected to have dimensions [ntimes by] nlev by nlat by nlon. The given input has '
                         'shape {}'.format(theta.shape))

    # This assumes that the same vector of pressures defines the levels for every column in the met data. This holds
    # true for GEOS-FP data, where the values are on a fixed pressure grid, but will need revisited if other met data
    # not on a fixed pressure grid is used.
    if np.size(target_pres) == 1:
        # levels always needs to be array-like so that we don't squeeze out the vertical dimension in theta when
        # averaging
        levels = [np.argmin(np.abs(pres_lev - target_pres))]
    elif np.size(target_pres) == 2:
        levels = (pres_lev >= np.max(target_pres)) & (pres_lev < np.min(target_pres))
    else:
        raise TypeError('target_pres must either be a scalar value or a two-element collection')

    n_times, n_lev, n_lat, n_lon = theta.shape

    lat_vec = np.zeros((n_times, n_lat, n_lon))
    lat_vec[:] = lat.reshape((1, n_lat, 1))
    theta_vec = np.mean(theta[:, levels, :, :], axis=1)

    theta_vec = theta_vec.flatten()
    lat_vec = lat_vec.flatten()

    # Now bin the latitude and theta and calculate the statistics on each bin
    bin_ops = {'mean': np.nanmean, 'median': np.nanmedian, 'std': np.nanstd, 'count': np.size,
               'percentiles': lambda x: np.percentile(x, percentiles)}
    # Figure out how large an array each operation returns - necessary to create the fill arrays if there's no data
    # in a bin
    dummy_arr = np.arange(100)
    bin_ops_size = {name: np.size(op(dummy_arr)) for name, op in bin_ops.items()}

    # digitize gives the indices of the upper edge of the bind, we want the bottom because that will work with the bin
    # centers
    if bin_var == 'theta':
        bin_centers = _theta_bin_centers
        bin_edges = _theta_bin_edges
        bin_inds = np.digitize(theta_vec, bin_edges) - 1
    elif bin_var == 'latitude':
        # Since the input latitude is always the same, we can calculate the bin edges from it
        lat_half_res = np.mean(np.diff(lat))/2.0
        bin_edges = np.concatenate((lat - lat_half_res, lat[-1] + lat_half_res))
        bin_centers = lat
        bin_inds = np.digitize(lat_vec, bin_edges) - 1
    else:
        raise ValueError('bin_var "{}" is not recognized'.format(bin_var))

    lat_bin_stats = []
    theta_bin_stats = []
    for bin in range(np.size(_theta_bin_edges) - 1):
        i_bin = bin_inds == bin
        bin_lats = lat_vec[i_bin]
        bin_thetas = theta_vec[i_bin]

        lat_stat_dict = dict()
        theta_stat_dict = dict()

        for name, op in bin_ops.items():
            # Some operations do not allow for an empty array, so deliberately set those to nans
            if np.size(bin_lats) == 0:
                n_values = bin_ops_size[name]
                lat_stat_dict[name] = np.full((n_values,), np.nan)
                theta_stat_dict[name] = np.full((n_values,), np.nan)
            else:
                lat_stat_dict[name] = op(bin_lats)
                theta_stat_dict[name] = op(bin_thetas)

        lat_bin_stats.append(lat_stat_dict)
        theta_bin_stats.append(theta_stat_dict)

    # Rearrange each stat into an array. First need to figure out the size of the array. For consistency, all will be
    # 2D and be bins-by-values, so the means will be n-by-1 and the percentiles n-by-9 by default.
    n_bins = np.size(theta_bin_stats)
    theta_bin_out = dict()
    lat_bin_out = dict()
    for op in bin_ops:
        # Get the result from the first bin to figure out the size in the 2nd dimension
        n_values = bin_ops_size[op]
        theta_op_results = np.full((n_bins, n_values), np.nan)
        lat_op_results = np.full((n_bins, n_values), np.nan)
        for i_bin in range(n_bins):
            theta_op_results[i_bin, :] = theta_bin_stats[i_bin][op]
            lat_op_results[i_bin, :] = lat_bin_stats[i_bin][op]

        theta_bin_out[op] = theta_op_results
        lat_bin_out[op] = lat_op_results

    return bin_centers, theta_bin_out, lat_bin_out, percentiles


def load_geos_data(geos_path, start_date, end_date, hours=None, product='fpit', skip_if_missing=False):
    if hours is None:
        # We always want hour to be iterable, so if it is a tuple with None in it, then when we pass None to
        # geosfp_file_names, we'll get files between the start and end dates for all hours.
        hours = (hours,)

    print('Loading GEOS files for {} to {}'.format(start_date, end_date))
    # Make the last day exclusive
    end_date -= pd.Timedelta(days=1)
    for hr in hours:
        geos_names = mod_utils.geosfp_file_names(product, 'Np', pd.date_range(start=start_date, end=end_date), utc_hours=hr)
        pbar = ProgressBar(len(geos_names), prefix='  Progress:', add_one=True, style='counter')
        for idx, fname in enumerate(geos_names):
            full_name = os.path.join(geos_path, fname)
            if not os.path.isfile(full_name):
                if skip_if_missing:
                    return None, None, None
                else:
                    raise GEOSError('Missing one of the files required for {} to {}'.format(start_date, end_date))
            with ncdf.Dataset(full_name, 'r') as nci:
                pbar.print_bar(idx)
                # Need just latitude, theta and pressure levels. For the first file, just load theta (and make sure it's
                # 4D). For the second file, append theta and verify that lat and pres levels are the same
                this_lat = nci.variables['lat'][:]
                this_plevs = nci.variables['lev'][:]
                this_temperature = nci.variables['T'][:]
                if np.ndim(this_temperature) == 3:
                    sz = (1,) + this_temperature.shape
                    this_temperature = this_temperature.reshape(sz)
                elif np.ndim(this_temperature) != 4:
                    raise GEOSDimensionError('Temperature from GEOS file expected to be 3- or 4- D, not {}D'
                                             .format(np.ndim(this_temperature)))

                this_theta = mod_utils.calculate_model_potential_temperature(this_temperature, pres_levels=this_plevs)
                this_theta = this_theta.filled(np.nan)

                if idx == 0:
                    lat = this_lat
                    plevs = this_plevs
                    theta = np.full((len(geos_names),) + this_theta.shape[1:], np.nan)
                    chk_file = full_name
                else:
                    if not np.allclose(lat, this_lat):
                        raise GEOSCoordinateError('Latitudes from "{new_file}" differs from those in "{chk_file}"'
                                                  .format(new_file=full_name, chk_file=chk_file))
                    elif not np.allclose(plevs, this_plevs):
                        raise GEOSCoordinateError('Pressure levels from "{new_file}" differ from those in "{chk_file}"'
                                                  .format(new_file=full_name, chk_file=chk_file))

                theta[idx] = this_theta

        pbar.finish()

    return lat, plevs, theta


def _cat_stats(stats):
    def cat_stat_dict():
        out_dict = dict()
        stat_keys = stats[0].keys()
        for k in stat_keys:
            cat_arr = np.concatenate([s[k][np.newaxis] for s in stats], axis=0)
            out_dict[k] = cat_arr

        return out_dict

    def cat_stat_simple():
        return np.concatenate([s[np.newaxis] for s in stats], axis=0)

    if isinstance(stats[0], dict):
        return cat_stat_dict()
    else:
        return cat_stat_simple()


def _convert_dates(dates, base_date, calendar='gregorian'):
    units_str = 'seconds since {}'.format(base_date.strftime('%Y-%m-%d %H:%M:%S'))
    date_arr = ncdf.date2num(dates, units_str, calendar=calendar)
    return date_arr, units_str, calendar


def save_bin_ncdf_file(save_name, dates, hours, bin_var, bin_centers, theta_stats, lat_stats, percentiles):
    def make_dim_helper(nc_handle, dim_name, dim_var, attrs=dict()):
        dim = nc_handle.createDimension(dim_name, np.size(dim_var))
        var = nc_handle.createVariable(dim_name, dim_var.dtype, dimensions=(dim_name,))
        var[:] = dim_var
        var.setncatts(attrs)
        return dim

    def make_var_helper(nc_handle, var_name, var_data, dims, attrs=dict()):
        dim_names = tuple([d.name for d in dims])
        var = nc_handle.createVariable(var_name, var_data.dtype, dimensions=dim_names)
        var[:] = var_data
        var.setncatts(attrs)

    def make_stat_dims(stat, std_dims, avail_dims):
        dim_size_map = {d.size: d for d in avail_dims}
        if len(dim_size_map) < len(avail_dims):
            raise NotImplementedError('One or more dimensions have the same size')

        dims = list(std_dims)
        for i in range(len(std_dims), np.ndim(stat)):
            dims.append(dim_size_map[stat.shape[i]])

        return dims

    if hours is None or hours == (None,):
        hours = np.full((1,), np.nan)
    else:
        hours = np.array(hours)

    # dates is a list of tuples of start, mid, and end dates. zip(*dates) will make an iterator that returns all start,
    # then all mid, then all end dates in turn.
    base_date = pd.Timestamp(1970, 1, 1)
    start_dates, mid_dates, end_dates = [_convert_dates(d, base_date)[0] for d in zip(*dates)]

    # convert one date to get the proper CF-style units string and calendar
    _, date_unit_str, date_calendar = _convert_dates(dates[0][0], base_date)
    common_date_attrs = {'units': date_unit_str, 'calendar': date_calendar}

    with ncdf.Dataset(save_name, 'w') as nch:
        # First create the dimensions. Some don't have corresponding variables, so we don't use the helper function
        # for those
        times_attr = dict(description='Datetime at the middle of the binning period', **common_date_attrs)
        times_dim = make_dim_helper(nch, 'times', mid_dates, attrs=times_attr)

        hours_dim = make_dim_helper(nch, 'hours', hours, attrs={'units': 'UTC hour', 'description': 'UTC hour of the GEOS files used'})
        bins_dim = nch.createDimension('bins', bin_centers.shape[2])
        percentiles_dim = make_dim_helper(nch, 'percentiles', percentiles)
        values_dim = nch.createDimension('stat_values', 1)

        std_3d_dims = (times_dim, hours_dim, bins_dim)
        available_dims = [values_dim, percentiles_dim]

        # Start writing the variables
        make_var_helper(nch, 'start_date', start_dates, (times_dim,), attrs=common_date_attrs)
        make_var_helper(nch, 'end_date', end_dates, (times_dim,), attrs=common_date_attrs)
        bin_center_var_name = 'theta_bin_centers' if bin_var == 'theta' else 'latitude_bin_centers'
        make_var_helper(nch, bin_center_var_name, bin_centers, std_3d_dims,
                        attrs={'units': 'K', 'description': 'Potential temperature at the center of the bin'})

        for var_name, var_unit, var in [('theta', 'K', theta_stats), ('latitude', 'degrees (south is negative)', lat_stats)]:
            for stat_name, stat in var.items():
                # For each of the stats variables, we need to add Nones to the dimension list to represent extra
                # dimensions that need to be created by make_var_helper
                stat_dims = make_stat_dims(stat, std_3d_dims, available_dims)
                stat_nc_name = '{var}_{stat}'.format(var=var_name, stat=stat_name)
                description = '{stat} of {var} in the bin'.format(stat=stat_name, var=var_name)
                stat_attrs = {'units': var_unit, 'description': description}
                make_var_helper(nch, stat_nc_name, stat, stat_dims, attrs=stat_attrs)


def iter_time_periods(year, freq):
    def calc_end_date(sdate):
        return sdate + freq

    def calc_mid_date(sdate):
        return sdate + freq / 2

    start_date = pd.Timestamp(year, 1, 1)
    end_date = calc_end_date(start_date)
    mid_date = calc_mid_date(start_date)
    while end_date.year == year:
        yield start_date, mid_date, end_date
        start_date = end_date
        end_date = calc_end_date(end_date)
        mid_date = calc_mid_date(start_date)


def make_geos_lat_v_theta_climatology(year, geos_path, save_path, freq=pd.Timedelta(days=14), by_hour=False,
                                      skip_if_missing=False, save_separate_files=False, bin_var='latitude'):
    if not by_hour:
        hours = (None,)
    else:
        # careful in python 3 - this will return an iterator, so you would need to reset it if you needed to iterate
        # over it again
        hours = range(0, 24, 3)

    all_bin_centers = []
    all_theta_stats = []
    all_lat_stats = []
    dates = []

    for start, midpoint, end in iter_time_periods(year, freq):
        bin_centers = []
        theta_stats = []
        lat_stats = []
        do_save = True
        for hr in hours:
            lat, plevs, theta = load_geos_data(geos_path, start, end, hours=(hr,), skip_if_missing=skip_if_missing)
            if lat is None:
                do_save = False
                continue
            this_bin_centers, this_theta_stats, this_lat_stats, percentiles = bin_lat_vs_theta(lat, theta, plevs,
                                                                                               bin_var=bin_var)
            bin_centers.append(this_bin_centers)
            theta_stats.append(this_theta_stats)
            lat_stats.append(this_lat_stats)

        if do_save:
            # Concatenate the individual hours in the theta_stats, lat_stats, and bin_centers lists into single arrays
            # that are (nhours)x(nbins)x(nvalues)
            if save_separate_files:
                bin_centers = _cat_stats(bin_centers)[np.newaxis]
                theta_stats = _cat_stats(theta_stats)
                for k, v in theta_stats.items():
                    theta_stats[k] = v[np.newaxis]
                lat_stats = _cat_stats(lat_stats)
                for k, v in lat_stats.items():
                    lat_stats[k] = v[np.newaxis]
                dates = [(start, midpoint, end)]

                if by_hour:
                    nc_file_name = 'GEOS_FPIT_lat_vs_theta_{}-{}_by_hour.nc'.format(start.strftime('%Y%m%d'), end.strftime('%Y%m%d'))
                else:
                    nc_file_name = 'GEOS_FPIT_lat_vs_theta_{}-{}.nc'.format(start.strftime('%Y%m%d'), end.strftime('%Y%m%d'))

                nc_file_name = os.path.join(save_path, nc_file_name)
                save_bin_ncdf_file(nc_file_name, dates, hours, bin_centers, theta_stats, lat_stats,
                                   percentiles=percentiles)
            else:
                all_bin_centers.append(_cat_stats(bin_centers))
                all_theta_stats.append(_cat_stats(theta_stats))
                all_lat_stats.append(_cat_stats(lat_stats))
                dates.append((start, midpoint, end))

    if not save_separate_files:
        # Convert to (ndays)x(nhours)x(nbins)x(nvalues)
        all_bin_centers = _cat_stats(all_bin_centers)
        all_theta_stats = _cat_stats(all_theta_stats)
        all_lat_stats = _cat_stats(all_lat_stats)
        if by_hour:
            nc_file_name = 'GEOS_FPIT_lat_vs_theta_{}_by_hour.nc'.format(year)
        else:
            nc_file_name = 'GEOS_FPIT_lat_vs_theta_{}.nc'.format(year)

        nc_file_name = os.path.join(save_path, nc_file_name)
        save_bin_ncdf_file(nc_file_name, dates, hours, bin_var, all_bin_centers, all_theta_stats, all_lat_stats,
                           percentiles=percentiles)

