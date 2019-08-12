"""
This module contains the code necessary to regenerate the GEOS_FPIT_lat_vs_theta_2018_500-700hPa.nc file.

For the tropospheric prior profiles, we need to know the latitude of the observation in order to determine the age of
the air (to know how far forward or back to look in the MLO/SMO record) and the seasonal cycle (which is strongest in
the northern midlatitudes). During testing, we found that some sites should have more tropical CO2 profiles that their
geographic latitude suggests. That is, the aircraft or aircore CO2 profiles were relatively flat throughout the
troposphere (which is hallmark of the tropics due to rapid vertical mixing) but the priors were assigning them a more
midlatitude-like seasonal cycle.

The correction for this was to use an "equivalent latitude" based on the mid-tropospheric potential temperature. This
originates from work by Gretchen Keppel-Aleks (Keppel-Aleks et al., Biogeosciences, 9, 875-891,
https://doi.org/10.5194/bg-9-875-2012, 2012) that showed that CO2 in the free troposphere correlates much better with
potential temperature at 700 hPa than it does simple geographic latitude (c.f. especially figures 9-11).

In order to use this equivalent latitude based on potential temperature, we need a dataset that allows us to map the
typical mid-tropospheric potential temperature to the equivalent latitude it represents. To get that, we make biweekly
zonal bins of mid-tropospheric potential temperature vs. latitude in this module. The main driver function is
`make_geos_lat_v_theta_climatology`. To reproduce the standard 2018 file mentioned above, it would be called as::

    make_geos_lat_v_theta_climatology(2018, geos_path, '.', target_pres=(500,700))

This would produce GEOS_FPIT_lat_vs_theta_2018_500-700hPa.nc in the current directory, which should then be placed
in the package :file:`data` directory.  2018 was chosen as the year for this climatology because it was the most recent
full year at the time of production, therefore it should be most representative looking backwards to the beginning of
the TCCON record and forward to the future TCCON record.

This method is necessary, rather than using potential vorticity based equivalent latitude, because Pan et al. 2012
(ACP, 12, p. 9187, doi: 10.5194/acp-12-9187-2012) note that when potential temperature surfaces intersect the surface,
potential temperature becomes a poor vertical coordinate. In general, they note that PV/EqLat can conceal tropospheric
chemical patterns.
"""


import netCDF4 as ncdf
import numpy as np
import os
import pandas as pd

from ...common_utils.ioutils import make_ncdim_helper, make_ncvar_helper
from ...common_utils import mod_utils
from ...common_utils.mod_utils import ProgressBar

_mydir = os.path.dirname(__file__)
_theta_bin_edges = np.arange(220.0, 350.0, 2.0)
_theta_bin_centers = (_theta_bin_edges[:-1] + _theta_bin_edges[1:])/2.0


class GEOSError(Exception):
    """
    Base error for problems with the GEOS met data
    """
    pass


class GEOSDimensionError(GEOSError):
    """
    Error for cases where the dimension of a GEOS variable is not as expected
    """
    pass


class GEOSCoordinateError(GEOSError):
    """
    Error for any problem with the coordinates in the GEOS met file
    """
    pass


def bin_lat_vs_theta(lat, theta, pres_lev, target_pres=700, percentiles=np.arange(10.0, 100.0, 10.0), bin_var='latitude'):
    """
    Create zonal bins of latitude and theta and calculate statistics on them.

    This is the main function that handles the calculation of the climatological latitude and potential temperature. It
    can use either latitude or theta as the variable that defines the bins; by default latitude is used.

    :param lat: a vector defining the latitude coordinates for the GEOS potential temperature fields. It will
     automatically be expanded to the proper size to match ``theta``, and is expected to be the same length as the
     second-to-last dimension of ``theta``.
    :type lat: :class:`numpy.ndarray`

    :param theta: an array defining the potential temperature from the GEOS met data.
    :type theta: :class:`numpy.ndarray`

    :param pres_lev: a vector defining the pressure levels as the vertical coordinate for ``theta``. It is expected to
     be the same length as the third-from last dimension of ``theta``.
    :type pres_lev: :class:`numpy.ndarray`

    :param target_pres: the pressure or pressures to calculate theta on. If given as a single number, the level of
     ``theta`` use for the theta bins will be the level closest to it. If given as a two-element collection (e.g.
     ``[500, 700]``), then all layers in theta between those values (inclusive) will be averaged along the vertical
     dimension before calculating the bins. In the second form, order does not matter (the lesser or greater number
     may come first).
    :type target_pres: int, float, or list(int or float)

    :param percentiles: a vector defining the percentiles to calculate for each bin. Note that these are true
     percentiles, i.e. `10` should be used, not `0.1`.
    :type percentiles: array-like

    :param bin_var: which variable to use to define the bins. May be ``'latitude'`` or ``'theta'``.
    :type bin_var: str

    :return: the bin centers, dictionaries for theta and lat containing the results of the binning as an array for each
     statistic (mean, median, std, percentiles, and count), and the vector of percentiles used. This last value is
     returned so that they can be written to the output file even if the default value was used. The statistics arrays
     will be nbins-by-nvalues. For all stats except "percentiles", nvalues is 1 (there is only 1 value per bin). For
     "percentiles", it will be equal to the number of percentiles requested.
    :rtype: :class:`numpy.ndarray`, dict, dict, array-like
    """
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
        levels = (pres_lev >= np.min(target_pres)) & (pres_lev <= np.max(target_pres))
    else:
        raise TypeError('target_pres must either be a scalar value or a two-element collection')

    n_times, n_lev, n_lat, n_lon = theta.shape

    # Average theta over the levels requested. Make lat into an array the same shape as the post-averaging theta. Then
    # flatten these arrays into vectors so that we can deal with them as a linear sequence of values.
    lat_vec = np.zeros((n_times, n_lat, n_lon))
    lat_vec[:] = lat.reshape((1, n_lat, 1))
    theta_vec = np.mean(theta[:, levels, :, :], axis=1)

    theta_vec = theta_vec.flatten()
    lat_vec = lat_vec.flatten()

    # Prepare for the binning
    bin_ops = {'mean': np.nanmean, 'median': np.nanmedian, 'std': np.nanstd, 'count': np.size,
               'percentiles': lambda x: np.percentile(x, percentiles)}
    # Figure out how large an array each operation returns - necessary to create the fill arrays if there's no data
    # in a bin
    dummy_arr = np.arange(100)
    bin_ops_size = {name: np.size(op(dummy_arr)) for name, op in bin_ops.items()}

    # Depending on which variables was told to be used for binning, get the correct bin centers and edges.
    # digitize gives the indices of the upper edge of the bind, we want the bottom because that will work with the bin
    # centers, hence the -1.
    if bin_var == 'theta':
        bin_centers = _theta_bin_centers
        bin_edges = _theta_bin_edges
        bin_inds = np.digitize(theta_vec, bin_edges) - 1
    elif bin_var == 'latitude':
        # Since the input latitude is always the same, we can calculate the bin edges from it
        lat_half_res = np.mean(np.diff(lat))/2.0
        bin_edges = np.concatenate((lat - lat_half_res, lat[-1:] + lat_half_res))
        bin_centers = lat
        bin_inds = np.digitize(lat_vec, bin_edges) - 1
    else:
        raise ValueError('bin_var "{}" is not recognized'.format(bin_var))

    # Actually do the binning. It was easier conceptually to store the calculation of the stats for each bin as a list
    # of dictionaries; later we rearrange these into one array per stat. This could probably be streamlined, but likely
    # isn't worth doing so unless this part is unnecessarily slow.
    lat_bin_stats = []
    theta_bin_stats = []
    for bin in range(np.size(bin_edges) - 1):
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
    """
    Custom function for loading GEOS data specifically for the binning procedure.

    :param geos_path: the path to where the GEOS FP-IT Np files are stored
    :type geos_path: str

    :param start_date: the first date to load GEOS data from
    :type start_date: datetime-like

    :param end_date: the last date (inclusive) to load GEOS data from
    :type end_date: datetime-like

    :param hours: which hours of the day (in UTC) to load GEOS data for. GEOS data is currently available at 0, 3, 6, 9,
     12, 15, 18, and 21 UTC. If this is ``None``, then all hours are loaded.
    :type hours: None or list(int)

    :param product: which GEOS product to load ('fp' or 'fpit')
    :type product: str

    :param skip_if_missing: set to ``True`` to allow there to be missing GEOS files in the date range. If there are,
     then this will simply return three ``None`` values rather than raising an error (which is the default behavior).
     Usually this is only set to ``True`` for testing this module with a subset of GEOS data.
    :type skip_if_missing: bool

    :return: latitude and pressure levels as vectors, theta as an array (ntimes by nlevels by nlat by nlon).
    :rtype: :class:`numpy.ndarray` x 3 or None x 3
    :raises GEOSError: if ``skip_if_missing`` is ``False`` and any expected GEOS files could not be found.

    Note that this function should *only* be used to load data for this module. :func:`mod_utils.read_geos_files` should
    be used elsewhere.
    """
    if hours is None:
        # We always want hour to be iterable, so if it is a tuple with None in it, then we can iterate over it and
        # when we pass None to geosfp_file_names, we'll get files between the start and end dates for all hours.
        hours = (hours,)

    print('Loading GEOS files for {} to {}'.format(start_date, end_date))
    # Make the last day exclusive
    end_date -= pd.Timedelta(days=1)
    for hr in hours:
        geos_names = mod_utils.geosfp_file_names_by_day(product, 'met', 'p', pd.date_range(start=start_date, end=end_date), utc_hours=hr)
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
                    # initialize theta as ntimes by nlevels by nlat by nlon. The last three come from the shape of the
                    # currently read in theta - its first dimension is always 1 because there's 1 time per GEOS file.
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
    """
    Helper function to concatenate lists of statistics along a new first dimension.

    :param stats: a list of arrays or list of dictionaries of arrays.
    :return: a single array or dict of arrays with the original arrays from each list element concatenated along a new
     first dimension.
    """
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
    """
    Helper function to convert dates into CF format.

    :param dates: a collection of dates acceptable to :func:`netCDF4.date2num`
    :param base_date: the date to use in the 'seconds since XX' unit
    :param calendar: which CF calendar to use
    :return: the array of dates as seconds since base_date, the units string, and the calendar name
    """
    units_str = 'seconds since {}'.format(base_date.strftime('%Y-%m-%d %H:%M:%S'))
    date_arr = ncdf.date2num(dates, units_str, calendar=calendar)
    return date_arr, units_str, calendar


def save_bin_ncdf_file(save_name, dates, hours, bin_var, bin_centers, theta_stats, lat_stats, percentiles, target_pres):
    """
    Save the netCDF file containing the theta and latitude bins.

    :param save_name: the name to give the file. If exists, will be overwritten without warning
    :type save_name: str

    :param dates: a list of tuples, which each tuple is the start, midpoint, and end date for each averaging period. The
     list must be the same length as the number of times in ``theta_stats`` and ``lat_stats``
    :type dates: list(tuple(datetime, datetime, datetime))

    :param hours: the vector of hours that corresponds to the second dimension of ``theta_stats`` and ``lat_stats``
     arrays. If ``None``, then this dimension will be given a length of 1 and a value of NaN in the netCDF file.
    :type hours: None or array-like

    :param bin_var: which variable was used to set the bins. Will affect the name of the bin center variable in the
     netCDF file.
    :type bin_var: str

    :param bin_centers: the vector of center points for the bins.
    :type bin_centers: :class:`numpy.ndarray`

    :param theta_stats: the dictionary of stats of theta. Each stat should be a key in the dictionary (e.g. 'mean',
     'std', etc.) and the value should be a ntimes x nhours x nbins x nvalues array. ntimes means the number of time
     periods the year is separated into; nhours will be 1 unless the data were separated into different UTC hours.
    :type theta_stats: dict

    :param lat_stats: the dictionary of bin-by-bin stats for latitude. Same format as ``theta_stats``.
    :type lat_stats: dict

    :param percentiles: the array of percentiles use in the statistics.
    :type percentiles: array-like

    :param target_pres: the pressure or pressures theta was averaged on or between. Will be stored as an attribute in
     the netCDF file.
    :type target_pres: int, float, or list(int or float)

    :return: None
    """
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
        # Record the target pressures as a root attribute
        if isinstance(target_pres, (int, float)):
            target_pres = (target_pres,)
        nch.theta_range = '-'.join(str(p) for p in target_pres)
        
        # First create the dimensions. Some don't have corresponding variables, so we don't use the helper function
        # for those
        times_attr = dict(description='Datetime at the middle of the binning period', **common_date_attrs)
        times_dim = make_ncdim_helper(nch, 'times', mid_dates, **times_attr)

        hours_dim = make_ncdim_helper(nch, 'hours', hours, units='UTC hour', description='UTC hour of the GEOS files used')
        bins_dim = nch.createDimension('bins', bin_centers.shape[2])
        percentiles_dim = make_ncdim_helper(nch, 'percentiles', percentiles)
        values_dim = nch.createDimension('stat_values', 1)

        std_3d_dims = (times_dim, hours_dim, bins_dim)
        available_dims = [values_dim, percentiles_dim]

        # Start writing the variables
        make_ncvar_helper(nch, 'start_date', start_dates, (times_dim,), **common_date_attrs)
        make_ncvar_helper(nch, 'end_date', end_dates, (times_dim,), **common_date_attrs)
        if bin_var == 'theta':
            bin_center_var_name = 'theta_bin_centers'
            bin_center_description = 'Potential temperature at the center of the bin'
            bin_center_units = 'K'
        elif bin_var == 'latitude':
            bin_center_var_name = 'latitude_bin_centers'
            bin_center_description = 'Latitude at the center of the bin'
            bin_center_units = 'degrees_north'

        make_ncvar_helper(nch, bin_center_var_name, bin_centers, std_3d_dims,
                          units=bin_center_units, description=bin_center_description)

        for var_name, var_unit, var in [('theta', 'K', theta_stats), ('latitude', 'degrees_north', lat_stats)]:
            for stat_name, stat in var.items():
                # For each of the stats variables, we need to add Nones to the dimension list to represent extra
                # dimensions that need to be created by make_var_helper
                stat_dims = make_stat_dims(stat, std_3d_dims, available_dims)
                stat_nc_name = '{var}_{stat}'.format(var=var_name, stat=stat_name)
                description = '{stat} of {var} in the bin'.format(stat=stat_name, var=var_name)

                stat_attrs = {'units': var_unit, 'description': description}
                if stat_name == 'count':
                    # Every stat should have the same units as the underlying quantity, except for count, which is just
                    # a number.
                    stat_attrs['units'] = 'number of data points'

                make_ncvar_helper(nch, stat_nc_name, stat, stat_dims, **stat_attrs)


def iter_time_periods(year, freq):
    """
    Iterate over the averaging time periods
    :param year: the year to use
    :type year: datetime-like

    :param freq: how long each time period is
    :type freq: timedelta-like

    :return: iterable over the time periods, yields the start, middle, and end dates of each time period.
    """
    def calc_end_date(sdate):
        return sdate + freq

    def calc_mid_date(sdate):
        return sdate + freq / 2

    start_date = pd.Timestamp(year, 1, 1)
    end_date = calc_end_date(start_date)
    mid_date = calc_mid_date(start_date)
    # Yes, this will omit a day or two at the end of the year. But since 365 % 14 == 1, that means only Dec 31st won't
    # get included in the climatology (or Dec 30-31st in a leap year). That is fine, because we're going to interpolate
    # between the time periods anyway in the main code.
    while end_date.year == year:
        yield start_date, mid_date, end_date
        start_date = end_date
        end_date = calc_end_date(end_date)
        mid_date = calc_mid_date(start_date)


def lat_v_theta_clim_name(year, target_pres, by_hour):
    # Include the target pressure levels in the name to avoid accidentally overwriting files if we want to experiment
    # with different pressure levels.
    if isinstance(target_pres, (int, float)):
        target_pres_name_str = '{}hPa'.format(target_pres)
    else:
        target_pres_name_str = '-'.join(str(p) for p in target_pres) + 'hPa'

    if by_hour:
        nc_file_name = 'GEOS_FPIT_lat_vs_theta_{}_{}_by_hour.nc'.format(year, target_pres_name_str)
    else:
        nc_file_name = 'GEOS_FPIT_lat_vs_theta_{}_{}.nc'.format(year, target_pres_name_str)

    return nc_file_name


def make_geos_lat_v_theta_climatology(year, geos_path, save_path, freq=pd.Timedelta(days=14), by_hour=False,
                                      skip_if_missing=False, bin_var='latitude', target_pres=700):
    """
    The main driver function to product the latitude vs. mid-tropospheric potential temperature climatology.

    :param year: which year to use to generate the climatology
    :type year: int

    :param geos_path: where the GEOS FP or FP-IT Np (i.e. 3D pressure level files) are stored.
    :type geos_path: str

    :param save_path: where to save the resulting netCDF file
    :type save_path: str

    :param freq: how long each averaging time period should be
    :type freq: timedelta-lik

    :param by_hour: set to True to bin different UTC hours' GEOS files separately. This will make the "hours" dimension
     in the netCDF file be > 1. This was an experimental feature and isn't used in the production climatology.
    :type by_hour: bool

    :param skip_if_missing: set to ``True`` to simply omit time periods missing one or more GEOS files from the
     climatology rather than raising an error. This is intended for testing this code with a subset of GEOS data and
     definitely should not be used in production.
    :type skip_if_missing: bool

    :param bin_var: which variable to use to define the bins. May be 'latitude' or 'theta'. Change this to 'theta' at
     your own risk, as it will cause northern and southern latitudes to get binned together, which makes no sense.
     Further, the main prior generating code expects the data to have been binned by latitude.
    :type bin_var: str

    :param target_pres: the pressure or pressures to calculate theta on. If given as a single number, the level of
     ``theta`` use for the theta bins will be the level closest to it. If given as a two-element collection (e.g.
     ``[500, 700]``), then all layers in theta between those values (inclusive) will be averaged along the vertical
     dimension before calculating the bins. In the second form, order does not matter (the lesser or greater number
     may come first).
    :type target_pres: int, float, or list(int or float)

    :return: None, saves netCDF file
    """
    if not by_hour:
        hours = (None,)
    else:
        hours = list(range(0, 24, 3))

    # The stats will be generated in several loops, one over time periods, and one over hours. As we finish each loop,
    # the lists from that loop will get concatenated into single arrays with a new first dimension.
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
            # Usually this will just loop once, but it's in here as an experimental option to treat each UTC time
            # separately. This never ended up being used, even in testing.
            lat, plevs, theta = load_geos_data(geos_path, start, end, hours=(hr,), skip_if_missing=skip_if_missing)
            if lat is None:
                do_save = False
                continue
            this_bin_centers, this_theta_stats, this_lat_stats, percentiles = bin_lat_vs_theta(lat, theta, plevs,
                                                                                               bin_var=bin_var, target_pres=target_pres)
            bin_centers.append(this_bin_centers)
            theta_stats.append(this_theta_stats)
            lat_stats.append(this_lat_stats)

        if do_save:
            # Concatenate the individual hours in the theta_stats, lat_stats, and bin_centers lists into single arrays
            # that are (nhours)x(nbins)x(nvalues)
            all_bin_centers.append(_cat_stats(bin_centers))
            all_theta_stats.append(_cat_stats(theta_stats))
            all_lat_stats.append(_cat_stats(lat_stats))
            dates.append((start, midpoint, end))

    # Convert to (ndays)x(nhours)x(nbins)x(nvalues)
    all_bin_centers = _cat_stats(all_bin_centers)
    all_theta_stats = _cat_stats(all_theta_stats)
    all_lat_stats = _cat_stats(all_lat_stats)

    nc_file_name = lat_v_theta_clim_name(year, target_pres, by_hour)
    nc_file_name = os.path.join(save_path, nc_file_name)
    save_bin_ncdf_file(nc_file_name, dates, hours, bin_var, all_bin_centers, all_theta_stats, all_lat_stats,
                       percentiles=percentiles, target_pres=target_pres)

