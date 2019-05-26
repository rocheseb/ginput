from __future__ import print_function, division

import datetime as dt
from hashlib import sha1
import netCDF4 as ncdf
import numpy as np

import mod_utils


def make_ncdim_helper(nc_handle, dim_name, dim_var, **attrs):
    """
    Create a netCDF dimension and its associated variable simultaneously

    Typically in a netCDF file, each dimension should have a variable with the same name that defines the coordinates
    for that dimension. This function streamlines the process of creating a dimension with its associated variable.

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param dim_name: the name to give both the dimension and its associated variable
    :type dim_name: str

    :param dim_var: the variable to use when defining the dimension's coordinates. The dimensions length will be set
     by the size of this vector. This must be a 1D numpy array or comparable type.
    :type dim_var: :class:`numpy.ndarray`

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the dimension object
    :rtype: :class:`netCDF4.Dimension`
    """
    if np.ndim(dim_var) != 1:
        raise ValueError('Dimension variables are expected to be 1D')
    dim = nc_handle.createDimension(dim_name, np.size(dim_var))
    var = nc_handle.createVariable(dim_name, dim_var.dtype, dimensions=(dim_name,))
    var[:] = dim_var
    var.setncatts(attrs)
    return dim


def make_nctimedim_helper(nc_handle, dim_name, dim_var, base_date=dt.datetime(1970, 1, 1), time_units='seconds',
                          calendar='gregorian', **attrs):
    """
    Create a CF-style time dimension.

    The Climate and Forecast (CF) Metadata Conventions define standardized conventions for how to represent geophysical
    data. Time is one of the trickiest since there are multiple ways of identifying dates that can be ambiguous. The
    standard representation is to give time in units of seconds/minutes/hours/days since a base time in a particular
    calendar. This function handles creating the a time dimension and associated variable from any array-like object of
    datetime-like object.

    For more information, see:

        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#time-coordinate

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param dim_name: the name to give both the dimension and its associated variable
    :type dim_name: str

    :param dim_var: the variable to use when defining the dimension's coordinates. The dimensions length will be set
     by the size of this vector. This must be a 1D numpy array or comparable type.
    :type dim_var: :class:`numpy.ndarray`

    :param base_date: the date and time to make the time coordinate relative to. The default is midnight, 1 Jan 1970.
    :type base_date: datetime-like object

    :param time_units: a string indicating what unit to use as the count of time between the base date and index date.
     Options are 'seconds', 'minutes', 'hours', or 'days'.  This is more restrictive than the CF convention, but avoids
     the potential pitfalls of using months or years.
    :type time_units: str

    :param calendar: one of the calendar types defined in the CF conventions document (section 4.4.1)
    :type calendar: str

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the dimension object, and a dictionary describing the units, calendar, and base date of the time dimension
    :rtype: :class:`netCDF4.Dimension`, dict
    """
    allowed_time_units = ('seconds', 'minutes', 'hours', 'days')
    if time_units not in allowed_time_units:
        raise ValueError('time_units must be one of: {}'.format(', '.join(allowed_time_units)))

    units_str = '{} since {}'.format(time_units, base_date.strftime('%Y-%m-%d %H:%M:%S'))
    # date2num requires that the dates be given as basic datetimes. We'll handle converting Pandas timestamps, either
    # as a series or datetime index, but other types will need handled by the user.
    try:
        date_arr = ncdf.date2num(dim_var, units_str, calendar=calendar)
    except TypeError:
        dim_var = [d.to_pydatetime() for d in dim_var]
        date_arr = ncdf.date2num(dim_var, units_str, calendar=calendar)
    dim = make_ncdim_helper(nc_handle, dim_name, date_arr, **attrs)
    time_info_dict = {'units': units_str, 'calendar': calendar, 'base_date': base_date}
    return dim, time_info_dict


def make_ncvar_helper(nc_handle, var_name, var_data, dims, **attrs):
    """
    Create a netCDF variable and store the data for it simultaneously.

    This function combines call to :func:`netCDF4.createVariable` and assigning the variable's data and attributes.

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param var_name: the name to give the variable
    :type var_name: str

    :param var_data: the array to store in the netCDF variable.
    :type var_data: :class:`numpy.ndarray`

    :param dims: the dimensions to associate with this variable. Must be a collection of either dimension names or
     dimension instances. Both types may be mixed. This works well with :func:`make_ncdim_helper`, since it returns the
     dimension instances.
    :type dims: list(:class:`netCDF4.Dimensions` or str)

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the variable object
    :rtype: :class:`netCDF4.Variable`
    """
    dim_names = tuple([d if isinstance(d, str) else d.name for d in dims])
    var = nc_handle.createVariable(var_name, var_data.dtype, dimensions=dim_names)
    var[:] = var_data
    var.setncatts(attrs)
    return var


def make_creation_info(nc_filename, creation_note=None):
    now = dt.datetime.now()
    commit_hash, branch, _ = mod_utils.hg_commit_info()
    clean_or_dirty = 'clean' if mod_utils.hg_is_commit_clean(ignore_files=[nc_filename]) else 'dirty'
    if creation_note is not None:
        description = 'Created by {note} on {date} (mercurial commit {commit} on branch {branch}, {cleanstate})'
    else:
        description = 'Created with commit {commit} on branch {branch} ({cleanstate}) on {date}'

    description = description.format(note=creation_note, date=now, commit=commit_hash, branch=branch,
                                     cleanstate=clean_or_dirty)
    return description


def add_creation_info(nc_handle, creation_note=None, creation_att_name='history'):
    description = make_creation_info(nc_handle.filepath(), creation_note=creation_note)
    nc_handle.setncattr(creation_att_name, description)


def make_dependent_file_hash(dependent_file):
    hashobj = sha1()
    with open(dependent_file, 'rb') as fobj:
        block = fobj.read(4096)
        while block:
            hashobj.update(block)
            block = fobj.read(4096)

    return hashobj.hexdigest()


def add_dependent_file_hash(nc_handle, hash_att_name, dependent_file):
    hash_hex = make_dependent_file_hash(dependent_file)
    nc_handle.setncattr(hash_att_name, hash_hex)
