from __future__ import print_function, division, absolute_import, unicode_literals
import netCDF4 as ncdf
import numpy as np

from ..tccon_priors import _clams_file
from ...common_utils import ioutils


def modify_clams_file():
    with ncdf.Dataset(_clams_file, 'r') as nhread:
        age = nhread.variables['age'][:]
        theta = nhread.variables['theta'][:]

    new_age, new_theta = _extend_clams_age(age, theta)

    with ncdf.Dataset(_clams_file, 'a') as nhwrite:
        # Can't use ioutils since we're not necessarily writing this file from scratch, in which case the variables
        # will already exist, and can't be recreated, but must be overwritten
        theta_dim_name = 'extended_theta'
        age_var_name = 'extended_age'
        if theta_dim_name not in nhwrite.dimensions:
            nhwrite.createDimension(theta_dim_name, new_theta.size)
        if theta_dim_name not in nhwrite.variables:
            theta_var = nhwrite.createVariable(theta_dim_name, new_theta.dtype, dimensions=(theta_dim_name,))
        else:
            theta_var = nhwrite.variables[theta_dim_name]
        if age_var_name not in nhwrite.variables:
            age_var = nhwrite.createVariable(age_var_name, new_age.dtype, dimensions=('doy', theta_dim_name, 'lat'))
        else:
            age_var = nhwrite.variables[age_var_name]

        theta_var[:] = new_theta
        theta_var.description = 'Potential temperature coordinate for the age array extended by extrapolation'

        age_var[:] = new_age
        age_var.units = 'years'
        age_var.long_name = 'CLaMS mean age climatology scaled to GSFC 2-D model'
        age_var.description = 'A version of the other age variable extended by extrapolation to greater theta values.'

        creation_description = ioutils.make_creation_info(_clams_file, 'backend_analysis.clams.modify_clams_file')
        # Make the note less confusing, since by default it says the netCDF file was "Created by" something, but we're
        # really only adding some variables.
        creation_description = creation_description.replace('Created', 'Extended variables added')
        nhwrite.history = creation_description


def _extend_clams_age(age, theta_vec, end_theta=7500.0, use_median=False):
    def calc_slopes(theta, age):
        return (age[1] - age[0]) / (theta[1] - theta[0])

    def fit_slopes(theta, age):
        m1 = calc_slopes(theta[-3:-1], age[-3:-1])
        m2 = calc_slopes(theta[-2:], age[-2:])
        return 0.5 * (m1 + m2)

    def get_additional_age(this_age):
        slope = fit_slopes(theta_vec, this_age)
        return (end_theta - theta_vec[-1]) * slope

    ndoy, ntheta, nlat = np.shape(age)
    extra_age = np.full([ndoy, 1, nlat], np.nan)
    for idoy in range(ndoy):
        if use_median:
            this_age = age[idoy, -3:, :].median(axis=1)
            extra_age[idoy, :, :] = age[-1, :] + get_additional_age(this_age)
        else:
            for ilat in range(nlat):
                this_age = age[idoy, -3:, ilat]
                extra_age[idoy, 0, ilat] = this_age[-1] + get_additional_age(this_age)

    return np.concatenate([age, extra_age], axis=1), np.concatenate([theta_vec, [end_theta]], axis=0)