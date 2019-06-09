from __future__ import print_function, division

import argparse
import datetime as dt
import os
import sys

from . import ace_fts_analysis as afa, geos_theta_lat as gtl
from .backend_utils import ACEFileError, find_ace_file
from ...common_utils.ggg_logging import logger

# TODO: once I've programmed a dependency package, make this smart about what files it regenerates

_my_dir = os.path.abspath(os.path.dirname(__file__))
_output_dir = os.path.abspath(os.path.join(_my_dir, '..', 'data'))


def create_lookup_files(ace_dir, geos_dir, backup_existing_files=True):
    _create_fch4_fn2o_lut(ace_dir, backup_existing_files=backup_existing_files)
    _create_hf_slopes_lut(ace_dir, backup_existing_files=backup_existing_files)
    #_create_geos_theta_vs_lat(geos_dir, backup_existing_files=backup_existing_files)


def _backup_file(filename):
    if not os.path.exists(filename):
        return

    new_name = filename + '.bak.{}'.format(dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.rename(filename, new_name)


def _create_fch4_fn2o_lut(ace_dir, backup_existing_files=True):
    logger.important('Creating F(CH4) vs. F(N2O) lookup table from ACE data')
    ace_n2o_file = find_ace_file(ace_dir, 'N2O')
    ace_ch4_file = find_ace_file(ace_dir, 'CH4')
    save_file = os.path.join(_output_dir, 'n2o_ch4_acefts.nc')

    if backup_existing_files:
        _backup_file(save_file)

    afa.make_fch4_fn2o_lookup_table(ace_n2o_file, ace_ch4_file, save_file)


def _create_hf_slopes_lut(ace_dir, backup_existing_files=True):
    logger.important('Creating HF vs. CH4 slope lookup table from ACE data')
    ace_ch4_file = find_ace_file(ace_dir, 'CH4')
    ace_hf_file = find_ace_file(ace_dir, 'HF')
    washenfelder_file = os.path.join(_output_dir, 'washenfelder03_table_s3.txt')
    save_file = os.path.join(_output_dir, 'ch4_hf_slopes.nc')

    if backup_existing_files:
        _backup_file(save_file)

    afa.make_hf_ch4_slopes(ace_ch4_file, ace_hf_file, washenfelder_file, save_file)


def _create_geos_theta_vs_lat(geos_dir, backup_existing_files=True):
    logger.important('Creating GEOS theta vs. latitude climatology')
    by_hour = False
    target_pres = [500, 700]
    year = 2018

    if backup_existing_files:
        clim_save_name = gtl.lat_v_theta_clim_name(year, target_pres, by_hour)
        _backup_file(clim_save_name)

    gtl.make_geos_lat_v_theta_climatology(year, os.path.join(geos_dir, 'Np'), _output_dir, by_hour=by_hour, target_pres=target_pres)


def parse_args():
    # TODO: add mechanic to force regenerating or not regenerating specific files
    parser = argparse.ArgumentParser(description='Regenerate the lookup tables and other data inputs necessary for '
                                                 'GGG priors')
    parser.add_argument('ace_dir', help='Directory containing ACE-FTS data')
    parser.add_argument('geos_dir', help='Directory containing the GEOS-FP or FPIT data must have subdirectories Np '
                                         'and Nx containing the 3D and 2D fields, respectively.')
    parser.add_argument('--no-backup', '-x', dest='backup_existing_files', action='store_false',
                        help='Do not backup existing output files')

    args = vars(parser.parse_args())
    return args


def main():
    args = parse_args()
    create_lookup_files(**args)


if __name__ == '__main__':
    main()
