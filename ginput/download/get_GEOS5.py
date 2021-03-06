#!~/anaconda2/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
from datetime import datetime, timedelta
from subprocess import Popen, PIPE, CalledProcessError
import sys

from . import download_utils as dlutils
from ..common_utils.ggg_logging import logger

####################
# Code Description #
####################
"""
Functions to create list of URLs and/or download them
"""

#############
# Functions #
#############
_file_types = ('met', 'chm')
_default_file_type = 'met'
_std_out_paths = {'surf': 'Nx', 'p': 'Np', 'eta': 'Nv'}
_level_types = tuple(_std_out_paths.keys())
_default_level_type = 'p'


def execute(cmd,cwd=os.getcwd()):
    '''
    function to execute a unix command and print the output as it is produced
    '''
    popen = Popen(cmd, stdout=PIPE, universal_newlines=True,cwd=cwd)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise CalledProcessError(return_code, cmd)


def URLlist_FP(start, end, timestep=timedelta(hours=3), outpath='', filetype=_default_file_type,
               levels=_default_level_type):
    """
    GEOS5-FP data has one global file every 3 hours (from 00:00 to 21:00 UTC each day)
    start: datetime object, start of the desired date range
    end: datetime object, end of the desired date range
    timestep: use the model time resolution to get all files, or a multiple of it to get less files
    outpath: full path to the file in which the list of URLs will be written
    """
    filetype = filetype.lower()
    if filetype == 'met':
        if levels == 'surf':
            fmt = "https://portal.nccs.nasa.gov/datashare/gmao_ops/pub/fp/das/Y{}/M{:0>2}/D{:0>2}/GEOS.fp.asm.inst3_2d_asm_Nx.{}_{:0>2}00.V01.nc4\n"
        elif levels == 'p':
            fmt = "https://portal.nccs.nasa.gov/datashare/gmao_ops/pub/fp/das/Y{}/M{:0>2}/D{:0>2}/GEOS.fp.asm.inst3_3d_asm_Np.{}_{:0>2}00.V01.nc4\n"
        else:
            raise ValueError('No FP URL format defined for filetype == {} and levels == {}'.format(filetype, levels))
    else:
        raise ValueError('No FP URL format defined for filetype == {}'.format(filetype))

    if outpath=='': # if no specified full path to make the file, just write a file in the current directory
        outpath = 'getFP.dat'

    print('Writting URL list in:',outpath)

    curdate = start
    with open(outpath,'w') as f:
        while curdate<end:
            f.write(fmt.format(curdate.year,curdate.month,curdate.day,datetime.strftime(curdate,'%Y%m%d'),curdate.hour))
            curdate += timestep


def URLlist_FPIT(start, end, timestep=timedelta(hours=3), outpath='', filetype=_default_file_type,
                 levels=_default_level_type):
    """
    GEOS5-FP-IT data has one global file every 3 hours (from 00:00 to 21:00 UTC each day)
    start: datetime object, start of the desired date range
    end: datetime object, end of the desired date range
    timestep: use the model time resolution to get all files, or a multiple of it to get less files
    outpath: full path to the file in which the list of URLs will be written
    """
    filetype = filetype.lower()
    levels = levels.lower()

    if filetype == 'met':
        if levels == 'p':
            fmt = "http://goldsfs1.gesdisc.eosdis.nasa.gov/data/GEOS5/DFPITI3NPASM.5.12.4/{yr}/{doy:0>3}/.hidden/GEOS.fpit.asm.inst3_3d_asm_Np.GEOS5124.{ymd}_{hr:0>2}00.V01.nc4\n"
        elif levels == 'eta':
            fmt = "http://goldsfs1.gesdisc.eosdis.nasa.gov/data/GEOS5/DFPITI3NVASM.5.12.4/{yr}/{doy:0>3}/.hidden/GEOS.fpit.asm.inst3_3d_asm_Nv.GEOS5124.{ymd}_{hr:0>2}00.V01.nc4\n"
        elif levels == 'surf':
            fmt = "http://goldsfs1.gesdisc.eosdis.nasa.gov/data/GEOS5/DFPITI3NXASM.5.12.4/{yr}/{doy:0>3}/.hidden/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.{ymd}_{hr:0>2}00.V01.nc4\n"
        else:
            raise ValueError('No FPIT URL format defined for filetype == {} and levels == {}'.format(filetype, levels))
    elif filetype == 'chm':
        if levels == 'eta':
            fmt = "http://goldsfs1.gesdisc.eosdis.nasa.gov/data/GEOS5/DFPITI3NVCHM.5.12.4/{yr}/{doy:0>3}/.hidden/GEOS.fpit.asm.inst3_3d_chm_Nv.GEOS5124.{ymd}_{hr:0>2}00.V01.nc4\n"
        else:
            raise ValueError('Chemistry files only available on eta levels')
    else:
        raise ValueError('No FPIT URL format defined for filetype == {}'.format(filetype))

    if outpath=='': # if no specified full path to make the file, just write a file in the current directory
        outpath = 'getFPIT.dat'

    print('Writing URL list in:',outpath)

    curdate = start
    with open(outpath, 'w') as f:
        while curdate < end:
            f.write(fmt.format(yr=curdate.year, doy=curdate.timetuple().tm_yday, ymd=datetime.strftime(curdate, '%Y%m%d'), hr=curdate.hour))
            curdate += timestep


# Define this here so that we can reference it for the command line help and in the driver function
_func_dict = {'FP':URLlist_FP, 'FPIT':URLlist_FPIT}


def _parse_file_types(clinput):
    if clinput not in _file_types:
        dlutils.eprint('{} is not an allowed file type. Allowed file types are: {}'
                       .format(clinput, ', '.join(_file_types)))
    return clinput


def _parse_level_types(clinput):
    if clinput not in _level_types:
        dlutils.eprint('{} is not an allowed level type. Allowed level types are: {}'
                       .format(clinput, ', '.join(_level_types)))
    return clinput


def parse_args(parser=None):
    description = 'Download GEOSFP or GEOSFP-IT reanalysis met data'
    if parser is None:
        parser = argparse.ArgumentParser(description=description)
        am_i_main = True
    else:
        parser.description = description
        am_i_main = False

    parser.add_argument('date_range', type=dlutils.parse_date_range_no_hm, help=dlutils.date_range_cl_help(False))
    parser.add_argument('--mode', choices=list(_func_dict.keys()), default='FP',
                        help='Which GEOS product to get. The default is %(default)s. Note that to retrieve FP-IT data '
                             'requires a subscription with NASA (https://gmao.gsfc.nasa.gov/GMAO_products/)')
    parser.add_argument('--path', default='.', help='Where to download the GEOS data to. Default is %(default)s. Data '
                                                    'will be placed in Np, Nv, and Nx subdirectories automatically '
                                                    'created in this directory.')
    parser.add_argument('-t', '--filetypes', default=None, choices=_file_types,
                        help='Which file types to download. Default is to download met files')
    parser.add_argument('-l', '--levels', default=None, choices=_level_types,
                        help='Which level type to download. Note that only "eta" levels are available for the "chm" '
                             'file type.')
    parser.epilog = 'If both --filetypes and --levels are omitted, then the legacy behavior is to download met data ' \
                    'for the surface and on fixed pressure levels. However, if one is given, then both must be given.'

    if am_i_main:
        args = vars(parser.parse_args())

        # Go ahead and separate out the two parts of the date range so this dictionary can be used directly for keyword
        # arguments to the driver function
        args['start'], args['end'] = args['date_range']
        return args
    else:
        parser.set_defaults(driver_fxn=driver)


def check_types_levels(filetypes, levels):
    if filetypes is None and levels is None:
        filetypes = ('met', 'met')
        levels = ('surf', 'p')
    elif (filetypes is None) != (levels is None):
        raise TypeError('Both or neither of filetypes and levels must be given (not None)')

    if isinstance(filetypes, str):
        filetypes = (filetypes,)
    if any(f not in _file_types for f in filetypes):
        raise ValueError('filetypes must be one of: {}'.format(', '.join(_file_types)))

    if isinstance(levels, str):
        levels = (levels,)
    if any(l not in _level_types for l in levels):
        raise ValueError('levels must be one of: {}'.format(', '.join(_level_types)))

    if len(filetypes) != len(levels):
        raise ValueError('filetypes and levels must be the same length, if multiple options are given as lists/tuples')

    return filetypes, levels


def driver(date_range, mode='FP', path='.', filetypes=_default_file_type, levels=_default_level_type, **kwargs):
    """
    Run get_GEOS5 as if called from the command line.

    A note on the ``filetypes`` and ``levels`` parameters. If both are ``None``, then this will automatically download
    met data for the surface and fixed pressure levels. Otherwise they can be strings or collections (lists, tuples)
    of strings. Both must be the same length if given as collections (a string counts as a 1-element collection).
    Specifying collections allows you to download multiple sets of files with a single call. For example, the ``None``
    behavior of downloading surface and fixed-pressure level met files is equivalent to setting
    ``filetypes=['met', 'met']`` and ``levels=['surf', 'p']``.

    :param date_range: the range of dates to retrieve as a two-element collection. The second date is exclusive.
    :type date_range: list(datetime-like)

    :param mode: one of the strings "FP" or "FPIT", determines which GEOS product to download.
    :type mode: str

    :param path: the path to download GEOS files to. Files will be automatically sorted into subdirectories "Nx", "Np",
     and "Nv" for surface, pressure-level, and native-level files, respectively. This directories are created if needed.
    :type path: str

    :param filetypes: a string or collection of strings specifying which types of file (meteorology: "met", chemistry:
     "chm") to download.  See the note above about the relationship between ``filetypes`` and ``levels``.
    :type filetypes: str, list(str), or None

    :param levels: a string of collection of strings specifying which level type to download. Options are "surf" for
     2D fields, "p" for fixed pressure level files, and "eta" for the native terrain following levels. See the note
     above about the relationship between ``filetypes`` and ``levels``.
    :type levels: str, list(str), or None

    :param kwargs: unused, swallows extra keyword arguments.

    :return: none, downloads GEOS files to ``path``.
    """
    filetypes, levels = check_types_levels(filetypes, levels)

    start, end = date_range
    for ftype, ltype in zip(filetypes, levels):
        outpath = os.path.join(path, _std_out_paths[ltype])
        if not os.path.exists(outpath):
            logger.info('Creating {}'.format(outpath))
            os.makedirs(outpath)

        _func_dict[mode](start, end, filetype=ftype, levels=ltype, outpath=os.path.join(outpath, 'getGEOS.dat'))
        for line in execute('wget -N -i getGEOS.dat'.split(), cwd=outpath):
            print(line, end="")


########
# Main #
########

if __name__=="__main__":
    arguments = parse_args()
    driver(**arguments)
