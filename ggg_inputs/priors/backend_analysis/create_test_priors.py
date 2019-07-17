from __future__ import print_function
import argparse
from datetime import datetime as dtime, timedelta as tdel, date as date
from glob import glob
from multiprocessing import Pool
import numpy as np
import os
import re
import shutil
from subprocess import check_call, CalledProcessError
import sys

from .. import tccon_priors
from ...mod_maker import mod_maker
from ...common_utils import mod_utils
from ...download import get_GEOS5

# These should match the arg names in driver()
_req_info_keys = ('gas_name', 'site_file', 'geos_top_dir', 'mod_top_dir', 'prior_top_dir')
_req_info_ispath = ('site_file', 'geos_top_dir', 'mod_top_dir', 'prior_top_dir')
_req_info_help = {'gas_name': 'The name of the gas to generate priors for.',
                  'site_file': 'A CSV file containing the header DATES,LATS,LONS and the date, latitude, and longitude '
                               'of each desired prior (one per line)',
                  'geos_top_dir': 'The directory containing the GEOS FP-IT data in subdirectories Np, Nx, and Nv. '
                                  'This is where it will be downloaded to, if --download is one of the actions.',
                  'mod_top_dir': 'The top directory to save .mod files to or read them from. Must contain '
                                 'subdirectories "fpit/xx/vertical" to read from, these will be automatically created '
                                 'if writing .mod files.',
                  'prior_top_dir': 'The directory to save the .mod files to. Will be organized into subdirectories by '
                                   'date, lat, and lon automatically.'}

_default_file_types = ('2dmet', '3dmet')

def unfmt_lon(lonstr):
    mod_utils.format_lon(lonstr)


def _date_range_str_to_dates(drange_str):
    start_dstr, end_dstr = drange_str.split('-')
    start_date = dtime.strptime(start_dstr, '%Y%m%d')
    end_date = dtime.strptime(end_dstr, '%Y%m%d')
    return start_date, end_date


def read_info_file(info_filename):
    info_dict = {k: None for k in _req_info_keys}
    info_file_dir = os.path.abspath(os.path.dirname(info_filename))
    with open(info_filename, 'r') as fobj:
        for line_num, line in enumerate(fobj):
            key, value = [el.strip() for el in line.split('=')]
            if re.match(r'\s*#', line):
                continue
            elif key not in _req_info_keys:
                print('Ignoring line {} of {}: key "{}" not one of the required keys'.format(line_num, info_filename, key))
                continue
            elif key in _req_info_ispath:
                value = value if os.path.isabs(value) else os.path.join(info_file_dir, value)

            info_dict[key] = value

    for key, value in info_dict.items():
        if value is None:
            raise RuntimeError('Key "{}" was missing from the input file {}'.format(key, info_filename))

    return info_dict


def read_date_lat_lon_file(acinfo_filename, date_fmt='str'):
    with open(acinfo_filename, 'r') as acfile:
        # Skip line 1 - header
        acfile.readline()
        acdates = []
        aclats = []
        aclons = []
        for line in acfile:
            if line.startswith('#'):
                continue
            elif '#' in line:
                line = line.split('#')[0].strip()
            line_parts = line.split(',')
            date_str = line_parts[0]
            date1 = dtime.strptime(date_str, '%Y-%m-%d')
            if date_fmt == 'str':
                date2 = date1 + tdel(days=1)
                acdates.append(date1.strftime('%Y%m%d') + '-' + date2.strftime('%Y%m%d'))
            elif date_fmt == 'datetime':
                acdates.append(date1)
            else:
                raise ValueError('date_fmt must be either "str" or "datetime"')

            aclats.append(float(line_parts[1]))
            aclons.append(float(line_parts[2]))

    return aclons, aclats, acdates


def make_full_mod_dir(top_dir, product):
    return os.path.join(top_dir, product.lower(), 'xx', 'vertical')


def check_geos_files(acdates, download_to_dir, file_type=get_GEOS5._default_file_type,
                     levels=get_GEOS5._default_level_type):
    acdates = [dtime.strptime(d.split('-')[0], '%Y%m%d') for d in acdates]
    is_chem = file_type == 'chm'
    types_name_args = {'surf': {'file_type': 'Nx', 'chem': is_chem},
                       'p': {'file_type': 'Np', 'chem': is_chem},
                       'eta': {'file_type': 'Nv', 'chem': is_chem}}
    missing_files = dict()
    name_args = types_name_args[levels]
    file_names, file_dates = mod_utils.geosfp_file_names_by_day('fpit', utc_dates=acdates, **name_args)
    for f, d in zip(file_names, file_dates):
        d = d.date()
        ffull = os.path.join(download_to_dir, name_args['file_type'], f)
        if not os.path.isfile(ffull):
            if d in missing_files:
                missing_files[d].append(f)
            else:
                missing_files[d] = [f]

    for d in sorted(missing_files.keys()):
        nmissing = len(missing_files[d])
        missingf = set(missing_files[d])
        print('{date}: {n} ({files})'.format(date=d.strftime('%Y-%m-%d'), n=min(8, nmissing), files=', '.join(missingf)))

    print('{} of {} dates missing at least one file'.format(len(missing_files), len(acdates)))


def download_geos(acdates, download_to_dir, file_type=get_GEOS5._default_file_type, levels=get_GEOS5._default_level_type):
    for dates in acdates:
        date_range = _date_range_str_to_dates(dates)
        get_GEOS5.driver(date_range, mode='FPIT', path=download_to_dir, filetypes=file_type, levels=levels)


def make_mod_files(acdates, aclons, aclats, geos_dir, out_dir, nprocs=0):

    print('Will save to', out_dir)
    mod_dir = make_full_mod_dir(out_dir, 'fpit')
    print('  (Listing GEOS files...)')
    geos_files = sorted(glob(os.path.join(geos_dir, 'Np', 'GEOS*.nc4')))
    geos_dates = set([dtime.strptime(re.search(r'\d{8}', f).group(), '%Y%m%d') for f in geos_files])

    mm_args = []

    print('  (Making list of .mod files to generate...)')
    for (dates, lon, lat) in zip(acdates, aclons, aclats):
        start_date, end_date = [dtime.strptime(d, '%Y%m%d') for d in dates.split('-')]
        if start_date not in geos_dates:
            print('Cannot run {}, no GEOS data'.format(start_date))
            continue
        files_complete = []
        for hr in range(0, 24, 3):
            date = start_date.replace(hour=hr)
            mod_file = mod_maker.mod_file_name(date, tdel(hours=3), lat, lon, 'E' if lon > 0 else 'W',
                                               'N' if lat > 0 else 'S', out_dir, round_latlon=False, in_utc=True)
            files_complete.append(os.path.isfile(os.path.join(mod_dir, mod_file)))
        if all(files_complete) and len(files_complete) == 8:
            print('All files for {} at {}/{} complete, skipping'.format(dates, lon, lat))
            continue
        else:
            print('One or more files for {} at {}/{} needs generated'.format(dates, lon, lat))

        these_args = ([start_date, end_date], lon, lat, geos_dir, out_dir, nprocs)
        mm_args.append(these_args)

    if nprocs == 0:
        print('Making .mod files in serial mode')
        for args in mm_args:
            mm_helper(*args)
    else:
        print('Making .mod file in parallel mode with {} processors'.format(nprocs))
        with Pool(processes=nprocs) as pool:
            pool.starmap(mm_helper, mm_args)


def mm_helper(date_range, mm_lon, mm_lat, geos_dir, out_dir, nprocs):
    date_fmt = '%Y-%m-%d'
    print('Generating .mod files at {}/{} for {} to {}'.format(mm_lon, mm_lat,
                                                               date_range[0].strftime(date_fmt),
                                                               date_range[1].strftime(date_fmt)))
    mod_maker.driver(date_range, geos_dir, out_dir, keep_latlon_prec=True, save_in_utc=True,
                     lon=mm_lon, lat=mm_lat, alt=0.0, muted=nprocs > 0)


def make_priors(prior_dir, mod_dir, gas_name, acdates, aclons, aclats, nprocs=0):
    print('Will save to', prior_dir)
    # Find all the .mod files, get unique date/lat/lon (should be 8 files per)
    # and make an output directory for that
    mod_files = glob(os.path.join(mod_dir, '*.mod'))
    grouped_mod_files = dict()
    acdates = [dtime.strptime(d.split('-')[0], '%Y%m%d').date() for d in acdates]
    aclons = np.array(aclons)
    aclats = np.array(aclats)

    for f in mod_files:
        fbase = os.path.basename(f)
        lonstr = re.search(r'\d{1,3}.\d\d[EW]', fbase).group()
        latstr = re.search(r'\d{1,2}.\d\d[NS]', fbase).group()
        datestr = re.search(r'^\d{8}_\d{4}Z', fbase).group()

        utc_datetime = dtime.strptime(datestr, '%Y%m%d_%H%MZ')
        utc_date = utc_datetime.date()
        utc_datestr = utc_datetime.date().strftime('%Y%m%d')
        lon = mod_utils.format_lon(lonstr)
        lat = mod_utils.format_lat(latstr)

        # If its one of the profiles in the info file, make it
        if utc_date in acdates and np.any(np.abs(aclons - lon) < 0.02) and np.any(np.abs(aclats - lat) < 0.02):
            print(f, 'matches one of the listed profiles!')
            keystr = '{}_{}_{}'.format(utc_datestr, lonstr, latstr)
            if keystr in grouped_mod_files:
                grouped_mod_files[keystr].append(f)
            else:
                grouped_mod_files[keystr] = [f]
                this_out_dir = os.path.join(prior_dir, keystr)
                if os.path.isdir(this_out_dir):
                    shutil.rmtree(this_out_dir)
                os.makedirs(this_out_dir)
        else:
            print(f, 'is not for one of the profiles listed in the lat/lon file; skipping')

    print('Instantiating {} record'.format(gas_name))
    if gas_name.lower() == 'co2':
        gas_rec = tccon_priors.CO2TropicsRecord()
    elif gas_name.lower() == 'n2o':
        gas_rec = tccon_priors.N2OTropicsRecord()
    elif gas_name.lower() == 'ch4':
        gas_rec = tccon_priors.CH4TropicsRecord()
    elif gas_name.lower() == 'hf':
        gas_rec = tccon_priors.HFTropicsRecord()
    elif gas_name.lower() == 'co':
        gas_rec = tccon_priors.COTropicsRecord()
    else:
        raise RuntimeError('No record defined for gas_name = "{}"'.format(gas_name))

    prior_args = []

    for k, files in grouped_mod_files.items():
        this_out_dir = os.path.join(prior_dir, k)
        for f in files:
            fbase = os.path.basename(f)
            datestr = re.search(r'^\d{8}_\d{4}Z', fbase).group()
            obs_date = dtime.strptime(datestr, '%Y%m%d_%H%MZ')

            these_args = (f, obs_date, this_out_dir, gas_rec)
            prior_args.append(these_args)

    if nprocs == 0:
        for args in prior_args:
            _prior_helper(*args)
    else:
        with Pool(processes=nprocs) as pool:
            pool.starmap(_prior_helper, prior_args)


def _prior_helper(ph_f, ph_obs_date, ph_out_dir, gas_rec):
    _fbase = os.path.basename(ph_f)
    print('Processing {} ({}), saving to {}'.format(_fbase, ph_obs_date.strftime('%Y-%m-%d'), ph_out_dir))
    tccon_priors.generate_single_tccon_prior(ph_f, ph_obs_date, tdel(hours=0), gas_rec, write_map=ph_out_dir,
                                             use_eqlat_strat=True)


def driver(check_geos, download, makemod, makepriors, site_file, geos_top_dir, mod_top_dir, prior_top_dir, gas_name,
           nprocs, dl_file_types, dl_levels):
    aclons, aclats, acdates = read_date_lat_lon_file(site_file)
    if check_geos:
        check_geos_files(acdates, geos_top_dir, file_type=dl_file_types, levels=dl_levels)

    if download:
        download_geos(acdates, geos_top_dir, file_type=dl_file_types,levels=dl_levels)
    else:
        print('Not downloading GEOS data')

    if makemod:
        make_mod_files(acdates, aclons, aclats, geos_top_dir, mod_top_dir, nprocs=nprocs)
    else:
        print('Not making .mod files')

    if makepriors:
        make_priors(prior_top_dir, make_full_mod_dir(mod_top_dir, 'fpit'), gas_name,
                    acdates=acdates, aclons=aclons, aclats=aclats, nprocs=nprocs)
    else:
        print('Not making priors')


def parse_args():
    parser = argparse.ArgumentParser('Run priors for a set of dates, lats, and lons')
    parser.add_argument('info_file', help='The file that defines the configuration variables. Pass "format" as this '
                                          'argument for more details on the format.')
    parser.add_argument('--check-geos', action='store_true', help='Check if the required GEOS files are already downloaded')
    parser.add_argument('--download', action='store_true', help='Download GEOS FP-IT files needed for these priors.')
    parser.add_argument('--makemod', action='store_true', help='Generate the .mod files for these priors.')
    parser.add_argument('--makepriors', action='store_true', help='Generate the priors as .map files.')
    parser.add_argument('-n', '--nprocs', default=0, type=int, help='Number of processors to use to run in parallel mode '
                                                          '(for --makemod and --makepriors only)')
    parser.add_argument('--dl-file-types', default=get_GEOS5._default_file_type, choices=get_GEOS5._file_types,
                        help='Which GEOS file types to download with --download (no effect if --download not specified).')
    parser.add_argument('--dl-levels', default=get_GEOS5._default_level_type, choices=get_GEOS5._level_types,
                        help='Which GEOS levels to download with --download (no effect if --download not specified).')

    return vars(parser.parse_args())


def print_config_help():
    prologue = """The info file is a simple text file where the lines follow the format

key = value

where key is one of {keys}. 
All keys are required; order does not matter. 
The value expected for each key is:""".format(keys=', '.join(_req_info_keys))
    epilogue = """The keys {paths} are file paths. 
These may be given as absolute paths, or as relative paths. 
If relative, they will be taken as relative to the 
location of the info file.""".format(paths=', '.join(_req_info_ispath))

    print(prologue + '\n')
    for key, value in _req_info_help.items():
        print('* {}: {}'.format(key, value))

    print('\n' + epilogue)


def main():
    args = parse_args()
    info_file = args.pop('info_file')
    if info_file == 'format':
        print_config_help()
        sys.exit(0)
    else:
        info_dict = read_info_file(info_file)

    args.update(info_dict)
    driver(**args)


if __name__ == '__main__':
    main()
