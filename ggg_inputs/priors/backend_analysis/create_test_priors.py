from __future__ import print_function
import argparse
from datetime import datetime as dtime, timedelta as tdel
from glob import glob
import os
import re
import shutil
from subprocess import check_call, CalledProcessError
import sys

from .. import tccon_priors
from ...mod_maker import mod_maker
from ...common_utils import mod_utils

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

def unfmt_lon(lonstr):
    mod_utils.format_lon(lonstr)


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


def read_date_lat_lon_file(acinfo_filename):
    with open(acinfo_filename, 'r') as acfile:
        # Skip line 1 - header
        acfile.readline()
        acdates = []
        aclats = []
        aclons = []
        for line in acfile:
            if line.startswith('#'):
                continue
            line_parts = line.split(',')
            date_str = line_parts[0]
            date1 = dtime.strptime(date_str, '%Y-%m-%d')
            date2 = date1 + tdel(days=1)
            acdates.append(date1.strftime('%Y%m%d') + '-' + date2.strftime('%Y%m%d'))

            aclats.append(float(line_parts[1]))
            aclons.append(float(line_parts[2]))

    return aclons, aclats, acdates


def make_full_mod_dir(top_dir, product):
    return os.path.join(top_dir, product.lower(), 'xx', 'vertical')


def download_geos(acdates, download_to_dir):
    raise NotImplementedError('Not updated to work with packaged ggg_inputs')
    for dates in acdates:
        try:
            check_call(['python', 'get_GEOS5.py', dates, 'mode=FPIT', 'path={}'.format(download_to_dir)],
                       cwd=download_bin_dir)
        except CalledProcessError:
            print('Could not download GEOS-FP for {}'.format(dates))


def make_mod_files(acdates, aclons, aclats, geos_dir, out_dir):
    print('Will save to', out_dir)
    mod_dir = make_full_mod_dir(out_dir, 'fpit')
    geos_files = sorted(glob(os.path.join(geos_dir, 'Np', 'GEOS*.nc4')))
    geos_dates = set([dtime.strptime(re.search(r'\d{8}', f).group(), '%Y%m%d') for f in geos_files])

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
            print('One or more files for {} at {}/{} needs generated, proceeding'.format(dates, lon, lat))

        mod_maker.driver(start_date, end_date, geos_dir, out_dir, keep_latlon_prec=True, save_in_utc=True,
                         lon=lon, lat=lat, alt=0.0)


def make_priors(prior_dir, mod_dir, gas_name):
    print('Will save to', prior_dir)
    # Find all the .mod files, get unique date/lat/lon (should be 8 files per)
    # and make an output directory for that
    mod_files = glob(os.path.join(mod_dir, '*.mod'))
    grouped_mod_files = dict()
    for f in mod_files:
        fbase = os.path.basename(f)
        lonstr = re.search(r'\d{1,3}.\d\d[EW]', fbase).group()
        latstr = re.search(r'\d{1,2}.\d\d[NS]', fbase).group()
        datestr = re.search(r'^\d{8}_\d{4}Z', fbase).group()
        utc_date = dtime.strptime(datestr, '%Y%m%d_%H%MZ')
        utc_datestr = utc_date.date().strftime('%Y%m%d')
        keystr = '{}_{}_{}'.format(utc_datestr, lonstr, latstr)
        if keystr in grouped_mod_files:
                grouped_mod_files[keystr].append(f)
        else:
                grouped_mod_files[keystr] = [f]
                this_out_dir = os.path.join(prior_dir, keystr)
                if os.path.isdir(this_out_dir):
                        shutil.rmtree(this_out_dir)
                os.makedirs(this_out_dir)

    print('Instantiating {} record'.format(gas_name))
    if gas_name.lower() == 'co2':
        gas_rec = tccon_priors.CO2TropicsRecord()
    elif gas_name.lower() == 'n2o':
        gas_rec = tccon_priors.N2OTropicsRecord()
    elif gas_name.lower() == 'ch4':
        gas_rec = tccon_priors.CH4TropicsRecord()
    elif gas_name.lower() == 'hf':
        gas_rec = tccon_priors.HFTropicsRecord()
    else:
        raise RuntimeError('No record defined for gas_name = "{}"'.format(gas_name))

    for k, files in grouped_mod_files.items():
        this_out_dir = os.path.join(prior_dir, k)
        for f in files:
            fbase = os.path.basename(f)
            print('Processing', fbase)
            datestr = re.search(r'^\d{8}_\d{4}Z', fbase).group()
            utc_offset = tdel(hours=0)  # calc_utc_offset(lon)
            obs_date = dtime.strptime(datestr, '%Y%m%d_%H%MZ')

            tccon_priors.generate_single_tccon_prior(f, obs_date, utc_offset, gas_rec, write_map=this_out_dir, use_eqlat_strat=True)


def driver(download, makemod, makepriors, site_file, geos_top_dir, mod_top_dir, prior_top_dir, gas_name):
    aclons, aclats, acdates = read_date_lat_lon_file(site_file)
    if download:
        download_geos(acdates, geos_top_dir)
    else:
        print('Not downloading GEOS data')

    if makemod:
        make_mod_files(acdates, aclons, aclats, geos_top_dir, mod_top_dir)
    else:
        print('Not making .mod files')

    if makepriors:
        make_priors(prior_top_dir, make_full_mod_dir(mod_top_dir, 'fpit'), gas_name)
    else:
        print('Not making priors')


def parse_args():
    parser = argparse.ArgumentParser('Run priors for a set of dates, lats, and lons')
    parser.add_argument('info_file', help='The file that defines the configuration variables. Pass "format" as this '
                                          'argument for more details on the format.')
    parser.add_argument('--download', action='store_true', help='Download GEOS FP-IT files needed for these priors.')
    parser.add_argument('--makemod', action='store_true', help='Generate the .mod files for these priors.')
    parser.add_argument('--makepriors', action='store_true', help='Generate the priors as .map files.')

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