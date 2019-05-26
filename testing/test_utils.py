from __future__ import print_function

import datetime as dt
from glob import glob
from hashlib import sha1
import os
import shutil
import sys

_mydir = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
# Add the paths to the mod maker and download utilities.
# TODO: replace with proper relative imports when this is a package
sys.path.append(os.path.abspath(os.path.join(_mydir, '..')))
sys.path.append(os.path.abspath(os.path.join(_mydir, '..', 'download')))

import get_GEOS5

input_data_dir = os.path.join(_mydir, 'test_input_data')
geos_fp_dir = os.path.join(input_data_dir, 'geosfp-it')
geos_sha_file = os.path.join(geos_fp_dir, 'fp_hashes.sha1')
mod_input_dir = os.path.join(input_data_dir, 'mod_files', 'fpit')

output_data_dir = os.path.join(_mydir, 'test_output_data')
mod_output_dir = os.path.join(output_data_dir, 'mod_files', 'fpit')

test_date = dt.datetime(2018, 1, 1)


# Python 2/3 compatibility. Never use "input" in Py2 because it executes whatever input it receives as code
if sys.version_info.major == 2:
    input = raw_input


class InputDataMismatchError(Exception):
    pass


class UserCancelledDownloadError(Exception):
    pass


def hash_file(filename):
    block_size = 2**16
    hash_obj = sha1()
    with open(filename, 'rb') as fobj:
        buf = fobj.read(block_size)
        while len(buf) > 0:
            hash_obj.update(buf)
            buf = fobj.read(block_size)

    return hash_obj.hexdigest()


def write_hash_list(hash_filename, files_to_hash):
    with open(hash_filename, 'w') as wobj:
        for f in files_to_hash:
            sha_sum = hash_file(f)
            # Strip off the leading path to the test data directory, when checking, we add that back on
            # so that the hash file can be used across machines
            local_f = f.replace(input_data_dir, '')
            local_f = local_f.lstrip('/')
            wobj.write('{hash}  {file}\n'.format(hash=sha_sum, file=local_f))


def read_hash_list(hash_filename):
    hash_dict = dict()
    with open(hash_filename, 'r') as fobj:
        for line in fobj:
            hash_obj, filename = line.split()
            filename = os.path.join(input_data_dir, filename)
            hash_dict[filename] = hash_obj

    return hash_dict


def check_hash_list(hash_list_filename):
    hash_dict = read_hash_list(hash_list_filename)
    for filename, hash_hex in hash_dict.items():
        new_hash_hex = hash_file(filename)
        if hash_hex != new_hash_hex:
            return False

    return True


def download_test_geos_data(rebuild=False, rebuild_hash_only=False):
    # Check if we need to download files. We should only need to do this if some of the files are missing. If redownload
    # is True though, that means we're recreating the test data set so we need to delete the existing directories
    # and also recreate the hash file.

    # check hashes, unless we're forcing a rebuild (where we redownload new files and write their hashes to the record)
    if not rebuild and not rebuild_hash_only:
        if not check_hash_list(geos_sha_file):
            print('One or more SHA1 hashes for GEOS FP files do not match expected, or files are missing.\n'
                  'Redownload now (~1 GB)? (y to download, anything else to abort)')
            user_ans = input('')
            if user_ans.lower().strip() != 'y':
                raise UserCancelledDownloadError('User elected not to download GEOS FP data')

        else:
            return

    if not rebuild_hash_only:
        if os.path.exists(os.path.join(geos_fp_dir, 'Np')):
            shutil.rmtree(os.path.join(geos_fp_dir, 'Np'))
        if os.path.exists(os.path.join(geos_fp_dir, 'Nx')):
            shutil.rmtree(os.path.join(geos_fp_dir, 'Nx'))

        get_GEOS5.driver(start=test_date, end=test_date+dt.timedelta(days=1), mode='FP', path=geos_fp_dir)

    if rebuild or rebuild_hash_only:
        surf_files = glob(os.path.join(geos_fp_dir, 'Nx', 'GEOS*.nc4'))
        prof_files = glob(os.path.join(geos_fp_dir, 'Np', 'GEOS*.nc4'))
        all_files = sorted(surf_files + prof_files)
        write_hash_list(geos_sha_file, all_files)
    elif not check_hash_list(geos_sha_file):
        raise InputDataMismatchError('After downloading the GEOS FP data, the hashes still do not match what is '
                                     'expected. The most likely explanation is that a new version of GEOS FP was '
                                     'released. If you are not a maintainer for GGG, please contact one and provide '
                                     'this error message. If you are a maintainer, then you will need to rerun the '
                                     'test suite with the new GEOS FP version and ensure the correct .mod files are '
                                     'produced, then update the SHA1 hash file.')


def iter_mod_file_pairs(base_dir, test_dir):
    site_dirs = sorted([os.path.basename(p) for p in glob(os.path.join(base_dir, '*')) if os.path.isdir(p)])
    for sdir in site_dirs:
        base_site_dir = os.path.join(base_dir, sdir, 'vertical')
        all_base_site_files = sorted(glob(os.path.join(base_site_dir, '*.mod')))

        for base_file in all_base_site_files:
            test_file = os.path.join(test_dir, sdir, 'vertical', os.path.basename(base_file))
            if not os.path.exists(test_file):
                raise InputDataMismatchError('Could not find a test file corresponding to the base file {}'
                                             .format(base_file))
            else:
                yield base_file, test_file