from __future__ import print_function, division
import datetime as dt
import numpy as np
import os
import sys
import unittest

_mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_mydir, '..'))

import test_utils, mod_utils
from mod_maker import driver as mmdriver

import pdb

class TestModMaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the GEOS FP/FP-IT data we need for the test if needed, check that the SHA1 sums are what is expected
        pdb.set_trace()
        test_utils.download_test_geos_data()

        # Run mod_maker for the standard test site
        mmdriver(test_utils.test_date, test_utils.test_date+dt.timedelta(days=1), test_utils.geos_fp_dir,
                 save_path=test_utils.mod_output_dir, keep_latlon_prec=True, save_in_utc=True, site_abbrv='oc')

    def test_mod_files(self):
        pdb.set_trace()
        for check_file, new_file in test_utils.iter_mod_file_pairs(test_utils.mod_input_dir, test_utils.mod_output_dir):
            check_mod_data = mod_utils.read_mod_file(check_file)
            new_mod_data = mod_utils.read_mod_file(new_file)

            for category_name, category_data in check_mod_data.items():
                for variable_name, variable_data in category_data.items():
                    new_data = new_mod_data[category_name][variable_name]

                    # Restore subtests in Python 3
                    #with self.subTest(check_file=check_file, new_file=new_file, category=category_name, variable=variable_name):
                    test_result = np.isclose(variable_data, new_data).all()
                    self.assertTrue(test_result, msg='"{variable}" in {filename} does not match the check data'
                                    .format(variable=variable_name, filename=new_file))

                    # print('"{}" in "{}" OK'.format(variable_name, new_file))


if __name__ == '__main__':
    unittest.main()
