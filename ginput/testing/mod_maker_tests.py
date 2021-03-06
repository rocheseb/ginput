from __future__ import print_function, division
import datetime as dt
import numpy as np
import os
import unittest

from . import test_utils
from ..common_utils import mod_utils
from ..priors import tccon_priors
from ..mod_maker.mod_maker import driver as mmdriver


_mydir = os.path.abspath(os.path.dirname(__file__))


class TestModMaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the GEOS FP/FP-IT data we need for the test if needed, check that the SHA1 sums are what is expected
        test_utils.download_test_geos_data()

        # Run mod_maker for the standard test site
        mmdriver([test_utils.test_date, test_utils.test_date+dt.timedelta(days=1)], test_utils.geos_fp_dir,
                 save_path=test_utils.mod_output_top_dir, keep_latlon_prec=True, save_in_utc=True,
                 site_abbrv=test_utils.test_site, include_chm=True, mode='fpit-eta')

        # Run the priors using the check mod files - that way even if mod_maker breaks we can still test the priors
        # Eventually we will probably need two testing modes - one that uses the precalculated strat LUTs and one that
        # recalculates them and either verifies them against the saved LUTs or runs the priors with them.
        mod_files = [f for f in test_utils.iter_mod_file_pairs(test_utils.mod_output_dir, None)]
        tccon_priors.generate_full_tccon_vmr_file(mod_files, dt.timedelta(hours=0), save_dir=test_utils.vmr_output_dir,
                                                  std_vmr_file=test_utils.std_vmr_file, site_abbrevs=test_utils.test_site)

    def test_mod_files(self):
        self._comparison_helper(test_utils.iter_mod_file_pairs, mod_utils.read_mod_file,
                                test_utils.mod_input_dir, test_utils.mod_output_dir)

    def test_vmr_files(self):
        self._comparison_helper(test_utils.iter_vmr_file_pairs, mod_utils.read_vmr_file,
                                test_utils.vmr_input_dir, test_utils.vmr_output_dir)

    def _comparison_helper(self, iter_fxn, read_fxn, input_dir, output_dir):
        for check_file, new_file in iter_fxn(input_dir, output_dir):
            check_data = read_fxn(check_file)
            new_data = read_fxn(new_file)

            for category_name, category_data in check_data.items():
                for variable_name, variable_data in category_data.items():
                    this_new_data = new_data[category_name][variable_name]

                    with self.subTest(check_file=check_file, new_file=new_file, category=category_name, variable=variable_name):
                        try:
                            test_result = np.isclose(variable_data, this_new_data).all()
                        except TypeError:
                            # Not all variables with be float arrays. If np.isclose() can't coerce the data to a numeric
                            # type, it'll raise a TypeError and we fall back on the equality test
                            test_result = np.all(variable_data == this_new_data)
                        self.assertTrue(test_result, msg='"{variable}" in {filename} does not match the check data'
                                        .format(variable=variable_name, filename=new_file))

                    #print('"{}" in "{}" OK'.format(variable_name, new_file))


if __name__ == '__main__':
    unittest.main()
