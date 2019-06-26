from glob import glob
import numpy as np
import os

# Extra analysis package
from sat_utils import mls

from ggg_inputs.common_utils import mod_utils


def write_mls_profile_list(mls_dirs, mls_specie, list_file, min_num_pts=10, every_n_days=1):
    with open(list_file, 'w') as wobj:
        wobj.write('DATES,LAT,LON\n')
        for this_dir in mls_dirs:
            mls_files = sorted(glob(os.path.join(this_dir, 'MLS*.he5')))
            mls_files = mls_files[::every_n_days]

            pbar = mod_utils.ProgressBar(len(mls_files), prefix='MLS file', style='counter')
            for i, f in enumerate(mls_files):
                pbar.print_bar(i)
                profiles = mls.read_mls_profiles(f, mls_specie)
                xx = np.sum(~np.isnan(profiles), axis=1) >= min_num_pts
                profiles = profiles[xx]
                prof_indices = np.arange(xx.size)[xx]

                timestamps = profiles.coords['time'].to_pandas()
                lines = ''
                for date, lat, lon, index, is_good in zip(timestamps, profiles.coords['lat'], profiles.coords['lon'],
                                                          prof_indices, xx):
                    if not is_good:
                        continue
                    lines += '{date},{lat:.3f},{lon:.3f} # {filename}, profile {profnum}\n'.format(
                        date=date.strftime('%Y-%m-%d'), lat=lat.item(), lon=lon.item(), filename=os.path.basename(f),
                        profnum=index
                    )
                wobj.write(lines)

            pbar.finish()
