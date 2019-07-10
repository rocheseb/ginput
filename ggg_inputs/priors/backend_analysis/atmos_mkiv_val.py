from datetime import datetime as dtime
import numpy as np
import os
import pandas as pd


from ggg_inputs.common_utils import mod_utils
from ggg_inputs.priors.backend_analysis import aircraft_aircore_val as aav


def make_atmos_occultation_list(atmos_file, list_file):
    atmos_dat = pd.read_csv(atmos_file, sep='\s+')
    with open(list_file, 'w') as wobj:
        wobj.write('DATES,LAT,LON,OCCULTATION\n')
        for occ, group in atmos_dat.groupby('Occultation'):
            lon = group['longitude'].to_numpy()
            lat = group['latitude'].to_numpy()
            date = group['yydoy'].to_numpy()

            # Get the profile info. Flag profiles with substantial variation in longitude because we might not be matching
            # it up to the priors very well
            flag = np.nanstd(lon) > 10
            lon = np.nanmean(lon)
            lat = np.nanmean(lat)
            date = np.unique(date)
            if date.size != 1:
                raise NotImplementedError('Occultation crosses multiple dates')
            date = dtime.strptime(str(date.item()), '%y%j')

            wobj.write('{},{:.3f},{:.3f},{}'.format(date.strftime('%Y-%m-%d'), lat, lon, occ))
            if flag:
                wobj.write(' # Caution: std(lon) exceeds 10 degrees')
            wobj.write('\n')


def make_mkiv_occultation_list(mkiv_file, list_file):
    mkiv_dat = pd.read_csv(mkiv_file, sep='\s+', header=8)
    with open(list_file, 'w') as wobj:
        wobj.write('DATES,LAT,LON,OCCULTATION\n')
        for occ, group in mkiv_dat.groupby('ocltn'):
            lon = group['tplong'].to_numpy()
            lat = group['tplat'].to_numpy()
            date = group['year'].to_numpy()

            lon = np.nanmean(lon)
            lat = np.nanmean(lat)
            date = mod_utils.decimal_year_to_date(np.nanmean(date))

            wobj.write('{},{:.3f},{:.3f},{}\n'.format(date.strftime('%Y-%m-%d'), lat, lon, occ))


def match_mkiv_to_priors(mkiv_file, list_file, map_dir, specie, max_error_ratio=0.2, allow_missing=False):
    # For each line in the list file, get the date/lat/lon to find the corresponding prior file and the occultation
    # number to get the occultation from the MkIV file. Interpolate the MkIV occultations to the prior altitudes.
    with open(list_file, 'r') as fobj:
        # discard the header
        fobj.readline()
        prof_list = pd.read_csv(list_file)
        prof_list.columns = [s.lower() for s in prof_list.columns]
        prof_list['dates'] = pd.to_datetime(prof_list['dates'])

    nprofs = prof_list.shape[0]
    out_shape = [nprofs, 42]
    priors = np.full(out_shape, np.nan)
    mkiv_profs = np.full(out_shape, np.nan)
    zarray = np.full(out_shape, np.nan)

    nhead = mod_utils.get_num_header_lines(mkiv_file)
    mk4dat = pd.read_csv(mkiv_file, header=nhead-1, sep='\s+')

    occ_key = 'ocltn'
    mk4_key = specie.lower()
    mk4_err_key = mk4_key + '_error'
    mk4_alt_key = 'altitude'

    before_2000 = np.zeros([nprofs], dtype=np.bool_)

    for i, row in enumerate(prof_list.itertuples(index=False)):
        if row.dates < pd.Timestamp(2000, 1, 1):
            before_2000[i] = True
            print('Assuming no profile for {} since before 2000'.format(row.dates))
            continue

        prior_file = os.path.join(map_dir, aav.py_map_file_subpath(row.lon, row.lat, row.dates))
        try:
            mapdat = mod_utils.read_map_file(prior_file)['profile']
        except FileNotFoundError:
            if allow_missing:
                print('Could not find {}, skipping'.format(prior_file))
                continue
            else:
                raise
        this_prior = mapdat[specie]
        this_prior_alt = mapdat['Height']

        xx_mk4 = mk4dat[occ_key] == row.occultation
        this_mk4_prof = mk4dat.loc[xx_mk4, mk4_key]
        this_mk4_err = mk4dat.loc[xx_mk4, mk4_err_key]
        rel_error = np.abs(this_mk4_err / this_mk4_prof)
        this_mk4_prof[rel_error > max_error_ratio] = np.nan
        this_mk4_alt = mk4dat.loc[xx_mk4, mk4_alt_key]

        this_mk4_prof = np.interp(this_prior_alt, this_mk4_alt.to_numpy(), this_mk4_prof.to_numpy(),
                                  left=np.nan, right=np.nan)

        priors[i, :] = this_prior
        mkiv_profs[i, :] = this_mk4_prof
        zarray[i, :] = this_prior_alt

    priors = priors[~before_2000]
    mkiv_profs = mkiv_profs[~before_2000]
    zarray = zarray[~before_2000]
    prof_info = {k: v.to_numpy()[~before_2000] for k, v in prof_list.items()}

    return priors, mkiv_profs, zarray, prof_info
