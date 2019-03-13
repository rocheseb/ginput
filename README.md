# README #

# mod_maker.py #
mod_maker.py was first translated from IDL

The code can be run in two ways:

## 1 - NEW: ##

used to generate MOD files on GEOS5-FP-IT times for all TCCON sites at once using GEOS5-FP-IT 3-hourly files

It will generate MOD files both along the vertical above the site location, and along the sun ray if required

Running the code like this will generate MOD files for ALL sites withtin the date range on GEOS5 times (every 3 hours) using GEOS5-FP-IT 3-hourly files

The first time the code is run the skyfield library will download 4 files named 'de421.bsp', 'deltat.data', 'deltat.preds', and 'Leap_Second.dat'

### How to run it ###

python mod_maker.py arg1 geos_path=arg2 site=arg3 lat=arg4 lon=arg5 alt=arg6 save_path=arg7

arg1: date range YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months; or a single date (YYYYMMDD) in which case the end date is +24h

You can also give YYYYMMDD_HH instead to specify the hour, but these must be exact GEOS5 times (UTC times 3 hourly from 00)

arg2: full path to directory containing the 3-hourly GEOS5-FP-IT files

arg3: (optional) two letter site abbreviation

arg4: (optional) latitude in [-90,90] range

arg5: (optional) longitude in [0,360] range

arg6: (optional) altitude (meters)

arg7: (optional) full path to directory where data will be saved (save_path/fpit will be created), defaults to GGGPATH/models/gnd

If arg3 is specified, MOD files will only be produced for that one site. See the dictionary in tccon_sites.py for site abbreviations of existing sites.

A custom site location can be given, in that case arg3,arg4,arg5, and arg6 must be specified

add 'mute' in the command line (somwehere after arg1) and there will be no print statements other than warnings and error messages

add 'slant' in the command line (somewhere after arg1) to generate both vertical and slant MOD files.

### Input files ###

two folders are expected in the geos_path directory:

in geos_path/Np you must have all the 42 levels GEOS5-FP-IT files (inst3_3d_asm_Np)

in geos_path/Nx you must have all the surface data files (inst3_2d_asm_Nx)

use the code get_GEOS5.py in the 'download' folder to get the necessary files.

### Where do files go? ###

MOD files will be generate both along the vertical, and along the sun ray if required

They will be saved under save_path/fpit/xx/yy

with xx the two letter site abbreviation and yy either 'vertical' or 'slant'

Some comparisons between vertical and slant profiles can be found here http://www.atmosp.physics.utoronto.ca/~sroche/mod_maker/fpit_vertical_vs_slant/

## 2 - OLD: ##

used to generate MOD files using ncep (like the IDL code), or using merra or fp or fpit data

This will not include equivalent latitude

Comparisons between mod profiles obtained from the different sources can be found here http://www.atmosp.physics.utoronto.ca/~sroche/mod_maker/ncep_merra_fp_fpit/

### How to run it ###

python mod_maker.py arg1 site=arg2 mode=arg3 time=arg4 step=arg5 lat=arg6 lon=arg7 alt=arg8 save_path=arg9 ncdf_path=arg10

arg1: date range (YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months) or single date (YYYYMMDD).

arg2: two letter site abbreviation (e.g. 'oc' for Lamont, Oklahoma; see the "site_dict" dictionary).

arg3: mode ('ncep', 'merradap42', 'merradap72', 'merraglob', 'fpglob', 'fpitglob'), ncep and 'glob' modes require local files.

the 'merradap' modes require a .netrc file in your home directory with credentials to connect to urs.earthdata.nasa.gov

arg4: (optional, default=12:00)  hour:minute (HH:MM) for the starting time in local time

arg5: (optional, default=24) time step in hours (can be decimal)

arg6: (optional) latitude in [-90,90] range

arg7: (optional) longitude in [0,360] range

arg8: (optional) altitude (meters)

arg9: (optional) full path to directory where data will be saved (save_path/fpit will be created), defaults to GGGPATH/models/gnd

arg10: (optional) full path to the directory where ncep/geos/merra files are located, defaults to GGGPATH/ncdf

A custom site location can be given, in that case arg6,arg7, and arg8 must be specified and a site name must be made up for arg2

add 'mute' in the command line (somwehere after arg1) and there will be no print statements other than warnings and error messages 

### Input files ###

The fpglob or fpitglob modes expect two files in 'ncdf_path' containing concatenated 3-hourly files for surface and multi-level data, the concatenated files need to be generated beforehand

e.g.

	GEOS_fpit_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
	GEOS_fpit_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

The merraglob mode works like the fpglob and fpitglob modes and will expect two files:

e.g.

	MERRA2_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
	MERRA2_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

### Where do files go? ###

If the 'save_path' argument is not specified MOD files will be saved under GGGPATH/models/gnd/xx

Otherwise MOD files will be saved under save_path/xx

with xx either 'ncep', 'merra', 'fp', or 'fpit'

### Other notes ###

The merradap modes require an internet connection and EarthData credentials

The ncep mode should produce files identical to the IDL mod_maker if 'time' and 'step' are kept as default

## Contact ##

sebastien.roche@mail.utoronto.ca
