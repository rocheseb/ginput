# README #

# mod_maker.py #
mod_maker.py was first translated from IDL

The code can be run in two ways:

## 1 - NEW: ##

used to generate MOD files on GEOS5-FP-IT times for all TCCON sites at once using GEOS5-FP-IT daily files

It will generate MOD files both along the vertical above the site location and along the sun ray

Running the code like this will generate MOD files for ALL sites withtin the date range on GEOS5 times (every 3 hours) using GEOS5-FP-IT daily files

The first time the code is run the skyfield library will download 4 files named 'de421.bsp', 'deltat.data', 'deltat.preds', and 'Leap_Second.dat'

### How to run it ###

python mod_maker.py arg1 geos_path=arg2

arg1: date range YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months; or single date (YYYYMMDD)

arg2: full path to directory containing the daily GEOS5-FP-IT files

### Input files ###

two folders are expected in the geos_path directory:

in geos_path/Np you must have all the 42 levels GEOS5-FP-IT files (inst3_3d_asm_Np)

in geos_path/Nx you must have all the surface data files (inst3_2d_asm_Nx)

use the code get_GEOS5.py in the 'download' folder to get the necessary files.

### Where do files go? ###

MOD files will be generate both along the vertical and along the sun ray

They will be saved under GGGPATH/models/gnd/fpit/xx/yy

with xx the two letter site abbreviation and yy either 'vertical' or 'slant'

Some comparisons between vertical and slant profiles can be found here http://www.atmosp.physics.utoronto.ca/~sroche/mod_maker/fpit_vertical_vs_slant/

## 2 - OLD: ##

used to generate MOD files using ncep (like the IDL code), or using merra or fp or fpit data

This will not include equivalent latitude

Comparisons between mod profiles obtained from the different sources can be found here http://www.atmosp.physics.utoronto.ca/~sroche/mod_maker/ncep_merra_fp_fpit/

### How to run it ###
python mod_maker.py arg1 site=arg2 mode=arg3 time=arg4 step=arg5

arg1: date range (YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months) or single date (YYYYMMDD).

arg2: two letter site abbreviation (e.g. 'oc' for Lamont, Oklahoma; see the "site_dict" dictionary).

arg3: mode ('ncep', 'merradap42', 'merradap72', 'merraglob', 'fpglob', 'fpitglob'), ncep and 'glob' modes require local files.

the 'merradap' modes require a .netrc file in your home directory with credentials to connect to urs.earthdata.nasa.gov

arg4: (optional, default=12:00)  hour:minute (HH:MM) for the starting time in local time

arg5: (optional, default=24) time step in hours (can be decimal)

### Input files ###

The fpglob or fpitglob modes expect two files in GGGPATH/ncdf containing concatenated daily files for surface and multi-level data, the concatenated files need to be generated beforehand

e.g.

	GEOS_fpit_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
	GEOS_fpit_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

The merraglob mode works like the fpglob and fpitglob modes and will expect two files:

e.g.

	MERRA2_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
	MERRA2_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

### Where do files go? ###

MOD files will be saved under GGGPATH/models/gnd/xx

with xx either 'ncep', 'merra', 'fp', or 'fpit'

The merradap modes require an internet connection and EarthData credentials

The ncep mode requires the global NCEP netcdf files of the given year to be present in GGGPATH/ncdf

The ncep mode should produce files identical to the IDL mod_maker if 'time' and 'step' are kept as default

## Contact ##

sebastien.roche@mail.utoronto.ca
