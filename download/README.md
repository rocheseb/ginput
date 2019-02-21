# README #

Utility programs to download MERRA2, GEOS5-FP, or GEOS5-FPIT data

Your machine needs to have been given permission to access GEOS5-FPIT data

## get_GEOS5.py ##

You can use this code to download GEOS5-FP or GEOS5-FPIT data

Run it with:

python get_GEOS5.py arg1 mode=arg2 path=arg3

arg1: date range YYYYMMDD-YYYYMMDD

arg2: either 'FP' or 'FPIT'

arg3: path where the data will be saved

.e.g. if arg3 is /home/sroche/geos_path/fpit then two folders will be created:

/home/sroche/geos_path/fpit/Nx  for surface data
/home/sroche/geos_path/fpit/Np  for profile data

The program uses wget to download the data

## get_MERRA2.py ##

Can be used to download MERRA2 files, see instructions in the code header
