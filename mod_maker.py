#!~/anaconda2/bin/python
 # -*- coding: utf-8 -*-

"""
README

mod_maker10.f translated into python

What's different?

#########################################################################################################################################################################

OLD: used to generate MOD files using ncep (like the IDL code), or using merra or fp or fpit data

python mod_maker.py arg1 site=arg2 mode=arg3 time=arg4 step=arg5

arg1: date range (YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months) or single date (YYYYMMDD)
arg2: two letter site abbreviation (e.g. 'oc' for Lamont, Oklahoma; see the "site_dict" dictionary)
arg3: mode ('ncep', 'merradap42', 'merradap72', 'merraglob', 'fpglob', 'fpitglob'), ncep and 'glob' modes require local files
the 'merradap' modes require a .netrc file in your home directory with credentials to connect to urs.earthdata.nasa.gov
arg4: (optional, default=12:00)  hour:minute (HH:MM) for the starting time in local time
arg5: (optional, default=24) time step in hours (can be decimal)

if the command line include 'mute' the program will not print outputs to the terminal, except error messages

MOD files will be saved under GGGPATH/models/gnd/xx.
With xx either 'ncep', 'merra', 'fp', or 'fpit'
The merradap modes require an internet connection and EarthData credentials
The ncep mode requires the global NCEP netcdf files of the given year to be present in GGGPATH/ncdf

The fpglob or fpitglob modes expect two files in GGGPATH/ncdf containing concatenated daily files for surface and multi-level data, the concatenated files need to be generated beforehand
e.g.
GEOS_fpit_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
GEOS_fpit_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

The merraglob mode works like the fpglob and fpitglob modes and will expect two files:
e.g.
MERRA2_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
MERRA2_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

The ncep mode should produce files identical to the IDL mod_maker if 'time' and 'step' are kept as default

#########################################################################################################################################################################

NEW: used to generate MOD files on GEOS5-FP-IT times for all TCCON sites at once using GEOS5-FP-IT daily files

python mod_maker.py arg1 geos_path=arg2

arg1: date range (YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months) or single date (YYYYMMDD)
arg2: full path to directory containing the daily GEOS5-FP-IT files

if the command line include 'mute' the program will not print outputs to the terminal, except error messages

two folders are expected in the geos_path directory:
in geos_path/Np you must have all the 42 levels GEOS5-FP-IT files
in geos_path/Nx you must have all the surface data files

Running the code like this will generate MOD files for ALL sites withtin the date range on GEOS5 times (every 3 hours) using GEOS5-FP-IT daily files
MOD files will be generate both along the vertical and along the sun ray

They will be saved under GGGPATH/models/gnd/fpit/xx/yy
with xx the two letter site abbreviation and yy either 'vertical' or 'slant'

The slant .mod files are only generated when the SZA is above 90 degrees.
#########################################################################################################################################################################

There is dictionary of sites with their respective lat/lon, so this works for all TCCON sites, lat/lon values were taken from the wiki page of each site.
"""

import os, sys
import numpy as np
import numpy.ma as ma
from numpy import cos,sin,tan,arctan,arccos,arcsin,arctan2,deg2rad,rad2deg
import pandas as pd
from scipy.interpolate import interp1d, interp2d, RectSphereBivariateSpline, griddata
import netCDF4 # netcdf I/O
import re # used to parse strings
import time
import netrc # used to connect to earthdata
from datetime import datetime, timedelta
from astropy.time import Time # this is essentialy like datetime, but with better methods for conversion of datetime to / from julian dates, can also be converted to datetime
from pydap.cas.urs import setup_session # used to connect to the merra opendap servers
from pydap.client import open_url
import xarray
from urllib2 import HTTPError
import pylab as pl
import pytz
import warnings

from slantify import * # code to make slant paths

def tccon_site_info():
	"""
	dictionary mapping TCCON site abbreviations to their lat-lon-alt data, and full names

	To add a new site make up a new two letter site abbreviation and add it to the dictionary following the same model of other sites.
	
	For sites the changed location, a 'time_spans' dictionary is used instead of the 'lat'/'lon'/'alt' keys.
	The keys of this dictionary are pairs of dates in tuples : tuple([start_date,end_date])
	The values are dictionaries of 'lat'/'lon'/'alt' for each time period.
	The first date is inclusive and the end date is exclusive. See Darwin for an example.

	If the instrument has moved enough so that the rounded lat and lon is different, then the mod file names will be different for the different time periods.

	the longitudes must be given in the range [0-360]
	"""
	site_dict = {
				'pa':{'name': 'Park Falls','loc':'Wisconsin, USA','lat':45.945,'lon':269.727,'alt':442},
				'oc':{'name': 'Lamont','loc':'Oklahoma, USA','lat':36.604,'lon':262.514,'alt':320},
				'wg':{'name': 'Wollongong','loc':'Australia','lat':-34.406,'lon':150.879,'alt':30},
				'db':{'name': 'Darwin','loc':'Australia','time_spans':{tuple([datetime(2005,8,1),datetime(2015,7,1)]):{'lat':-12.422445,'lon':130.89154,'alt':30},
														tuple([datetime(2015,7,1),datetime.now()]):{'lat':-12.45606,'lon':130.92658,'alt':37}
														}
				},#,'lat':-12.45606,'lon':130.92658,'alt':37},
				'or':{'name': 'Orleans','loc':'France','lat':47.97,'lon':2.113,'alt':130},
				'bi':{'name': 'Bialystok','loc':'Poland','lat':53.23,'lon':23.025,'alt':180},
				'br':{'name': 'Bremen','loc':'Germany','lat':53.1037,'lon':8.849517,'alt':30},
				'jc':{'name': 'JPL 01','loc':'California, USA','lat':34.202,'lon':241.825,'alt':390},
				'jf':{'name': 'JPL 02','loc':'California, USA','lat':34.202,'lon':241.825,'alt':390},
				'ra':{'name': 'Reunion Island','loc':'France','lat':-20.901,'lon':55.485,'alt':87},
				'gm':{'name': 'Garmisch','loc':'Germany','lat':47.476,'lon':11.063,'alt':743},
				'lh':{'name': 'Lauder 01','loc':'New Zealand','lat':-45.038,'lon':169.684,'alt':370},
				'll':{'name': 'Lauder 02','loc':'New Zealand','lat':-45.038,'lon':169.684,'alt':370},
				'tk':{'name': 'Tsukuba 02','loc':'Japan','lat':63.0513,'lon':140.1215,'alt':31},
				'ka':{'name': 'Karlsruhe','loc':'Germany','lat':49.1002,'lon':8.4385,'alt':119},
				'ae':{'name': 'Ascenssion Island','loc':'United Kingdom','lat':-7.933333,'lon':345.583333,'alt':0},
				'eu':{'name': 'Eureka','loc':'Canada','lat':80.05,'lon':273.58,'alt':610},
				'so':{'name': 'Sodankyla','loc':'Finland','lat':67.3668,'lon':26.6310,'alt':188},
				'iz':{'name': 'Izana','loc':'Spain','lat':28.0,'lon':344.0,'alt':2370},
				'if':{'name': 'Idianapolis','loc':'Indiana, USA','lat':39.861389,'lon':273.996389,'alt':270},
				'df':{'name': 'Dryden','loc':'California, USA','lat':34.959917,'lon':242.118931,'alt':700},
				'js':{'name': 'Saga','loc':'Japan','lat':33.240962,'lon':130.288239,'alt':7},
				'fc':{'name': 'Four Corners','loc':'USA','lat':36.79749,'lon':251.51991,'alt':1643},
				#'ci':{'name': 'Pasadena','loc':'California, USA','lat':34.13623,'lon':241.873103,'alt':230},
				'ci':{'name': 'Pasadena','loc':'California, USA','lat':34.136,'lon':241.873,'alt':230},
				'rj':{'name': 'Rikubetsu','loc':'Japan','lat':43.4567,'lon':143.7661,'alt':380},
				'pr':{'name': 'Paris','loc':'France','lat':48.846,'lon':2.356,'alt':60},
				'ma':{'name': 'Manaus','loc':'Brazil','lat':-3.2133,'lon':299.4017,'alt':50},
				'sp':{'name': 'Ny-Alesund','loc':'Norway','lat':78.92324,'lon':11.92298,'alt':20},
				'et':{'name': 'East Trout Lake','loc':'Canada','lat':54.353738,'lon':255.013333,'alt':501.8},
				'an':{'name': 'Anmyeondo','loc':'Korea','lat':36.5382,'lon':126.331,'alt':30},
				'bu':{'name': 'Burgos','loc':'Philippines','lat':18.5325,'lon':120.6496,'alt':35},
				'we':{'name': 'Jena','loc':'Austria','lat':50.91,'lon':11.57,'alt':211.6},
				'ha':{'name':'Harwell','loc':'UK','lat':51.57133,'lon':341.10683,'alt':123},
				'he':{'name':'Hefei','loc':'China','lat':31.9,'lon':117.17,'alt':34.5},
				'yk':{'name':'Yekaterinburg','loc':'Russia','lat':57.03833,'lon':59.54500,'alt':0}, # needs alt update
				'he':{'name':'Hefei','loc':'China','lat':31.9,'lon':117.17,'alt':34.5},
				'zs':{'name':'Zugspitze','loc':'Germany','lat':47.42,'lon':10.98,'alt':34.5},
				}

	for site in site_dict:
		if 'time_spans' in site_dict[site].keys():
			for time_span in site_dict[site]['time_spans']:
				if site_dict[site]['time_spans'][time_span]['lon']>180:
					site_dict[site]['time_spans'][time_span]['lon_180'] = site_dict[site]['time_spans'][time_span]['lon']-360
				else:
					site_dict[site]['time_spans'][time_span]['lon_180'] = site_dict[site]['time_spans'][time_span]['lon']			
		else:
			if site_dict[site]['lon']>180:
				site_dict[site]['lon_180'] = site_dict[site]['lon']-360
			else:
				site_dict[site]['lon_180'] = site_dict[site]['lon']

	return site_dict

def compute_h2o_dmf(qv,rmm):
	"""
	compute h2o dry mole fraction from specific humidity
	"""
	return rmm*qv/(1-qv)

def compute_h2o_wmf(h2o_dmf):
	"""
	compute h2o wet mole fraction from h2o dry mole fraction
	"""
	return h2o_dmf/(1+h2o_dmf)

def compute_rh(t,h2o_wmf,p):
	"""
	compute relative humidity from h2o wet mole fraction, pressure and temperature
	"""
	svp = svp_wv_over_ice(t)
	return 100*h2o_wmf*p/svp

def compute_mmw(h2o_wmf):
	"""
	compute mean molecular weight of air from h2o wet mole fraction
	"""
	return 28.964*(1-h2o_wmf)+18.02*h2o_wmf

def svp_wv_over_ice(temp):
	"""	
	Uses the Goff-Gratch equation to calculate the saturation vapor
	pressure of water vapor over ice at a user-specified temperature.
		Input:  temp (K)
		Output: svp (mbar)
	"""
	t0 = 273.16	# triple point temperature
	tr = t0/temp 
	yy = -9.09718*(tr-1)-3.56654*np.log10(tr)+0.876793*(1-1/tr)
	svp = 6.1173*10**yy # saturation vapor pressure over ice (mbar)

	return svp

def write_mod(mod_path,version,site_lat,data=0,surf_data=0,func=None,muted=False):
	"""
	Creates a GGG-format .mod file
	INPUTS:
		mod_path: full path to write the .mod file
		version: the mod_maker version
		site_lat: site latitude (-90 to 90)
		data: dictionary of the inputs
		surf_data: dictionary of the surface inputs (for merra/geos5)
	"""

	# Define US Standard Atmosphere (USSA) for use above 10 mbar
	p_ussa=[10.0,  5.0,   2.0,   1.0,   0.5,    0.2,   0.1,   0.01,  0.001, 0.0001]
	t_ussa=[227.7, 239.2, 257.9, 270.6, 264.3, 245.2, 231.6, 198.0, 189.8, 235.0]
	z_ussa=[31.1,  36.8,  42.4,  47.8,  53.3,  60.1,  64.9,  79.3,  92.0,  106.3]

	if type(surf_data)==int: # ncep mode

		# The head of the .mod file	
		fmt = '{:8.3f} {:11.4e} {:7.3f} {:5.3f} {:8.3f} {:8.3f} {:8.3f}\n'
		mod_content = []
		mod_content+=[	'5  6\n',
						fmt.format(6378.137,6.000E-05,site_lat,9.81,data['H'][0],1013.25,data['TROPP']),
						version+'\n',
						' mbar        Kelvin         km      g/mole      DMF       %\n',
						'Pressure  Temperature     Height     MMW        H2O      RH\n',	]

		fmt = '{:9.3e}    {:7.3f}    {:7.3f}    {:7.4f}    {:9.3e}{:>6.1f}\n' # format for writting the lines

		# Export the Pressure, Temp and SHum for lower levels (1000 to 300 mbar)
		for k,elem in enumerate(data['H2O_DMF']):
			svp = svp_wv_over_ice(data['T'][k])
			h2o_wmf = compute_h2o_wmf(data['H2O_DMF'][k]) # wet mole fraction of h2o
			frh = h2o_wmf*data['T'][k]/svp # Fractional relative humidity

			# Relace H2O mole fractions that are too small
			if (frh < 30./data['T'][k]):
				if not muted:
					print 'Replacing too small H2O ',mod_path, data['lev'][k],h2o_wmf,svp*30./data['lev'][k]/data['lev'][k],frh,30./data['lev'][k]				
				frh = 30./data['lev'][k]
				h2o_wmf = svp*frh/data['lev'][k]
				data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)

			# Relace H2O mole fractions that are too large (super-saturated)  GCT 2015-08-05
			if (frh > 1.0):
				if not muted:	
					print 'Replacing too large H2O ',mod_path,data['lev'][k],h2o_wmf,svp/data['lev'][k],frh,1.0
				frh=1.0
				h2o_wmf = svp*frh/data['lev'][k]
				data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)

			mmw = compute_mmw(h2o_wmf)

			mod_content += [fmt.format(data['lev'][k], data['T'][k],data['H'][k],mmw,data['H2O_DMF'][k],100*frh)]

		# Export Pressure and Temp for middle levels (250 to 10 mbar)
		# which have no SHum reanalysis
		ptop = data['lev'][k] # Top pressure level
		frh_top = frh  # remember the FRH at the top (300 mbar) level

		for k in range(len(data['H2O_DMF']),len(data['T'])): 
			zz = np.log10(data['lev'][k])  # log10[pressure]
			strat_wmf = 7.5E-06*np.exp(-0.16*zz**2)
			svp = svp_wv_over_ice(data['T'][k])
			trop_wmf = frh_top*svp/data['lev'][k]
			wt = (data['lev'][k]/ptop)**3
			avg_wmf = trop_wmf*wt + strat_wmf*(1-wt)
			avg_frh = avg_wmf*data['lev'][k]/svp
			if (avg_frh > 1.0):
				if not muted:	
					print 'Replacing super-saturated H2O ',mod_path, data['lev'][k],avg_wmf,svp*avg_frh/data['lev'][k],avg_frh,1.0
				avg_frh = 1.0
				avg_wmf = svp*avg_frh/data['lev'][k]

			mmw = compute_mmw(avg_wmf)
			
			mod_content += [fmt.format(data['lev'][k],data['T'][k],data['H'][k],mmw,avg_wmf/(1-avg_wmf),100*avg_frh)]

		# Get the difference between the USSA and given site temperature at 10 mbar,
		Delta_T=data['T'][16]-t_ussa[0]

		# Export the P-T profile above 10mbar
		for k in range(1,len(t_ussa)):
			Delta_T=Delta_T/2
			zz = np.log10(p_ussa[k])  # log10[pressure]
			strat_wmf = 7.5E-06*np.exp(-0.16*zz**2)
			svp = svp_wv_over_ice(data['T'][k])
			mmw = compute_mmw(strat_wmf)
			mod_content += [fmt.format(p_ussa[k],t_ussa[k]+Delta_T,z_ussa[k],mmw,strat_wmf,100*strat_wmf*p_ussa[k]/svp)]

	else: # merra/geos mode

		# The head of the .mod file	
		fmt1 = '{:8.3f} {:11.4e} {:7.3f} {:5.3f} {:8.3f} {:8.3f} {:8.3f}\n'
		mod_content = []
		if func is None:
			fmt2 = '{:9.3e}    {:7.3f}    {:7.3f}    {:7.4f}    {:9.3e}{:>6.1f}    {:9.3e}    {:9.3e}    {:9.3e}    {:9.3e}    {:7.3f}\n'
			mod_content+=[	'7  10\n',
							fmt1.format(6378.137,6.000E-05,site_lat,9.81,data['H'][0],1013.25,surf_data['TROPPB']),
							'Pressure  Temperature     Height     MMW        H2O      RH         SLP        TROPPB        TROPPV      TROPPT       TROPT\n',
							fmt2.format(*[surf_data[key] for key in ['PS','T2M','H','MMW','H2O_DMF','RH','SLP','TROPPB','TROPPV','TROPPT','TROPT']]),
							version+'\n',
							' mbar        Kelvin         km      g/mole        DMF        %       k.m+2/kg/s   Kelvin     kg/kg\n',
							'Pressure  Temperature     Height     MMW          H2O       RH          EPV         PT         O3\n',	]

			fmt = '{:9.3e}    {:7.3f}    {:7.3f}    {:7.4f}    {:10.3e} {:>6.1f}    {:10.3e}    {:8.3f}    {:9.3e}\n' # format for writting the lines

		else:
			fmt2 = '{:9.3e}    {:7.3f}    {:7.3f}    {:7.4f}    {:9.3e}{:>6.1f}    {:9.3e}    {:9.3e}    {:9.3e}    {:9.3e}    {:7.3f}    {:7.3f}\n'
			mod_content+=[	'7  10\n',
							fmt1.format(6378.137,6.000E-05,site_lat,9.81,data['H'][0],1013.25,surf_data['TROPPB']),
							'Pressure  Temperature     Height     MMW        H2O      RH         SLP        TROPPB        TROPPV      TROPPT       TROPT       SZA\n',
							fmt2.format(*[surf_data[key] for key in ['PS','T2M','H','MMW','H2O_DMF','RH','SLP','TROPPB','TROPPV','TROPPT','TROPT','SZA']]),
							version+'\n',
							' mbar        Kelvin         km      g/mole        DMF        %       k.m+2/kg/s   Kelvin      degrees     kg/kg\n',
							'Pressure  Temperature     Height     MMW          H2O       RH          EPV         PT          EL         O3\n',	]

			fmt = '{:9.3e}    {:7.3f}    {:7.3f}    {:7.4f}    {:10.3e} {:>6.1f}    {:10.3e}    {:8.3f}    {:7.3f}    {:9.3e}\n' # format for writting the lines

		# not sure if merra needs all the filters/corrections used for ncep data?

		# Export the Pressure, Temp and SHum
		for k,elem in enumerate(data['H2O_DMF']):
			svp = svp_wv_over_ice(data['T'][k])
			h2o_wmf = compute_h2o_wmf(data['H2O_DMF'][k]) # wet mole fraction of h2o

			if 300<=data['lev'][k]<=1000:
				# Relace H2O mole fractions that are too small
				if (data['RH'][k] < 30./data['T'][k]):
					if not muted:
						print 'Replacing too small H2O at {:.2f} hPa; H2O_WMF={:.3e}; {:.3e}; RH={:.3f}'.format(data['lev'][k],h2o_wmf,svp/data['T'][k],data['RH'][k],1.0)	
					data['RH'][k] = 30./data['lev'][k]
					h2o_wmf = svp*data['RH'][k]/data['lev'][k]
					data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)
					if not muted:
						print 'svp,h2o_wmf,h2o_dmf',svp,h2o_wmf,data['H2O_DMF'][k],data['RH'][k]

			# Relace H2O mole fractions that are too large (super-saturated)  GCT 2015-08-05
			if (data['RH'][k] > 1.0):
				if not muted:
					print 'Replacing too large H2O at {:.2f} hPa; H2O_WMF={:.3e}; {:.3e}; RH={:.3f}'.format(data['lev'][k],h2o_wmf,svp/data['T'][k],data['RH'][k],1.0)
				data['RH'][k] = 1.0
				h2o_wmf = svp*data['RH'][k]/data['T'][k]
				data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)

			mmw = compute_mmw(h2o_wmf)

			# compute potential temperature
			PT = data['T'][k]*(1000.0/data['lev'][k])**0.286

			if func is None:
				mod_content += [fmt.format(data['lev'][k],data['T'][k],data['H'][k],mmw,data['H2O_DMF'][k],100*data['RH'][k],data['EPV'][k],PT,data['O3'][k])]
			else: # compute equivalent latitude; 1e6 converts EPV to PVU (1e-6 K . m2 / kg / s)
				EL = func(data['EPV'][k]*1e6,PT)[0]
				mod_content += [fmt.format(data['lev'][k],data['T'][k],data['H'][k],mmw,data['H2O_DMF'][k],100*data['RH'][k],data['EPV'][k],PT,EL,data['O3'][k])]

	with open(mod_path,'w') as outfile:
		outfile.writelines(mod_content)

	if not muted:
		print mod_path

def trilinear_interp(DATA,varlist,site_lon_360,site_lat,site_tim):
	"""
	Evaluates  fout = fin(xx,yy,*,tt) 
	Result is a 1-vector
	"""
	INTERP_DATA = {}

	dx = DATA['lon'][1]-DATA['lon'][0]
	dy = DATA['lat'][1]-DATA['lat'][0]
	dt = DATA['time'][1]-DATA['time'][0]

	xx = (site_lon_360-DATA['lon'][0])/dx
	yy = (site_lat-DATA['lat'][0])/dy
	tt = (site_tim-DATA['time'][0])/dt

	nxx =  len(DATA['lon'])
	nyy =  len(DATA['lat'])
	ntt =  len(DATA['time'])

	index_xx = int(xx)
	ixpomnxx = (index_xx+1) % nxx
	fr_xx = xx-index_xx	

	index_yy = int(yy)
	if index_yy > nyy-2:
		index_yy = nyy-2  #  avoid array-bound violation at SP
	fr_yy = yy-index_yy

	index_tt = int(tt)
	
	if index_tt < 0:
		index_tt = 0          # Prevent Jan 1 problem
	if index_tt+1 > ntt-1:
		index_tt = ntt-2  # Prevent Dec 31 problem

	fr_tt=tt-index_tt  #  Should be between 0 and 1 when interpolating in time

	if (fr_tt < -1) or (fr_tt > 2):
	   print 'Excessive time extrapolation:',fr_tt,' time-steps   =',fr_tt*dt,' days'
	   print ' tt= ',tt,'  index_tt=',index_tt,'  fr_tt=',fr_tt
	   print 'input file does not cover the full range of dates'
	   print 'site_tim',site_tim
	   print 'tim_XX',DATA['time']
	   raw_input() # will hold the program until something is typed in commandline

	if (fr_xx < 0) or (fr_xx > 1):
	   print 'Excessive longitude extrapolation:',fr_xx,' steps   =',fr_xx*dx,' deg'
	   print ' xx= ',xx,'  index_xx=',index_xx,'  fr_xx=',fr_xx
	   print 'input file does not cover the full range of longitudes'
	   raw_input() # will hold the program until something is typed in commandline

	if (fr_yy < 0) or (fr_yy > 1):
	   print 'Excessive latitude extrapolation:',fr_yy-1,' steps   =',(fr_yy-1)*dy,' deg'
	   print ' yy= ',yy,'  index_yy=',index_yy,'  fr_yy=',fr_yy
	   print 'input file does not cover the full range of latitudes'
	   raw_input() # will hold the program until something is typed in commandline

	if (fr_tt < 0) or (fr_tt > 1):
		print ' Warning: time extrapolation of ',fr_tt,' time-steps'
	if (fr_xx < 0) or (fr_xx > 1):
		print ' Warning: longitude extrapolation of ',fr_xx,' steps'
	if (fr_yy < 0) or (fr_yy > 1):
		print ' Warning: latitude extrapolation of ',fr_yy,' steps'

	for varname in varlist:

		fin = DATA[varname]
		
		if fin.ndim==4:
			fout =	((fin[index_tt,:,index_yy,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt,:,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
			+ (fin[index_tt,:,index_yy+1,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt,:,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*(1.0-fr_tt) \
			+ ((fin[index_tt+1,:,index_yy,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt+1,:,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
			+ (fin[index_tt+1,:,index_yy+1,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt+1,:,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*fr_tt
		elif fin.ndim==3: # for data that do not have the vertical dimension
			fout =	((fin[index_tt,index_yy,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
			+ (fin[index_tt,index_yy+1,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*(1.0-fr_tt) \
			+ ((fin[index_tt+1,index_yy,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt+1,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
			+ (fin[index_tt+1,index_yy+1,index_xx]*(1.0-fr_xx) \
			+ fin[index_tt+1,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*fr_tt
		else:
			print 'Data has unexpected dimensions, ndim =',fin.ndim
			sys.exit()

		INTERP_DATA[varname] = fout*DATA['scale_factor_'+varname] + DATA['add_offset_'+varname]

	return INTERP_DATA

def read_data(dataset, varlist, lat_lon_box=0):
	"""
	for ncep files "dataset" is the full path to the netcdf file

	for merra files "dataset" is a pydap.model.DatasetType object
	"""
	DATA = {}

	opendap = type(dataset)!=netCDF4._netCDF4.Dataset

	if lat_lon_box == 0: # ncep mode

		varlist += ['level','lat','lon','time']

		for varname in varlist:
			DATA[varname] = dataset[varname][:]

			for attribute in ['add_offset','scale_factor']:
				try:
					DATA['add_offset_'+varname] = dataset[varname].getncattr('add_offset')
				except:
					DATA['add_offset_'+varname] = 0.0
					DATA['scale_factor_'+varname] = 1.0

	else: # merra/geos5 mode
		
		min_lat_ID, max_lat_ID, min_lon_ID, max_lon_ID = lat_lon_box

		DATA['lat'] = dataset['lat'][min_lat_ID:max_lat_ID] 	# Read in variable 'lat'
		DATA['lon'] = dataset['lon'][min_lon_ID:max_lon_ID] 	# Read in variable 'lon'
		DATA['time'] = dataset['time'][:]	# Read in variable 'time'

		try:
			dataset['lev']
		except:
			pass
		else:
			if dataset['lev'].shape[0] == 72:
				DATA['lev'] = dataset['PL'][:,:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID]	# the merra 72 mid level pressures are not fixed
			elif dataset['lev'].shape[0] == 42:
				DATA['lev'] = dataset['lev'][:]	# the 42 levels data is on a fixed pressure grid
			else:
				DATA['lev'] = dataset['PS'][:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID] # surface data doesn't have a 'lev' variable

		if opendap:
			for varname in ['time','lev','lat','lon']:
				try:
					DATA[varname] = DATA[varname].data	
				except KeyError,IndexError:
					pass

		# get longitudes as 0 -> 360 instead of -180 -> 180, needed for trilinear_interp
		for i,elem in enumerate(DATA['lon']):
			if elem < 0:
				DATA['lon'][i] = elem + 360.0

		for varname in varlist:

			if dataset[varname].ndim == 4:
				DATA[varname] = dataset[varname][:,:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID] 	# Read in variable varname
			else:
				DATA[varname] = dataset[varname][:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID] 	# Read in variable varname
			if opendap or ('Masked' in str(type(DATA[varname]))):
				DATA[varname] = DATA[varname].data

			for attribute in ['add_offset','scale_factor']:
				try:
					DATA[attribute+'_'+varname] = dataset[varname].getncattr(attribute)
				except:
					DATA['add_offset_'+varname] = 0.0
					DATA['scale_factor_'+varname] = 1.0

			print varname, DATA[varname].shape

	time_units = dataset['time'].units  # string containing definition of time units

	# two lines to parse the date (no longer need to worry about before/after 2014)
	date_list = re.findall(r"[\w]+",time_units.split(' since ')[1])
	common_date_format = '{:0>4}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}'.format(*date_list)

	start_date = datetime.strptime(common_date_format,'%Y-%m-%d %H:%M:%S')
	astropy_start_date = Time(start_date)

	DATA['julday0'] = astropy_start_date.jd # gives same results as IDL's JULDAY function

	return DATA

def querry_indices(dataset,site_lat,site_lon_180,box_lat_half_width,box_lon_half_width):
	"""	
	Set up a lat-lon box for the data querry

	Unlike with ncep, this will only use daily files for interpolation, so no time box is defined
	
	NOTE: merra lat -90 -> +90 ;  merra lon -180 -> +179.375

	To be certain to get two points on both side of the site lat and lon, use the grid resolution
	"""
	# define 2xbox_lat_half_width°x2xbox_lon_half_width° lat-lon box centered on the site lat-lon
	min_lat, max_lat = site_lat-box_lat_half_width, site_lat+box_lat_half_width
	min_lon, max_lon = site_lon_180-box_lon_half_width, site_lon_180+box_lon_half_width
	
	# handle edge cases
	if min_lat < -90:
		min_lat = -(90-abs(90-min_lat))
	if max_lat > 90:
		max_lat = 90-abs(90-max_lat)

	if max_lat < min_lat:
		swap = max_lat
		max_lat = min_lat
		min_lat = swap
	
	if min_lon < -180:
		min_lon = 180 - abs(180-min_lon)
	if max_lon > 180:
		max_lon = - (180 - abs(180-max_lon))

	if max_lon < min_lon:
		swap = max_lon
		max_lon = min_lon
		min_lon = swap

	# read the latitudes and longitudes from the merra file
	if type(dataset)==netCDF4._netCDF4.Dataset:
		merra_lon = dataset['lon'][:]
		merra_lat = dataset['lat'][:]
	elif type(dataset)==list:
		merra_lat = dataset[0]
		merra_lon = dataset[1]
	else: # for opendap datasets
		merra_lon = dataset['lon'][:].data
		merra_lat = dataset['lat'][:].data

	# get the indices of merra longitudes and latitudes that fit in the lat-lon box
	merra_lon_in_box_IDs = np.where((merra_lon>=min_lon) & (merra_lon<=max_lon))[0]
	merra_lat_in_box_IDs = np.where((merra_lat>=min_lat) & (merra_lat<=max_lat))[0]

	min_lat_ID, max_lat_ID = merra_lat_in_box_IDs[0], merra_lat_in_box_IDs[-1]+1
	min_lon_ID, max_lon_ID = merra_lon_in_box_IDs[0], merra_lon_in_box_IDs[-1]+1
	# +1 because ARRAY[i:j] in python will return elements i to j-1

	return [min_lat_ID, max_lat_ID, min_lon_ID, max_lon_ID]

# ncep has geopotential height profiles, not merra(?, only surface), so I need to convert geometric heights to geopotential heights
# the idl code uses a fixed radius for the radius of earth (6378.137 km), below the gravity routine of gsetup is used
# also the surface geopotential height of merra is in units of m2 s-2, so it must be divided by surface gravity
def gravity(gdlat,altit):
	"""
	copy/pasted from fortran routine comments
	This is used to convert

	Input Parameters:
	    gdlat       GeoDetric Latitude (degrees)
	    altit       Geometric Altitude (km)
	
	Output Parameter:
	    gravity     Effective Gravitational Acceleration (m/s2)
	    radius 		Radius of earth at gdlat
	
	Computes the effective Earth gravity at a given latitude and altitude.
	This is the sum of the gravitational and centripital accelerations.
	These are based on equation I.2.4-(17) in US Standard Atmosphere 1962
	The Earth is assumed to be an oblate ellipsoid, with a ratio of the
	major to minor axes = sqrt(1+con) where con=.006738
	This eccentricity makes the Earth's gravititational field smaller at
	the poles and larger at the equator than if the Earth were a sphere
	of the same mass. [At the equator, more of the mass is directly
	below, whereas at the poles more is off to the sides). This effect
	also makes the local mid-latitude gravity field not point towards
	the center of mass.
	
	The equation used in this subroutine agrees with the International
	Gravitational Formula of 1967 (Helmert's equation) within 0.005%.
	
	Interestingly, since the centripital effect of the Earth's rotation
	(-ve at equator, 0 at poles) has almost the opposite shape to the
	second order gravitational field (+ve at equator, -ve at poles),
	their sum is almost constant so that the surface gravity could be
	approximated (.07%) by the simple expression g=0.99746*GM/radius^2,
	the latitude variation coming entirely from the variation of surface
	r with latitude. This simple equation is not used in this subroutine.
	"""

	d2r=3.14159265/180.0	# Conversion from degrees to radians
	gm=3.9862216e+14  		# Gravitational constant times Earth's Mass (m3/s2)
	omega=7.292116E-05		# Earth's angular rotational velocity (radians/s)
	con=0.006738       		# (a/b)**2-1 where a & b are equatorial & polar radii
	shc=1.6235e-03  		# 2nd harmonic coefficient of Earth's gravity field 
	eqrad=6378178.0   		# Equatorial Radius (meters)

	gclat=arctan(tan(d2r*gdlat)/(1.0+con))  # radians

	radius=1000.0*altit+eqrad/np.sqrt(1.0+con*sin(gclat)**2)
	ff=(radius/eqrad)**2
	hh=radius*omega**2
	ge=gm/eqrad**2                      # = gravity at Re

	gravity=(ge*(1-shc*(3.0*sin(gclat)**2-1)/ff)/ff-hh*cos(gclat)**2)*(1+0.5*(sin(gclat)*cos(gclat)*(hh/ge+2.0*shc/ff**2))**2)

	return gravity, radius

def read_merradap(username,password,mode,site_lon_180,site_lat,gravity_at_lat,date,end_date,time_step,varlist,surf_varlist,muted):
	"""
	Read MERRA2 data via opendap.

	This has to connect to the daily netcdf files, and then concatenate the subsetted datasets.

	This is EXTREMELY slow, should probably make that use separate to generate local files, and then use to files in mod_maker
	"""
	DATA = {}
	SURF_DATA = {}

	if '42' in mode:
		letter = 'P'
	elif '72' in mode:
		letter = 'V'
		varlist += ['PL']

	old_UTC_date = ''
	urllist = []
	surface_urllist = []
	if not muted:
		print '\n\t-Making lists of URLs'
	while date < end_date:
		UTC_date = date + timedelta(hours = -site_lon_180/15.0) # merra times are in UTC, so the date may be different than the local date, make sure to use the UTC date to querry the file
		if (UTC_date.strftime('%Y%m%d') != old_UTC_date):
			if not muted:
				print '\t\t',UTC_date.strftime('%Y-%m-%d')
			urllist += ['https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I3N{}ASM.5.12.4/{:0>4}/{:0>2}/MERRA2_400.inst3_3d_asm_N{}.{:0>4}{:0>2}{:0>2}.nc4'.format(letter,UTC_date.year,UTC_date.month,letter.lower(),UTC_date.year,UTC_date.month,UTC_date.day)]
			surface_urllist += ['https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I1NXASM.5.12.4/{:0>4}/{:0>2}/MERRA2_400.inst1_2d_asm_Nx.{:0>4}{:0>2}{:0>2}.nc4'.format(UTC_date.year,UTC_date.month,UTC_date.year,UTC_date.month,UTC_date.day)]
			if old_UTC_date == '':
				session = setup_session(username,password,check_url=urllist[0]) # just need to setup the authentication session once
		old_UTC_date = UTC_date.strftime('%Y%m%d')
		date = date + time_step

	# multi-level data
	if not muted:
		print '\nNow doing multi-level data'
		print '\t-Connecting to datasets ...'
	store_list = [xarray.backends.PydapDataStore.open(url,session) for url in urllist]
	dataset_list = [xarray.open_dataset(store) for store in store_list]
	if not muted:
		print '\t-Datasets opened'
	min_lat_ID,max_lat_ID,min_lon_ID,max_lon_ID = querry_indices(dataset_list[0],site_lat,site_lon_180,2.5,2.5) # just need to get the lat/lon box once
	subsest_dataset_list = [dataset[{'lat':range(min_lat_ID,max_lat_ID+1),'lon':range(min_lon_ID,max_lon_ID+1)}] for dataset in dataset_list]
	if not muted:
		print '\t-Datasets subsetted'
		print '\t-Merging datasets (time consuming)'
	merged_dataset = xarray.concat(subsest_dataset_list,'time')
	merged_dataset = merged_dataset.fillna(1e15)

	# single-level data
	if not muted:
		print '\nNow doing single-level data'
		print '\t-Connecting to datasets ...'
	surface_store_list = [xarray.backends.PydapDataStore.open(url,session) for url in surface_urllist]
	surface_dataset_list = [xarray.open_dataset(store) for store in surface_store_list]
	if not muted:
		print '\t-Datasets opened'
	subsest_surface_dataset_list = [dataset[{'lat':range(min_lat_ID,max_lat_ID+1),'lon':range(min_lon_ID,max_lon_ID+1)}] for dataset in surface_dataset_list]
	if not muted:
		print '\t-Datasets subsetted'
		print '\t-Merging datasets (time consuming)'
	merged_surface_dataset = xarray.concat(subsest_surface_dataset_list,'time')
	merged_surface_dataset = merged_surface_dataset.fillna(1e15)

	for varname in varlist:
		DATA[varname] = merged_dataset[varname].data	
		DATA['add_offset_'+varname] = 0.0
		DATA['scale_factor_'+varname] = 1.0
	for varname in surf_varlist:
		SURF_DATA[varname] = merged_surface_dataset[varname].data
		SURF_DATA['add_offset_'+varname] = 0.0
		SURF_DATA['scale_factor_'+varname] = 1.0

	for varname in ['time','lat','lon']:
		DATA[varname] = merged_dataset[varname].data
		SURF_DATA[varname] = merged_surface_dataset[varname].data
	DATA['lev'] = merged_dataset['lev'].data

	DATA['PHIS'] = DATA['PHIS'] / gravity_at_lat # convert from m2 s-2 to m

	delta_time = [(i-DATA['time'][0]).astype('timedelta64[h]') / np.timedelta64(1,'h') for i in DATA['time']] # hours since base time
	surf_delta_time = [(i-SURF_DATA['time'][0]).astype('timedelta64[h]') / np.timedelta64(1,'h') for i in SURF_DATA['time']] # hours since base time

	DATA['julday0'] = Time(str(DATA['time'][0]),format="isot").jd
	SURF_DATA['julday0'] = Time(str(SURF_DATA['time'][0]),format="isot").jd

	DATA['time'] = delta_time
	SURF_DATA['time'] = surf_delta_time

	# get longitudes as 0 -> 360 instead of -180 -> 180, needed for trilinear_interp
	for i,elem in enumerate(DATA['lon']):
		if elem < 0:
			DATA['lon'][i] = elem + 360.0
	for i,elem in enumerate(SURF_DATA['lon']):
		if elem < 0:
			SURF_DATA['lon'][i] = elem + 360.0

	return DATA, SURF_DATA

def read_ncep(ncdf_path,year):
	"""
	Read data from yearly NCEP netcdf files and return it in one dictionary
	"""

	# path to the netcdf files
	ncdf_AT_file = os.path.join(ncdf_path,'.'.join(['air','{:0>4}'.format(year),'nc']))
	ncdf_GH_file = os.path.join(ncdf_path,'.'.join(['hgt','{:0>4}'.format(year),'nc']))
	ncdf_SH_file = os.path.join(ncdf_path,'.'.join(['shum','{:0>4}'.format(year),'nc']))

	print 'Read global',year,'NCEP data ...'
	# Air Temperature
	DATA = read_data(netCDF4.Dataset(ncdf_AT_file,'r'), ['air'])
	if len(DATA['air']) < 17:
		print 'Need 17 levels of AT data: found only ',len(lev_AT)

	# Specific Humidity
	SHUM_DATA = read_data(netCDF4.Dataset(ncdf_SH_file,'r'), ['shum'])
	if len(SHUM_DATA['level']) <  8:
		print 'Need  8 levels of SH data: found only ',len(lev_SH)

	if list(SHUM_DATA['level'])!=list(DATA['level'][:len(SHUM_DATA['level'])]):
		print 'Warning: air and shum do not share the same lower pressure levels'
	
	DATA.update(SHUM_DATA)
	
	# Geopotential Height
	GH_DATA = read_data(netCDF4.Dataset(ncdf_GH_file,'r'), ['hgt'])
	if len(GH_DATA['level']) < 17:
		print 'Need 17 levels of GH data: found only ',len(lev_GH)
	
	DATA.update(GH_DATA)

	for key in DATA:
		if 'air' in key:
			DATA[key.replace('air','T')] = DATA[key]
			del DATA[key]
		if 'hgt' in key:
			DATA[key.replace('hgt','H')] = DATA[key]
			del DATA[key]
		if 'shum' in key:
			DATA[key.replace('shum','QV')] = DATA[key]
			del DATA[key]
	
	DATA['lev'] = DATA['level']
	del DATA['level']

	return DATA

def read_global(ncdf_path,mode,site_lat,site_lon_180,gravity_at_lat,varlist,surf_varlist,muted):
	"""
	Read data from GEOS5 and MERRA2 datasets

	This assumes those are saved locally in GGGPATH/ncdf with two files per dataset (inst3_3d_asm_np and inst3_2d_asm_nx)
	"""

	key_dict = {'merraglob':'MERRA','fpglob':'_fp_','fpitglob':'_fpit_'}

	# assumes only one file with all the data exists in the GGGPATH/ncdf folder
	# path to the netcdf file
	ncdf_list = [i for i in os.listdir(ncdf_path) if key_dict[mode] in i]
	
	ncdf_file = [i for i in ncdf_list if '3d' in i][0]
	dataset = netCDF4.Dataset(os.path.join(ncdf_path,ncdf_file),'r')
	
	surf_file = [i for i in ncdf_list if '2d' in i][0]
	surface_dataset = netCDF4.Dataset(os.path.join(ncdf_path,surf_file),'r')

	if not muted:
		print ncdf_file
		print surf_file

	# get the min/max lat-lon indices of merra lat-lon that lies within a given box.
	# geos5-fp has a smaller grid than merra2 amd geos5-fp-it
	box_lat_half_width = float(dataset.LatitudeResolution)
	box_lon_half_width = float(dataset.LongitudeResolution)
	lat_lon_box = querry_indices(dataset,site_lat,site_lon_180,box_lat_half_width,box_lon_half_width)

	# multi-level data
	if not muted:
		print 'Read global',mode,'multi-level data ...'
	DATA = read_data(dataset,varlist,lat_lon_box)
	DATA['PHIS'] = DATA['PHIS'] / gravity_at_lat # convert from m2 s-2 to m

	# single level data
	if not muted:
		print 'Read global',mode,'single-level data ...'
	SURF_DATA = read_data(surface_dataset,surf_varlist,lat_lon_box)
	
	# merra/geos time is minutes since base time, need to convert to hours
	DATA['time'] = DATA['time'] / 60.0 
	SURF_DATA['time'] = SURF_DATA['time'] / 60.0

	return DATA,SURF_DATA

def equivalent_latitude_functions(ncdf_path,mode,start=None,end=None,muted=False):
	"""
	Inputs:
		- dataset: global dataset for fp, fp-it, or merra

	Outputs:
		- func_dict: list of functions, at each dataset time, to get equivalent latitude for a given PV and PT

	e.g. for the ith time, to get equivalent latitude for PV and PT: eq_lat = func_dict[i](PV,PT)

	takes ~ 3-4 minutes per date
	"""

	key_dict = {'merraglob':'MERRA','fpglob':'_fp_','fpitglob':'_fpit_'}

	ncdf_list = [i for i in os.listdir(ncdf_path) if key_dict[mode] in i]
	
	ncdf_file = [i for i in ncdf_list if '3d' in i][0]
	dataset = netCDF4.Dataset(os.path.join(ncdf_path,ncdf_file),'r')

	if not muted:
		print '\nGenerating equivalent latitude functions ...'

	lat = dataset['lat'][:]
	lat[180] = 0.0
	lon = dataset['lon'][:]
	pres = dataset['lev'][:]
	date = netCDF4.num2date(dataset['time'][:],dataset['time'].units)

	EPV = (dataset['EPV'][0]*1e6).data

	ntim,nlev,nlat,nlon = [dataset.dimensions[i].size for i in dataset.dimensions]
	if not muted:
		print 'time,lev,lat,lon',(ntim,nlev,nlat,nlon)

	select_dates = date[date>=start]
	select_dates = select_dates[select_dates<=end]
	date_inds = [np.where(date==np.datetime64(i))[0][0] for i in select_dates]
	ntim = len(date_inds)

	# Get the area of each grid cell
	lat_res = float(dataset.LatitudeResolution)
	lon_res = float(dataset.LongitudeResolution)

	lon_res = np.radians(lon_res)
	lat_half_res = 0.5*lat_res

	area = np.zeros([nlat,nlon])
	earth_area = 0
	for j in range(nlat):
		Slat = lat[j]-lat_half_res
		Nlat = lat[j]+lat_half_res

		Slat = np.radians(Slat)
		Nlat = np.radians(Nlat)
		for i in range(nlon):
			area[j,i] = lon_res*np.abs(sin(Slat)-sin(Nlat))

	earth_area = np.sum(area)

	if abs(np.sum(area)-earth_area)>0.0001:
		area = area*4*np.pi/earth_area

	# used to compute potential temperature PT = T*(P0/P)**0.286; this is the (P0/P)**0.286 which is computed once here instead of many times in the dates loop
	coeff = (1000.0/pres)**0.286
	coeff_mat = np.zeros([nlev,nlat,nlon]) 
	for i in range(nlat):
		for j in range(nlon):
			coeff_mat[:,i,j] = coeff
	        
	nmin = [0.125]
	func_dict = {} # dictionary mapping each time to the corresponding equivalent latitude function
	total_start = time.time()
	for t in date_inds: # loop over dates
		start = time.time()
		if not muted:
			sys.stdout.write('\r\tDate {:4d} / {:4d} ; finish in about {:.1f} minutes'.format(t+1,ntim,np.mean(nmin)*(ntim-t)))
			sys.stdout.flush()

		# Compute potential temperature
		PT = (dataset['T'][t]*coeff_mat).data

		EPV = (dataset['EPV'][t].data)*1e6 # Potential vorticity in PVU = 1e-6 K . m2 / kg / s	

		# Get rid of fill values, this fills the bottom of profiles with the first valid value
		PT[PT>1e4]=np.nan
		EPV[EPV>1e8]=np.nan
		for i in range(nlat):
			pd.DataFrame(PT[:,i,:]).fillna(method='bfill',axis=0,inplace=True)
			pd.DataFrame(EPV[:,i,:]).fillna(method='bfill',axis=0,inplace=True)

		# Define a fixed potential temperature grid, with increasing spacing
		#fixed_PT = np.arange(np.min(PT),np.max(PT),20) # fixed potential temperature grid
		fixed_PT = sorted(list(set(range(int(np.min(PT)),300,2)+range(300,350,5)+range(350,500,10)+range(500,750,20)+range(750,1000,30)+range(1000,int(np.max(PT)),100))))
		new_nlev = len(fixed_PT)

		# Get PV on the fixed PT levels
		new_EPV = np.zeros([new_nlev,nlat,nlon])
		for i in range(nlat):
			for j in range(nlon):
				new_EPV[:,i,j] = np.interp(fixed_PT,PT[:,i,j],EPV[:,i,j])

		# Compute equivalent latitudes
		EL = np.zeros([new_nlev,100])
		EPV_thresh = np.zeros([new_nlev,100])
		for k in range(new_nlev): # loop over potential temperature levels
			maxPV = np.max(new_EPV[k]) # global max PV
			minPV = np.min(new_EPV[k]) # global min PV

			# define 100 PV values between the min and max PV
			EPV_thresh[k] = np.linspace(minPV,maxPV,100)

			for l,thresh in enumerate(EPV_thresh[k]):
				area_total = np.sum(area[new_EPV[k]>=thresh])
				EL[k,l] = arcsin(1-area_total/(2*np.pi))*90.0*2/np.pi

		# Define a fixed potentital vorticity grid, with increasing spacing away from 0
		#fixed_PV = np.arange(np.min(EPV_thresh),np.max(EPV_thresh)+10,10) # fixed PV grid
		fixed_PV = sorted(list(set(range(int(np.min(EPV_thresh)-50),-1000,50)+range(-1000,-500,20)+range(-500,-100,10)+range(-100,-10,1)+list(np.arange(-10,-1,0.1))+list(np.arange(-1,1,0.01))+list(np.arange(1,10,0.1))+range(10,100,1)+range(100,500,10)+range(500,1000,20)+range(1000,int(np.max(EPV_thresh)+50),50))))
		if 0.0 not in fixed_PV: # need a point at 0.0 for the interpolations to work better
			fixed_PV = np.sort(np.append(fixed_PV,0.0))

		# Generate interpolating function to get EL for a given PV and PT
		interp_EL = np.zeros([new_nlev,len(fixed_PV)])
		for k in range(new_nlev):
			interp_EL[k] = np.interp(fixed_PV,EPV_thresh[k],EL[k])

		func_dict[date[t]] = interp2d(fixed_PV,fixed_PT,interp_EL)

		end = time.time()
		nmin.append(int(end-start)/60.0)

	actual_time = (time.time()-total_start)/60.0
	predicted_time = 0.125*ntim
	if not muted:
		print '\nPredicted to finish in {:.1f} minutes\nActually finished in {:.1f} minutes'.format(predicted_time,actual_time)

	dataset.close()

	return func_dict

def parse_args(argu=sys.argv):
	"""
	parse commandline arguments (see code header)
	"""

	arg_dict = {}

	muted = False
	if 'mute' in argu:
		muted = True
	arg_dict['muted'] = muted

	# parse the selected range of dates for which .mod files will be generated
	date_range = argu[1].split('-')
	start_date = datetime.strptime(date_range[0],'%Y%m%d')
	try:
		end_date = datetime.strptime(date_range[1],'%Y%m%d')
	except IndexError: # if a single date is given set the end date as the next day
		end_date = start_date + timedelta(days=1)
	if start_date>=end_date:
		print 'Error: the second argument must be a date range YYYYMMDD-YYYYMMDD or a single date YYYYMMDD'
		sys.exit()
	if not muted:
		print 'Date range: from',start_date.strftime('%Y-%m-%d'),'to',end_date.strftime('%Y-%m-%d')

	arg_dict['start_date'] = start_date
	arg_dict['end_date'] = end_date

	for arg in argu:
		if 'geos_path=' in arg:
			geos_path = arg.split('=')[1]
			if not os.path.exists(geos_path):
				print 'Wrong path given for geos_path:',geos_path
				sys.exit()
			arg_dict['geos_path'] = geos_path
		if 'site=' in arg:
			site_abbrv = arg.split('=')[1].lower()
			arg_dict['site_abbrv'] = site_abbrv
		if 'mode=' in arg:
			mode = arg.split('=')[1].lower()
			arg_dict['mode'] = mode
			if False not in [elem not in mode for elem in ['merradap','merraglob','fpglob','fpitglob','ncep']]:
				print 'Wrong mode, must be one of [ncep, merradap42, merradap72, merraglob, fpglob, fpitglob]'
				sys.exit()
			print 'Mode:',mode.upper()
		if 'time=' in arg:
			HHMM = arg.split('=')[1]
			# hour and minute for time interpolation, default is local noon
			time_input = re.search('([0-9][0-9]):([0-9][0-9])',HHMM).groups()

			HH = int(time_input[0])
			MM = int(time_input[1])

			# small checks
			if HH>=24 or HH<0:
				print 'Need 0<=H<24'
				sys.exit()
			if MM>=60 or MM<0:
				print 'Need 0<=MM<60'
				sys.exit()
		else: # use local noon by default
			HH = 12
			MM = 0

		arg_dict['HH'] = HH
		arg_dict['MM'] = MM

		if 'step=' in arg:
			arg_dict['time_step'] = float(arg.split('=')[1])
		else: # use 1 day by default
			arg_dict['time_step'] = 24

	return arg_dict

def mod_file_name(date,time_step,site_lat,site_lon_180,ew,ns,mod_path):

	YYYYMMDD = date.strftime('%Y%m%d')
	HHMM = date.strftime('%H%M')
	if time_step < timedelta(days=1):
		mod_name = '{}_{}_{:0>2.0f}{:>1}_{:0>3.0f}{:>1}.mod'.format(YYYYMMDD,HHMM,round(abs(site_lat)),ns,round(abs(site_lon_180)),ew)
	else:
		mod_name = '{}_{:0>2.0f}{:>1}_{:0>3.0f}{:>1}.mod'.format(YYYYMMDD,round(abs(site_lat)),ns,round(abs(site_lon_180)),ew)

	return mod_name

def GEOS_files(GEOS_path,start_date,end_date):

	# all GEOS5-FPIT Np files and their dates
	ncdf_list = np.array(os.listdir(GEOS_path))
	ncdf_dates = np.array([datetime.strptime(elem.split('.')[-3],'%Y%m%d_%H%M') for elem in ncdf_list])

	# just the one between the 'start_date' and 'end_date' dates
	select_files = ncdf_list[(ncdf_dates>=start_date) & (ncdf_dates<=end_date)]
	select_dates = ncdf_dates[(ncdf_dates>=start_date) & (ncdf_dates<=end_date)]

	if len(select_dates) == 0:
		print 'No GEOS files between',start_date,end_date
		sys.exit()

	return select_files,select_dates

def equivalent_latitude_functions_new(ncdf_path,start_date=None,end_date=None,muted=False):
	"""
	Inputs:
		- dataset: global dataset for fp, fp-it, or merra

	Outputs:
		- func_dict: list of functions, at each dataset time, to get equivalent latitude for a given PV and PT

	e.g. for the ith time, to get equivalent latitude for PV and PT: eq_lat = func_dict[i](PV,PT)

	takes ~ 3-4 minutes per date
	"""

	select_files, select_dates = GEOS_files(ncdf_path,start_date,end_date)

	if not muted:
		print '\nGenerating equivalent latitude functions for {} times'.format(len(select_dates))

	# Use any file for stuff that is the same in all files
	with netCDF4.Dataset(os.path.join(ncdf_path,select_files[0]),'r') as dataset:
		lat = dataset['lat'][:]
		lat[180] = 0.0
		lon = dataset['lon'][:]
		pres = dataset['lev'][:]
		ntim,nlev,nlat,nlon = dataset['EPV'].shape

		# Get the area of each grid cell
		lat_res = float(dataset.LatitudeResolution)
		lon_res = float(dataset.LongitudeResolution)

	lon_res = np.radians(lon_res)
	lat_half_res = 0.5*lat_res

	# Compute area of each grid cell and the total area
	area = np.zeros([nlat,nlon])
	earth_area = 0
	for j in range(nlat):
		Slat = lat[j]-lat_half_res
		Nlat = lat[j]+lat_half_res

		Slat = np.radians(Slat)
		Nlat = np.radians(Nlat)
		for i in range(nlon):
			area[j,i] = lon_res*np.abs(sin(Slat)-sin(Nlat))

	earth_area = np.sum(area)

	if abs(np.sum(area)-earth_area)>0.0001: # ensure proper normalization so the total area of Earth is 4*pi
		area = area*4*np.pi/earth_area # in units of squared Earth radius

	# pre-compute pressure coefficients for calculating potential temperature, this is the (Po/P)^(R/Cp) term
	coeff = (1000.0/pres)**0.286
	coeff_mat = np.zeros([nlev,nlat,nlon])
	for i in range(nlat):
		for j in range(nlon):
			coeff_mat[:,i,j] = coeff
	
	ntim = len(select_dates)
	nmin = [0.125]
	func_dict = {}
	total_start = time.time()
	for date_ID,date in enumerate(select_dates):

		start = time.time()
		if not muted:
			sys.stdout.write('\r\tDate {:4d} / {:4d} ; finish in about {:.1f} minutes'.format(date_ID+1,ntim,np.mean(nmin)*(ntim-date_ID)))
			sys.stdout.flush()
		
		with netCDF4.Dataset(os.path.join(ncdf_path,select_files[date_ID])) as dataset:
		        
			PT = (dataset['T'][0]*coeff_mat).data # Compute potential temperature

			EPV = (dataset['EPV'][0].data)*1e6 # Potential vorticity in PVU = 1e-6 K . m2 / kg / s	

		# Get rid of fill values, this fills the bottom of profiles with the first valid value
		PT[PT>1e4]=np.nan
		EPV[EPV>1e8]=np.nan
		for i in range(nlat):
			pd.DataFrame(PT[:,i,:]).fillna(method='bfill',axis=0,inplace=True)
			pd.DataFrame(EPV[:,i,:]).fillna(method='bfill',axis=0,inplace=True)

		# Define a fixed potential temperature grid, with increasing spacing
		# this is done arbitrarily to get sufficient levels for the interpolation to work well, and not too much for the computations to take less time
		#fixed_PT = np.arange(np.min(PT),np.max(PT),20) # fixed potential temperature grid
		fixed_PT = sorted(list(set(range(int(np.min(PT)),300,2)+range(300,350,5)+range(350,500,10)+range(500,750,20)+range(750,1000,30)+range(1000,int(np.max(PT)),100))))
		new_nlev = len(fixed_PT)

		# Get PV on the fixed PT levels ~ 2 seconds per date
		new_EPV = np.zeros([new_nlev,nlat,nlon])
		for i in range(nlat):
			for j in range(nlon):
				new_EPV[:,i,j] = np.interp(fixed_PT,PT[:,i,j],EPV[:,i,j])

		# Compute equivalent latitudes
		EL = np.zeros([new_nlev,100])
		EPV_thresh = np.zeros([new_nlev,100])
		for k in range(new_nlev): # loop over potential temperature levels
			maxPV = np.max(new_EPV[k]) # global max PV
			minPV = np.min(new_EPV[k]) # global min PV

			# define 100 PV values between the min and max PV
			EPV_thresh[k] = np.linspace(minPV,maxPV,100)

			for l,thresh in enumerate(EPV_thresh[k]):
				area_total = np.sum(area[new_EPV[k]>=thresh])
				EL[k,l] = arcsin(1-area_total/(2*np.pi))*90.0*2/np.pi

		# Define a fixed potentital vorticity grid, with increasing spacing away from 0
		#fixed_PV = np.arange(np.min(EPV_thresh),np.max(EPV_thresh)+10,10) # fixed PV grid
		fixed_PV = sorted(list(set(range(int(np.min(EPV_thresh)-50),-1000,50)+range(-1000,-500,20)+range(-500,-100,10)+range(-100,-10,1)+list(np.arange(-10,-1,0.1))+list(np.arange(-1,1,0.01))+list(np.arange(1,10,0.1))+range(10,100,1)+range(100,500,10)+range(500,1000,20)+range(1000,int(np.max(EPV_thresh)+50),50))))
		if 0.0 not in fixed_PV: # need a point at 0.0 for the interpolations to work better
			fixed_PV = np.sort(np.append(fixed_PV,0.0))

		# Generate interpolating function to get EL for a given PV and PT
		interp_EL = np.zeros([new_nlev,len(fixed_PV)])
		for k in range(new_nlev):
			interp_EL[k] = np.interp(fixed_PV,EPV_thresh[k],EL[k])

		func_dict[date] = interp2d(fixed_PV,fixed_PT,interp_EL)

		end = time.time()
		nmin.append(int(end-start)/60.0)
		# end of loop over dates

	actual_time = (time.time()-total_start)/60.0
	predicted_time = 0.125*ntim
	if not muted:
		print '\nPredicted to finish in {:.1f} minutes\nActually finished in {:.1f} minutes'.format(predicted_time,actual_time)

	return func_dict

def lat_lon_interp(data_old,lat_old,lon_old,lat_new,lon_new,IDs_list):
	"""
	Use RectSphereBivariateSpline to interpolate in a latitude-longitude grid (rectangle over of sphere)
	
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.html

	lat_old: array of latitude within [-90,90] degrees
	lon_old: array of longitudes within [-180,+180[ degrees (+180 excluded)

	lat_new: array of latitudest o interpolate to [-90,90]
	lon_new: array of longitudes to interpolate to [-180,180[
	"""

	# make copies of input arrays to not modify them in place
	data_old = data_old.copy()
	lat_old = lat_old.copy()
	lon_old = lon_old.copy()
	lat_new = lat_new.copy()
	lon_new = lon_new.copy()
	
	"""	
	# Using interpolation in rectangles on a sphere
	lat_old = deg2rad(lat_old+90)[1:-1]
	lon_old = deg2rad(lon_old)

	data_old = data_old[1:-1,:]

	func = RectSphereBivariateSpline(lat_old,lon_old,data_old)

	for i,elem in enumerate(lon_new):
		if elem<0:
			lon_new[i] = elem + 360

	data_new = func.ev(deg2rad(lat_new+90),deg2rad(lon_new))
	"""
	
	data_new = []
	count = 0
	for IDs in IDs_list:
		lat1,lat2,lon1,lon2 = IDs

		lat = np.array([lat_old[lat1],lat_old[lat2]])
		lon = np.array([lon_old[lon1],lon_old[lon2]])

		data = np.array([[data_old[lat1,lon1],data_old[lat1,lon2]],[data_old[lat2,lon2],data_old[lat2,lon1]]])

		data = ma.masked_where(np.isnan(data),data)

		func = interp2d(lat,lon,data)

		data_new.append(func(lat_new[count],lon_new[count]))

		count+=1
		
	return data_new

def show_interp(data,x,y,interp_data,ilev):

	max = data[ilev].max()
	min = data[ilev].min()

	pl.imshow(data[ilev],extent=(-180,180,90,-90),vmin=min,vmax=max)
	try:
		pl.scatter(x,y,c=np.diag(interp_data[ilev]),vmax=max,vmin=min,edgecolor='black')
	except:
		pl.scatter(x,y,c=interp_data[ilev],vmax=max,vmin=min,edgecolor='black')
	pl.gca().invert_yaxis()
	pl.xlabel('Longitude')
	pl.ylabel('Latitude')
	pl.title('Level {}'.format(ilev+1))
	pl.colorbar()
	pl.show()

def mod_maker_new(start_date=None,end_date=None,func_dict=None,GEOS_path=None,muted=False):
	"""
	This code only works with GEOS-5 FP-IT data.
	It generates MOD files for all sites between start_date and end_date on GEOS-5 times (every 3 hours)

	start_date: datetime object for first date
	end_date:  datetime object for last date
	func_dict: output of equivalent_latitude_functions
	GEOS_path: full path to the directory containing all the GEOS5-FP-IT files
	"""

	site_dict = tccon_site_info()

	GGGPATH = os.environ['GGGPATH']
	mod_path = os.path.join(GGGPATH,'models','gnd','fpit')
	if not os.path.exists(mod_path):
		if not muted:
			print 'Creating',mod_path
		os.makedirs(mod_path)

	varlist = ['T','QV','RH','H','EPV','O3','PHIS']
	surf_varlist = ['T2M','QV2M','PS','SLP','TROPPB','TROPPV','TROPPT','TROPT']

	select_files, select_dates = GEOS_files(os.path.join(GEOS_path,'Np'),start_date,end_date)
	select_surf_files, select_surf_dates = GEOS_files(os.path.join(GEOS_path,'Nx'),start_date,end_date)

	nsite = len(site_dict.keys())

	rmm = 28.964/18.02	# Ratio of Molecular Masses (Dry_Air/H2O)

	start = time.time()
	for date_ID,UTC_date in enumerate(select_dates):
		start_it = time.time()

		DATA = {}
		if not muted:
			print '\nNOW DOING date {:4d} / {} :'.format(date_ID+1,len(select_dates)),UTC_date.strftime("%Y-%m-%d %H:%M"),' UTC'
			print '\t-Read global data ...'

		with netCDF4.Dataset(os.path.join(GEOS_path,'Np',select_files[date_ID]),'r') as dataset:

			for var in varlist:
				DATA[var] = dataset[var][0]
			DATA['lev'] = dataset['lev'][:]

			lat = dataset['lat'][:]
			lon = dataset['lon'][:]
			nlev = dataset.dimensions['lev'].size
			nlat = dataset.dimensions['lat'].size
			nlon = dataset.dimensions['lon'].size
			if date_ID == 0:
				box_lat_half_width = 0.5*float(dataset.LatitudeResolution)
				box_lon_half_width = 0.5*float(dataset.LongitudeResolution)

		for site in site_dict:
			if 'time_spans' in site_dict[site].keys(): # instruments with different locations for different time periods
				for time_span in site_dict[site]['time_spans']:
					if time_span[0]<=UTC_date<time_span[1]:
						site_dict[site]['IDs'] = querry_indices([lat,lon],site_dict[site]['time_spans'][time_span]['lat'],site_dict[site]['time_spans'][time_span]['lon_180'],box_lat_half_width,box_lon_half_width)
						site_dict[site]['lat'] = site_dict[site]['time_spans'][time_span]['lat']
						site_dict[site]['lon'] = site_dict[site]['time_spans'][time_span]['lon']
						site_dict[site]['lon_180'] = site_dict[site]['time_spans'][time_span]['lon_180']
						site_dict[site]['alt'] = site_dict[site]['time_spans'][time_span]['alt']
						break
			else:
				site_dict[site]['IDs'] = querry_indices([lat,lon],site_dict[site]['lat'],site_dict[site]['lon_180'],box_lat_half_width,box_lon_half_width)

		new_lats = np.array([site_dict[site]['lat'] for site in site_dict])
		new_lons = np.array([site_dict[site]['lon_180'] for site in site_dict])

		SURF_DATA = {}
		with netCDF4.Dataset(os.path.join(GEOS_path,'Nx',select_surf_files[date_ID]),'r') as dataset:
			for var in surf_varlist:
				SURF_DATA[var] = dataset[var][0]

		IDs_list = [site_dict[site]['IDs'] for site in site_dict]

		if not muted:
			print '\t-Interpolate to (lat,lon) of sites ...'
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")		
			INTERP_DATA = {}
			for var in varlist:
				if not muted:
					sys.stdout.write('\r\t\tNow doing : {:<10s}'.format(var))
					sys.stdout.flush()
				if DATA[var].ndim==2:
					INTERP_DATA[var] = lat_lon_interp(DATA[var],lat,lon,new_lats,new_lons,IDs_list)
					continue

				INTERP_DATA[var] = np.zeros([nlev,nsite])
				for ilev,level_data in enumerate(DATA[var]):
					INTERP_DATA[var][ilev] = lat_lon_interp(level_data,lat,lon,new_lats,new_lons,IDs_list)

			INTERP_SURF_DATA = {}
			for var in surf_varlist:
				if not muted:
					sys.stdout.write('\r\t\tNow doing : {:<10s}'.format(var))
					sys.stdout.flush()
				INTERP_SURF_DATA[var] = lat_lon_interp(SURF_DATA[var],lat,lon,new_lats,new_lons,IDs_list)
			if not muted:
				print '\r\t\t{:<40s}'.format('DONE')

		# setup masks
		for var in varlist:
			for i in range(nsite):
				INTERP_DATA[var] = ma.masked_where(np.isnan(INTERP_DATA[var]),INTERP_DATA[var])

		for var in surf_varlist:
			for i in range(nsite):
				INTERP_SURF_DATA[var] = ma.masked_where(np.isnan(INTERP_SURF_DATA[var]),INTERP_SURF_DATA[var]).reshape(nsite)

		INTERP_DATA['lev'] = DATA['lev'].copy()

		for varname in ['PS','SLP','TROPPB','TROPPV','TROPPT']:
			INTERP_SURF_DATA[varname] = INTERP_SURF_DATA[varname] / 100.0 # convert Pa to hPa

		INTERP_DATA['H2O_DMF'] = rmm*INTERP_DATA['QV']/(1-INTERP_DATA['QV']) # Convert specific humidity, a wet mass mixing ratio, to dry mole fraction
		INTERP_DATA['H'] = INTERP_DATA['H']/1000.0	# Convert m to km
		INTERP_DATA['PHIS'] = INTERP_DATA['PHIS'] / 1000.0

		INTERP_SURF_DATA['H2O_DMF'] = compute_h2o_dmf(INTERP_SURF_DATA['QV2M'],rmm)
		
		# compute surface relative humidity
		svp = svp_wv_over_ice(INTERP_SURF_DATA['T2M'])
		INTERP_SURF_DATA['H2O_WMF'] = compute_h2o_wmf(INTERP_SURF_DATA['H2O_DMF'])  # wet mole fraction of h2o
		INTERP_SURF_DATA['RH'] = compute_rh(INTERP_SURF_DATA['T2M'],INTERP_SURF_DATA['H2O_WMF'],INTERP_SURF_DATA['PS']) # Fractional relative humidity
		INTERP_SURF_DATA['MMW'] = compute_mmw(INTERP_SURF_DATA['H2O_WMF'])
		INTERP_SURF_DATA['H'] = INTERP_DATA['PHIS']

		# restructure the data
		temp_data = {}
		for i,site in enumerate(site_dict.keys()):
			gravity_at_site,r = gravity(site_dict[site]['lat'],site_dict[site]['alt']/1000.0)
			temp_data[site] = {}
			temp_data[site]['prof'] = {}
			for var in INTERP_DATA:
				if var == 'lev':
					temp_data[site]['prof'][var] = INTERP_DATA[var]
				elif var == 'PHIS':
					continue
				else:
					temp_data[site]['prof'][var] = INTERP_DATA[var][:,i]
			
			temp_data[site]['surf'] = {}
			for var in INTERP_SURF_DATA:
				if var == 'H':
					temp_data[site]['surf'][var] = INTERP_SURF_DATA[var][i][0] / gravity_at_site
				else:
					temp_data[site]['surf'][var] = INTERP_SURF_DATA[var][i]
		INTERP_DATA = temp_data

		# add a mask for temperature = 0 K
		for site in site_dict:
			for var in INTERP_DATA[site]['prof']:
				if var not in ['T','lev']:
					INTERP_DATA[site]['prof'][var] = ma.masked_where(INTERP_DATA[site]['prof']['T']==0,INTERP_DATA[site]['prof'][var])
			INTERP_DATA[site]['prof']['T'] =  ma.masked_where(INTERP_DATA[site]['prof']['T']==0,INTERP_DATA[site]['prof']['T'])
		
		# get slant path coordinates corresponding to the altitude levels above each site
		if not muted:
			print '\t-Slantify:'
		for i,site in enumerate(site_dict.keys()): # loops over sites
			if not muted:
				sys.stdout.write('\r\t\t site {:3d} / {}  {:>20}'.format(i+1,nsite,site_dict[site]['name']))
				sys.stdout.flush()

			site_alt = site_dict[site]['alt']
			site_lat = site_dict[site]['lat']
			site_lon = site_dict[site]['lon_180']

			# vertical grid above site
			H = INTERP_DATA[site]['prof']['H']*1000.0
			pres = INTERP_DATA[site]['surf']['PS'] # surface pressure (hPa)
			temp = INTERP_DATA[site]['surf']['T2M']-273.15 # surface temperature (celsius)

			# get the (lat,lon,alt) of points on sunray correspondings to the vertical altitudes
			site_dict[site]['slant_coords'] = slantify(UTC_date,site_lat,site_lon,site_alt,H,pres=pres,temp=temp)
			for var in ['lat','lon','alt','vertical','slant']:
				site_dict[site]['slant_coords'][var] = ma.masked_where(H.mask,site_dict[site]['slant_coords'][var])
		if not muted:
			print '\r\t\t{:<40s}'.format('DONE')

		# Set two lists with all the latitudes and longitudes of all sites at all slant levels
		slant_lat = []
		slant_lon = []
		slat_slon = []
		for site in site_dict:
			if site_dict[site]['slant_coords']['sza']<90: # only make profiles where sun is above the horizon
				slat = site_dict[site]['slant_coords']['lat']
				slon = site_dict[site]['slant_coords']['lon']
				slant_lat.extend(slat)
				slant_lon.extend(slon)
				for i in range(len(slat)):
					if slat[i] is ma.masked:
						continue
					if (slat[i],slon[i]) not in slat_slon:
						slat_slon.append((slat[i],slon[i]))

		IDs_list = np.array([querry_indices([lat,lon],slat,slon,box_lat_half_width,box_lon_half_width) for slat,slon in slat_slon])

		slant_lat = np.array([slat for slat,slon in slat_slon])
		slant_lon = np.array([slon for slat,slon in slat_slon])

		# Interpolate to each slant level (lat,lon)
		# This will give a vertical profile at every (lat,lon) of all the slant levels
		if not muted:
			print '\t-Interpolate to each slant level (lat,lon) ...'
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			NEW_INTERP_DATA = {}
			for var in varlist:
				if not muted:
					sys.stdout.write('\r\t\tNow doing : {:<10s}'.format(var))
					sys.stdout.flush()
				if DATA[var].ndim==2:
					NEW_INTERP_DATA[var] = lat_lon_interp(DATA[var],lat,lon,slant_lat,slant_lon,IDs_list)
					continue

				NEW_INTERP_DATA[var] = np.zeros([nlev,len(IDs_list)])
				for ilev,level_data in enumerate(DATA[var]):
					NEW_INTERP_DATA[var][ilev] = lat_lon_interp(level_data,lat,lon,slant_lat,slant_lon,IDs_list)
			if not muted:
				print '\r\t\t{:<40s}'.format('DONE')
		# setup masks
		for var in set(varlist)-set(['PHIS']):
			NEW_INTERP_DATA[var] = ma.masked_where(np.isnan(NEW_INTERP_DATA[var]),NEW_INTERP_DATA[var])

		# Now just get the data along the slant paths
		if not muted:
			print '\t-Get data along slant paths ...'
		SLANT_DATA = {}
		for site in site_dict: # for each site
			if site_dict[site]['slant_coords']['sza']<90:
				SLANT_DATA[site] = site_dict[site]['slant_coords']
				SLANT_DATA[site]['H'] = SLANT_DATA[site]['alt']
				SLANT_DATA[site]['lev'] = INTERP_DATA[site]['prof']['lev']
				for var in set(varlist)-set(['H','PHIS']): # for each variable
					SLANT_DATA[site][var] = np.array([])				
					for i in range(len(SLANT_DATA[site]['H'])): # for each slant point
						slat,slon = SLANT_DATA[site]['lat'][i] , SLANT_DATA[site]['lon'][i]
						try:
							ID = slat_slon.index((slat,slon))
						except ValueError:
							SLANT_DATA[site][var] = np.append(SLANT_DATA[site][var],np.nan)
						else:
							SLANT_DATA[site][var] = np.append(SLANT_DATA[site][var],NEW_INTERP_DATA[var][i,ID])
					
					SLANT_DATA[site][var] = ma.masked_where(np.isnan(SLANT_DATA[site][var]),SLANT_DATA[site][var])

		# Interpolate T,RH,MMW down to 1000 hPa
		# Or extrapolate so that when linearly interpolating to the surface level using two levels that bracket the surface pressure, the values match the surface values
		# if surface pressure is larger than 1000 hPa, interpolate between surface pressure and the first valid level
		# if surface pressure is smaller than 1000 hPa, extrapolate to 1000 hPa using the surface data and the first valid level above it
		if not muted:
			print '\t-Patch fill values ...'	
		for site in site_dict:
			try:
				all_data = [SLANT_DATA[site],INTERP_DATA[site]['prof']]
			except KeyError:
				all_data = [INTERP_DATA[site]['prof']]

			for elem in all_data:
				H = elem['H']
				patch = {'surf':{},'first':{}}
				missing_p = elem['lev'][H.mask] # pressure levels with masked data
				if len(missing_p)!=0:
					surf_p = INTERP_DATA[site]['surf']['PS'] # surface pressure

					first_p = elem['lev'][~H.mask][0] # first valid pressure level
					first_ID = np.where(elem['lev']==first_p)[0]

					patch['surf']['H'] = INTERP_DATA[site]['surf']['H']
					patch['surf']['RH'] = INTERP_DATA[site]['surf']['RH']/100.0
					patch['surf']['QV'] = INTERP_DATA[site]['surf']['QV2M']
					patch['surf']['T'] = INTERP_DATA[site]['surf']['T2M']

					patch['first']['H'] =  elem['H'][first_ID].data[0]
					patch['first']['RH'] =  elem['RH'][first_ID].data[0]
					patch['first']['QV'] =  elem['QV'][first_ID].data[0]
					patch['first']['T'] =  elem['T'][first_ID].data[0]

					#for variables with no surface data, just use the value of the first valid level
					for var in ['EPV','O3']: 
						elem[var][H.mask] = elem[var][~H.mask][0]

					# interpolate/extrapolate using the first valid level and surface data
					if surf_p>first_p:
						patch_p = [surf_p,first_p]
						for var in ['RH','QV','T','H']: # must do H last since the last line will get rid of its masks
							patch_var = [patch['surf'][var],patch['first'][var]]
							f = interp1d(patch_p,patch_var,fill_value='extrapolate')
							elem[var][H.mask] = f(missing_p)
					else:
						first_2p = elem['lev'][~H.mask][:2]
						patch_p = np.insert(first_2p,1,surf_p)
						for var in ['RH','QV','T','H']: # must do H last since the last line will get rid of its masks
							first_2var = elem[var][~H.mask][:2]
							patch_var = np.insert(first_2var,1,patch['surf'][var])
							f = interp1d(patch_p,patch_var,fill_value='extrapolate')
							elem[var][H.mask] = f(missing_p)
				elem['H2O_DMF'] = compute_h2o_dmf(elem['QV'],rmm) # Convert specific humidity, a wet mass mixing ratio, to dry mole fraction
		
		if not muted:
			print '\t\t {:3d} / {} sites with SZA<90'.format(len(SLANT_DATA.keys()),nsite)
			print '\t-Write mod files ...'
		
		# write the .mod files
		version = 'mod_maker_10.6   2017-04-11   GCT'

		for site in INTERP_DATA:

			INTERP_DATA[site]['surf']['SZA'] = site_dict[site]['slant_coords']['sza']

			site_lat = site_dict[site]['lat']
			site_lon_180 = site_dict[site]['lon_180']

			utc_offset = timedelta(hours=site_dict[site]['lon_180']/15.0)
			local_date = UTC_date + utc_offset

			vertical_mod_path =  os.path.join(mod_path,site,'vertical')
			if not os.path.exists(vertical_mod_path):
				os.makedirs(vertical_mod_path)

			slant_mod_path =  os.path.join(mod_path,site,'slant')
			if not os.path.exists(slant_mod_path):
				os.makedirs(slant_mod_path)
				
			# directions for .mod file name
			if site_lat > 0:
				ns = 'N'
			else:
				ns = 'S'

			if site_lon_180>0:
				ew = 'E'
			else:
				ew = 'W'

			mod_name = mod_file_name(local_date,timedelta(hours=3),site_lat,site_lon_180,ew,ns,mod_path)
			if not muted:
				print '\t\t\t{:<20s} : {}'.format(site_dict[site]['name'], mod_name)
			
			# write vertical mod file
			mod_file_path = os.path.join(vertical_mod_path,mod_name)
			write_mod(mod_file_path,version,site_lat,data=INTERP_DATA[site]['prof'],surf_data=INTERP_DATA[site]['surf'],func=func_dict[UTC_date],muted=muted)
			# write slant mod_file
			if site in SLANT_DATA.keys():
				if not muted:
					print '\t\t\t{:>20s} + slant'.format('')
				mod_file_path = os.path.join(slant_mod_path,mod_name)
				write_mod(mod_file_path,version,site_lat,data=SLANT_DATA[site],surf_data=INTERP_DATA[site]['surf'],func=func_dict[UTC_date],muted=muted)
		if not muted:
			print '\ndate {:4d} / {} DONE in {:.0f} seconds'.format(date_ID+1,len(select_dates),time.time()-start_it)
	if not muted:
		print 'ALL DONE in {:.1f} minutes'.format((time.time()-start)/60.0)

def mod_maker(site_abbrv=None,start_date=None,end_date=None,mode=None,HH=None,MM=None,time_step=None,muted=False):
	"""
	Inputs:
		- site_abbvr: two letter site abbreviation
		- start_date: YYYYMMDD (set from date_range in parse_args)
		- end_date: YYYYMMDD  (set from date_range in parse_args)
		- mode: one of ncep, merradap42, merradap72, merraglob, fpglob, fpitglob
		- HH:MM (default: '12:00' ): string of starting time 'HH:MM' in local time
		- time_step (default: 24): time step in hours
		- func_dict: for the 'glob' modes, a list of functions to get equivalent latitude for a given potential vorticity and potential temperature
	Outputs:
		- .mod files at every time_step within the given date range
	"""
	if 'merradap' in mode: # get the earthdata credentials
		try:
			username,account,password = netrc.netrc().authenticators('urs.earthdata.nasa.gov')
		except:
			print 'When using MERRA mode, you need a ~/.netrc file to connect to urs.earthdata.nasa.gov'
			sys.exit()

	GGGPATH = os.environ['GGGPATH'] # reads the GGGPATH environment variable
	if not muted:
		print 'GGGPATH =',GGGPATH

	ncdf_path = os.path.join(GGGPATH,'ncdf')

	site_dict = tccon_site_info()

	try:
		print 'Site:',site_dict[site_abbrv]['name'],site_dict[site_abbrv]['loc']
	except KeyError:
		print 'Wrong 2 letter site abbreviation (check the site_dict dictionary)'
		sys.exit()
	
	if not muted:
		print 'lat,lon,masl:',site_dict[site_abbrv]['lat'],site_dict[site_abbrv]['lon'],site_dict[site_abbrv]['alt']

	simple = {'merradap42':'merra','merradap72':'merra','merraglob':'merra','ncep':'ncep','fpglob':'fp','fpitglob':'fpit'}
	mod_path = os.path.join(GGGPATH,'models','gnd',simple[mode],site_abbrv)	# .mod files will be saved here
	if not os.path.exists(mod_path):
		os.makedirs(mod_path)
	if not muted:
		print 'MOD files will be saved in:',mod_path

	local_date = start_date + timedelta(hours=HH,minutes=MM) # date with local time
	astropy_date = Time(local_date)
	if not muted:
		print 'Starting local time for interpolation:',local_date.strftime('%Y-%m-%d %H:%M')

	time_step = timedelta(hours=time_step) # time step between mod files; will need to change the mod file naming and gsetup to do sub-daily files
	if not muted:
		print 'Time step:',time_step.total_seconds()/3600.0,'hours'
	
	total_time = end_date-start_date
	n_step = int(total_time.total_seconds()/time_step.total_seconds())
	local_date_list = np.array([start_date+i*time_step for i in range(n_step)])

	site_moved = False
	if 'time_spans' in site_dict[site_abbrv].keys(): # instruments with different locations for different time periods
		site_moved = True
		for time_span in site_dict[site]['time_spans']:
			if time_span[0]<=UTC_date<time_span[1]:
				site_lat = site_dict[site]['time_spans'][time_span]['lat']
				site_lon_360 = site_dict[site]['time_spans'][time_span]['lon']
				site_lon_180 = site_dict[site]['time_spans'][time_span]['lon_180']
				site_alt = site_dict[site]['time_spans'][time_span]['alt']
				break
	else:
		site_lat = site_dict[site_abbrv]['lat']
		site_lon_360 = site_dict[site_abbrv]['lon']
		site_lon_180 = site_dict[site_abbrv]['lon_180']
		site_alt = site_dict[site_abbrv]['alt']
	
	rmm = 28.964/18.02	# Ratio of Molecular Masses (Dry_Air/H2O)
	gravity_at_lat, earth_radius_at_lat = gravity(site_lat,site_alt/1000.0) # used in merra/fp mode

	if 'ncep' in mode:
		DATA = read_ncep(ncdf_path,start_date.year)
		varlist = ['T','H','QV']
	elif 'glob' in mode:
		varlist = ['T','QV','RH','H','EPV','O3','PHIS']
		surf_varlist = ['T2M','QV2M','PS','SLP','TROPPB','TROPPV','TROPPT','TROPT']	
		DATA,SURF_DATA = read_global(ncdf_path,mode,site_lat,site_lon_180,gravity_at_lat,varlist,surf_varlist,muted)
	elif 'merradap' in mode: # read all the data first, this could take a while ...	
		if not muted:
			print 'Reading MERRA2 data via opendap'
		varlist = ['T','QV','RH','H','EPV','O3','PHIS']
		surf_varlist = ['T2M','QV2M','PS','SLP','TROPPB','TROPPV','TROPPT','TROPT']	
		DATA,SURF_DATA = read_merradap(username,password,mode,site_lon_180,site_lat,gravity_at_lat,local_date,end_date,time_step,varlist,surf_varlist,muted)

	for local_date in local_date_list:

		if site_moved:
			for time_span in site_dict[site]['time_spans']:
				if time_span[0]<=local_date<time_span[1]:
					site_lat = site_dict[site]['time_spans'][time_span]['lat']
					site_lon_360 = site_dict[site]['time_spans'][time_span]['lon']
					site_lon_180 = site_dict[site]['time_spans'][time_span]['lon_180']
					site_alt = site_dict[site]['time_spans'][time_span]['alt']
					break
		
		astropy_date = Time(local_date)

		utc_offset = timedelta(hours=site_lon_180/15.0)
		UTC_date = local_date - utc_offset
		
		"""
		Interpolation time:
			julday0 is the fractional julian day number of the base time of the dataset: dataset times are in hours since base UTC time
			astropy_date.jd is the fractional julian day number of the current local day
			(astropy_date.jd-julday0)*24.0 = local hours since julday0
		"""
		site_tim = (astropy_date.jd-DATA['julday0'])*24.0 - utc_offset.total_seconds()/3600.0 # UTC hours since julday0
		# interpolate the data to the site's location and the desired time
		INTERP_DATA = trilinear_interp(DATA,varlist,site_lon_360,site_lat,site_tim)

		if 'ncep' in mode:
			INTERP_DATA['lev'] = np.copy(DATA['lev'])
			INTERP_DATA['TROPP'] = 0  # tropopause pressure not used with NCEP data
			INTERP_DATA['RH'] = 0 # won't be used, just to feed something to write_mod frh
		else: # merra/geos5
			if 'lev' not in varlist:
				INTERP_DATA['lev'] = np.copy(DATA['lev'])
			
			# get rid of fill values
			without_fill_IDs = np.where(INTERP_DATA['T']<1e10) # merra/geos fill value is 1e15
			for varname in list(set(varlist+['lev'])):
				try:
					INTERP_DATA[varname] = INTERP_DATA[varname][without_fill_IDs]
				except IndexError:
 					pass

		if ('merradap' in mode) or ('glob' in mode):
			site_tim = (astropy_date.jd-SURF_DATA['julday0'])*24.0 - utc_offset.total_seconds()/3600.0 # UTC hours since julday0
			INTERP_SURF_DATA = trilinear_interp(SURF_DATA,surf_varlist,site_lon_360,site_lat,site_tim)
			for varname in ['PS','SLP','TROPPB','TROPPV','TROPPT']:
				INTERP_SURF_DATA[varname] = INTERP_SURF_DATA[varname] / 100.0 # convert Pa to hPa

			if 'merradap72' in mode: # merra42 and ncep go from high pressure to low pressure, but merra 72 does the reverse
				# reverse merra72 profiles
				INTERP_DATA['lev'] = INTERP_DATA['PL'] / 100.0
				for varname in list(set(varlist+['lev'])):
					try:
						INTERP_DATA[varname] = INTERP_DATA[varname][::-1]
					except IndexError:
						pass

		INTERP_DATA['H2O_DMF'] = rmm*INTERP_DATA['QV']/(1-INTERP_DATA['QV']) # Convert specific humidity, a wet mass mixing ratio, to dry mole fraction
		INTERP_DATA['H'] = INTERP_DATA['H']/1000.0	# Convert m to km

		if ('merradap' in mode) or ('glob' in mode):
			INTERP_SURF_DATA['H2O_DMF'] = compute_h2o_dmf(INTERP_SURF_DATA['QV2M'],rmm)
			INTERP_DATA['PHIS'] = INTERP_DATA['PHIS']/1000.0
			# compute surface relative humidity
			INTERP_SURF_DATA['H2O_WMF'] = compute_h2o_wmf(INTERP_SURF_DATA['H2O_DMF']) # wet mole fraction of h2o
			INTERP_SURF_DATA['RH'] = compute_rh(INTERP_SURF_DATA['T2M'],INTERP_SURF_DATA['H2O_WMF'],INTERP_SURF_DATA['PS']) # Fractional relative humidity
			INTERP_SURF_DATA['MMW'] = compute_mmw(INTERP_SURF_DATA['H2O_WMF'])
			INTERP_SURF_DATA['H'] = INTERP_DATA['PHIS']

		## write the .mod file
		# directions for .mod file name
		if site_lat > 0:
			ns = 'N'
		else:
			ns = 'S'

		if site_lon_180>0:
			ew = 'E'
		else:
			ew = 'W'

		# use the local date for the name of the .mod file
		mod_name = mod_file_name(local_date,time_step,site_lat,site_lon_180,ew,ns,mod_path)
		mod_file_path = os.path.join(mod_path,mod_name)
		if not muted:
			print '\n',mod_name

		version = 'mod_maker_10.6   2017-04-11   GCT'
		if 'ncep' in mode:
			write_mod(mod_file_path,version,site_lat,data=INTERP_DATA,muted=muted)
		else:
			write_mod(mod_file_path,version,site_lat,data=INTERP_DATA,surf_data=INTERP_SURF_DATA,muted=muted)

		if ((UTC_date+time_step).year!=UTC_date.year) and ('ncep' in mode):
			DATA = read_ncep(ncdf_path,(UTC_date+time_step).year)
	
	if not muted:
		print len(local_date_list),'mod files written'

if __name__ == "__main__": # this is only executed when the code is used directly (e.g. not executed when imported from another python code)

	GGGPATH = os.environ['GGGPATH']
	ncdf_path = os.path.join(GGGPATH,'ncdf')

	arguments = parse_args()

	if 'mode' in arguments.keys(): # the fp / fpit mode works with concatenated files

		mod_maker(**arguments)

	else: # using fp-it daily files
		### New code that can generate slant paths and uses GEOS5-FP-IT daily files
		arguments['func_dict'] = equivalent_latitude_functions_new(os.path.join(arguments['geos_path'],'Np'),start_date=arguments['start_date'],end_date=arguments['end_date'],muted=arguments['muted'])
		mod_maker_new(start_date=arguments['start_date'],end_date=arguments['end_date'],func_dict=arguments['func_dict'],GEOS_path=arguments['geos_path'],muted=arguments['muted'])
