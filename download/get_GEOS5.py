#!~/anaconda2/bin/python
 # -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
import sys
import urllib
import requests
import shutil
import xarray
import netrc
from pydap.cas.urs import setup_session
from subprocess import Popen,PIPE,CalledProcessError

####################
# Code Description #
####################
"""
Functions to create list of URLs and/or download them
"""

#############
# Functions #
#############

def execute(cmd,cwd=os.getcwd()):
	'''
	function to execute a unix command and print the output as it is produced
	'''
	popen = Popen(cmd, stdout=PIPE, universal_newlines=True,cwd=cwd)
	for stdout_line in iter(popen.stdout.readline, ""):
	    yield stdout_line 
	popen.stdout.close()
	return_code = popen.wait()
	if return_code:
	    raise CalledProcessError(return_code, cmd)

def URLlist_FP(start,end,timestep=timedelta(hours=3),outpath='',surf=False):
	"""
	GEOS5-FP data has one global file every 3 hours (from 00:00 to 21:00 UTC each day)
	start: datetime object, start of the desired date range
	end: datetime object, end of the desired date range
	timestep: use the model time resolution to get all files, or a multiple of it to get less files
	outpath: full path to the file in which the list of URLs will be written
	"""
	if surf:
		fmt = "https://portal.nccs.nasa.gov/datashare/gmao_ops/pub/fp/das/Y{}/M{:0>2}/D{:0>2}/GEOS.fp.asm.inst3_2d_asm_Nx.{}_{:0>2}00.V01.nc4\n"
	else:
		fmt = "https://portal.nccs.nasa.gov/datashare/gmao_ops/pub/fp/das/Y{}/M{:0>2}/D{:0>2}/GEOS.fp.asm.inst3_3d_asm_Np.{}_{:0>2}00.V01.nc4\n"
	
	if outpath=='': # if no specified full path to make the file, just write a file in the current directory 
		outpath = 'getFP.dat'

	print('Writting URL list in:',outpath)

	curdate = start
	with open(outpath,'w') as f:
		while curdate<end:
			f.write(fmt.format(curdate.year,curdate.month,curdate.day,datetime.strftime(curdate,'%Y%m%d'),curdate.hour))
			curdate += timestep

def URLlist_FPIT(start,end,timestep=timedelta(hours=3),outpath='',surf=False):
	"""
	GEOS5-FP-IT data has one global file every 3 hours (from 00:00 to 21:00 UTC each day)
	start: datetime object, start of the desired date range
	end: datetime object, end of the desired date range
	timestep: use the model time resolution to get all files, or a multiple of it to get less files
	outpath: full path to the file in which the list of URLs will be written
	"""
	if surf:
		fmt = "http://goldsfs1.gesdisc.eosdis.nasa.gov/data/GEOS5/DFPITI3NXASM.5.12.4/{}/{:0>3}/.hidden/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.{}_{:0>2}00.V01.nc4\n"
	else:
		fmt = "http://goldsfs1.gesdisc.eosdis.nasa.gov/data/GEOS5/DFPITI3NPASM.5.12.4/{}/{:0>3}/.hidden/GEOS.fpit.asm.inst3_3d_asm_Np.GEOS5124.{}_{:0>2}00.V01.nc4\n"

	if outpath=='': # if no specified full path to make the file, just write a file in the current directory 
		outpath = 'getFPIT.dat'

	print('Writting URL list in:',outpath)

	curdate = start
	with open(outpath,'w') as f:
		while curdate<end:
			f.write(fmt.format(curdate.year,curdate.timetuple().tm_yday,datetime.strftime(curdate,'%Y%m%d'),curdate.hour))
			curdate += timestep

########
# Main #
########

if __name__=="__main__":
	argu = sys.argv

	func_dict = {'FP':URLlist_FP,'FPIT':URLlist_FPIT}

	argu = sys.argv

	start = datetime.strptime(argu[1].split('-')[0],'%Y%m%d') # YYYYMMDD
	end = datetime.strptime(argu[1].split('-')[1],'%Y%m%d') # YYYYMMDD

	for arg in argu:
		if 'mode=' in arg:
			mode = arg.split('=')[1]
		if 'path=' in arg:
			path = arg.split('=')[1]

	# surface data
	outpath = os.path.join(path,'Nx')
	if not os.path.exists(outpath):
		print('Creating',outpath)
		os.makedirs(outpath)
	func_dict[mode](start,end,surf=True,outpath=os.path.join(outpath,'getFPIT.dat'))

	for line in execute('wget -i getFPIT.dat'.split(),cwd=outpath):
		print(line, end="")

	# profile data
	outpath = os.path.join(path,'Np')
	if not os.path.exists(outpath):
		print('Creating',outpath)
		os.makedirs(outpath)
	func_dict[mode](start,end,surf=False,outpath=os.path.join(outpath,'getFPIT.dat'))

	for line in execute('wget -i getFPIT.dat'.split(),cwd=outpath):
		print(line, end="")
