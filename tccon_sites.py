from datetime import datetime

"""
site_dict is a dictionary mapping TCCON site abbreviations to their lat-lon-alt data, and full names

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

def tccon_site_info(site_dict):
	"""
	Takes the site_dict dictionary and adds longitudes in the [-180,180] range
	"""

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