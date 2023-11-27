import pandas as pd
import numpy as np
from obspy import read_events, Catalog
from pyproj import Proj
from glob import glob
from tqdm import tqdm
import os

def cat2df(cat):
	"""
	Convert an ObsPy Catalog into a Pandas DataFrame with specific origin data
	"""
	t0 = []; lat = []; lon = []; dep = []; 
	deperr = []; herrmax = []; herrmin = []; 
	nphz = []; nsta = []; src = []
	for i_ in range(len(cat)):
		# Pull Origin
		orig = cat[i_]['origins'][0]
		# Append data
		t0.append(pd.Timestamp(orig['time'].datetime))
		lat.append(orig['latitude'])
		lon.append(orig['longitude'])
		dep.append(orig['depth'])
		deperr.append(orig['depth_errors']['uncertainty'])
		herrmax.append(orig['origin_uncertainty']['max_horizontal_uncertainty'])
		herrmin.append(orig['origin_uncertainty']['min_horizontal_uncertainty'])
		nphz.append(orig['quality']['used_phase_count'])
		nsta.append(orig['quality']['used_station_count'])
		src.append(orig['creation_info']['author'])

	_df_ = pd.DataFrame({'t0':t0,'lat':lat,'lon':lon,'dep':dep,\
					   'deperr':deperr,'herrmax':herrmax,'herrmin':herrmin,\
					   'nphz':nphz,'nsta':nsta,'src':src})
	return _df_



### MAP DATA SOURCES ###
ROOT = os.path.join('..','..','..')
NLLDIR = os.path.join(ROOT,'processed_data','NLLoc','PROJECT','IH_ice_LOC_v5')
flist = []
for l_ in ['loc_k0_merr','loc_k1_merr','loc_k3_merr','loc_k-1_merr']:
	iflist = glob(os.path.join(NLLDIR,l_,'grp*','sggs.2019*.hyp'))
	flist += iflist

### Build Catalog ### - THIS TAKES AWHILE
cat = Catalog()
for f_ in tqdm(flist):
	cat += read_events(f_)

### Write Catalog 
print('Writing catalog')
cat.write(os.path.join(NLLDIR,'catalog.xml'),format='QUAKEML')
print('Catalog written as QuakeML')
### Convert into 
print('Extracting Origin Features for CSV')
df = cat2df(cat)

### Convert LLH to ENZ Rotation ###
print('Processing Locations into UTM Zone 11U Coordinates')
myproj = Proj("+proj=utm +zone=11U, +south +ellipse=WGS84 +datum=WGS84 +units=m +no_defs")
mE,mN = myproj(df['lon'].values,df['lat'].values)
mZ = df['dep'].values*-1

df_UTM = pd.DataFrame({'mE':mE,'mN':mN,'mZ':mZ,'hypfile':flist})

df_out = pd.concat([df,df_UTM],axis=1,ignore_index=False)
print("Writing Origin DataFrame to CSV")
df_out.to_csv(os.path.join(NLLDIR,'origin_catalog.csv'),header=True,index=False)

