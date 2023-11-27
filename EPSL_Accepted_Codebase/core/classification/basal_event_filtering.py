import pandas as pd
import numpy as np
import scipy as sp
from obspy import read_events, Catalog
from scipy import spatial
from pyproj import Proj
from glob import glob
from tqdm import tqdm
import os

### MAP DATA SOURCES ###
ROOT = os.path.join('..','..','..')
NLLDIR = os.path.join(ROOT,'processed_data','NLLoc','PROJECT','IH_ice_LOC_v5')
GRDDIR = os.path.join(ROOT,'processed_data','seismic','HV','grids')

### LOAD ORIGIN CATALOG & GRIDS ###
df_EQ = pd.read_csv(os.path.join(NLLDIR,'origin_catalog.csv'),parse_dates=True,index_col='t0')

surfs = {}
for f_ in glob(os.path.join(GRDDIR,'*_Grid.npy')):
	k_ = os.path.split(f_)[-1].split('_Grid')[0]
	surfs.update({k_:np.load(f_)})

mE0,mN0 = surfs['mE'].mean(), surfs['mN'].mean()
### GENERATE cKDTrees ###
eps = 1e-3
z_dict = {}
tree_dict = {}
vmE = np.ravel(surfs['mE'] - mE0)
vmN = np.ravel(surfs['mN'] - mN0)
for k_ in surfs.keys():
	if k_ in ['H_med','H_p05','H_p95','H_man','H_gau_u']:
		Z_mod = (surfs['mZsurf'] - surfs[k_])*surfs['MASK']
		vmZi = np.ravel(Z_mod)
		z_dict.update({'Z_%s'%(k_.split('H_')[-1]):vmZi})
		M_xyz = np.c_[vmE,vmN,vmZi]
		i_tree = sp.spatial.cKDTree(M_xyz)
		tree_dict.update({'Z_%s'%(k_.split('H_')[-1]):i_tree})

	## Create perturbations for shallow bed estiamtes
	if k_ == 'H_p05':
		Z_mod = (surfs['mZsurf'] - surfs[k_])*surfs['MASK']
		vmZi = np.ravel(Z_mod) - eps
		z_dict.update({'Z_%s_pert'%(k_.split('H_')[-1]):vmZi})
		M_xyz = np.c_[vmE,vmN,vmZi]
		i_tree = sp.spatial.cKDTree(M_xyz)
		tree_dict.update({'Z_%s_pert'%(k_.split('H_')[-1]):i_tree})

		Z_mod = (surfs['mZsurf'] - surfs['H_gau_u'] - 1.96*surfs[k_] - eps)*surfs['MASK']
		vmZi = np.ravel(Z_mod)
		z_dict.update({'Z_gau_um2o_pert':vmZi})
		M_xyz = np.c_[vmE,vmN,vmZi]
		i_tree = sp.spatial.cKDTree(M_xyz)
		tree_dict.update({'Z_gau_um2o_pert':i_tree})

	## Create perturbations for deep bed estimates
	if k_ == 'H_p95':
		Z_mod = (surfs['mZsurf'] - surfs[k_])*surfs['MASK']
		vmZi = np.ravel(Z_mod) + eps
		z_dict.update({'Z_%s_pert'%(k_.split('H_')[-1]):vmZi})
		M_xyz = np.c_[vmE,vmN,vmZi]
		i_tree = sp.spatial.cKDTree(M_xyz)
		tree_dict.update({'Z_%s_pert'%(k_.split('H_')[-1]):i_tree})

		Z_mod = (surfs['mZsurf'] - surfs['H_gau_u'] + 1.96*surfs[k_] + eps)*surfs['MASK']
		vmZi = np.ravel(Z_mod)
		z_dict.update({'Z_gau_up2o_pert':vmZi})
		M_xyz = np.c_[vmE,vmN,vmZi]
		i_tree = sp.spatial.cKDTree(M_xyz)
		tree_dict.update({'Z_gau_up2o_pert':i_tree})



	if k_ == 'H_gau_o':
		Z_mod = (surfs['mZsurf'] - surfs['H_gau_u'] + 1.96*surfs[k_])*surfs['MASK']
		vmZi = np.ravel(Z_mod)
		z_dict.update({'Z_gau_up2o':vmZi})
		M_xyz = np.c_[vmE,vmN,vmZi]
		i_tree = sp.spatial.cKDTree(M_xyz)
		tree_dict.update({'Z_gau_up2o':i_tree})
		Z_mod = (surfs['mZsurf'] - surfs['H_gau_u'] - 1.96*surfs[k_])*surfs['MASK']
		vmZi = np.ravel(Z_mod)
		z_dict.update({'Z_gau_um2o':vmZi})
		M_xyz = np.c_[vmE,vmN,vmZi]
		i_tree = sp.spatial.cKDTree(M_xyz)
		tree_dict.update({'Z_gau_um2o':i_tree})


### SEARCH DISTANCES ACROSS TREES ###
search_dict = {}
df_dist = pd.DataFrame()
O_xyz = np.c_[df_EQ['mE'].values - mE0,\
			  df_EQ['mN'].values - mN0,\
			  df_EQ['mZ'].values]

for k_ in tree_dict:
	print('Searching %s'%(k_))
	i_tree = tree_dict[k_]
	i_search = i_tree.query(O_xyz)
	iS = pd.Series(i_search[0],name=k_)
	search_dict.update({k_:i_search})
	df_dist = pd.concat([df_dist,iS],axis=1)


### 
# Get index for Maximum Likelihood Solutions falling inside the 0.05 < p < 0.95
DN_ind = ((df_dist['Z_p95'] <= df_dist['Z_p95_pert'])).values

IN_ind = ((df_dist['Z_p95'] > df_dist['Z_p95_pert']) &\
		     (df_dist['Z_p05'] > df_dist['Z_p05_pert'])).values

UP_ind = ((df_dist['Z_p05'] <= df_dist['Z_p05_pert'])).values
### For those falling outside this envelope, calculate vertical Z-scores ###
NN_med_elev = z_dict['Z_med'][search_dict['Z_med'][1]]
NN_mea_elev = z_dict['Z_gau_u'][search_dict['Z_gau_u'][1]]
### Calculate Approximation of 1 sigma ###
NN_qrg = (NN_med_elev - z_dict['Z_p05'][search_dict['Z_med'][1]])/1.96
NN_sig = (NN_mea_elev - z_dict['Z_gau_um2o'][search_dict['Z_gau_u'][1]])/2.

Zs_Q = (df_EQ['mZ'].values - NN_med_elev)/np.sqrt(NN_qrg**2. + df_EQ['deperr'].values**2.)
Zs_G = (df_EQ['mZ'].values - NN_mea_elev)/np.sqrt(NN_sig**2. + df_EQ['deperr'].values**2.)

### Apply maximum error & Z-score filter ###
errmax = 30.
IND_Q = (np.abs(Zs_Q) <= 2) & (df_EQ[['herrmax','herrmin','deperr']].max(axis=1).values <= errmax)
IND_G = (np.abs(Zs_G) <= 2) & (df_EQ[['herrmax','herrmin','deperr']].max(axis=1).values <= errmax)

### Conduct Filtering ###
iS_Q_Bool = pd.Series(IND_Q,index=df_EQ.index,name='Q_filt')
iS_G_Bool = pd.Series(IND_G,index=df_EQ.index,name='G_filt')

df_EQc = pd.concat([df_EQ,iS_Q_Bool,iS_G_Bool],axis=1,ignore_index=False)

df_EQc.to_csv(os.path.join(ROOT,'processed_data','seismic','basal_catalog','well_located_%dm_basal_catalog_merr.csv'%(errmax)),\
			  index=True,header=True)



