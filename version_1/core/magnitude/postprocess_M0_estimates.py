"""
postprocess_M0_estimates.py

This script does some light distillation of M_0 estimates from 3-C data,
selecting preferred-channel estimates of M_0 and SNR for P- and SH- wave
estimates and appends station/event geometric data

It then calculates and appends a series of normalized weights 
for use in testing different averaging approaches for "event-average M_0"
conducted in a later processing script

:AUTH: Nathan T. Stevens
:EMAIL: ntstevens@wisc.edu

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Define root directory
ROOT = os.path.join('..','..','data')
# Map concatenated M0 file
M0_DAT = os.path.join(ROOT,'M0_ests_cat.csv')
# Map SITE file
SITE_DAT = os.path.join(ROOT,'tables','site.csv')
# Map ORIGIN file'
ORIG_DAT = os.path.join(ROOT,'tables','well_located_30m_basal_catalog_merr.csv')


## LOAD DATA ##
df_M0 = pd.read_csv(M0_DAT,parse_dates=['t0'])
df_OR = pd.read_csv(ORIG_DAT,parse_dates=['t0'])
df_SITE = pd.read_csv(SITE_DAT)


## SET FILTERS ##
P_chan = '_L'  	# [-] Channel designation for M0^P estimates
S_chan = '_T'	# [-] Channel designation for M0^S estimates
SNR_min = 5	 	# [dB] Minimum signal to noise ratio associate with an M0 estimate 
				#  	   to consider in further calculations
SNR_max = None
AGE_max = 3 	# [days] Maximum age of a station install before discounting observations
min_samps = 5 	# [-] Minimum number of samples for doing full statistics

## OBSOLITED FILTER PARAMETERS - this gets done in a later script
# SNR_min = 5 	# [dB] minimum acceptable SNR
# min_est = 8		# [-] minimum number of combined M0^P and M0^S estimates to use for further analysis
# max_age = 3 	# [days] maximum age of a deployment to use estimates
# age_wts = [1.,2./3.,1./3.] # [-] Age-weighting components



def calculate_SNR_AGE_weights(M0v,SNRv,AGEv,SNR_min=3,SNR_ref=None,AGE_max=3):
	"""
	Calculate unit scalar weights for a population of M_0 estimates using each estimate's
	SNR and the age of the seismic deployment

	:: INPUTS ::
	:param M0v: [array-like] set of M_0 estimates
	:param SNRv: [array-like] set of associated SNR values for M_0 estimates
	:param AGEv: [array-like] set of associated deployment ages for M_0 estimates
	:param SNR_min: [float] minimum SNR value corresponding to a weight of 0
	:param SNR_ref: [float or None] SNR value corresponding to a weight of 1
	:param AGE_max: [int] maximum age (in days) at which point age weighting becomes wt=0
						 with weights being calculated as:
						 1 - (AGEv[i_]//1)/AGE_max
	"""
	IDX = (SNRv >= SNR_min) & (AGEv <= AGE_max)
	if SNR_ref is not None:
		SNR_max = SNR_ref
	else:
		SNR_max = np.nanmax(SNRv)
	
	wtSNR, wtAGE = [],[]
	for i_ in range(len(M0v)):
		iA = AGEv[i_]
		iS = SNRv[i_]
		# Get inverse, stepped age weighting factors
		if iA > AGE_max:
			wtAGE.append(0)
		else:
			wtAGE.append(1. - (AGEv[i_]//1)/np.ceil(AGE_max))
		# Get linear-scaled SNR weighting factors
		if iS < SNR_min:
			wtSNR.append(0)
		else:
			wtSNR.append((SNRv[i_] - SNR_min) / (SNR_max - SNR_min))

	return np.array(wtSNR),np.array(wtAGE)

def weighted_nanmean(values,weights):
	"""
	Calculate the weighted median of a set of values, ignoring 
	value-weight pairs that contain NaN
	"""
	ind = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
	vv = values[ind]
	ww = weights[ind]
	summed_weight = np.nansum(ww)
	nmean = np.nansum(vv*ww)/summed_weight
	return nmean

def weighted_nanstd(values,weights):
	"""
	Calculate the weighted standard deviation of a set of values,
	ignoring value-weight pairs that contain NaN
	"""
	ind = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
	vv = values[ind]
	ww = weights[ind]
	Np = len(vv)
	part1 = np.nansum(ww*(vv - weighted_nanmean(vv,ww))**2)
	part2 = (Np - 1.)*np.nansum(ww)/Np
	if part1/part2 < 0 or np.isnan(part1/part2):
		breakpoint()

	sdw = np.sqrt(part1/part2)
	return sdw

def calculate_scalar_weights(df,fields=['SNR_L','SNR_T','age','dd m'],mins=[3,3,0,0],maxs=[np.inf,np.inf,3,2000],methods=['cont','cont','step','inv2']):
	
	# Create a dictionary holder for weight fields
	h_dict = {}
	# Iterate across fields to use for weight calculations
	for i_,f_ in enumerate(fields):
		# Create new column name for output
		wkey = 'wt_'+f_
		# Get reference range for each vector
		wmin = mins[i_]
		wmax = maxs[i_]
		# Get weight calculation method
		wmeth = method[i_]
		# Create list holder for this given weight series
		wt = []
		# Iterate across each data instance 
		for j_ in range(len(df)):
			# Grab a single value
			val = df[f_].values[j_]
			# If the value is within the reference range
			if wmin <= val <= wmax:
				# Do stepped calculation if specified
				if wmeth == 'step':
					wt.append((1. - (val//1)/np.ceil(wmax)))
				# Do continuous (linear) calculation if specified
				elif wmeth == 'cont':
					wt.append((val - wmin)/(wmax - wmin))
				# Do inverse squared calculation if specified
				elif wmeth == 'inv2':
					wt.append(((val - wmin)/(wmax - wmin))**-2)
			# Otherwise, weight is zero
			else:
				wt.append(0)
		# Append weight vector to dictionary with new column name
		h_dict.update({wkey:wt})
	# Create weight DataFrame
	df_OUT = pd.DataFrame(h_dict,index=df.index)
	# Append to input dataframe keying off index
	df_OUT = pd.concat([df,df_OUT],axis=1,ignore_index=False)

	return df_OUT

# Subset phase estimates & filter by SNR_min & maximum permissable age
df_P = df_M0[(df_M0['phase']=='P')&(df_M0['SNR'+P_chan]>= SNR_min)&(df_M0['age']<=AGE_max)]
df_S = df_M0[(df_M0['phase']=='S')&(df_M0['SNR'+S_chan]>= SNR_min)&(df_M0['age']<=AGE_max)]

df_F = pd.concat([df_P,df_S],axis=0,ignore_index=False)

cols = ['EVID','t0','nP','nS','mE','mN','mZ','hemin','hemax','verr',\
		'mean SNR','mean R','mean age',\
		'mean','std','median','Q025','Q975','Q1','Q3','min','max',\
		'Pmean','Pstd','Pmedian','PQ025','PQ975','PQ1','PQ3','Pmin','Pmax',\
		'Smean','Sstd','Smedian','SQ025','SQ975','SQ1','SQ3','Smin','Smax',\
		'SNR_wt_mean','SNR_wt_std','AGE_wt_mean','AGE_wt_std','SA_wt_mean','SA_wt_std',\
		'P_SNR_wt_mean','P_SNR_wt_std','P_AGE_wt_mean','P_AGE_wt_std','P_SA_wt_mean','P_SA_wt_std',\
		'S_SNR_wt_mean','S_SNR_wt_std','S_AGE_wt_mean','S_AGE_wt_std','S_SA_wt_mean','S_SA_wt_std']

# Iterate across events and append location data
holder = []
# Key off origin times in case EVID in ORIG table gets corrupted
for t_ in tqdm(df_F['t0'].unique()):
	# Subset data
	idf_OR = df_OR[df_OR['t0']==t_]
	idf_P = df_P[df_P['t0']==t_]
	idf_S = df_S[df_S['t0']==t_]
	idf_F = pd.concat([idf_P,idf_S],axis=0,ignore_index=False)
	# Concatenate measures for calculating \\bar{M_0} for both methods
	# Individual M_0 estimates
	iM0v = np.abs(np.r_[idf_P['M0'+P_chan].values,idf_S['M0'+S_chan].values])
	iM0vP = np.abs(idf_P['M0'+P_chan].values)
	iM0vS = np.abs(idf_S['M0'+S_chan].values)
	# Individual SNR estimates
	iSNRv = np.r_[idf_P['SNR'+P_chan].values,idf_S['SNR'+S_chan].values]
	iSNRvP = idf_P['SNR'+P_chan].values
	iSNRvS = idf_S['SNR'+S_chan].values
	# Individual age values
	iAGEv = np.r_[idf_P['age'].values,idf_S['age'].values]
	iAGEvP = idf_P['age'].values
	iAGEvS = idf_S['age'].values

	# Compose output data
	line = []
	# Append EVID and source timing
	line += [idf_OR.index[0],t_]
	# Append data counts
	line += [len(idf_P),len(idf_S)]
	# Append source location information
	line += list(idf_OR[['mE','mN','mZ','herrmin','herrmax','deperr']].values[0])
	# Append average SNR
	line += [np.nanmean(iSNRv)]
	# Append average distance and data age
	line += list(idf_F[['dd m','age']].mean().values)
	# Iterate across estimate method slices - All, P-wave, SH-wave
	for D_ in [iM0v,idf_P['M0'+P_chan].values,idf_S['M0'+S_chan].values]:
		# Ensure some data is present (could increase this later)
		if len(D_) > 0:
			# Impose absolute value on M0 values
			D_ = np.abs(D_)
			if len(D_) > min_samps:
				# Run statistics calculations
				line += [np.nanmean(D_),np.nanstd(D_),np.nanmedian(D_),\
						 np.nanquantile(D_,0.025),np.nanquantile(D_,0.975),\
						 np.nanquantile(D_,0.25),np.nanquantile(D_,0.75),\
						 np.nanmin(D_),np.nanmax(D_)]
			else:
				line += [np.nan,np.nan,np.nanmedian(D_),\
						 np.nan,np.nan,np.nan,np.nan,\
						 np.nanmin(D_),np.nanmax(D_)]
		else:
			line += [np.nan]*9

	## CONDUCT SNR-/AGE-WEIGHTED STATISTICS ##
	# All-data
	if len(iM0v) >= min_samps:
		all_wt_SNR, all_wt_AGE = calculate_SNR_AGE_weights(iM0v,iSNRv,iAGEv,SNR_min=SNR_min,SNR_ref=SNR_max,AGE_max=AGE_max)
		line += [weighted_nanmean(iM0v,all_wt_SNR),weighted_nanstd(iM0v,all_wt_SNR),\
				 weighted_nanmean(iM0v,all_wt_AGE),weighted_nanstd(iM0v,all_wt_AGE),\
				 weighted_nanmean(iM0v,all_wt_SNR*all_wt_AGE),weighted_nanstd(iM0v,all_wt_SNR*all_wt_AGE)]
	else:
		line += [np.nan]*6
	# P-phase data
	if len(iM0vP) >= min_samps:
		P_wt_SNR, P_wt_AGE = calculate_SNR_AGE_weights(iM0vP,iSNRvP,iAGEvP,SNR_min=SNR_min,SNR_ref=SNR_max,AGE_max=AGE_max)
		line += [weighted_nanmean(iM0vP,P_wt_SNR),weighted_nanstd(iM0vP,P_wt_SNR),\
				 weighted_nanmean(iM0vP,P_wt_AGE),weighted_nanstd(iM0vP,P_wt_AGE),\
				 weighted_nanmean(iM0vP,P_wt_SNR*P_wt_AGE),weighted_nanstd(iM0vP,P_wt_SNR*P_wt_AGE)]
	else:
		line += [np.nan]*6
	# S-phase data
	if len(iM0vS) >= min_samps:
		S_wt_SNR, S_wt_AGE = calculate_SNR_AGE_weights(iM0vS,iSNRvS,iAGEvS,SNR_min=SNR_min,SNR_ref=SNR_max,AGE_max=AGE_max)
		line += [weighted_nanmean(iM0vS,S_wt_SNR),weighted_nanstd(iM0vS,S_wt_SNR),\
				 weighted_nanmean(iM0vS,S_wt_AGE),weighted_nanstd(iM0vS,S_wt_AGE),\
				 weighted_nanmean(iM0vS,S_wt_SNR*S_wt_AGE),weighted_nanstd(iM0vS,S_wt_SNR*S_wt_AGE)]
	else:
		line += [np.nan]*6
	holder.append(line)

df_SUM = pd.DataFrame(holder,columns=cols)

OUT_NAME = 'M0_stats_SNR_ge%d_ND_ge%d_MA_le%ddays_SA_wt.csv'%(SNR_min,min_samps,AGE_max)
OUT_FPATH = os.path.join(ROOT,'M0_results')
try:
	os.makedirs(OUT_FPATH)
except:
	pass

df_SUM.to_csv(os.path.join(OUT_FPATH,OUT_NAME),header=True,index=False)




