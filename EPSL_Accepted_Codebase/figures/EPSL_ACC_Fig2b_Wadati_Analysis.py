"""
:module: EPSL_ACC_FigS5_Wadati_Analysis.py
:purpose: produce a Wadati Analysis style plot from manual pick data - Fig S5 in Stevens and others (accepted; EPSL)
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu

"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.odr as odr
from scipy.optimize import curve_fit
from obspy import UTCDateTime, read_events, Catalog
from glob import glob
from tqdm import tqdm


ROOT = os.path.join('..')
GSTR = os.path.join(ROOT,'inputs','data_files','seismic','manual_picks','test_5node_v7','sggs.2019*.hyp')
ODIR = os.path.join(ROOT,'outputs')
DPI = 300
FMT = 'PNG'
issave = True

### PROCESSING FUNCTIONS ###

def cat2df(cat):
	"""
	Extract relevant phase-arrival and event time data from an
	ObsPy Catalog and write to a pandas.DataFrame for ease-of-use
	filtering & association workflows

	:: INPUT ::
	:param cat: obspy.Catalog (with events, origins, and picks)

	:: OUTPUT ::
	:return df_out: pandas.DataFrame containing event-phase data with the following columns:
				'event index' - index of event from `cat` [int]
				't0 ref' - event origin time [pandas.Timestamp]
				'sta' - station name for pick [str]
				'chan' - channel name for pick [str]
				't' - arrival time of pick [pandas.Timestamp]
				'phz' - phase name [str]
				'sigma_t' - phase arrival uncertainty [float]
				'time_residual' - data-model residual

	"""
	holder = []
	cols = ['event index','t0 ref','sta','chan','t','phz','sigma_t','time_residual']
	for i_,e_ in tqdm(enumerate(cat.events)):
		t0 = pd.Timestamp(str(e_.origins[0].time))
		for p_ in e_.picks:
			pid = p_.resource_id
			for a_ in e_.origins[0].arrivals:
				if pid == a_.pick_id:
					arr = a_
			line = [i_,t0,\
					p_.waveform_id['station_code'],\
					p_.waveform_id['channel_code'],\
					pd.Timestamp(str(p_.time)),\
					p_.phase_hint,\
					p_.time_errors['uncertainty'],\
					arr.time_residual]
			holder.append(line)
	df_out = pd.DataFrame(holder,columns=cols)

	return df_out

def find_wadati_pairs(df):
	"""
	Find event-station sets with both P and S wave arrival time picks using
	a pandas.DataFrame output from cat2df() and calculate the following data
	points for Wadati analysis (tS - tP, tP - t0)

	:: INPUT ::
	:param df: pandas.DataFrame as described above

	:: OUTPUT ::
	:param df_w: pandas.DataFrame with the following columns:
			'event index' - index inherited from the initial catalog conversion [int]
			'sta' - station name [str]
			'P index' - index from `df` for P-wave arrival time data [int]
			'S index' - index from `df` for S-wave arrival time data [int]
			'tS-tP' - differential arrival times [float]
			'tP-t0' - initial estimate of P-wave travel time [float]
			'sigSP' - cumulative uncertainty on tS-tP
			'sigP0' - cumulative uncertainty on tP - t0
	"""
	holder = []
	cols = ['event index','sta','P index','S index','tS-tP','tP-t0','sigSP','sigP0','resSP','resP0']
	for eid in tqdm(df['event index'].unique()):
		df_P = df[(df['phz']=='P')&(df['event index']==eid)]
		for i_ in range(len(df_P)):
			S_P = df_P.iloc[i_,:]
			df_S = df[(df['phz']=='S')&(df['sta']==S_P['sta'])&(df['event index']==eid)]
			if len(df_S) == 1:
				sta = S_P['sta']
				Pind = S_P.name
				Sind = df_S.index[0]
				tStP = (df_S['t'] - S_P['t'])[Sind].total_seconds()
				sigSP = (S_P['sigma_t']**2 + df_S['sigma_t'].values[0]**2)**0.5
				sigP0 = S_P['sigma_t']
				resSP = (S_P['time_residual']**2 + df_S['time_residual'].values[0]**2)**0.5
				resP0 = S_P['time_residual']
				tPt0 = (S_P['t'] - S_P['t0 ref']).total_seconds()
				line = [eid,sta,Pind,Sind,tStP,tPt0,sigSP,sigP0,resSP,resP0]
				holder.append(line)

	df_w = pd.DataFrame(holder,columns=cols)
	return df_w

def lin_fun_odr(beta,xx):
	return (beta[0]*xx) + beta[1]

def slope_fun_odr(beta,xx):
	return beta*xx

def lin_fun(xx,mm,bb):
	return (xx*mm) + bb

def slope_fun(xx,mm):
	return xx*mm

def wadati_odr(tStP,tPt0,sigSP,sigP0,beta0=None,fun=lin_fun_odr,fit_type=0):
	"""
	Estimate function parameters from observations
	of <tP-t0,tS-tP> observations and their uncertainties
	using an Orthogonal Distance Regression fitting.

	:: INPUTS ::
	:param tStP: differential tS - tP travel times [array-like]
	:param tPt0: P-wave travel time [array-like]
	:param sigSP: uncertainties on tS - tP data [array-like]
	:param sigP0: uncertainties on tP - t0 data [array-like]
	:param beta0: initial guesses on parameter values [array-like]
	:param fun: Function defining global matrix for the general inverse formatted
				for use with scipy.odr.ODR (i.e., arguments of (beta,xx))
	"""
	# Write data and standard errors for fitting to data object
	data = odr.RealData(tPt0,tStP,sigP0,sigSP)
	# Define model
	model = odr.Model(fun)
	# Compose Orthogonal Distance Regression 
	_odr_ = odr.ODR(data,model,beta0=beta0)
	# Set solution type
	_odr_.set_job(fit_type=fit_type)

	# Run regression
	output = _odr_.run()

	return output

def wadati_PP(output):
	"""
	Convert slope fitting to Vp/Vs ratio and uncertainties
	ASSUMES USE OF lin_fun() FOR GLOBAL MATRIX GENERATION
	:: INPUT ::
	:param output: output report from scipy.odr.ODR

	:: OUTUPT ::
	:return VpVs: Expected value of Vp/Vs from inversion results
	:return sigVpVs: Standard error for VpVs from inversion results
	"""
	m = output.beta[0]
	sm = output.cov_beta[0]**0.5
	VpVs = m + 1.
	return VpVs,sm



### DATA PROCESSING SECTION ###
flist = glob(GSTR)
flist.sort()
cat = Catalog()

for f_ in tqdm(flist):
	cat += read_events(f_)

df = cat2df(cat)
df_w = find_wadati_pairs(df)
output = wadati_odr(df_w['tS-tP'].values,df_w['tP-t0'].values,\
					df_w['resSP'].values,df_w['resP0'].values,fun=slope_fun_odr,beta0=[0.95])
popt,pcov = curve_fit(slope_fun,df_w['tP-t0'].values,df_w['tS-tP'].values,p0=[0.95],method='trf',bounds=(0.75,1))
VpVs_odr,sm_odr = wadati_PP(output)

manual_fit = (.079-.0426)/(.0822-.0451)
VpVs_man = manual_fit + 1

### PLOTTING SECTION ###
plt.figure(figsize=(4.8,4.8))
plt.plot(df_w['tP-t0'].values*1e3,df_w['tS-tP'].values*1e3,'k.',alpha=0.1)
xv = np.linspace(df_w['tP-t0'].min(),df_w['tP-t0'].max(),100)*1e3
plt.plot(xv,slope_fun(xv,*popt),'r--')

plt.fill_between(xv,slope_fun(xv,popt[0] - 2*pcov[0]**0.5),\
					slope_fun(xv,popt[0] + 2*pcov[0]**0.5),\
				 	alpha=0.3,color='r')
plt.xlabel('$t_P - t_0$ [msec]')
plt.ylabel('$t_S - t_P$ [msec]')
plt.text(xv[0]+10,slope_fun(xv[-1],popt[0])-10,'$V_P / V_S$: %.3f $\pm$ %.3f'%(popt[0]+1,pcov**0.5))


if issave:
	I_OFILE = 'SGEQ_Fig2B_Wadati_Analysis_%ddpi.%s'%(DPI,FMT.lower())
	plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)
	print('Figure saved: %s'%(os.path.join(ODIR,I_OFILE)))

plt.show()