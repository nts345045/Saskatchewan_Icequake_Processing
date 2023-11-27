"""
EPSL_ACC_FigS5_SmartSolo_LOG_Ages.py

This script does basic processing of parsed SoloLite.LOG files and
survey files to demonstrate relative changes in instrument orientation

This support selection of age-dependent filtering/weighting parameters
for M0 estimation presented in the main text.

This particular analysis was added in response to reviewer comments on
the initial submission of "Icequake insights on transient glacier slip
mechanics" to Earth and Planetary Science Letters

:CORRESPONDING AUTHOR: 	Nathan T. Stevens
:EMAIL: 				ntstevens@wisc.edu
:REVIEWERS: 			Thomas Hudson
						Ugo Nanni
:EDITOR: 				Jean-Philippe Avouac
"""
import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

issave = True
FMT = 'PNG'
DPI = 300


## TIME ZONE CORRECTION ##
TZC = pd.Timedelta(6,unit='hour')
# --- USE --- #
# Add this to SURVEY times to convert to UTC that LOG data use
# Subtract this from LOG times to convert to local times used in SURVEY 

## LOCAL TIME TO SOLAR NOON CORRECTION ##
SNC = pd.Timedelta(1,unit='hour')
# --- USE --- #
# Subtract this from LOCAL times to convert to SOLAR TIMES

# Project root directory
ROOT = os.path.join('..','..')
# Directory containing parsed logs
LOGS = os.path.join(ROOT,'data','LOG')
# Station orientation surveys
SURV = os.path.join(ROOT,'data','tables','Sorted_Passive_Servicing.csv')
# Figure output directory
ODIR = os.path.join(ROOT,'outputs')


# Get most instruments' parsed LOG files
flist = glob(os.path.join(LOGS,'Parsed_DigiSolo_LOGSS????.csv'))
# Get parsed LOG file from 2nd data pull of DAS 4530001427 (includes R108 and R2108)
flist += glob(os.path.join(LOGS,'Parsed_DigiSolo_*L2.csv'))
flist.sort()

# Make Quick-reference series for *.csv files
SNlist = [f_.split('_')[-1].split('.')[0][5:9] for f_ in flist]

S_SNF = pd.Series(flist,index=SNlist,name='file')



# Import SURVEY file
df_SURV = pd.read_csv(SURV,parse_dates=['DateTime Mean'],index_col='DateTime Mean')
# Subset SURVEY file to Smart Solos
df_SURV = df_SURV[df_SURV['sta'].str.contains('R')]

# Initialize a summative dictionary for all results
D_STATS = dict()

# Iterate across unique station combinations
for SN_ in S_SNF.index:
	# Load relevant LOG file
	df_LOG = pd.read_csv(S_SNF.loc[SN_],parse_dates=[0],index_col=0)
	# Subset relevant survey entries
	idf_SURV = df_SURV[df_SURV['Serial 1']==SN_]
	# Initialize an DataFrame to hold individual sensor-survey-log merge results
	idf_STATS = pd.DataFrame()
	# Iterate across survey intervals
	for i_ in range(len(idf_SURV) - 1):
		# Get deployment bounding times from survey data
		ti = idf_SURV.index[i_] + TZC
		to = idf_SURV.index[i_+1] + TZC
		# Pull relevant LOG entries
		idf_LOG = df_LOG[(df_LOG.index >= ti) & (df_LOG.index <= to)]
		if len(idf_LOG) > 0:
			if idf_LOG.index[-1] - idf_LOG.index[0] > pd.Timedelta(12,unit='hour'):
				# Conduct calculations on time representations and pull relevant relative orientation data
				IDXo = (idf_LOG.index >= idf_LOG.index[0] + pd.Timedelta(1,unit='hour'))&\
					   (idf_LOG.index <= idf_LOG.index[0] + pd.Timedelta(12,unit='hour'))
				try:
					jdf_STATS = pd.DataFrame({	'age days':(idf_LOG.index - ti).total_seconds()/(3600*24),\
												'dAZ':idf_LOG['eCompass North'].values - np.nanmedian(idf_LOG[IDXo]['eCompass North'].values),\
												'dTA':idf_LOG['Tilted Angle'].values - np.nanmedian(idf_LOG[IDXo]['Tilted Angle'].values),\
												'localtime h': [t_.hour + t_.minute/60 for t_ in pd.DatetimeIndex(idf_LOG.index) - TZC - SNC],\
												'SN':[SN_]*len(idf_LOG),\
												'sta':[idf_SURV['sta'][i_]]*len(idf_LOG)},\
											index=idf_LOG.index)
					idf_STATS = pd.concat([idf_STATS,jdf_STATS],axis=0,ignore_index=False)
				except:
					pass
			# Tag S/N specific stats to summative dictionary
			if len(idf_STATS) > 0:
				D_STATS.update({SN_:idf_STATS})
		
### PLOTTING SECTION ###
fig = plt.figure(figsize=(8.5,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for SN_ in D_STATS.keys():
	idf = D_STATS[SN_]
	# AZo = np.nanmedian(idf[(idf['age days'] > 0.1) & (idf['age days'] < 1)]['AZ_log'].values)
	# TAo = np.nanmedian(idf[(idf['age days'] > 0.1) & (idf['age days'] < 1)]['TA_log'].values)
	ch1 = ax1.scatter(idf['age days'],idf['dAZ'],s=1,\
				c=np.abs(idf['localtime h']-12),vmin=0,vmax=24,cmap='YlGnBu')
	# ax1.plot(idf['age days'])
	ax2.scatter(idf['age days'],idf['dTA'],s=1,\
				c=np.abs(idf['localtime h']-12),vmin=0,vmax=24,cmap='YlGnBu')

for ax in [ax1,ax2]:
	for i_ in range(4):
		ax.text(i_ + 0.5,-20,'%.2f'%(1 - i_/3),ha='center',color='red')
	
	ax.text(4.5,-20,'0',ha='center',color='red')
	ax.text(2.5,-25,'$M_0$ averaging weights',color='red',ha='center')

ax1.set_xlabel('Deployment age (days)')
ax1.set_ylabel('Logged azimuth change ($^oE$)')
ax2.set_xlabel('Deployment age (days)')
ax2.set_ylabel('Logged tilt changes ($^o$ from vertical)',labelpad=-3)
ax1.set_ylim([-30,30])
ax2.set_ylim([-30,30])
ax1.grid(linestyle=':')
ax2.grid(linestyle=':')
ax1.text(0,22.5,'A',fontstyle='italic',fontweight='extra bold',fontsize=14)
ax2.text(0,22.5,'B',fontstyle='italic',fontweight='extra bold',fontsize=14)
ax2.text(1.5,10,'Midnights',rotation=90,ha='center',va='bottom')
ax2.text(1,10,'Middays',rotation=90,ha='center',va='bottom')

if issave:
	I_OFILE = 'EPSL_REV1_FigS3_SmartSolo_LOG_Orientation_Ages_%ddpi.%s'%(DPI,FMT.lower())
	plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)
	print('Figure saved: %s'%(os.path.join(ODIR,I_OFILE)))




plt.show()

