import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as PathEffects
import matplotlib.dates as mdates
from copy import deepcopy


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),linewidth=3, alpha=1.0):
	"""
	http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
	http://matplotlib.org/examples/pylab_examples/multicolored_line.html
	Plot a colored line with coordinates x and y
	Optionally specify colors in the array z
	Optionally specify a colormap, a norm function and a line width

	SOURCE: https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
	"""

	# Default colors equally spaced on [0,1]:
	if z is None:
		z = np.linspace(0.0, 1.0, len(x))

	# Special case if a single number:
	if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
		z = np.array([z])

	z = np.asarray(z)

	segments = make_segments(x, y)
	lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,\
							  linewidth=linewidth, alpha=alpha)

	ax = plt.gca()
	ax.add_collection(lc)

	return lc

def make_segments(x, y):
	"""
	Create list of line segments from x and y coordinates, in the correct format
	for LineCollection: an array of the form numlines x (points per line) x 2 (x
	and y) array
	SOURCE: https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
	"""

	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	return segments


def yaxis_only(ax,unusedy='right',includeX=False):
	"""
	Wrapper for variably displaying already-rendered X and Y axes
	"""
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(includeX)
	ax.xaxis.set_visible(includeX)
	if unusedy == 'left':
		ax.yaxis.set_ticks_position('right')
		ax.yaxis.set_label_position('right')
	if unusedy == 'right':
		ax.yaxis.set_ticks_position('left')
		ax.yaxis.set_label_position('left')
	ax.spines[unusedy].set_visible(False)


def series_pearson_xcorr(s_A,s_B,nlags,lagdt=pd.Timedelta(15,unit='min')):
	ccvs = []
	lagind = np.arange(int(-nlags),int(nlags+1))
	for lag in lagind:
		ccvs.append(s_A.corr(s_B.shift(lag)))
	S_out = pd.Series(ccvs,index=lagind*lagdt.seconds)
	return S_out

############################################################################################################
######### DATA MAPPING SECTION #############################################################################
############################################################################################################

### MAP DATA SOURCES ###
# Map revision root directory
ROOT = os.path.join('..','..')
# Map 15 min (up-)sampled BEQ, S_w, and slip rates
TSFL = os.path.join(ROOT,'data','timeseries','ERV_0.25hr_G_filt_resampled_UTC_merr.csv')
# Icequake hypocenter catalog location
EQFL = os.path.join(ROOT,'data','seismic','well_located_30m_basal_catalog_merr.csv')
# Get seismic stations for display purposes
STFL = os.path.join(ROOT,'data','seismic','Stations.csv')
# Get filtered moment estimates
M0FL = os.path.join(ROOT,'data','M0_results','M0_stats_SNR_ge5_ND_ge5_MA_le3_SA_wt.csv')
# XDIR = os.path.join(ROOT,'data','timeseries','xcorr')

### OUTPUT DIRECTORY ###
ODIR = os.path.join(ROOT,'outputs')
DPI = 300
FMT = 'PNG'
issave = True

############################################################################################################
######### LOADING SECTION ##################################################################################
############################################################################################################

# Read in data
# Evenly sampled time-series data
df_T = pd.read_csv(TSFL,parse_dates=True,index_col=[0])
# Station Deploy file (for showing )
df_S = pd.read_csv(STFL)
# Event Catalog 
df_EQ = pd.read_csv(EQFL,parse_dates=['t0'],index_col='t0')
# Filtered M0 event estimates
df_M = pd.read_csv(M0FL,parse_dates=['t0'],index_col='t0')
df_M = df_M.sort_index()

## Postprocessing for basal event catalog
# Subset to gaussian uncertainty based filtering for bed-proximal, well located events
df_EQ = df_EQ[df_EQ['G_filt']]
# Multi-sort by timing and solution uncertainty (ascending)
df_EQ = df_EQ.sort_index().sort_values(['deperr','herrmax','herrmin'],ascending=True)
# Reject idential origin times, keeping first entry, which should favor minimized uncertainties
df_EQ = df_EQ[~df_EQ.index.duplicated(keep='first')]

# Output postprocessed event catalog for other plottings
df_EQ.to_csv(os.path.join(ROOT,'data','seismic','well_located_30m_basal_catalog_merr_dupQCd.csv'),header=True,index=True)

############################################################################################################
######### TIME ZONE PROCESSING SECTION #####################################################################
############################################################################################################

# Time Shifts to Local Solar Times
TZ0 = pd.Timedelta(-6,unit='hour') 					# UTC Time Offset
SNt = pd.Timedelta(-1,unit='hour') 					# Solar Noontime
CTt0 = pd.Timestamp('2019-07-31T22:06:00') + SNt	# Civil Twilight reference time 1
CTt1 = pd.Timestamp('2019-08-01T05:51:00') + SNt 	# Civil Twilight reference time 2
SNo = pd.Timestamp('2019-07-31T13:48:00') + SNt
DT = pd.Timedelta(15,unit='min')


# Adjust time-series to local solar time
df_T.index += (TZ0 + SNt)
# patch small holes
df_T = df_T.interpolate(direction='both',limit=2)

# Adjust origin times to local solar time
df_EQ.index += (TZ0 + SNt)
# Resample EQ catalog for basal rates
df_ER = pd.concat([pd.DataFrame(index=[df_T.index[0]]),df_EQ.copy()],axis=0,ignore_index=False)
df_ER = df_ER.resample(DT).count()


# Do time shift of M0 timestamps
df_M.index += (TZ0 + SNt)


# Do binning/resampling of M0 results
df_Mi = pd.concat([pd.DataFrame(index=[df_T.index[0]]),df_M.copy()],axis=0,ignore_index=False)
# Get median values of binned values
df_bM = df_Mi.resample(DT).median()
# Get count of events with M0 in each bin
df_bC = df_Mi.resample(DT).count()
# # Get 95% CI for binned M0
# df_b0 = df_Mi.resample(DT).quantile(q=0.025)
# df_b1 = df_Mi.resample(DT).quantile(q=0.975)


# Shift station deploy times from UTC to local solar time
df_S['Ondate'] = pd.to_datetime(df_S['Ondate'],unit='s') + TZ0 + SNt
df_S['Offdate'] = pd.to_datetime(df_S['Offdate'],unit='s') + TZ0 + SNt
df_S.index = df_S['Ondate']
df_S = df_S.sort_index()

## DO SIMPLE CROSS_CORRELATIONS
df_X_INs = pd.concat([df_T[['SR(mmWE h-1)','V_H(cm d-1)']],(2/3)*np.log10(np.abs(df_bM['SA_wt_mean'])) - 6.03])#,df_bC['mean']*4])
df_X_INs = df_X_INs.rename(columns={0:'mw'})
df_X_INs = pd.concat([df_X_INs,df_bC['mean']*4])
df_X_INs = df_X_INs.rename(columns={0:'dotE'})

# CC = []
# LAG = []

# for A_ in df_X_INs.columns:
# 	LL = []
# 	CL = []
# 	for B_ in df_X_INs.columns:
# 		S_ = series_pearson_xcorr(df_X_INs[A_],df_X_INs[B_],nlags=60)
# 		AMAX = S_[S_==np.nanmax(S_)]
# 		if len(AMAX) > 0:
# 			LL.append(AMAX.index[0])
# 			CL.append(AMAX.values[0])
# 		else:
# 			LL.append(np.nan)
# 			CL.append(np.nan)		
# 	CC.append(CL)
# 	LAG.append(LL)

# df_CC = pd.DataFrame(CC,columns=df_X_INs.columns,index=df_X_INs.columns)
# df_LAG = pd.DataFrame(LAG,columns=df_X_INs.columns,index=df_X_INs.columns)
############################################################################################################
######### PLOTTING SECTION #################################################################################
############################################################################################################
### INITIALIZE PLOT ###
fig = plt.figure(figsize=(8,9))
axs = []
gs = GridSpec(nrows=14,ncols=1,hspace=0,wspace=0)

# Set number of plots that will be stacked
nstack = 4
# Overlay Axis
# axO = fig.add_axes([.1,.1,.8,.8])
axO = fig.add_subplot(gs[:,:])
axs.append(axO)
# Axes for each subplot
ax1 = fig.add_subplot(gs[:3,:])
ax2 = fig.add_subplot(gs[3:6,:])
ax3 = fig.add_subplot(gs[6:10,:])
ax4 = fig.add_subplot(gs[10:,:])

axs += [ax1,ax2,ax3,ax4]


# for i_ in range(nstack):
# 	iax = fig.add_axes([.1,.9 - (.8/nstack)*(i_ + 1),.8,.8/nstack],sharex=axO)
# 	if i_ < nstack - 1:
# 		iax.xaxis.set_visible(False)
# 		iax.spines['bottom'].set_visible(False)
# 	axs.append(iax)


I_ = 0
## BACKGROUND ##
axs[I_].patch.set_alpha(0)
for SL_ in ['top','bottom','right','left']:
	axs[I_].spines[SL_].set_visible(False)
axs[I_].xaxis.set_visible(False)
axs[I_].yaxis.set_visible(False)
axs[I_].set_ylim([0,1])

### Plot Nighttime Hours and Solar Noontimes ###
for i_ in range(20):
	# Plot Civil Twilight Bounds for Nighttime
	axs[I_].fill_between([CTt0,CTt1],np.zeros(2),np.ones(2),color='midnightblue',alpha=0.1)
	# Plot Solar Noon Lines
	axs[I_].plot([SNo,SNo],[0,1],color='orange',alpha=0.5)
	# Advance counts
	CTt0 += pd.Timedelta(24,unit='hour')
	CTt1 += pd.Timedelta(24,unit='hour')
	SNo += pd.Timedelta(24,unit='hour')

### Plot Array Deployment & Demobilization ###
idf_S = df_S[(df_S['Offdate'].values - df_S['Ondate'].values) > pd.Timedelta(4,unit='day')]
st_alpha = 0.05; st_col='k'
for i_ in range(len(idf_S)):
	axs[I_].fill_between([pd.Timestamp("2019-08-01"),idf_S['Ondate'].values[i_]],\
						 np.zeros(2),np.ones(2),color=st_col,alpha=st_alpha)
	axs[I_].fill_between([idf_S['Offdate'].values[i_],pd.Timestamp("2019-08-20T12:00:00")],\
						 np.zeros(2),np.ones(2),color=st_col,alpha=st_alpha)


f34t = [pd.Timestamp('2019-08-08T05'),pd.Timestamp('2019-08-09T05')]
# axs[I_].fill_between(fig34_times,np.zeros(2,),np.ones(2,),linewidth=1,edgecolor='cyan',alpha=0.25)
axs[I_].plot([f34t[0],f34t[0],f34t[1],f34t[1],f34t[0]],\
			 [0,1,1,0,0],'.-',linewidth=2,color='royalblue')
axs[I_].text(f34t[0] + (f34t[1] - f34t[0])/2,0.6,'Fig. 5',\
			 ha='center',va='center',color='royalblue',fontweight='extra bold',\
			 rotation=90,fontsize=14)


period_times = [pd.Timestamp('2019-08-05T04'),pd.Timestamp("2019-08-09T04"),\
				pd.Timestamp("2019-08-14T04"),pd.Timestamp("2019-08-19T12")]

for i_ in range(3):
	axs[I_].plot([period_times[i_+1],period_times[i_+1]],\
				 [0,1],'k-.',linewidth=2)
	ts_u = period_times[i_] + (period_times[i_+1] - period_times[i_])/2
	if i_ == 0:
		ts_u -= pd.Timedelta(0.25,unit='day')
	axs[I_].text(ts_u,0.5,'Interval %s'%(['I','II','III'][i_]),ha='center',va='top',\
				 fontsize=14)

# y_coords = np.linspace(1,0,5)
# for i_,l_ in enumerate(['A','B','C','D']):
# 	axs[I_].text(pd.Timestamp("2019-08-05T06"),y_coords[i_]-0.025,l_,\
# 				 fontsize=16,fontweight='extra bold',fontstyle='italic',\
# 				 va = 'top')

### DATA OVERLAYS ###

I_ += 1
## MELTWATER SUPPLY RATE
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['SR(mmWE h-1)'],'k',zorder=2)#label='15-minute estimates',zorder=2)
# axs[I_].plot(df_T24['SR(mmWE h-1)']*(df_T['SR(mmWE h-1)']/df_T['SR(mmWE h-1)']),'r',label='24-hour averages',zorder=1)
axs[I_].set_ylabel('Meltwater\nsupply rate [$\dot{S}_w$]\n($mm w.e.$ $h^{-1}$)')
axs[I_].set_ylim([-0.1,3.9])
axs[I_].grid(axis='y',alpha=0.5)


I_ += 1
## SURFACE VELOCITY PLOTTING ##
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['V_H(cm d-1)']*3.6524 - 8.9,'k',zorder=2)#label='15-minute estimates',zorder=2)
axs[I_].set_ylabel('Slip velocity [$V$]\n($m$ $a^{-1}$)' ,labelpad=0)
axs[I_].grid(axis='y',alpha=0.5)
axs[I_].set_ylim([-5,160])

I_ += 1
## BASAL EVENT RATE PLOTTING ##
axs[I_].patch.set_alpha(0)
# axs[I_].plot(df_ER['mE']/0.25e3,'k',zorder=1,label='All icequakes')
# axs[I_].plot(df_T24M['ER(N h-1)']/1e3,'r',label='24-hour maximum')
# axs[I_].plot(df_T24['ER(N h-1)']/1e3,'r',label='24-hour averages',zorder=1)
axs[I_].set_ylabel('Icequake rate [$\dot{E}$]\n($ct.$ $h^{-1}$ x1000)')
axs[I_].set_ylim([-0.1,2.1])
axs[I_].grid(axis='y',alpha=0.5)
# Do events with M0 estimates
axs[I_].plot(df_bC['median']/0.25e3,'k-',label='Icequakes with $M_0$',zorder=2)
# axs[I_].legend(ncol=2)

I_ += 1
# MEDIAN BINNED MOMENT RATE
axs[I_].patch.set_alpha(0)
# Plot 95% CI for binned  M0
# axs[I_].fill_between(df_b0.index,np.log10(df_b0['median'].values),\
# 					np.log10(df_b1['median'].values),\
# 					step='mid',color='k',alpha=0.33,zorder=2)

# Plot median of binned M0 estimates
axs[I_].plot((2/3)*np.log10(df_bM['median']) - 6.03,'ko-',ms=2,lw=1,zorder=3,label='Binned $\\tilde{m_w}$')
# Plot individual samples
axs[I_].plot((2/3)*np.log10(df_M['SA_wt_mean']) - 6.03,'r*',ms=3,alpha=0.25,zorder=1,label='Individual icequakes')
# axs[I_].plot(np.log10(df_M[df_M['EVID']==21414]['SA_wt_mean']),'*',color='dodgerblue',ms=9,zorder=2,label='Fig. 2 icequake')
# axs[I_].legend(loc='lower center',ncol=3)
# Format Y-axis scaling
axs[I_].set_ylim([(2/3)*2.5 - 6.03,(2/3)*8 - 6.03])
axs[I_].grid(axis='y',alpha=0.5)
axs[I_].set_ylabel('Moment magnitude [$m_w$ / $\\tilde{m_w}$]')

# axb = axs[I_].twinx()
# axb.set_ylim((2/3)*np.array(axs[I_].get_ylim()) - 6.03)
# axb.set_ylabel('Moment magnitude [$m_w$]',rotation=270,labelpad=15,fontsize=10)
# yaxis_only(axb,unusedy='left')


## FORMAT SHARED X-AXIS LABEL
axs[I_].set_xlabel('Day of August 2019 (UTC - 7 hours)')
axs[I_].xaxis.set_major_locator(mdates.DayLocator())
axs[I_].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
axs[I_].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
axs[I_].format_xdata = mdates.DateFormatter('%d')

for i_ in range(len(axs)):
	# axs[i_].set_xlim([pd.Timestamp("2019-08-05"),pd.Timestamp("2019-08-20")])
	axs[i_].set_xlim([pd.Timestamp("2019-08-05"),period_times[-1]])
	if i_ < len(axs) - 1:
		yaxis_only(axs[i_])
	else:
		yaxis_only(axs[i_],includeX=True)

# Put in Subplot Labels
for i_,l_ in enumerate(['A','B','C','D']):
	ylims = np.array(axs[i_+1].get_ylim())
	y_coord = np.min(ylims) + 0.975*(np.max(ylims) - np.min(ylims))
	axs[i_+1].text(pd.Timestamp("2019-08-05T06"),y_coord,l_,\
				 fontsize=16,fontweight='extra bold',fontstyle='italic',\
				 va = 'top')


if issave:
	save_file = 'EPSL_REV1.3_Fig4_TimeSeries_TightTimeline_%ddpi.%s'%(DPI,FMT.lower())
	plt.savefig(os.path.join(ODIR,save_file),dpi=DPI,format=FMT.lower())
	print('Figure saved as %s'%(os.path.join(ODIR,save_file)))


# ### Create Labels
# text = bxs[0].text(pd.Timestamp("2019-08-01T12:00:00"),2.5,'Deploy\nPeriod',color='white',fontweight='extra bold',fontsize=14)
# text.set_path_effects([path_effects.Stroke(linewidth=1,foreground='black'),path_effects.Normal()])

# text = bxs[0].text(pd.Timestamp("2019-08-20T00:00:00"),2.5,'Demobilization',\
# 				   rotation=90,color='white',fontweight='extra bold',fontsize=12)
# text.set_path_effects([path_effects.Stroke(linewidth=1,foreground='black'),path_effects.Normal()])

# for i_,D_ in enumerate([pd.Timestamp("2019-08-06"),pd.Timestamp("2019-08-10T12:00:00"),pd.Timestamp("2019-08-15")]):
# 	text = bxs[2].text(D_,.01,'Period %d'%(i_ + 1),color='white',fontweight='extra bold',fontsize=14)
# 	text.set_path_effects([path_effects.Stroke(linewidth=1,foreground='black'),path_effects.Normal()])

# for D_ in [pd.Timestamp('2019-08-04T16:38:00'),pd.Timestamp('2019-08-09T06:00:00'),pd.Timestamp('2019-08-14T06:00:00')]:
# 	axs[I_].plot([D_,D_],[0,1],'k:')


plt.show()
