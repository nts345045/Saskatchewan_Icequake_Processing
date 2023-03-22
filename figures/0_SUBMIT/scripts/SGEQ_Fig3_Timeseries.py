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

############################################################################################################
######### DATA MAPPING SECTION #############################################################################
############################################################################################################

### MAP DATA SOURCES ###
ROOT = os.path.join('..','..','..','..')
TSFL = os.path.join(ROOT,'results','data_files','ERV_0.25hr_G_filt_resampled_UTC_merr.csv')
STFL = os.path.join(ROOT,'data','seismic','sites','Stations.csv')
XDIR = os.path.join(ROOT,'processed_data','timeseries','xcorr')

### OUTPUT DIRECTORY ###
ODIR = os.path.join(ROOT,'results','figures','Manuscript_Figures','v25_render')
DPI = 300
FMT = 'PNG'
issave = True


# Read in data
# Evenly sampled time-series data
df_T = pd.read_csv(TSFL,parse_dates=True,index_col=[0])
# Station Deploy file (for showing )
df_S = pd.read_csv(STFL)
df_P = pd.read_csv(os.path.join(XDIR,'pxcorr_l15h_v2.csv'),parse_dates=True,index_col=[0])
df_VxE = pd.read_csv(os.path.join(XDIR,'V_H_x_ER_rxcorr_l15h_s1h.csv'),parse_dates=True,index_col=[0])
df_VxR = pd.read_csv(os.path.join(XDIR,'V_H_x_SR_rxcorr_l15h_s1h.csv'),parse_dates=True,index_col=[0])



############################################################################################################
######### LOADING SECTION ##################################################################################
############################################################################################################

# Time Shifts to Local Solar Times
TZ0 = pd.Timedelta(-6,unit='hour') 					# UTC Time Offset
SNt = pd.Timedelta(-1,unit='hour') 					# Solar Noontime
CTt0 = pd.Timestamp('2019-07-31T22:06:00') + SNt	# Civil Twilight reference time 1
CTt1 = pd.Timestamp('2019-08-01T05:51:00') + SNt 	# Civil Twilight reference time 2
SNo = pd.Timestamp('2019-07-31T13:48:00') + SNt

# Adjust time-series to local solar time
df_T.index += (TZ0 + SNt)
# patch small holes
df_T = df_T.interpolate(direction='both',limit=2)
# Do 24 Hour rolling Average Values of time-series
df_T24 = df_T.copy().rolling(pd.Timedelta(24,unit='hour')).mean()
df_T24M = df_T.copy().rolling(pd.Timedelta(24,unit='hour')).max()
# center sampling window
# df_T24.index -= pd.Timedelta(12,unit='hour')
# df_T24M.index -= pd.Timedelta(12,unit='hour')
# Shift rolling x-corr to local solar times (prior used UTC)
df_VxE.index += (TZ0 + SNt)
df_VxR.index += (TZ0 + SNt)

# Extract peak correlation values and lags
PXC_dict = {}
for C_ in df_P.columns:
	PXC_dict.update({C_:{'IND':df_P[C_].argmax(),\
						 'LAG':df_P[C_].index[df_P[C_].argmax()],\
						 'CC':df_P[C_].max()}})

# FRe = Fraction of Real-Valued Data points in Window
IND = []; LAG = []; CC = []; FRe = []; F0 = [];
for i_ in range(len(df_VxE)):
	idf = df_VxE.iloc[i_]
	FRe.append(np.sum(np.isfinite(idf))/len(idf))
	IND.append(idf.argmax())
	LAG.append(float(idf.index[idf.argmax()]))
	CC.append(idf.max())
VxEmax = {'IND':IND,\
		  'LAG':LAG,\
		  'CC':CC,\
		  'FRe':FRe}

IND = []; LAG = []; CC = []; FRe = [];
for i_ in range(len(df_VxR)):
	idf = df_VxR.iloc[i_]
	FRe.append(np.sum(np.isfinite(idf))/len(idf))
	IND.append(idf.argmax())
	LAG.append(float(idf.index[idf.argmax()]))
	CC.append(idf.max())
VxRmax = {'IND':IND,\
		  'LAG':LAG,\
		  'CC':CC,\
		  'FRe':FRe}

df_VxE_max = pd.DataFrame(VxEmax,index=df_VxE.index)
df_VxR_max = pd.DataFrame(VxRmax,index=df_VxR.index)

# Shift station deploy times from UTC to local solar time
df_S['Ondate'] = pd.to_datetime(df_S['Ondate'],unit='s') + TZ0 + SNt
df_S['Offdate'] = pd.to_datetime(df_S['Offdate'],unit='s') + TZ0 + SNt
df_S.index = df_S['Ondate']
df_S = df_S.sort_index()





############################################################################################################
######### PLOTTING SECTION #################################################################################
############################################################################################################
### INITIALIZE PLOT ###
fig = plt.figure(figsize=(8,8))
axs = []
# Set number of plots that will be stacked
nstack = 4
# Overlay Axis
axO = fig.add_axes([.1,.1,.8,.8])
axs.append(axO)
for i_ in range(nstack):
	iax = fig.add_axes([.1,.9 - (.8/nstack)*(i_ + 1),.8,.8/nstack],sharex=axO)
	if i_ < nstack - 1:
		iax.xaxis.set_visible(False)
		iax.spines['bottom'].set_visible(False)
	axs.append(iax)


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
axs[I_].text(f34t[0] + (f34t[1] - f34t[0])/2,0.525,'Fig. 3',\
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
	axs[I_].text(ts_u,0.5,'Interval %s'%(['A','B','C'][i_]),ha='center',va='center',\
				 fontsize=14)

y_coords = np.linspace(1,0,5)
for i_,l_ in enumerate(['a','b','c','d']):
	axs[I_].text(pd.Timestamp("2019-08-05T06"),y_coords[i_]-0.025,l_,\
				 fontsize=16,fontweight='extra bold',fontstyle='italic',\
				 va = 'top')

### DATA OVERLAYS ###

I_ += 1
## MELTWATER SUPPLY RATE
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['SR(mmWE h-1)'],'k',label='15-minute estimates',zorder=2)
axs[I_].plot(df_T24['SR(mmWE h-1)']*(df_T['SR(mmWE h-1)']/df_T['SR(mmWE h-1)']),'r',label='24-hour averages',zorder=1)
axs[I_].set_ylabel('Meltwater supply rate\n[$\dot{S}_{w}$] ($mm w.e.$ $h^{-1}$)')
axs[I_].set_ylim([0.1,3.9])



I_ += 1
## SURFACE VELOCITY PLOTTING ##
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['V_H(cm d-1)']*3.6524 - 8.9,'k',label='15-minute estimates',zorder=2)
axs[I_].plot((df_T24['V_H(cm d-1)']*3.6524 - 8.9)*(df_T['V_H(cm d-1)']/df_T['V_H(cm d-1)']),'r',label='24-hour averages',zorder=1)
axs[I_].set_ylabel('Slip velocity\n[$V$] ($m$ $a^{-1}$)' ,labelpad=0)
# axs[I_].set_ylim([-0.2,2.2])
# axs[I_].set_yticks([0,1,2])
axs[I_].legend(loc='upper right')

I_ += 1
## BASAL EVENT RATE PLOTTING ##
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['ER(N h-1)']/1e3,'k',label='15-minute estimates',zorder=2)
# axs[I_].plot(df_T24M['ER(N h-1)']/1e3,'r',label='24-hour maximum')
axs[I_].plot(df_T24['ER(N h-1)']/1e3,'r',label='24-hour averages',zorder=1)
axs[I_].set_ylabel('Ice-quake rate\n[$\dot{E}$] ($ct.$ $h^{-1}$ x1000)')
axs[I_].set_ylim([-0.1,4.5])




I_ += 1
## CROSS CORRELATIONS
axs[I_].patch.set_alpha(0)
idf_VxRm = df_VxR_max[(df_VxR_max['FRe'] > 0.75)&(df_VxR_max['CC'] >= 0.55)].resample(pd.Timedelta(1,unit='hour')).mean()
idf_VxEm = df_VxE_max[(df_VxE_max['FRe'] > 0.75)&(df_VxE_max['CC'] >= 0.55)].resample(pd.Timedelta(1,unit='hour')).mean()
cax = axs[I_].scatter(idf_VxRm.index,idf_VxRm['LAG']/-3600,c=idf_VxRm['CC'],vmin=0.5,vmax=1,\
				label='Meltwater',cmap='Blues',alpha=1)
cbx = axs[I_].scatter(idf_VxEm.index,idf_VxEm['LAG']/-3600,c=idf_VxEm['CC'],vmin=0.5,vmax=1,\
				label='Ice-quakes',cmap='Reds',alpha=1)
axs[I_].plot([pd.Timestamp('2019-08-01'),pd.Timestamp("2019-08-20")],\
			 np.ones(2,)*PXC_dict['V_H(cm d-1)xSR(mmWE h-1)']['LAG']/-3600,\
			 'b:',linewidth=2,alpha=0.75)
axs[I_].plot([pd.Timestamp('2019-08-01'),pd.Timestamp("2019-08-20")],\
			 np.ones(2,)*PXC_dict['V_H(cm d-1)xER(N h-1)']['LAG']/-3600,\
			 ':',color='firebrick',linewidth=2,alpha=0.75)
# axs[I_].plot(df_VxE_max[df_VxE_max['FRe'] > 0.75]['LAG']/-3600,'r-',label='c')
axs[I_].set_ylabel('Lag after\nslip velocity ($h$)',labelpad=-10)
axs[I_].set_xlabel('Day of August 2019 (UTC - 7 hours)')
axs[I_].xaxis.set_major_locator(mdates.DayLocator())
axs[I_].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
axs[I_].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
axs[I_].format_xdata = mdates.DateFormatter('%d')
# axs[I_].xticks()
## CROSS CORRELATION COEFFICIENT LEGEND

# caax = fig.add_axes([.1,.105,.02,.1])
# cbax = fig.add_axes([.12,.105,.02,.1])
caax = fig.add_axes([.115,.15,.11,.02])
cbax = fig.add_axes([.115,.13,.11,.02])
plt.colorbar(cax,cax=cbax,ticks=[0.5,1],orientation='horizontal')
plt.colorbar(cbx,cax=caax,ticks=[],orientation='horizontal')
txt1 = cbax.text(0.75,0.45,'V x S',ha='center',va='center')#rotation=90,va='center')
txt1.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='w')])
txt2 = caax.text(0.75,0.45,'V x E',ha='center',va='center')#rotation=90,va='center')
txt2.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='w')])
cbax.text(0.75,-0.75,'CC',ha='center',va='center')#rotation=90,va='center')
axs[I_].set_ylim([-18,18])




for i_ in range(len(axs)):
	# axs[i_].set_xlim([pd.Timestamp("2019-08-05"),pd.Timestamp("2019-08-20")])
	axs[i_].set_xlim([pd.Timestamp("2019-08-05"),period_times[-1]])
	if i_ < len(axs) - 1:
		yaxis_only(axs[i_])
	else:
		yaxis_only(axs[i_],includeX=True)


if issave:
	save_file = 'SGEQ_v25_Fig2_TimeSeriesXcorr_TightTimeline_%ddpi.%s'%(DPI,FMT.lower())
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
