"""
EPSL_ACC_Mov1_SeismicEvolution_FrameMaker.py
:purpose: Generate a series of PNGs with frame number naming to turn into a
movie using ffmpeg or QuickTime

"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerTuple
from matplotlib import cm
from pyproj import Proj
from tqdm import tqdm

### TODO ###
"""
 - Reinstate reticles
 - Add annotations on mw statistics to C
 - Shift B label up a bit
 - Correct fill color for mw \\in [-3.5,-2.0] in C

"""


############################################################################################################
############################################################################################################
######### RENDERING CONTROLS & OUTPUT DIRECTORY ############################################################
############################################################################################################
############################################################################################################

# Write Controls
FMT='PNG'
issave = True
# Rendering controls
cmap = 'tab20c'#'terrain'#'viridis'
eqalpha=0.5
LW = 1
ALP = 0.75
ECV = ['black','red','dodgerblue','darkorange']
DPI = 100
mw_lims =[-3.5,-1.5]
# Framespacing controls
Frame_STEP = pd.Timedelta(10,unit='min') # How much to advance times between frames
Frame_LENG = pd.Timedelta(2,unit='hour') # How long the viewing window should be
StartTime = pd.Timestamp("2019-08-01T00:00:00") # 
EndTime = pd.Timestamp("2019-08-19T06:00:00")
frame_count = 'auto'#90 #'auto'

### CALCULATE NUMBER OF FRAMES IF 'auto' ###
if frame_count == 'auto':
	frame_count = 0
	its = StartTime + Frame_LENG
	while its < EndTime:
		its += Frame_STEP
		frame_count += 1
else:
	print('Manually set frame count: %d'%(frame_count)) 






############################################################################################################
############################################################################################################
######### DATA MAPPING SECTION #############################################################################
############################################################################################################
############################################################################################################

### MAP DATA SOURCES ###
ROOT = os.path.join('..','..')
# Path to time-series file
TSFL = os.path.join(ROOT,'data','timeseries','ERV_0.25hr_G_filt_resampled_UTC_merr.csv')
# Bed Map Directory
BMDR = os.path.join(ROOT,'data','seismic','HVSR','grids')
# Icequake hypocenter catalog location
EQFL = os.path.join(ROOT,'data','seismic','well_located_30m_basal_catalog_merr_dupQCd.csv')
# Station locations
STFL = os.path.join(ROOT,'data','seismic','Stations.csv')
# Station orientation surveys
SURV = os.path.join(ROOT,'data','tables','Sorted_Passive_Servicing.csv')
# Phase Instruction File
PFIL = os.path.join(ROOT,'data','timeseries','times_and_phases.csv')
# Get filtered moment estimates
M0FL = os.path.join(ROOT,'data','M0_results','M0_stats_SNR_ge5_ND_ge5_MA_le3_SA_wt.csv')


# Figure output directory
ODIR = os.path.join(ROOT,'outputs','MOV_Frames_SAwt')




############################################################################################################
############################################################################################################
######### LOADING SECTION ##################################################################################
############################################################################################################
############################################################################################################

## LOAD DATA
# Evenly sampled time-series data
df_T = pd.read_csv(TSFL,parse_dates=True,index_col=[0])
## Import Station Location and Survey Data
df_STA = pd.read_csv(STFL)
df_SURV = pd.read_csv(SURV,parse_dates=['DateTime Mean'],index_col='DateTime Mean')

# Filtered basal events
df_EQ = pd.read_csv(EQFL,parse_dates=['t0'],index_col=['t0'])
# Filtered M0 event estimates
df_M0 = pd.read_csv(M0FL,parse_dates=['t0'],index_col='t0')
df_M0 = df_M0.sort_index()


# Define Local Solar Times
TZ0 = pd.Timedelta(-6,unit='hour') 					# UTC Time Offset
SNt = pd.Timedelta(-1,unit='hour') 					# Solar Noontime
CTt0 = pd.Timestamp('2019-07-31T22:06:00') + SNt    # Civil Twilight reference time 1
CTt1 = pd.Timestamp('2019-08-01T05:51:00') + SNt 	# Civil Twilight reference time 2
SNo = pd.Timestamp('2019-07-31T13:48:00') + SNt
DT = pd.Timedelta(15,unit='min')
# Adjust time-series to local solar time
df_T.index += (TZ0 + SNt)
# patch small holes
df_T = df_T.interpolate(direction='both',limit=2)

# Shift basal event origin times to local solar time
df_EQ.index += (TZ0 + SNt)
df_EQ = df_EQ.sort_index()
df_M0.index += (TZ0 + SNt)

# Shift station on/off dates to local solar time
df_STA['Ondate'] = pd.to_datetime(df_STA['Ondate'],unit='s') + TZ0 + SNt
df_STA['Offdate'] = pd.to_datetime(df_STA['Offdate'],unit='s') + TZ0 + SNt
df_STA.index = df_STA['Ondate']
df_STA = df_STA.sort_index()

# Shift survey times from local daylight savings time to local solar time
df_SURV.index += SNt


## RESAMPLE M0 DATA TO GET BINNED STATISTICS##
# Do binning/resampling of M0 results
df_M0i = pd.concat([pd.DataFrame(index=[df_T.index[0]]),df_M0.copy()],axis=0,ignore_index=False)
# Get median values of binned values
df_bM = df_M0i.resample(DT).median()
# Get count of events with M0 in each bin
df_bC = df_M0i.resample(DT).count()

## Convert Station Locations to UTM
myproj = Proj(proj='utm',zone=11,ellipse='WGS84',preserve_units=False)
SmE,SmN = myproj(df_STA['Lon'].values,df_STA['Lat'].values)
df_STA = pd.concat([df_STA,pd.DataFrame({'mE':SmE,'mN':SmN},index=df_STA.index)],axis=1,ignore_index=False)

# ## Import Phase Instructions 
# df_p = pd.read_csv(PFIL,parse_dates=['T0','T1'],index_col=[0])

# Merge binned data 
df_T = pd.concat([df_T,df_bC['SA_wt_mean']],axis=1,ignore_index=False).rename(columns={'SA_wt_mean':'M0 count'})

df_T = pd.concat([df_T,df_bM['SA_wt_mean']],axis=1,ignore_index=False).rename(columns={'SA_wt_mean':'M0 median'})

## LOAD GRIDS
## Import Grids
grd_E = np.load(os.path.join(BMDR,'mE_Grid.npy'))
grd_N = np.load(os.path.join(BMDR,'mN_Grid.npy'))
grd_Hm = np.load(os.path.join(BMDR,'H_med_Grid.npy'))
grd_Hu = np.load(os.path.join(BMDR,'H_gau_u_Grid.npy'))
grd_Ho = np.load(os.path.join(BMDR,'H_gau_o_Grid.npy'))
grd_Zs = np.load(os.path.join(BMDR,'mZsurf_Grid.npy'))
grd_M = np.load(os.path.join(BMDR,'MASK_Grid.npy'))
## Localized Reference Coordinates
mE0 = 4.87e5
mN0 = 5.777e6
mEm = np.min(grd_E)
mEM = np.max(grd_E)
mNm = np.min(grd_N)
mNM = np.max(grd_N)



############################################################################################################
############################################################################################################
######### PLOTTING SECTION #################################################################################
############################################################################################################
############################################################################################################
#### SUPPORTING SUBROUTINES ####

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

def compound_scalebar(ax,ssteps=6,srange=[0,4000],sscalar=1,sslbl='1-100',slbl='Event rate [$\\dot{E}$] (ct. $h^{-1}$)',csteps=3,crange=[-3.25,-1.5],clbl='(Binned) median $m_w$',cmap='viridis',linewidth=2,alpha=0.75):
	# Grab cmap
	icmap = cm.get_cmap(cmap)
	# Ensure minmax order
	srange = [min(srange),max(srange)]
	crange = [min(crange),max(crange)]
	## Create Size Indexing
	# Size labeling position vector
	slv = np.linspace(0,1,ssteps)
	# Size label values
	sll = np.linspace(srange[0],srange[1],ssteps)

	## Create Color Indexing
	# Color labeling position vector
	clv = np.linspace(0,1,csteps)
	# Color label values
	cll = np.linspace(crange[0],crange[1],csteps)


	# Plot size & color ramp
	ax.scatter(np.linspace(0,1,101),np.zeros(101),s=np.linspace(min(srange),max(srange),101)*sscalar,\
            c=np.linspace(0,1,101),cmap=cmap)
	# Plot reference rings on scale
	ax.scatter(slv,np.zeros(ssteps),s=np.linspace(min(srange),max(srange),ssteps)*sscalar,\
	            facecolors='none',edgecolors='k',linewidth=2)
	# Plot Near-Zero symbol
	ax.scatter(1/srange[1],0,s=50,edgecolors='k',facecolors='none',marker='d',linewidth=2)

	# Render Color range labels (lower register)
	for c_ in range(csteps):
		text = ax.text(clv[c_],-0.02,'%.2f'%(cll[c_]),color=icmap((clv[c_] - clv[0])/(clv[-1] - clv[0])),ha='center',va='top')	
		text.set_path_effects([path_effects.Stroke(linewidth=1,foreground='black'),path_effects.Normal()])

	# Render size range labels (upper register)
	for s_ in range(ssteps):
		if s_ == 0:
			ax.text(slv[s_],0.02,sslbl,ha='center')
		else:
		    ax.text(slv[s_],0.02,'%d'%(sll[s_]),ha='center')

	# Plot index labels
	ax.text(np.mean(slv),0.0325,slbl,ha='center',va='bottom')
	ax.text(np.mean(slv),-0.0325,clbl,ha='center',va='top')
	

	# ax.set_ylim([-.12,.06])
	# Custom scale size
	ax.set_ylim([-0.05,0.125])
	ax.set_xlim([-0.03,1.1])


	# Ensure no borders and background is transparent
	ax.patch.set_alpha(0)
	for s_ in ['top','bottom','left','right']:
	    ax.spines[s_].set_visible(False)
	    ax.spines[s_].set_visible(False)
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])


def mw_inv(mw):
	M0 = 10**(1.5*(mw + 6.03))
	return M0

def mw(M0):
	mw = (2/3)*np.log10(M0) - 6.03
	return mw

class HandlerThreeMarkers(HandlerTuple):
	"""
	Class for making compound legend markers - generated by ChatGPT 3.5, checked by human review and modification
	"""
	def create_artists(self, legend, orig_handle,xdescent,ydescent,width,height,fontsize,trans):
		sizes = [fontsize,fontsize*1.5,fontsize*2]
		markers=['o','s','v']
		colors=['blue','orange','green']

		artists = []
		for xO, marker, color in zip(linspace(0,width,3), markers, colors):
			artists.append(plt.Line2D([xO],[0.5*height],marker=marker,color=color,markersize=sizes[markers.index(marker)]))

		return artists

### INITIALIZE PLOT ###
fig = plt.figure(figsize=(10,10))
axs = []
gs = GridSpec(nrows=32,ncols=8)

### CREATE AXIS FOR TIMESERIES DATA BACKGROUND ##
ax0 = fig.add_subplot(gs[:12,:])

### CREATE AXES FOR TIMESERIES DATA ###
for i_ in range(4):
	iax = fig.add_subplot(gs[0+i_*3:3+i_*3,:])
	if i_ < 3:
		iax.xaxis.set_visible(False)
		iax.spines['bottom'].set_visible(False)
		iax.spines['top'].set_visible(False)
	else:
		iax.spines['top'].set_visible(False)
	iax.set_xlim([StartTime,EndTime])
	axs.append(iax)
### CREATE AXIS FOR  TIMESERIES DATA OVERLAY ##
ax1 = fig.add_subplot(gs[:12,:])
ax1.set_xlim([StartTime,EndTime])
### CREATE AXIS FOR MAP ###
axm = fig.add_subplot(gs[14:,4:])

### CREATE AXIS FOR CROSSPLOT ###
axx = fig.add_subplot(gs[18:,:4])
### CREATE AXIS FOR SCALE BAR
axl = fig.add_subplot(gs[13:17,:4])


### FORMAT UNDERLAY FOR TIME SERIES ###
ax0.patch.set_alpha(0)
for SL_ in ['top','bottom','right','left']:
	ax0.spines[SL_].set_visible(False)
ax0.xaxis.set_visible(False)
ax0.yaxis.set_visible(False)
ax0.set_ylim([0,1])



############################################################################################################
######### PERMANENT BACKDROPS ##############################################################################
############################################################################################################

### Plot Nighttime Hours and Solar Noontimes ###
for i_ in range(20):
	# Plot Civil Twilight Bounds for Nighttime
	ax0.fill_between([CTt0,CTt1],np.zeros(2),np.ones(2),color='midnightblue',alpha=0.1)
	# Plot Solar Noon Lines
	ax0.plot([SNo,SNo],[0,1],color='orange',alpha=0.5)
	# Advance counts
	CTt0 += pd.Timedelta(24,unit='hour')
	CTt1 += pd.Timedelta(24,unit='hour')
	SNo += pd.Timedelta(24,unit='hour')

### Plot Array Deployment & Demobilization Periods ###
idf_STA = df_STA[(df_STA['Offdate'].values - df_STA['Ondate'].values) > pd.Timedelta(4,unit='day')]
st_alpha = 0.05; st_col='k'
for i_ in range(len(idf_STA)):
	ax0.fill_between([pd.Timestamp("2019-08-01"),idf_STA['Ondate'].values[i_]],\
						 np.zeros(2),np.ones(2),color=st_col,alpha=st_alpha)
	ax0.fill_between([idf_STA['Offdate'].values[i_],pd.Timestamp("2019-08-20T12:00:00")],\
						 np.zeros(2),np.ones(2),color=st_col,alpha=st_alpha)


period_times = [pd.Timestamp("2019-08-01"),pd.Timestamp('2019-08-05T04'),pd.Timestamp("2019-08-09T04"),\
				pd.Timestamp("2019-08-14T04"),pd.Timestamp("2019-08-19T12")]

plbl = ['Deploy Period','Period I','Period II','Period III']
for i_ in range(len(plbl)):
	ax0.plot([period_times[i_+1],period_times[i_+1]],\
				 [0,1],'k-.',linewidth=2)
	ts_u = period_times[i_] + (period_times[i_+1] - period_times[i_])/2
	ax0.text(ts_u,0.25,plbl[i_],ha='center',va='center',fontsize=10)

# TIME SERIES LABELS
y_coords = np.linspace(1,-.05,5)
ax0.text(EndTime-pd.Timedelta(30,unit='hour'),.9,'A',fontsize=14,fontweight='extra bold',fontstyle='italic')
for i_,l_ in enumerate(['$\\dot{E}$','$\\tilde{m_w}$','$\\dot{S} _w$','$V$']):

	text = ax0.text(StartTime+pd.Timedelta(4,unit='hour'),y_coords[i_]-0.1-0.025,l_,\
				 fontsize=10,fontweight='extra bold',fontstyle='italic',\
				 va = 'top')
	text.set_path_effects([path_effects.Stroke(linewidth=2,foreground='white'),path_effects.Normal()])

ax0.set_xlim([StartTime,EndTime])

### PLOT DATA ###
I_ = 0
## BASAL EVENT RATE PLOTTING ##
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_bC['SA_wt_mean']/0.25e3,'k',label='Measurements')
# axs[I_].plot(df_T24M['ER(N h-1)']/1e3,'r',label='24-hour maximum')
axs[I_].set_ylabel('$ct.$ $h^{-1}$\n(x1000)')
axs[I_].set_ylim([-0.1,2.25])
axs[I_].set_yticks([0,2,4])

I_ += 1
## MAGNITUDE PLOTTING ##
axs[I_].patch.set_alpha(0)
axs[I_].plot((2/3)*np.log10(df_bM['SA_wt_mean']) - 6.03,'k.-',markersize=3)
axs[I_].set_ylabel('Median\nmag.')
axs[I_].set_ylim([-4.1,-1.75])
axs[I_].set_yticks([-4,-3,-2])

I_ += 1
## MELTWATER SUPPLY RATE
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['SR(mmWE h-1)'],'k')
axs[I_].set_ylabel('$mm w.e.$ $h^{-1}$')
axs[I_].set_ylim([0.1,3.9])
axs[I_].set_yticks([1,2,3])
I_ += 1
## SURFACE VELOCITY PLOTTING ##
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['V_H(cm d-1)']/24,'k',label='Measurements')
axs[I_].set_ylabel('$cm$ $h^{-1}$')
axs[I_].set_ylim([-0.2,2.2])
axs[I_].set_yticks([0,1,2])

axs[I_].set_xlabel('Day of August 2019 (UTC - 7 hours)')
axs[I_].xaxis.set_major_locator(mdates.DayLocator())
axs[I_].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
axs[I_].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
axs[I_].format_xdata = mdates.DateFormatter('%d')


### FORMAT AXES FOR MOVING WINDOW OVERLAY ##

ax1.patch.set_alpha(0)
for SL_ in ['top','bottom','right','left']:
	ax0.spines[SL_].set_visible(False)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_ylim([0,1])
ax1.set_xlim([StartTime,EndTime])


# Create formatted compound scalebar
compound_scalebar(axl,ssteps=3,srange=[0,2000],sscalar=1/2.5,csteps=6,crange=mw_lims,linewidth=LW,cmap=cmap)
axl.set_xlim([-0.4380119590852878, 1.477656022704268])
axl.set_ylim([-0.02685406151929237, 0.06157684475093339])

# Add Annotations for v-large and v-small mw
# axl.scatter(-0.25,0.00,marker='*',edgecolors='dodgerblue',facecolors='none')
# axl.text(-0.25,-0.007,'$m_w$ < %.1f'%(min(mw_lims)),ha='center',va='top')
# axl.scatter(1.3,0.00,edgecolors='firebrick',s=2000/2.5,facecolors='none')
# axl.text(1.3,-0.02,'$m_w$ > %.1f'%(max(mw_lims)),ha='center',va='top')



################################################################################################
##### ITERATE ACROSS DESIRED FRAMES ############################################################
################################################################################################
framerange = np.arange(0,frame_count,1)
print('Starting render from frame %d/%d'%(framerange[0],frame_count))
for fn_ in tqdm(framerange):

	### DO ITERATION-SPECIFIC PROCESSING ###
	# Get frame time indices
	t0 = StartTime + fn_*Frame_STEP
	tf = StartTime + Frame_LENG + fn_*Frame_STEP

	### INDIVIDUAL ICEQUAKE PROCESSING ###
	# Subset Individual Icequakes
	idf_M0 = df_M0[(df_M0.index >= t0) & (df_M0.index < tf)]
	# Make subset indices for splitting out formatting
	IDX_M0s = idf_M0['SA_wt_mean']<mw_inv(min(mw_lims))
	IDX_M0m = (idf_M0['SA_wt_mean']>=mw_inv(min(mw_lims)))&(idf_M0['SA_wt_mean']<=mw_inv(max(mw_lims)))
	IDX_M0l = idf_M0['SA_wt_mean']>mw_inv(max(mw_lims))
	# Get opacity range for events based on age
	iM0_alpha = np.array(0.01 + 0.99*((idf_M0.index - t0)/Frame_LENG))

	### STATION PROCESSING ###
	# Subset Active Stations
	idf_STA = df_STA[(df_STA['Ondate'] < tf) & (df_STA['Offdate'] > t0)]
	# Get station deploy ages
	ages = []
	for S_ in idf_STA['Station']:
		idf_SURV = df_SURV[(df_SURV.index <= tf) & (df_SURV['sta']==S_)]
		if len(idf_SURV) > 0:
			tslast = (tf - idf_SURV.index[-1]).total_seconds()/(3600*24)
			Az2last = idf_SURV['Azimuth 2'].values[-1]
			valid_az = np.isfinite(Az2last)
			ages.append([S_,tslast,valid_az])
		else:
			ages.append([S_,0,False])
	idf_STA = pd.concat([idf_STA,\
						 pd.DataFrame(ages,\
						 			  columns=['sta_ck','age days','valid Az'],\
						 			  index=idf_STA.index)],\
						 axis=1,ignore_index=False)

	### TIME-SERIES PROCESSING ###
	# Subset binned time-series data
	idf_T = df_T[(df_T.index >= t0) & (df_T.index < tf)]
	# Subset points with event rates greater than DPT
	IDX_Tm = idf_T['M0 count'].values*4 > 100
	# Subset points with event rates \in [1,DPT]
	IDX_Ts = (idf_T['M0 count'].values*4 <= 100) & (idf_T['M0 count'].values*4 > 0)
	# Subset points with event rates = 0
	IDX_To = idf_T['M0 count'].values == 0

	# Subset path of last trailing 24 hours
	idf_T24 = df_T[(df_T.index >= tf - pd.Timedelta(24,unit='hour')) &\
				 (df_T.index < tf)]


	######################################################################################
	######## UPDATE FIGURES ##############################################################
	######################################################################################

	### UPDATE TIMESERIES SUBPLOT ###
	# Clear Time bar Overlay
	ax1.clear()
	### Plot new reference window
	ax1.fill_between([t0,tf],[0,0],[1,1],alpha=0.33,color='red')

	# Re-establish ax1 overlay formatting
	ax1.patch.set_alpha(0)
	for SL_ in ['top','bottom','right','left']:
		ax0.spines[SL_].set_visible(False)
	ax1.xaxis.set_visible(False)
	ax1.yaxis.set_visible(False)
	ax1.set_ylim([0,1])
	ax1.set_xlim([StartTime,EndTime])



	### UPDATE CROSS-PLOT ###

	# Clear cross-plot figure
	axx.clear()
	axx.set_aspect('auto','box')
	axx.set_xlabel('Meltwater supply rate [$\dot{S}_{w}$]\n($mm$ $w.e.$ $h^{-1}$)')
	axx.set_ylabel('Slip velocity [$V$] ($m$ $a^{-1}$)')
	axx.set_xlim([1.1,4])
	Vm = np.nanmin(df_T['V_H(cm d-1)'])
	VM = np.nanmax(df_T['V_H(cm d-1)'])
	axx.set_ylim([(Vm - 0.05*(VM - Vm))*3.6524 - 8.9,(VM + 0.15*(VM - Vm))*3.6524 - 8.9 + 30])
	
	## PLOT RETICLES ##
	# Get current EQ rate to dictate formatting
	Edotcurrent = idf_T24['M0 count'].values[-1]*4

	# Plot reticles
	axx.plot([1.1,4],np.ones(2)*idf_T24['V_H(cm d-1)'].values[-1]*3.6524 - 8.9,'r:',alpha=0.4)
	axx.plot(idf_T24['SR(mmWE h-1)'].values[-1]*np.ones(2),axx.get_ylim(),'r:',alpha=0.4)

	# Plot reticle edge markers
	# If there are no current icequakes, plot a pip at the base of each reticle
	if Edotcurrent == 0:
		# Marker for V X \dot{E} reticle
		axx.scatter(1.15,idf_T24['V_H(cm d-1)'].values[-1]*3.6524 - 8.9,alpha=ALP,\
				marker='.',c='red',linewidth=LW,zorder=10)
		# \dot{S}_w X \dot{E}
		axx.scatter(idf_T24['SR(mmWE h-1)'].values[-1],0,alpha=ALP,\
				marker='.',c='red',linewidth=LW,zorder=10)

	# If Icequake rate is in the small category, plot diamond
	elif Edotcurrent < 100:
		# V X \dot{E}
		axx.scatter(1.15,idf_T24['V_H(cm d-1)'].values[-1]*3.6524 - 8.9,alpha=ALP,\
				marker='d',c=mw(idf_T24['M0 median'].values[-1]),cmap=cmap,\
				edgecolor='red',linewidth=LW,zorder=10,vmin=min(mw_lims),vmax=max(mw_lims))
		# \dot{S}_w X \dot{E}
		axx.scatter(idf_T24['SR(mmWE h-1)'].values[-1],0,alpha=ALP,\
				marker='d',c=mw(idf_T24['M0 median'].values[-1]),cmap=cmap,\
				edgecolor='red',linewidth=LW,zorder=10,vmin=min(mw_lims),vmax=max(mw_lims))
	
	# Otherwise, plot appropriately scaled reticle
	else:
		# V X \dot{E}
		axx.scatter(1.15,idf_T24['V_H(cm d-1)'].values[-1]*3.6524 - 8.9,alpha=ALP,\
				s=idf_T24['M0 count'].values[-1]*(4/2.5),\
				c=mw(idf_T24['M0 median'].values[-1]),cmap=cmap,\
				edgecolor='red',linewidth=LW,zorder=10,vmin=min(mw_lims),vmax=max(mw_lims))
		# \dot{S}_w X \dot{E}
		axx.scatter(idf_T24['SR(mmWE h-1)'].values[-1],0,alpha=ALP,\
				s=idf_T24['M0 count'].values[-1]*(4/2.5),\
				c=mw(idf_T24['M0 median'].values[-1]),cmap=cmap,\
				edgecolor='red',linewidth=LW,zorder=10,vmin=min(mw_lims),vmax=max(mw_lims))


	## PLOT STRINGER OF DATA ##
	# Plot background trace
	axx.plot(idf_T24['SR(mmWE h-1)'],idf_T24['V_H(cm d-1)']*3.6524 - 8.9,'k-',alpha=0.25)

	# Plot zero markers
	if sum(IDX_To) > 0:
		axx.scatter(idf_T[IDX_To]['SR(mmWE h-1)'],idf_T[IDX_To]['V_H(cm d-1)']*3.6524 - 8.9,\
					marker='.',c='k',s=10,zorder=5)
	
	# Plot low event-rate diamonds
	if sum(IDX_Ts) > 0:
		axx.scatter(idf_T[IDX_Ts]['SR(mmWE h-1)'],idf_T[IDX_Ts]['V_H(cm d-1)']*3.6524 - 8.9,\
					c=mw(idf_T[IDX_Ts]['M0 median'].values),edgecolor='black',\
					s=50,marker='d',\
					alpha=ALP,cmap=cmap,\
					linewidth=LW,zorder=10,\
					vmin=min(mw_lims),vmax=max(mw_lims))

	# Plot high event-rate bubbles
	if sum(IDX_Tm) > 0:
		axx.scatter(idf_T[IDX_Tm]['SR(mmWE h-1)'],idf_T[IDX_Tm]['V_H(cm d-1)']*3.6524 - 8.9,\
					c=mw(idf_T[IDX_Tm]['M0 median'].values),edgecolor='black',\
					s=idf_T[IDX_Tm]['M0 count'].values*(4/2.5),\
					alpha=ALP,cmap=cmap,linewidth=LW,zorder=10,\
					vmin=min(mw_lims),vmax=max(mw_lims))
	
	axx.text(3.75,180,'B',fontsize=14,fontweight='extra bold',fontstyle='italic')




	##############################################################
	### UPDATE ICEQUAKE EVENT MAP ################################
	##############################################################
	# Clear map figure
	axm.clear()
	axm.set_aspect('equal','box')
	# Plot subglacial topography background
	axm.pcolor(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hm)*grd_M,cmap='gray',zorder=2)
	cl = axm.contour(grd_E - mE0,grd_N - mN0,(grd_Zs - grd_Hm)*grd_M,\
					 np.arange(1700,1900,10),colors='w',zorder=3,alpha=0.5)

	# Calculate mw
	lsize = 150
	msize = 50
	ssize = 30
	cvect = (2/3)*np.log10(idf_M0['SA_wt_mean'].values) - 6.03
	svect = ((msize-ssize)/(max(mw_lims) - min(mw_lims)))*(cvect - min(mw_lims)) + ssize
	# svect /= (max(mw_lims) - min(mw_lims))
	# svect += 10


	# Plot very large icequakes
	if sum(IDX_M0l) > 0:
		M0l = idf_M0[IDX_M0l]
		axm.scatter(M0l['mE'] - mE0,M0l['mN'] - mN0,\
					s=lsize,c='firebrick',facecolors='none',\
					marker='s',alpha=(M0l.index - t0)/Frame_LENG,\
					zorder=13)
		# except:
		# 	breakpoint()

	# Plot icequakes within m_w bounds
	if sum(IDX_M0m) > 0:
		# breakpoint()
		axm.scatter(idf_M0[IDX_M0m]['mE'] - mE0,idf_M0[IDX_M0m]['mN'] - mN0,\
					s=msize,c=cvect[IDX_M0m],\
					cmap=cmap,vmin=min(mw_lims),vmax=max(mw_lims),\
					alpha=iM0_alpha[IDX_M0m],\
					zorder=11)

	# Plot Small Icequakes
	if sum(IDX_M0s) > 0:
		axm.scatter(idf_M0[IDX_M0s]['mE'] - mE0,idf_M0[IDX_M0s]['mN'] - mN0,\
					s=ssize,marker='*',c='m',facecolors='none',\
					alpha=iM0_alpha[IDX_M0s],\
					zorder=12)

	# Maintain legend entries
	sbig = axm.scatter(0,0,s=lsize,edgecolors='firebrick',\
					marker='s',facecolors='none',\
					zorder=1,label='$m_w$ > %.1f'%(max(mw_lims)))
	sm0 = axm.scatter(0,0,s=msize,c=0,cmap=cmap,vmin=0,vmax=1,facecolors='none',\
					zorder=1,label='$m_w\\in$ [%.1f,%.1f] \n(see colorbar)'%(min(mw_lims),max(mw_lims)))
	sm1 = axm.scatter(0,0,s=msize,c=0.5,cmap=cmap,vmin=0,vmax=1,facecolors='none',\
					zorder=1,label='$m_w\\in$ [%.1f,%.1f]'%(min(mw_lims),max(mw_lims)))
	sm2 = axm.scatter(0,0,s=msize,c=1,cmap=cmap,vmin=0,vmax=1,facecolors='none',\
					zorder=1,label='$m_w\\in$ [%.1f,%.1f]'%(min(mw_lims),max(mw_lims)))
	ssm = axm.scatter(0,0,s=ssize,marker='*',edgecolors='m',facecolors='none',\
					zorder=1,label='$m_w$ < %.1f'%(min(mw_lims)))
	pnS1, = axm.plot(0,0,'v',markersize=6,mfc='cyan',mec='black',label='Age $\\leq$ 1 days',zorder=1)
	pnS2, = axm.plot(0,0,'v',markersize=6,mfc='yellow',mec='black',label='Age $\\in$ (1,2] days',zorder=1)
	pnS3, = axm.plot(0,0,'v',markersize=6,mfc='magenta',mec='black',label='Age $\\in$ (2,3] days',zorder=1)
	poS, = axm.plot(0,0,'v',markersize=6,mfc='darkred',mec='black',label='Age > 3 days',zorder=1)


	axm.legend([pnS1,pnS2,pnS3,poS,sbig,(sm0,sm1,sm2),ssm],\
			   ['Install age $\\leq$ 1 day','Install age $\\in$ (1,2] days','Install age $\\in$ (2,3] days','Install age > 3 days',\
			    '$m_w$ > %.1f'%(max(mw_lims)),'$m_w \\in$ [%.1f,%.1f]\n(see colorbar)'%(min(mw_lims),max(mw_lims)),'$m_w$ < %.1f'%(min(mw_lims))],\
				loc='lower center',bbox_to_anchor=(0.5,-0.25),ncol=2,handler_map={tuple:HandlerTuple(ndivide=None)})



	# # Plot Active Stations
	IDX_nS1 = (idf_STA['valid Az'])&(idf_STA['age days'] <= 1)
	IDX_nS2 = (idf_STA['valid Az'])&(idf_STA['age days'] <= 2)&(idf_STA['age days'] > 1)
	IDX_nS3 = (idf_STA['valid Az'])&(idf_STA['age days'] <= 3)&(idf_STA['age days'] > 2)

	IDX_oS = (idf_STA['valid Az'])&(idf_STA['age days'] > 3)
	if sum(IDX_nS1) > 0:
		axm.plot(idf_STA[IDX_nS1]['mE'] - mE0,idf_STA[IDX_nS1]['mN'] - mN0,'v',\
				 markersize=6,mfc='cyan',mec='black',zorder=50)

	if sum(IDX_nS2) > 0:
		axm.plot(idf_STA[IDX_nS2]['mE'] - mE0,idf_STA[IDX_nS2]['mN'] - mN0,'v',\
				 markersize=6,mfc='yellow',mec='black',zorder=50)

	if sum(IDX_nS3) > 0:
		axm.plot(idf_STA[IDX_nS3]['mE'] - mE0,idf_STA[IDX_nS3]['mN'] - mN0,'v',\
				 markersize=6,mfc='magenta',mec='black',zorder=50)



	if sum(IDX_oS) > 0:
		axm.plot(idf_STA[IDX_oS]['mE'] - mE0,idf_STA[IDX_oS]['mN'] - mN0,'v',\
				 markersize=6,mfc='darkred',mec='black',zorder=50)
		
	# Plot Scale Bar
	XSFT = -200
	axm.fill_between([mEM - mE0 - 300 + XSFT,mEM - mE0 - 100 + XSFT],\
					 [mNm - mN0,mNm - mN0],[mNm - mN0 + 20,\
					 mNm - mN0 + 20],facecolor='k')
	axm.fill_between([mEM - mE0 - 199 + XSFT,mEM - mE0 - 102 + XSFT],\
					 [mNm - mN0 + 3,mNm - mN0 + 3],\
					 [mNm - mN0 + 17,mNm - mN0 + 17],\
					 facecolor='white')
	axm.text(mEM - mE0 - 300 + XSFT,mNm - mN0 + 25,0,ha='center')
	axm.text(mEM - mE0 - 100 + XSFT,mNm - mN0 + 25,200,ha='center')
	


	## Annotate map with:
	# Station count, and those providiing M_0 observations
	if len(idf_M0) > 0:
		axm.text(1008,0,'Station ct.:%d\n($M_0$ observing: %d)\nIcequake ct.:%d\n$m_w$ max: %.02f\n$\\tilde{m_w}$: %.02f\n$m_w$ min: %.02f'%\
				(len(idf_STA),sum(IDX_nS1) + sum(IDX_nS2) + sum(IDX_nS3),len(idf_M0),mw(np.nanmax(idf_M0['SA_wt_mean'].values)),\
					mw(np.nanmedian(idf_M0['SA_wt_mean'].values)),mw(np.nanmin(idf_M0['SA_wt_mean'].values))),\
				ha='left',va='bottom')
	else:
		axm.text(1008,0,'Station ct.:%d\n($M_0$ observing: %d)\nIcequake ct.:%d\n$m_w$ max: - \n$\\tilde{m_w}$: - \n$m_w$ min: -'%\
				(len(idf_STA),sum(IDX_nS1) + sum(IDX_nS2) + sum(IDX_nS3),len(idf_M0)),\
				ha='left',va='bottom')
	# axm.text(mEM - mE0 - 300,mNm - mN0 + 65,'Average $\\dot{E}$:%d $h^{-1}$'%(len(idf_M0)/((tf - t0).total_seconds()/3600)))
	
	# Do Formatting
	plt.setp(axm.get_xticklabels(),visible=False)
	plt.setp(axm.get_yticklabels(),visible=False)
	axm.tick_params(axis='both',which='both',length=0)

	
	# Plot Flow direction
	axm.arrow(600,500,150,160,width=10)
	axm.text(675,580,'Flow\n\ndirection',fontsize=8,rotation=45,ha='center',va='center')
	axm.text(600,700,'C',fontsize=14,fontweight='extra bold',fontstyle='italic')
	axm.set_xlim([mEm - 0.05*(mEM - mEm) - mE0,mEM + 0.05*(mEM - mEm) - mE0])
	axm.set_ylim([mNm - 0.05*(mNM - mNm) - mN0,mNM + 0.05*(mNM - mNm) - mN0])

	ax1.set_title('%s -- %s'%(str(t0),str(tf)))



	# breakpoint()

	# plt.show()
	# breakpoint()

	if issave:
		try:
			I_OFILE = '%04d__Mov1__%ddpi.%s'%(fn_,DPI,FMT.lower())
			plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)
		except:
			breakpoint()
	# plt.show()
	# plt.show()
	# breakpoint()

	# # Do Colorbar Rendering
	# if I_ < 3:
	# 	ceax = fig.add_axes([.15,.075,.3,.02])
	# 	cbax = fig.add_axes([.55,.075,.3,.02])
	# elif I_ == 3:
	# 	ceax = fig.add_axes([.15,.375,.3,.02])
	# 	cbax = fig.add_axes([.55,.375,.3,.02])
	# cbar_b = plt.colorbar(cbh,cax=cbax,orientation='horizontal',ticks=[-conmax,0.0,conmax])
	# cbar_b.ax.set_xticklabels(['Convex\n(likely rock)','Curvature\n(interpretation)','Concave\n(likely till)'])
	
	# cbar_e = plt.colorbar(ceh,cax=ceax,orientation='horizontal',ticks=[0,1,2,3,4,5,6])
	# cbar_e.ax.set_xticklabels(['0 h','1 h','2 h','3 h\nRelative origin time','4 h','5 h','6 h'])

	# Remove whitespace
	# plt.subplots_adjust(wspace=0,hspace=0)


	# if fn_ == 3:
	# 	plt.show()


# 	if issave:
# 		I_OFILE = 'SGEQ_v16_FigS3_DiurnalSeismicMapCatalog_Page%d_%ddpi.%s'%(I_,DPI,FMT.lower())
# 		plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)



# plt.show()

