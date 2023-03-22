import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from pyproj import Proj
from tqdm import tqdm

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


### RENDERING CONTROLS & OUTPUT DIRECTORY ###
ROOT = os.path.join('..','..','..','..')
# Write Controls
ODIR = os.path.join(ROOT,'results','figures','Manuscript_Figures','v25_render','MOV')
FMT='PNG'
issave = True
# Rendering controls
cmap = 'viridis'
DPI = 100
eqalpha=0.5
Frame_STEP = pd.Timedelta(10,unit='min') # How much to advance times between frames
Frame_LENG = pd.Timedelta(2,unit='hour') # How long the viewing window should be
StartTime = pd.Timestamp("2019-08-04T20:00:00") # 
EndTime = pd.Timestamp("2019-08-19T20:00:00")
frame_count = 'auto'#90 #'auto'
eqscale = 'newness'
### CALCULATE NUMBER OF FRAMES IF 'auto' ###
if frame_count == 'auto':
	frame_count = 0
	its = StartTime + Frame_LENG
	while its < EndTime:
		its += Frame_STEP
		frame_count += 1
else:
	print('Manually set frame count: %d'%(frame_count)) 


### MAP DATA SOURCES ###

# Path to time-series data (evenly sampled)
TSFL = os.path.join(ROOT,'results','data_files','ERV_0.25hr_G_filt_resampled_UTC_merr.csv')
# Path to station SITE table
STFL = os.path.join(ROOT,'data','seismic','sites','Stations.csv')
# Bed Map Directory
BMDR = os.path.join(ROOT,'processed_data','seismic','HV','grids')
# Moulin location directory
MLDR = os.path.join(ROOT,'processed_data','hydro')
# Filtered earthquake catalog file
EQFL = os.path.join(ROOT,'processed_data','seismic','basal_catalog','well_located_30m_basal_catalog_merr.csv')
# Phase Instruction File
PFIL = './times_and_phases.csv'


############################################################################################################
######### LOADING SECTION ##################################################################################
############################################################################################################

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


## Import Moulin, Station, and Basal Event Location Files
# Evenly sampled time-series data
df_T = pd.read_csv(TSFL,parse_dates=True,index_col=[0])
df_M = pd.read_csv(os.path.join(MLDR,'Moulin_Locations_Cleaned.csv'),parse_dates=True,index_col=['ID'])
df_S = pd.read_csv(os.path.join(ROOT,'data','seismic','sites','Stations.csv'))
df_EQ = pd.read_csv(EQFL,parse_dates=['t0'],index_col=['t0'])

# Define Local Solar Times
TZ0 = pd.Timedelta(-6,unit='hour') 					# UTC Time Offset
SNt = pd.Timedelta(-1,unit='hour') 					# Solar Noontime
CTt0 = pd.Timestamp('2019-07-31T22:06:00') + SNt    # Civil Twilight reference time 1
CTt1 = pd.Timestamp('2019-08-01T05:51:00') + SNt 	# Civil Twilight reference time 2
SNo = pd.Timestamp('2019-07-31T13:48:00') + SNt

# Adjust time-series to local solar time
df_T.index += (TZ0 + SNt)
# patch small holes
df_T = df_T.interpolate(direction='both',limit=2)

# Shift earthquake origin times to local solar time
df_EQ.index += (TZ0 + SNt)
df_EQ = df_EQ.sort_index()
# Shift station on/off dates to local solar time
df_S['Ondate'] = pd.to_datetime(df_S['Ondate'],unit='s') + TZ0 + SNt
df_S['Offdate'] = pd.to_datetime(df_S['Offdate'],unit='s') + TZ0 + SNt
df_S.index = df_S['Ondate']
df_S = df_S.sort_index()

## Convert Moulin Locations to UTM
myproj = Proj(proj='utm',zone=11,ellipse='WGS84',preserve_units=False)
SmE,SmN = myproj(df_S['Lon'].values,df_S['Lat'].values)
df_S = pd.concat([df_S,pd.DataFrame({'mE':SmE,'mN':SmN},index=df_S.index)],axis=1,ignore_index=False)

## Import Phase Instructions 
df_p = pd.read_csv(PFIL,parse_dates=['T0','T1'],index_col=[0])



############################################################################################################
######### PLOTTING SECTION #################################################################################
############################################################################################################
### INITIALIZE PLOT ###
fig = plt.figure(figsize=(8,10))
axs = []
gs = GridSpec(ncols=1,nrows=8)

### CREATE AXIS FOR TIMESERIES DATA BACKGROUND ##
ax0 = fig.add_subplot(gs[-3:])
### CREATE AXES FOR TIMESERIES DATA ###
for i_ in range(3):
	iax = fig.add_subplot(gs[-3+i_])
	if i_ < 2:
		iax.xaxis.set_visible(False)
		iax.spines['bottom'].set_visible(False)
	iax.set_xlim([StartTime,EndTime])
	axs.append(iax)
### CREATE AXIS FOR  TIMESERIES DATA OVERLAY ##
ax1 = fig.add_subplot(gs[-3:])

### CREATE AXIS FOR MAP ###
axm = fig.add_subplot(gs[:-3])




### FORMAT UNDERLAY FOR TIME SERIES ###
ax0.patch.set_alpha(0)
for SL_ in ['top','bottom','right','left']:
	ax0.spines[SL_].set_visible(False)
ax0.xaxis.set_visible(False)
ax0.yaxis.set_visible(False)
ax0.set_ylim([0,1])

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
idf_S = df_S[(df_S['Offdate'].values - df_S['Ondate'].values) > pd.Timedelta(4,unit='day')]
st_alpha = 0.05; st_col='k'
for i_ in range(len(idf_S)):
	ax0.fill_between([pd.Timestamp("2019-08-01"),idf_S['Ondate'].values[i_]],\
						 np.zeros(2),np.ones(2),color=st_col,alpha=st_alpha)
	ax0.fill_between([idf_S['Offdate'].values[i_],pd.Timestamp("2019-08-20T12:00:00")],\
						 np.zeros(2),np.ones(2),color=st_col,alpha=st_alpha)


period_times = [pd.Timestamp('2019-08-05T04'),pd.Timestamp("2019-08-09T04"),\
				pd.Timestamp("2019-08-14T04"),pd.Timestamp("2019-08-19T12")]

for i_ in range(3):
	ax0.plot([period_times[i_+1],period_times[i_+1]],\
				 [0,1],'k-.',linewidth=2)
	ts_u = period_times[i_] + (period_times[i_+1] - period_times[i_])/2
	ax0.text(ts_u,0.33,'Period %d'%(i_+1),ha='center',va='center',\
				 fontsize=10)

y_coords = np.linspace(1,-.05,4)
for i_,l_ in enumerate(['EQ rate','Melt','Velocity']):
	ax0.text(pd.Timestamp("2019-08-05T06"),y_coords[i_]-0.025,l_,\
				 fontsize=10,fontweight='extra bold',fontstyle='italic',\
				 va = 'top')

ax0.set_xlim([StartTime,EndTime])

### PLOT DATA ###
I_ = 0
## BASAL EVENT RATE PLOTTING ##
axs[I_].patch.set_alpha(0)
axs[I_].plot(df_T['ER(N h-1)']/1e3,'k',label='Measurements')
# axs[I_].plot(df_T24M['ER(N h-1)']/1e3,'r',label='24-hour maximum')
axs[I_].set_ylabel('$ct.$ $h^{-1}$\n(x1000)')
axs[I_].set_ylim([-0.1,4.5])
axs[I_].set_yticks([0,2,4])

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


### FORMAT AXES FOR MOVING WINDOW OVERLAY ##

ax1.patch.set_alpha(0)
for SL_ in ['top','bottom','right','left']:
	ax0.spines[SL_].set_visible(False)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_ylim([0,1])
ax1.set_xlim([StartTime,EndTime])






for fn_ in tqdm(range(frame_count)):
	# Get frame time indices
	t0 = StartTime + fn_*Frame_STEP
	tf = StartTime + Frame_LENG + fn_*Frame_STEP
	# Subset Icequakes
	idf_EQ = df_EQ[(df_EQ.index >= t0) & (df_EQ.index < tf) & df_EQ['G_filt']]
	# Subset Active Stations
	idf_S = df_S[(df_S['Ondate'] <= tf) & (df_S['Offdate'] >= t0)]


	# Clear Time bar Overlay
	ax1.clear()
	ax1.patch.set_alpha(0)
	for SL_ in ['top','bottom','right','left']:
		ax0.spines[SL_].set_visible(False)
	ax1.xaxis.set_visible(False)
	ax1.yaxis.set_visible(False)
	ax1.set_ylim([0,1])
	ax1.set_xlim([StartTime,EndTime])
	# Plot new  reference window
	ax1.fill_between([t0,tf],[0,0],[1,1],alpha=0.33,color='red')
	# # Gradient fill_between from: https://stackoverflow.com/questions/68002782/pyplot-fill-between-gradient
	# poly = ax1.fill_between([t0,tf],[0,0],[1,1],alpha=0.33)
	# verts = np.vstack([p.vertices for p in poly.get_paths()])
	# gradient = plt.imshow(np.linspace(0,1,256).reshape(1,-1),cmap=cmap,aspect='auto',\
	# 					  extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
	# gradient.set_clip_path(poly.get_paths()[0], transform=plt.gca().transData)
	ax1.set_ylim([0,1])
	ax1.set_xlim([StartTime,EndTime])


	# Clear map figure
	axm.clear()
	axm.set_aspect('equal','box')

	# Plot subglacial topography background

	# Plot concavity background (negative signed for figure rendering purposes only)
	axm.pcolor(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hm)*grd_M,cmap='gray',zorder=1)
	cl = axm.contour(grd_E - mE0,grd_N - mN0,(grd_Zs - grd_Hm)*grd_M,np.arange(1700,1900,10),colors='w',zorder=10,alpha=0.5)

	## GENERATE ICEQUAKE SCALING CRITERION
	if eqscale == 'newness':
		svect = 20*(idf_EQ.index - t0)/Frame_LENG + 1
	elif eqscale == 'nass':
		svect = idf_EQ['nphz']

	# Plot Icequakes
	ceh = axm.scatter(idf_EQ['mE']-mE0,idf_EQ['mN']-mN0,s=svect,c=(idf_EQ.index - t0).total_seconds()/3600,\
			   vmin=0,vmax=(tf - t0).total_seconds()/3600,cmap=cmap,alpha=eqalpha,zorder=1000)
	# Plot Active Stations
	axm.plot(idf_S['mE'] - mE0,idf_S['mN'] - mN0,'kv',markersize=3)
	# Plot Scale Bar
	axm.fill_between([mEM - mE0 - 300,mEM - mE0 - 100],[mNm - mN0,mNm - mN0],[mNm - mN0 + 20,mNm - mN0 + 20],facecolor='k')
	axm.fill_between([mEM - mE0 - 199,mEM - mE0 - 102],[mNm - mN0 + 1,mNm - mN0 + 1],[mNm - mN0 + 19,mNm - mN0 + 19],facecolor='white')
	
	axm.text(mEM - mE0 - 300,mNm - mN0 + 25,0,ha='center')
	axm.text(mEM - mE0 - 100,mNm - mN0 + 25,200,ha='center')
	# Print event count
	axm.text(mEM - mE0 - 300,mNm - mN0 + 100,'ct.:%d'%(len(idf_EQ)))

	# Do Formatting
	plt.setp(axm.get_xticklabels(),visible=False)
	plt.setp(axm.get_yticklabels(),visible=False)
	axm.tick_params(axis='both',which='both',length=0)


	axm.set_xlim([mEm - 0.05*(mEM - mEm) - mE0,mEM + 0.05*(mEM - mEm) - mE0])
	axm.set_ylim([mNm - 0.05*(mNM - mNm) - mN0,mNM + 0.05*(mNM - mNm) - mN0])

	axm.set_title('%s -- %s'%(str(t0),str(tf)))

	if issave:
		I_OFILE = '%04d__MovS1__%ddpi.%s'%(fn_,DPI,FMT.lower())
		plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)

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

