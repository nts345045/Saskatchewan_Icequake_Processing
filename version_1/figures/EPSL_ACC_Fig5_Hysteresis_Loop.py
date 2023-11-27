import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from copy import deepcopy
from pyproj import Proj

### Supporting Cutsom Plotting Routine ###

def compound_scalebar(ax,ssteps=6,srange=[0,4000],sscalar=1,sslbl='1-100',slbl='Event rate (ct. $h^{-1}$)',csteps=3,crange=[-3.25,-2.25],clbl='Median $m_w$',cmap='viridis',linewidth=2,alpha=0.75):
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

###################################################################################

### MAP DATA FILES ###
ROOT = os.path.join('..','..')
# Path to time-series file
TSFL = os.path.join(ROOT,'data','timeseries','ERV_0.25hr_G_filt_resampled_UTC_merr.csv')
# Bed Map Directory
BMDR = os.path.join(ROOT,'data','seismic','HVSR','grids')
# Icequake hypocenter catalog location
EQFL = os.path.join(ROOT,'data','seismic','well_located_30m_basal_catalog_merr_dupQCd.csv')
# Station locations
STFL = os.path.join(ROOT,'data','seismic','Stations.csv')
# Get filtered moment estimates
M0FL = os.path.join(ROOT,'data','M0_results','M0_stats_SNR_ge5_ND_ge8_MA_le2days.csv')


### PLOTTING CONTROLS ###
DPI = 300
FMT = 'PNG'
issave = True

# Figure output directory
ODIR = os.path.join(ROOT,'outputs')

### READ DATA ###
# Read in time-series data
df_T = pd.read_csv(TSFL,parse_dates=True,index_col=[0])
## Import Station and Basal Event Location/Magnitude Files
df_S = pd.read_csv(STFL)
df_EQ = pd.read_csv(EQFL,parse_dates=['t0'],index_col=['t0'])
# Filtered M0 event estimates
df_M = pd.read_csv(M0FL,parse_dates=['t0'],index_col='t0')
df_M = df_M.sort_index()

## Import Grids
grd_E = np.load(os.path.join(BMDR,'mE_Grid.npy'))
grd_N = np.load(os.path.join(BMDR,'mN_Grid.npy'))
grd_Hm = np.load(os.path.join(BMDR,'H_med_Grid.npy'))
grd_Hu = np.load(os.path.join(BMDR,'H_gau_u_Grid.npy'))
grd_Ho = np.load(os.path.join(BMDR,'H_gau_o_Grid.npy'))
grd_Zs = np.load(os.path.join(BMDR,'mZsurf_Grid.npy'))
grd_M = np.load(os.path.join(BMDR,'MASK_Grid.npy'))

### TIME (ZONE) (RE)FORMATTING ###
# Define Local Solar Times
TZ0 = pd.Timedelta(-6,unit='hour') 					# UTC Time Offset
SNt = pd.Timedelta(-1,unit='hour') 					# Solar Noontime
CTt0 = pd.Timestamp('2019-07-31T22:06:00') + SNt    # Civil Twilight reference time 1
CTt1 = pd.Timestamp('2019-08-01T05:51:00') + SNt 	# Civil Twilight reference time 2
SNo = pd.Timestamp('2019-07-31T13:48:00') + SNt
DT = pd.Timedelta(15,unit='min') 					# Resampling period for M0 data

## Apply Time Shifts ##
# Update time-series index to UTC-7
df_T.index += (TZ0 + SNt)
df_T = df_T.interpolate(direction='both',limit=3)
# Update origin times to UTC-7
df_EQ.index += (TZ0 + SNt)
df_EQ = df_EQ.sort_index()
# Update station ON/OFF times to UTC-7
df_S['Ondate'] = pd.to_datetime(df_S['Ondate'],unit='s') + TZ0 + SNt
df_S['Offdate'] = pd.to_datetime(df_S['Offdate'],unit='s') + TZ0 + SNt
df_S.index = df_S['Ondate']
df_S = df_S.sort_index()

# Do time shift of M0 timestamps
df_M.index += (TZ0 + SNt)

## RESAMPLE M0 DATA TO GET BINNED STATISTICS##
# Do binning/resampling of M0 results
df_Mi = pd.concat([pd.DataFrame(index=[df_T.index[0]]),df_M.copy()],axis=0,ignore_index=False)
# Get median values of binned values
df_bM = df_Mi.resample(DT).median()
# Get count of events with M0 in each bin
df_bC = df_Mi.resample(DT).count()
# Get MinMax
df_b0 = df_Mi.resample(DT).apply(np.nanmin)#quantile(q=0.025)
df_b1 = df_Mi.resample(DT).apply(np.nanmax)#quantile(q=0.975)


## GEOGRAPHIC PROCESSING SECTION
## Convert Station Locations to UTM
myproj = Proj(proj='utm',zone=11,ellipse='WGS84',preserve_units=False)
SmE,SmN = myproj(df_S['Lon'].values,df_S['Lat'].values)
df_S = pd.concat([df_S,pd.DataFrame({'mE':SmE,'mN':SmN},index=df_S.index)],axis=1,ignore_index=False)

# Get bounds
mE0 = 4.87e5
mN0 = 5.777e6
mEm = np.min(grd_E)
mEM = np.max(grd_E)
mNm = np.min(grd_N)
mNM = np.max(grd_N)


### INITIALIZE FIGURE ###
fig = plt.figure(figsize=(7.5,7.5*(4/3)))
gs = GridSpec(nrows=8,ncols=3)
AX = {}
## Create Axis for Crossplot
AX.update({'X':fig.add_subplot(gs[1:7,1:])})
## Create Axis for Common Legend
# AX.update({'L':fig.add_subplot(gs[3,0:2])})
AX.update({'L':fig.add_subplot(gs[1,1])})
## Create Axes for Seismicity Maps
for i_,l_ in enumerate(['A','B','C','D']):
	AX.update({l_:fig.add_subplot(gs[i_*2:(i_+1)*2,0])})

# AX.update({'A':fig.add_subplot(gs[0,0]),'B':fig.add_subplot(gs[0,2]),\
		   # 'C':fig.add_subplot(gs[1,2]),'D':fig.add_subplot(gs[2,2])})

mw_lims = [-3.5,-2]
LW = 1
ALP = 0.75
# DPT = 100 # Diamond point threshold value (less than DPT ct. h-1 ice-quakes? make it a diamond!)
cmap = 'tab20c'#'terrain'#'nipy_spectral'
ECV = ['black','red','dodgerblue','darkorange']
# T0 = pd.Timestamp("2019-08-08T05")
# T1 = T0 + pd.Timedelta(6,unit='hour')
TIMES = [('A',pd.Timestamp("2019-08-08T04"), pd.Timestamp("2019-08-08T10")),\
		 ('B',pd.Timestamp("2019-08-08T10"), pd.Timestamp("2019-08-08T16")),\
		 ('C',pd.Timestamp("2019-08-08T16"), pd.Timestamp("2019-08-08T22")),\
		 ('D',pd.Timestamp("2019-08-08T22"), pd.Timestamp("2019-08-09T04"))]

To,Tf = TIMES[0][1],TIMES[-1][-1]
# idf_T = df_T[(df_T.index >= To) & (df_T.index < Tf)]
# # Plot background line in crossplot
# AX['X'].plot(idf_T['SR(mmWE h-1)'],idf_T['V_H(cm d-1)']*3.6524 - 8.9,'k-',zorder=1)
# Get maximum event rate 
ratemax = np.nanmax(df_bC[(df_bC.index >= To) & (df_bC.index <= Tf)]['median'].values)*4.
M0o = np.nanmin(df_M[(df_M.index >= To) & (df_M.index <= Tf)]['median'].values)*0.9
for i_,T_ in enumerate(TIMES):
	# Pull bounding times
	L_,T0,T1 = T_

	# Subset Active Stations
	idf_S = df_S[(df_S['Ondate'] <= T1) & (df_S['Offdate'] >= T0)]

	# Subset timeseries
	idf_T = df_T[(df_T.index >= T0)&(df_T.index <= T1)]

	# Subset individual events with M0 estimates
	idf_M = df_M[(df_M.index >= T0) & (df_M.index <= T1)]

	## Subset Binned M0 statistics
	# Binned median
	idf_bM = df_bM[(df_bM.index >= T0) & (df_bM.index <= T1)]
	# Binned minimum
	idf_b0 = df_b0[(df_b0.index >= T0) & (df_b0.index <= T1)]
	# Binned maximum
	idf_b1 = df_b1[(df_b1.index >= T0) & (df_b1.index <= T1)]
	# Binned count
	idf_bC = df_bC[(df_bC.index >= T0) & (df_bC.index <= T1)]
	# Big event count
	IDX_BE = (idf_bC['median']>25)
	# Small event count
	IDX_SE = (idf_bC['median']<=25)&(idf_bC['median']>0)
	# No-event count
	IDX_NE = (idf_bC['median']==0)


	# Subset Icequake Locations w/o M0
	idf_EQ = df_EQ[(df_EQ.index >= T0) & (df_EQ.index < T1) & df_EQ['G_filt']]
	idf_EQ = idf_EQ[~idf_EQ.index.isin(idf_M.index)]


	#############################
	## ALIAS DATA FOR PLOTTING ##
	#############################
	## CROSS PLOT DATA ##########
	#############################
	# X-coordinates (meltwater supply during high rate seismic periods)
	XX = idf_T[IDX_BE]['SR(mmWE h-1)'].values
	# X-coordinates (meltwater supply during low rate seismic periods)
	Xx = idf_T[IDX_SE]['SR(mmWE h-1)'].values
	# X-coordinates (meltwater supply during aseismic periods)
	xx = idf_T[IDX_NE]['SR(mmWE h-1)'].values
	# Y-coordinates (slip velocity during seismic periods)
	XY = idf_T[IDX_BE]['V_H(cm d-1)'].values*3.6524 - 8.9
	# Y-coordinates (slip velocity for aseismic periods)
	Xy = idf_T[IDX_SE]['V_H(cm d-1)'].values*3.6524 - 8.9
	# Y-coordinates (slip velocity for aseismic periods)
	xy = idf_T[IDX_NE]['V_H(cm d-1)'].values*3.6524 - 8.9
	# Color coordinates (median binned M0)
	XC = (2/3)*np.log10(idf_bM[IDX_BE]['median'].values) - 6.07# - M0o)
	Xc = (2/3)*np.log10(idf_bM[IDX_SE]['median'].values) - 6.07
	# # Size coordinates (maximum binned M0)
	# XSO = np.sqrt(idf_b1[~IDX_NE]['median'].values - M0o)/5
	# Size coordinates (event count per hour = rate)
	XS = idf_bC[IDX_BE]['median'].values*4/2.5
	Xs = 16
	xs = 1
	
	#################################
	## MAP PLOT DATA ################
	#################################
	LX = idf_M['mE'].values - mE0
	# Lx = idf_EQ
	lx = idf_EQ[~(idf_EQ.index.isin(idf_M))]['mE'].values - mE0
	LY = idf_M['mN'].values - mN0
	ly = idf_EQ[~(idf_EQ.index.isin(idf_M))]['mN'].values - mN0
	LS = np.ones(len(LX))*8
	LC = (2/3)*np.log10(idf_M['median'].values) - 6.07# - M0o)/5

	# Sort datapoints by ascending magnitude for rendering order
	RIND = pd.Series(LC,name='mw').sort_values().index
	LXR = [LX[i_] for i_ in RIND]
	LYR = [LY[i_] for i_ in RIND]
	LSR = [LS[i_] for i_ in RIND]
	LCR = [LC[i_] for i_ in RIND]


	#################################
	#### PLOT CROSSPLOT ELEMENTS ####
	#################################
	# Plot line segments
	AX['X'].plot(idf_T['SR(mmWE h-1)'],idf_T['V_H(cm d-1)']*3.6524 - 8.9,'-',color=ECV[i_],zorder=1)
	# Plot truly aseismic markers
	if len(xx) == len(xy) > 0:
		AX['X'].scatter(xx,xy,marker='.',s=1,c=ECV[i_],zorder=2)
	# Plot seismic markers w/o M0 estimates (if valid)
	if len(Xx) == len(Xy) > 0:
		AX['X'].scatter(Xx,Xy,c=Xc,s=50,edgecolor=ECV[i_],\
						marker='d',cmap=cmap,alpha=ALP,\
						vmin=mw_lims[0],vmax=mw_lims[1],\
						zorder=3)
	# Plot seismic markers for binned median M0
	if len(XX) == len(XY) > 0:
		AX['X'].scatter(XX,XY,c=XC,s=XS,edgecolor=ECV[i_],\
						cmap=cmap,alpha=ALP,\
						vmin=mw_lims[0],vmax=mw_lims[1],\
						zorder=4)
	# # Plot seismic markers for binned maximum M0
	# AX['X'].scatter(XX,XY,s=XSO,edgecolor=ECV[i_],facecolors='none',linestyle=':',zorder=5)




	#### PLOT EVENT SUB-MAP ####
	# Plot bed topography background
	cbh = AX[L_].pcolor(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hu)*grd_M,cmap='gray',zorder=1)#,alpha=0.5)
	cbh.set_clim([1800,1880])
	AX[L_].contour(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hu)*grd_M,np.arange(1800,1900,20),colors='w',zorder=2,alpha=0.5)

	# Plot Icequakes with M0
	ceh = AX[L_].scatter(LXR,LYR,s=(np.array(LCR)+4.75)**4,c=LCR,alpha=np.array(LCR)/-4.5,cmap=cmap,zorder=3,vmin=mw_lims[0],vmax=mw_lims[1])#,vmin=0,vmax=(T1 - T0).total_seconds())#,edgecolor=ECV[i_])
	# # Plot Icequakes without M0
	# AX[L_].scatter(lx,ly,marker='d',alpha=0.25,cmap=cmap,zorder=4,vmin=mw_lims[0],vmax=mw_lims[1])#,vmin=0,vmax=(T1 - T0).total_seconds(),edgecolor='k')

	# Plot Active Stations
	AX[L_].plot(idf_S['mE'] - mE0,idf_S['mN'] - mN0,'kv',markersize=3,zorder=5)
	# Do Formatting
	plt.setp(AX[L_].get_xticklabels(),visible=False)
	plt.setp(AX[L_].get_yticklabels(),visible=False)
	AX[L_].tick_params(axis='both',which='both',length=0)
	AX[L_].yaxis.set_label_position("left")
	AX[L_].set_ylabel('%02d:00-%02d:00'%(T0.hour,T1.hour), rotation=270,labelpad=15)
	xlims = AX[L_].get_xlim()
	ylims = AX[L_].get_ylim()
	AX[L_].plot(xlims[0],ylims[1],'o',mec=ECV[i_],ms=16,mfc="none")

	AX[L_].text(xlims[0],ylims[1],T_[0].upper(),ha='center',va='center',\
				fontweight='extra bold',fontstyle='italic',fontsize=14)

	# RATE = len(idf_EQ) / ((T_[-1] - T_[-2]).total_seconds()/3600)
	# AX[L_].text(xlims[1],ylims[0],'ct.: %d\n$\dot{E}$: %d ct. $h^{-1}$'%\
	# 							  (len(LX),len(LX)/((T1 - T0).total_seconds()/3600)),ha='right',va='bottom')
	
	# Plot mw statistics
	AX[L_].text(xlims[1]+20,ylims[0]-20,\
				'ct.: %d\n$m_w$ max: %.2f\n$m_w$ med: %.2f'%\
				(len(LX),LC.max(),np.nanmedian(LC)),\
				ha='right',va='bottom')

	# Fix spatial limits for viewing
	AX[L_].set_xlim([xlims[0]-50,xlims[1]+50])
	AX[L_].set_ylim([ylims[0]-50,ylims[1]+50])
	# Plot Scale Bar
	AX[L_].fill_between([1050,1150],[650,650],[675,675],color='k',ec='k')
	AX[L_].fill_between([1150,1250],[650,650],[675,675],color='w',ec='k')
	AX[L_].text(1050,680,'0 m',ha='center',fontsize=8)
	AX[L_].text(1250,680,'200',ha='center',fontsize=8)
	# Plot Flow direction
	AX[L_].arrow(600,500,150,160,width=10)
	AX[L_].text(675,580,'Flow\n\ndirection',fontsize=8,rotation=45,ha='center',va='center')


# Create formatted compound scalebar
compound_scalebar(AX['L'],ssteps=3,srange=[0,2000],sscalar=1/2.5,csteps=4,crange=mw_lims,linewidth=LW,alpha=ALP,cmap=cmap)
AX['L'].set_xlim([-0.166,1.153])
AX['L'].set_ylim([-0.02,0.057])

# Plot Axis Labels for Crossplot
AX['X'].set_xlim([1.3,3.6])

## Plot Labels on Crossplot
AX['X'].text(1.8,70,'Phase 1\n(A)',ha='center')
AX['X'].text(3.4,130,'Phase 2\n(B)',ha='center')
AX['X'].text(3.2,80,'Phase 3\n(C)',ha='center')
AX['X'].text(2.1,35,'Phase 4\n(D)',ha='center')
AX['X'].set_xlabel('Meltwater supply rate ($mm$ $w.e.$ $h^{-1}$)')
AX['X'].yaxis.set_label_position('right')
AX['X'].yaxis.set_ticks_position('right')
AX['X'].set_ylabel('Slip velocity ($m$ $a^{-1}$)',rotation=270,labelpad=15)
plt.subplots_adjust(wspace=0,hspace=0)

if issave:
	I_OFILE = 'EPSL_REV1_Fig6_Hysteresis_Loop_and_Maps_%ddpi.%s'%(DPI,FMT.lower())
	plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)
	print('Figure saved: %s'%(os.path.join(ODIR,I_OFILE)))

plt.show()