import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from copy import deepcopy
from pyproj import Proj

def compound_scalebar(ax,ssteps=6,DT=6,cmap='viridis',niter=4,colors=['black','red','blue','orange'],T0=4,linewidth=2,alpha=0.75):
	## Create Size Indexing
	slv = np.linspace(0,1,ssteps)
	sll = np.linspace(0,4000,ssteps)
	ax.scatter(np.linspace(0,1,101),np.zeros(101),s=np.linspace(0,4000/5,101),\
            c=np.linspace(0,1,101),cmap=cmap)


	ax.scatter(slv,np.zeros(ssteps),s=np.linspace(0,4000/5,ssteps),\
	            facecolors='none',edgecolors='k',linewidth=linewidth)

	ax.scatter(25/4000,0,s=50,edgecolors='k',facecolors='none',marker='d',linewidth=linewidth)

	for s_ in range(ssteps):
	    if s_ == 0:
	        ax.text(25/4000,0.02,'1-100',ha='center')
	    else:
	        text = ax.text(slv[s_],0.02,'%d'%(sll[s_]),ha='center')    
	    #text.set_path_effects([path_effects.Stroke(linewidth=1,foreground='black'),path_effects.Normal()])
	ax.text(np.mean(slv),0.0325,'Ice-quake rate [$\dot{E}$] (ct. $h^{-1}$)',ha='center')

	## Create Color Indexing
	if DT is None:
		ax.text(0.5,-0.0325,'Relative time in phase',color='red')
	else:
		slv = np.linspace(0,1,ssteps)
		sll = np.linspace(0,DT,ssteps)
		icmap = cm.get_cmap(cmap)

		for s_ in range(ssteps):
		    text = ax.text(slv[s_],-0.02,'%d'%(sll[s_]),color=icmap((slv[s_] - slv[0])/(slv[-1] - slv[0])),ha='center',va='top')    
		    text.set_path_effects([path_effects.Stroke(linewidth=1,foreground='black'),path_effects.Normal()])
		ax.text(np.mean(slv),-0.0325,'Relative time ($h$)',ha='center',va='top')

	ax.patch.set_alpha(0)


	# ax.set_ylim([-.12,.06])
	ax.set_ylim([-0.05,0.125])
	ax.set_xlim([-0.03,1.1])

	x0 = 0
	y0 = -0.07
	# yv = np.linspace(y0,-0.11,niter)
	# ax.scatter(np.ones(niter,)*(x0 + 0.05),yv,s=10,edgecolors=colors,alpha=alpha,marker='.')
	# ax.scatter(np.ones(niter,)*(x0 + 0.10),yv,s=50,edgecolors=colors,alpha=alpha,marker='d',linewidth=linewidth,facecolors='none')
	# ax.scatter(np.ones(niter,)*(x0 + 0.15),yv,s=50,edgecolors=colors,alpha=alpha,linewidth=linewidth,facecolors='none')
	# ax.text(-0.025,-0.06,'Phase interval times (UTC-7 $h$)')
	# for N_ in range(niter):
	# 	ax.text(0,yv[N_],str(N_+1) + ')',ha='center',va='center')
	# 	if N_ < niter - 1:
	# 		ax.text(0.20,yv[N_],'%02d:00 - %02d:00'%(T0 + N_*6, T0 + (1 + N_)*6),\
	# 			    va='center')
	# 	else:
	# 		ax.text(0.20,yv[N_],'%02d:00 - %02d:00+1d'%(T0 + N_*6, T0), va='center')


	for s_ in ['top','bottom','left','right']:
	    ax.spines[s_].set_visible(False)
	    ax.spines[s_].set_visible(False)
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])


DPI = 300
FMT = 'PNG'
issave = True
ROOT = os.path.join('..','..','..','..')
# Path to time-series file
TSFL = os.path.join(ROOT,'results','data_files','ERV_0.25hr_G_filt_resampled_UTC_merr.csv')
# Bed Map Directory
BMDR = os.path.join(ROOT,'processed_data','seismic','HV','grids')
# Moulin location dierctory
MLDR = os.path.join(ROOT,'processed_data','hydro')
# Icequake hypocenter catalog location
EQFL = os.path.join(ROOT,'processed_data','seismic','basal_catalog','well_located_30m_basal_catalog_merr.csv')

# Figure output directory
ODIR = os.path.join(ROOT,'results','figures','Manuscript_Figures','v25_render')


# Read in time-series data
df_T = pd.read_csv(TSFL,parse_dates=True,index_col=[0])
## Import Grids
grd_E = np.load(os.path.join(BMDR,'mE_Grid.npy'))
grd_N = np.load(os.path.join(BMDR,'mN_Grid.npy'))
grd_Hm = np.load(os.path.join(BMDR,'H_med_Grid.npy'))
grd_Hu = np.load(os.path.join(BMDR,'H_gau_u_Grid.npy'))
grd_Ho = np.load(os.path.join(BMDR,'H_gau_o_Grid.npy'))
grd_Zs = np.load(os.path.join(BMDR,'mZsurf_Grid.npy'))
grd_M = np.load(os.path.join(BMDR,'MASK_Grid.npy'))

## Import Moulin, Station, and Basal Event Location Files
df_M = pd.read_csv(os.path.join(MLDR,'Moulin_Locations_Cleaned.csv'),parse_dates=True,index_col=['ID'])
df_S = pd.read_csv(os.path.join(ROOT,'data','seismic','sites','Stations.csv'))
df_EQ = pd.read_csv(EQFL,parse_dates=['t0'],index_col=['t0'])


# Define Local Solar Times
TZ0 = pd.Timedelta(-6,unit='hour') 					# UTC Time Offset
SNt = pd.Timedelta(-1,unit='hour') 					# Solar Noontime
CTt0 = pd.Timestamp('2019-07-31T22:06:00') + SNt    # Civil Twilight reference time 1
CTt1 = pd.Timestamp('2019-08-01T05:51:00') + SNt 	# Civil Twilight reference time 2
SNo = pd.Timestamp('2019-07-31T13:48:00') + SNt

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
AX.update({'X':fig.add_subplot(gs[1:7,0:2])})
## Create Axis for Common Legend
# AX.update({'L':fig.add_subplot(gs[3,0:2])})
AX.update({'L':fig.add_subplot(gs[1,0])})
## Create Axes for Seismicity Maps
for i_,l_ in enumerate(['A','B','C','D']):
	AX.update({l_:fig.add_subplot(gs[i_*2:(i_+1)*2,2])})

# AX.update({'A':fig.add_subplot(gs[0,0]),'B':fig.add_subplot(gs[0,2]),\
		   # 'C':fig.add_subplot(gs[1,2]),'D':fig.add_subplot(gs[2,2])})


LW = 1
ALP = 0.75
DPT = 100 # Diamond point threshold value (less than DPT ct. h-1 ice-quakes? make it a diamond!)
cmap = 'Reds'#'viridis'
ECV = ['black','red','dodgerblue','darkorange']
# T0 = pd.Timestamp("2019-08-08T05")
# T1 = T0 + pd.Timedelta(6,unit='hour')
TIMES = [('A',pd.Timestamp("2019-08-08T04"), pd.Timestamp("2019-08-08T10")),\
		 ('B',pd.Timestamp("2019-08-08T10"), pd.Timestamp("2019-08-08T16")),\
		 ('C',pd.Timestamp("2019-08-08T16"), pd.Timestamp("2019-08-08T22")),\
		 ('D',pd.Timestamp("2019-08-08T22"), pd.Timestamp("2019-08-09T04"))]

To,Tf = TIMES[0][1],TIMES[-1][-1]
idf_T = df_T[(df_T.index >= To) & (df_T.index < Tf)]
# Plot background line in crossplot
AX['X'].plot(idf_T['SR(mmWE h-1)'],idf_T['V_H(cm d-1)']*3.6524 - 8.9,'k-')


for i_,T_ in enumerate(TIMES):
	# Pull bounding times
	L_,T0,T1 = T_
	T0 -= pd.Timedelta(0,unit='day')
	T1 -= pd.Timedelta(0,unit='day')
	# Subset points with event rates greater than DPT
	jdf_T = df_T[(df_T.index >= T0) & (df_T.index < T1) & (df_T['ER(N h-1)'] > DPT)]
	# Subset points with event rates \in [1,DPT]
	kdf_T = df_T[(df_T.index >= T0) & (df_T.index < T1) &\
				 (df_T['ER(N h-1)'] <= DPT) & (df_T['ER(N h-1)'] > 0)]
	# Subset points with event rates = 0
	ldf_T = df_T[(df_T.index >= T0) & (df_T.index < T1) &\
				 (df_T['ER(N h-1)'] == 0)]
	# Subset Icequake Locations
	idf_EQ = df_EQ[(df_EQ.index >= T0) & (df_EQ.index < T1) & df_EQ['G_filt']]
	# Subset Active Stations
	idf_S = df_S[(df_S['Ondate'] <= T1) & (df_S['Offdate'] >= T0)]

	#### PLOT CROSSPLOT ELEMENTS ####
	# Plot general markers
	AX['X'].scatter(ldf_T['SR(mmWE h-1)'],ldf_T['V_H(cm d-1)']*3.6524 - 8.9,marker='.',c=ECV[i_],s=10,zorder=5)
	# Plot high event-rate bubbles
	AX['X'].scatter(jdf_T['SR(mmWE h-1)'],jdf_T['V_H(cm d-1)']*3.6524 - 8.9,alpha=ALP,\
			s=jdf_T['ER(N h-1)']/5,c=(jdf_T.index - T0).total_seconds()/3600,cmap=cmap,\
			edgecolor=ECV[i_],linewidth=LW,zorder=10,vmin=0,vmax=(T1-T0).total_seconds()/3060)

	# 		vmin=(T0 - To).total_seconds(),\
	# 												 vmax=(T1 - To).total_seconds())
	# # vmin=0,vmax=(T1 - T0).total_seconds())
	# Plot low event-rate diamonds
	ch2 = AX['X'].scatter(kdf_T['SR(mmWE h-1)'],kdf_T['V_H(cm d-1)']*3.6524 - 8.9,alpha=ALP,\
			marker='d',c=(kdf_T.index - T0).total_seconds()/3600,cmap=cmap,\
			edgecolor=ECV[i_],linewidth=LW,zorder=10,s=50,vmin=0,vmax=(T1 - T0).total_seconds()/3600)

	# 		vmin=(T0 - To).total_seconds(),\
	# 												 	  vmax=(T1 - To).total_seconds())
	# # vmin=0,vmax=(T1 - T0).total_seconds())
	#### PLOT EVENT SUB-MAP ####
	# Plot bed topography background
	cbh = AX[L_].pcolor(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hu)*grd_M,cmap='gray',zorder=1)#,alpha=0.5)
	cbh.set_clim([1800,1880])
	AX[L_].contour(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hu)*grd_M,np.arange(1800,1900,20),colors='w',zorder=2)

	# Plot Icequakes
	ceh = AX[L_].scatter(idf_EQ['mE']-mE0,idf_EQ['mN']-mN0,s=1,c=(idf_EQ.index - T0).total_seconds()/3600,\
			   vmin=0,vmax=(T1 - T0).total_seconds()/3600,cmap=cmap,alpha=0.5,zorder=3)
	# AX[L_].plot(idf_EQ['mE'] - mE0, idf_EQ['mN'] - mN0,'r.',alpha=0.1)
	# Plot Active Stations
	AX[L_].plot(idf_S['mE'] - mE0,idf_S['mN'] - mN0,'kv',markersize=3,zorder=4)
	# Do Formatting
	plt.setp(AX[L_].get_xticklabels(),visible=False)
	plt.setp(AX[L_].get_yticklabels(),visible=False)
	AX[L_].tick_params(axis='both',which='both',length=0)
	AX[L_].yaxis.set_label_position("right")
	AX[L_].set_ylabel('%02d:00-%02d:00'%(T0.hour,T1.hour), rotation=270,labelpad=15)
	xlims = AX[L_].get_xlim()
	ylims = AX[L_].get_ylim()
	AX[L_].plot(xlims[0],ylims[1],'o',mec=ECV[i_],ms=16,mfc="none")

	AX[L_].text(xlims[0],ylims[1],T_[0].lower(),ha='center',va='center',\
				fontweight='extra bold',fontstyle='italic',fontsize=14)

	RATE = len(idf_EQ) / ((T_[-1] - T_[-2]).total_seconds()/3600)
	AX[L_].text(xlims[1],ylims[0],'ct.: %d\n$\dot{E}$: %d ct. $h^{-1}$'%(len(idf_EQ),RATE),ha='right',va='bottom')
	AX[L_].set_xlim([xlims[0]-50,xlims[1]+50])
	AX[L_].set_ylim([ylims[0]-50,ylims[1]+50])

	AX[L_].fill_between([1050,1150],[650,650],[675,675],color='k',ec='k')
	AX[L_].fill_between([1150,1250],[650,650],[675,675],color='w',ec='k')
	AX[L_].text(1050,680,'0',ha='center',fontsize=8)
	AX[L_].text(1250,680,'200',ha='center',fontsize=8)
	AX[L_].arrow(600,500,150,160,width=10)
	AX[L_].text(675,580,'Flow\n\ndirection',fontsize=8,rotation=45,ha='center',va='center')
# Create formatted compound scalebar
compound_scalebar(AX['L'],colors=ECV,ssteps=3,DT=6,linewidth=LW,alpha=ALP,cmap=cmap,T0=5)
AX['L'].set_xlim([-0.166,1.153])
AX['L'].set_ylim([-0.02,0.057])

# Complete window dressing
AX['X'].set_xlim([1.3,3.6])
Vm = np.nanmin(idf_T['V_H(cm d-1)'])
VM = np.nanmax(idf_T['V_H(cm d-1)'])
# axa.set_ylim([(Vm - 0.05*(VM - Vm))/24,(VM + 0.1*(VM - Vm))/24])

AX['X'].text(1.8,70,'Phase 1\n(a)',ha='center')
AX['X'].text(3.4,130,'Phase 2\n(b)',ha='center')
AX['X'].text(3.2,80,'Phase 3\n(c)',ha='center')
AX['X'].text(2.1,35,'Phase 4\n(d)',ha='center')
AX['X'].set_xlabel('Meltwater supply rate [$\dot{S}_{w}$]\n($mm$ $w.e.$ $h^{-1}$)')
AX['X'].set_ylabel('Slip velocity [$V$] ($m$ $a^{-1}$)')
plt.subplots_adjust(wspace=0,hspace=0)

if issave:
	I_OFILE = 'SGEQ_v25_Fig3_Hysteresis_Loop_and_Maps_%ddpi.%s'%(DPI,FMT.lower())
	plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)
	print('Figure saved: %s'%(os.path.join(ODIR,I_OFILE)))

plt.show()