import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from copy import deepcopy

def compound_scalebar(ax,ssteps=7,DT=6,cmap='viridis',niter=4,colors=['black','red','blue','orange'],T0=4,linewidth=2,alpha=0.75):
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
	ax.text(np.mean(slv),0.0325,'Event rate ($h^{-1}$)',ha='center')

	## Create Color Indexing
	slv = np.linspace(0,1,ssteps)
	sll = np.linspace(0,DT,ssteps)
	icmap = cm.get_cmap(cmap)

	for s_ in range(ssteps):
	    text = ax.text(slv[s_],-0.02,'%d'%(sll[s_]),color=icmap((slv[s_] - slv[0])/(slv[-1] - slv[0])),ha='center',va='top')    
	    text.set_path_effects([path_effects.Stroke(linewidth=1,foreground='black'),path_effects.Normal()])
	ax.text(np.mean(slv),-0.0325,'Elapsed time ($h$)',ha='center',va='top')

	ax.patch.set_alpha(0)


	ax.set_ylim([-.12,.06])
	ax.set_xlim([-0.03,1.1])

	x0 = 0
	y0 = -0.07
	yv = np.linspace(y0,-0.11,niter)
	ax.scatter(np.ones(niter,)*(x0 + 0.00),yv,s=10,edgecolors=colors,alpha=alpha,marker='.')
	ax.scatter(np.ones(niter,)*(x0 + 0.05),yv,s=50,edgecolors=colors,alpha=alpha,marker='d',linewidth=linewidth,facecolors='none')
	ax.scatter(np.ones(niter,)*(x0 + 0.10),yv,s=50,edgecolors=colors,alpha=alpha,linewidth=linewidth,facecolors='none')
	ax.text(0.1,-0.06,'Phase time intervals (UTC-7 $h$)')
	for N_ in range(niter):
		if N_ < niter - 1:
			ax.text(0.15,yv[N_],'%02d:00 - %02d:00  Phase %d'%(T0 + N_*DT, T0 + (1 + N_)*DT,N_+1),\
				    va='center')
		else:
			ax.text(0.15,yv[N_],'%02d:00 - %02d:00+1d  Phase 4'%(T0 + N_*DT, T0), va='center')
	for s_ in ['top','bottom','left','right']:
	    ax.spines[s_].set_visible(False)
	    ax.spines[s_].set_visible(False)
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])



ROOT = os.path.join('..','..','..','..')
TSFL = os.path.join(ROOT,'results','data_files','ERV_0.25hr_G_filt_resampled_UTC_merr.csv')
EQFL = os.path.join(ROOT,'processed_data','seismic','basal_catalog','well_located_30m_basal_catalog_merr.csv')
STFL = os.path.join(ROOT,'data','seismic','sites','Stations.csv')

ODIR = os.path.join(ROOT,'results','figures','Manuscript_Figures','v25_render')
DPI = 300
FMT='PNG'
issave = True

# Read in data
df_T = pd.read_csv(TSFL,parse_dates=True,index_col=[0])
df_N = pd.read_csv(EQFL,parse_dates=True,index_col=[0])
df_S = pd.read_csv(STFL)

# Define Local Solar Times
TZ0 = pd.Timedelta(-6,unit='hour') 					# UTC Time Offset
SNt = pd.Timedelta(-1,unit='hour') 					# Solar Noontime
CTt0 = pd.Timestamp('2019-07-31T22:06:00') + SNt    # Civil Twilight reference time 1
CTt1 = pd.Timestamp('2019-08-01T05:51:00') + SNt 	# Civil Twilight reference time 2
SNo = pd.Timestamp('2019-07-31T13:48:00') + SNt

df_T.index += (TZ0 + SNt)
df_T = df_T.interpolate(direction='both',limit=3)

df_N.index += (TZ0 + SNt)
df_N = df_N.sort_index()
df_S['Ondate'] = pd.to_datetime(df_S['Ondate'],unit='s') + TZ0 + SNt
df_S['Offdate'] = pd.to_datetime(df_S['Offdate'],unit='s') + TZ0 + SNt
df_S.index = df_S['Ondate']
df_S = df_S.sort_index()

T0 = pd.Timestamp("2019-08-06T05")
T1 = T0 + pd.Timedelta(6,unit='hour')
TF = pd.Timestamp("2019-08-20T05")


gs = GridSpec(ncols=2,nrows=3)
LW = 1
ALP = 0.75
cmap = 'Reds'#'viridis'
ECV = ['black','red','dodgerblue','darkorange']
for I_ in range(2):
	# Initialize Plot
	fig = plt.figure(figsize=(8,10))
	# Iterate across rows
	for R_ in range(3):
		# Iterate across columns
		for C_ in range(2):
			ax = fig.add_subplot(gs[R_,C_])
			print('I: %d, R: %d, C: %d: %s-%s'%(I_,R_,C_,str(T0),str(T1)))
			if R_ == 2 and C_ == 1:
				compound_scalebar(ax,colors=ECV,T0=4,linewidth=LW,alpha=ALP,cmap=cmap)
			# If not the lower-right section, render crossplots
			else:
				# Subset time-series data into 24-hour segments
				idf_T = df_T[(df_T.index >= T0) & (df_T.index < T0 + pd.Timedelta(1,unit='day'))]

				# Check for data content
				# mreal = np.sum(np.isfinite(idf_T.values),axis=0)

				# if np.min(mreal)/len(idf_T) >= 0.25:

				# Plot background points
				ax.plot(idf_T['SR(mmWE h-1)'],idf_T['V_H(cm d-1)']*3.6524 - 8.9,'k-',zorder=1)

				# Label axes
				if R_ == 2:
					ax.set_xlabel('Meltwater supply rate\n($mm w.e.$ $h^{-1}$)')

				if C_ == 0:
					ax.set_ylabel('Slip velocity ($m$ $a^{-1}$)')

				if R_ == 1 and C_ == 1:
					ax.set_xlabel('Meltwater supply rate\n($mm w.e.$ $h^{-1}$)')


				ax.set_title('2019-08-%02d to 2019-08-%02d'%(T0.day,T0.day+1))
				# Iterate across sub-periods
				for P_ in range(4):


					# Subset points with event rates greater than 250
					jdf_T = df_T[(df_T.index >= T0) & (df_T.index < T1) & (df_T['ER(N h-1)'] > 100)]
					# Subset points with event rates \in [1,250]
					kdf_T = df_T[(df_T.index >= T0) & (df_T.index < T1) &\
								 (df_T['ER(N h-1)'] <= 100) & (df_T['ER(N h-1)'] > 0)]
					ldf_T = df_T[(df_T.index >= T0) & (df_T.index < T1) &\
								 (df_T['ER(N h-1)'] == 0)]
					# Plot general markers
					ax.scatter(ldf_T['SR(mmWE h-1)'],ldf_T['V_H(cm d-1)']*3.6524 - 8.9,marker='.',c=ECV[P_],s=10,zorder=5)
					# Plot high event-rate bubbles
					ax.scatter(jdf_T['SR(mmWE h-1)'],jdf_T['V_H(cm d-1)']*3.6524 - 8.9,alpha=ALP,\
							s=jdf_T['ER(N h-1)']/5,c=(jdf_T.index - T0).total_seconds(),cmap=cmap,\
							edgecolor=ECV[P_],linewidth=LW,zorder=10,vmin=0,vmax=(T1 - T0).total_seconds())
					# Plot low event-rate diamonds
					ch2 = ax.scatter(kdf_T['SR(mmWE h-1)'],kdf_T['V_H(cm d-1)']*3.6524 - 8.9,alpha=ALP,\
							marker='d',c=(kdf_T.index - T0).total_seconds(),cmap=cmap,\
							edgecolor=ECV[P_],linewidth=LW,zorder=10,s=50,vmin=0,vmax=(T1 - T0).total_seconds())
					T0 += pd.Timedelta(6,unit='hour')
					T1 += pd.Timedelta(6,unit='hour')
				# Complete window dressing
				ax.set_xlim([1.1,3.5])
				Vm = np.nanmin(idf_T['V_H(cm d-1)'])
				VM = np.nanmax(idf_T['V_H(cm d-1)'])
				ax.set_ylim([(Vm - 0.05*(VM - Vm))*3.6524 - 8.9,(VM + 0.1*(VM - Vm))*3.6524 - 8.9])

	if issave:

		I_OFILE = 'SGEQ_v25_FigS2_CrossPlotCatalog_Page%d_%ddpi.%s'%(I_,DPI,FMT.lower())
		plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)



plt.show()