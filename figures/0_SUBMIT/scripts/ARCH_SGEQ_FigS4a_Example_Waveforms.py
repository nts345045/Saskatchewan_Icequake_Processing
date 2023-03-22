import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

issave = True
DPI = 120
FMT = 'PNG'

ROOT = os.path.join('..','..','..','..')
WFDIR = os.path.join(ROOT,'processed_data','seismic','basal_catalog')
ODIR = os.path.join(ROOT,'results','figures','Manuscript_Figures','v16_render')

# Load distance-sorted data-brick
WFM = np.load(os.path.join(WFDIR,'Cluster77_E10m_BRICK.npy'))
# # Load distance vector for brick
# dv = np.load(os.path.join(WFDIR,'Cluster77_E10m_Dvect.npy'))
# Load station list (in sorted order)
df_S = pd.read_csv(os.path.join(WFDIR,'Cluster77_E10m_stas.csv'),parse_dates=True,index_col=[0])

# Channel order 
#(todo: include these in an updated metadata write from when the brick is formed)
ORIENTS = ['Vertical','Northward','Eastward'] 
CHANS = ['??[Z3]','??1','??2']
CODES = ['a','b','c']

NAMP = 10
SR = 1000
ALP = 0.01


# fig = plt.figure(figsize=(10,8))
# gs = GridSpec(ncols=3,nrows=1)
# axs = []
# for a_ in range(3):
# 	if a_ > 0:
# 		ax = fig.add_subplot(gs[a_],sharex=axs[0],sharey=axs[0])
# 	else:
# 		ax = fig.add_subplot(gs[a_])
# 	axs.append(ax)
# figs = []
for k_ in range(3):
	fig = plt.figure(figsize=(7,7))
	ax =  fig.add_subplot(111)

	for i_ in range(len(df_S)):
	# for k_ in range(len(CHANS)):
		for l_ in range(WFM.shape[3]):
			nv = WFM[i_,:,k_,l_]
			# Remove median and re-normalize
			nv -= np.median(nv)
			nv /= np.max(np.abs(nv))
			# nv = (av/np.max(np.abs(av)))*NAMP
			di = df_S['D_m'].values[i_]
			ax.plot(np.arange(0,len(nv))/SR,nv*NAMP + di,'k-',alpha=ALP)
		EV = np.median(WFM[i_,:,k_,:],axis=1)
		EV -= np.median(EV)
		EV /= (np.max(np.abs(EV)))
		ax.plot(np.arange(0,len(nv))/SR,EV*NAMP + di,color='firebrick')

# for a_ in range(3):
	ax.set_title('%s) %s Channels: %s'%(CODES[k_],ORIENTS[k_],CHANS[k_]))
	ax.set_ylabel('Source-receiver distance (m)\nNormalized ground accelerations')
	ax.set_xlabel('Elapsed time since event origin (sec)')
	ax.set_xlim([0,.25])
	ax.set_ylim([150,500])

# for k_ in range(3):
	if issave:
		I_OFILE = 'SGEQ_v16_FigS4%s_Normalized_Waveforms_%s_%ddpi.%s'%(CODES[k_],ORIENTS[k_],DPI,FMT.lower())
		plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)
		print('Figure saved: %s'%(os.path.join(ODIR,I_OFILE)))


plt.show()