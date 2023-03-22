import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from pyproj import Proj


## MAP PATHS
ROOT = os.path.join('..','..','..','..')
# Bed Map Directory
BMDR = os.path.join(ROOT,'processed_data','seismic','HV','grids')
MLDR = os.path.join(ROOT,'processed_data','hydro')
EQFL = os.path.join(ROOT,'processed_data','seismic','basal_catalog','well_located_30m_basal_catalog_merr.csv')

ODIR = os.path.join(ROOT,'results','figures','Manuscript_Figures','v25_render')
DPI = 300
FMT='PNG'
issave = True


## LOAD SURFACES & MOULIN LOCATIONS
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

df_EQ.index += (TZ0 + SNt)
df_EQ = df_EQ.sort_index()
df_S['Ondate'] = pd.to_datetime(df_S['Ondate'],unit='s') + TZ0 + SNt
df_S['Offdate'] = pd.to_datetime(df_S['Offdate'],unit='s') + TZ0 + SNt
df_S.index = df_S['Ondate']
df_S = df_S.sort_index()

## Convert Moulin Locations to UTM
myproj = Proj(proj='utm',zone=11,ellipse='WGS84',preserve_units=False)
SmE,SmN = myproj(df_S['Lon'].values,df_S['Lat'].values)
df_S = pd.concat([df_S,pd.DataFrame({'mE':SmE,'mN':SmN},index=df_S.index)],axis=1,ignore_index=False)


### ESTIMATE LAPLACIAN OF BED ###
Uz,Vz = np.gradient(grd_Zs - grd_Hu,5)
UUz,UVz = np.gradient(Uz,5)
_,VVz = np.gradient(Vz,5)

grd_LAP = -1*(UUz + VVz) + 0.001

mE0 = 4.87e5
mN0 = 5.777e6
mEm = np.min(grd_E)
mEM = np.max(grd_E)
mNm = np.min(grd_N)
mNM = np.max(grd_N)

T0 = pd.Timestamp("2019-08-02T04")
T1 = T0 + pd.Timedelta(6,unit='hour')
TF = pd.Timestamp("2019-08-20T04")
conmax = 0.02375
conmax = 0.03
cmap = 'Reds'#'viridis'
gs = GridSpec(ncols=4,nrows=5)
for I_ in range(4):
	fig = plt.figure(figsize=(8,10))
	for R_ in range(5):
		for C_ in range(4):
			if T1 < TF:
				# Create Axis
				ax = fig.add_subplot(gs[R_,C_])
				# Subset Icequakes
				idf_EQ = df_EQ[(df_EQ.index >= T0) & (df_EQ.index < T1) & df_EQ['G_filt']]
				# Subset Active Stations
				idf_S = df_S[(df_S['Ondate'] <= T1) & (df_S['Offdate'] >= T0)]
				# # Plot concavity background (negative signed for figure rendering purposes only)
				# cbh = ax.pcolor(grd_E-mE0,grd_N-mN0,-1*grd_M*(np.abs(grd_LAP) > 0.001)*grd_LAP,cmap='RdGy_r')#,alpha=0.5)
				# cbh.set_clim([-conmax,conmax])
				# Plot bed topography background
				cbh = ax.pcolor(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hu)*grd_M,cmap='gray',zorder=1)#,alpha=0.5)
				cbh.set_clim([1800,1880])
				ax.contour(grd_E-mE0,grd_N-mN0,(grd_Zs - grd_Hu)*grd_M,np.arange(1800,1900,20),colors='w',zorder=2)

				# Plot Icequakes
				ceh = ax.scatter(idf_EQ['mE']-mE0,idf_EQ['mN']-mN0,s=1,c=(idf_EQ.index - T0).total_seconds()/3600,\
						   vmin=0,vmax=(T1 - T0).total_seconds()/3600,cmap=cmap,alpha=0.5,zorder=5)
				# Plot Active Stations
				plt.plot(idf_S['mE'] - mE0,idf_S['mN'] - mN0,'kv',markersize=3,zorder=4)
				# Do Formatting
				plt.setp(ax.get_xticklabels(),visible=False)
				plt.setp(ax.get_yticklabels(),visible=False)
				ax.tick_params(axis='both',which='both',length=0)
				# Put times as column heads
				if R_ == 0:
					ax.set_title('Phase %d\n%02d:00-%02d:00'%(C_+1,T0.hour,T1.hour),fontsize=12)
				# Put dates as left y-label
				if C_ == 0:
					ax.set_ylabel('2019-08-%02d'%(T0.day),fontsize=12)

				if C_ == 3:
					ax.yaxis.set_label_position("right")
					if pd.Timestamp('2019-08-01') <= T0 < pd.Timestamp("2019-08-05"):
						ax.set_ylabel('Deploy Interval',rotation=270,labelpad=15)
					elif pd.Timestamp("2019-08-05") <= T0 < pd.Timestamp("2019-08-08"):
						ax.set_ylabel('Interval A',rotation=270,labelpad=15)
					elif pd.Timestamp("2019-08-08") <= T0 < pd.Timestamp("2019-08-09"):
						ax.set_ylabel("Fig. 3\nInterval A",rotation=270,labelpad=30)
					elif pd.Timestamp("2019-08-09") <= T0 < pd.Timestamp("2019-08-14"):
						ax.set_ylabel("Interval B",rotation=270,labelpad=15)
					elif pd.Timestamp("2019-08-14") <= T0 < pd.Timestamp("2019-08-20"):
						ax.set_ylabel("Interval C",rotation=270,labelpad=15)

				if I_ == 3 and R_ == 2 and C_ == 2:
					ax.yaxis.set_label_position("right")
					ax.set_ylabel("Demobilization",rotation=270,labelpad=15)

				# Advance Times
				T0 += pd.Timedelta(6,unit='hour')
				T1 += pd.Timedelta(6,unit='hour')
					   # x0  y0  dx  dy
				# Equally Scale Axes
				ax.set_xlim([mEm - 0.05*(mEM - mEm) - mE0,mEM + 0.05*(mEM - mEm) - mE0])
				ax.set_ylim([mNm - 0.05*(mNM - mNm) - mN0,mNM + 0.05*(mNM - mNm) - mN0])
				# ax.fill_between([mEM - mE0 - 300,mEM - mE0 - 100],[mNm - mN0,mNm - mN0],[mNm - mN0 + 20,mNm - mN0 + 20],facecolor='k')
				# ax.text(mEM - mE0 - 300,mNm - mN0 + 25,0,ha='center')
				# ax.text(mEM - mE0 - 100,mNm - mN0 + 25,200,ha='center')
				# ax.text(mEM - mE0 - 300,mNm - mN0 + 100,'%d ct./h'%(np.ceil(len(idf_EQ)/6)))
				xlims = ax.get_xlim()
				ylims = ax.get_ylim()
				ax.fill_between([1050,1125],[650,650],[675,675],color='k',ec='k')
				ax.fill_between([1125,1200],[650,650],[675,675],color='w',ec='k')
				ax.text(1050,680,'0',ha='center',fontsize=8)
				ax.text(1200,680,'150',ha='center',fontsize=8)
				ax.arrow(600,500,150,160,width=10)
				ax.text(675,580,'Flow\n\n',fontsize=8,rotation=45,ha='center',va='center')
				RATE = len(idf_EQ) / ((T1 - T0).total_seconds()/3600)
				ax.text(xlims[1]-20,ylims[0],'ct.: %d\n$\dot{E}$: %d ct. $h^{-1}$'%(len(idf_EQ),RATE),ha='right',va='bottom')

	# Do Colorbar Rendering
	if I_ < 3:
		ceax = fig.add_axes([.15,.075,.3,.02])
		cbax = fig.add_axes([.55,.075,.3,.02])
	elif I_ == 3:
		ceax = fig.add_axes([.15,.375,.3,.02])
		cbax = fig.add_axes([.55,.375,.3,.02])
	# cbar_b = plt.colorbar(cbh,cax=cbax,orientation='horizontal',ticks=[-conmax,0.0,conmax])
	# cbar_b.ax.set_xticklabels(['Convex\n(likely rock)','Curvature\n(interpretation)','Concave\n(likely till)'])
	cbar_b = plt.colorbar(cbh,cax=cbax,orientation='horizontal',ticks=[1800,1840,1880])
	cbar_b.ax.set_xticklabels(['1800','1840\nBed elevation (m a.s.l.)','1880'])
	cbar_e = plt.colorbar(ceh,cax=ceax,orientation='horizontal',ticks=[0,1,2,3,4,5,6])
	cbar_e.ax.set_xticklabels(['0 h','1 h','2 h','3 h\nRelative origin time','4 h','5 h','6 h'])


	plt.subplots_adjust(wspace=0,hspace=0)
	if issave:
		I_OFILE = 'SGEQ_diss_rev_FigS3_DiurnalSeismicMapCatalog_Page%d_%ddpi.%s'%(I_,DPI,FMT.lower())
		plt.savefig(os.path.join(ODIR,I_OFILE),dpi=DPI,format=FMT)

plt.show()

