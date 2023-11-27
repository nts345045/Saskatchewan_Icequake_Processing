"""
process_magnitudes_3C.py

This script makes SH wave coda integral estimates of mw using inputs of:
	Station locations			- r(x), \sigma_r(x)
	Event locations				- s(x), \sigma_s(x)
	Station azimuth corrections - \theta(t)
	Bed topography				- z(x), \sigma_z(x)
	displacement waveforms 		- u(t)

Key References
Aki & Richards (1979) - Quantitative Seismology, Volume 1, 1st Edition
Aki & Richards (2002) - Quantitative Seismology, 2nd Edition
Boatwright (1980) - BSSA - A spectral theory for circular seismic sources; simple estimates of source dimension, dynamic stress drop
						   and radiated seismic energy
Roeoesli and others (2016) - JGR-Earth Surface - Meltwater influences on deep stick-slip icequakes near the base of the Greenland Ice Sheet
Walter and others (2009) - BSSA - Moment tensor inversions of icequakes on Gornergletscher, Switzerland


UPDATED VERSION THAT DOES A 3C rotation 

"""
import os
import pandas as pd
import numpy as np
from glob import glob
from obspy import read, UTCDateTime, Stream
from obspy.signal.rotate import rotate_zne_lqt
from pyproj import Proj
from tqdm import tqdm
import pyrocko.gui.marker as pm


### PHYSICS SUBROUTINES ###

def calc_mw(M0,p0=[2./3.,6.07]):
	"""
	Calculate moment magnitude (mw) from seismic moment (M0) in N-m as:
	mw = 2/3 * log10(M0) - 6.07

	:: INPUTS ::
	:param M0: 	[N-m] seismic moment (float)
	:param p0: 	[-] equation coefficients (array-like)

	:: OUTPUT ::
	:return mw: [log10(N-n)] moment magnitude
	"""
	mw = p0[0]*np.log10(M0) - p0[1]
	return mw

def calc_M0_Boatwright(tr_u,RR,iT1,iT2,FF,SF=2.,p_i=910.,p_b=2550.,B_i=1790.,B_b=2650.,int_kwargs={'method':'cumtrapz'},signed=True):
	"""
	Calculate the seismic moment from recordings of the SH displacement coda integral
	using the method from Boatwright (1980) and presented in Walter and others(2009)
	and Roeoesli and others (2016). Note that these later studies ignore the source-region
	impedance, assuming p_b = p_i and B_b = B_i

	M0 = [[4*pi*RR*(p_i*p_b*B_i*B_b**5)**0.5]/[FF*SSH]] int_{iT1}^{iT2}{tr_u dt}

	:: INPUTS ::
	:param tr_u: 	[m] Phase-appropriate displacement seismogram (obspy.Trace)
	:param RR: 		[m] Phase-appropriate pathlength (float)
	:param iT1: 	[timestamp] lower limit of integration (obspy.UTCDateTime) - should be at the SH wave onset time
	:param iT2: 	[timestamp] upper limit of integration (obspy.UTCDateTime) - Stork et al. (2014) suggest tS + (tS - tP)
	:param FF: 		[-] Phase-appropriate spreading factor coefficient (float) - see calc_FSH()
	:param SF: 		[-] surface amplification factor (float) - see Roeoesli and others (2016) and p.130 in Aki & Richards (2002)
	:param p_i: 	[kg m**-3] density of ice (float) - e.g., Cuffey & Patterson (2010)
	:param p_b: 	[kg m**-3] density of bedrock (float) - after field measurements and samples collected by McCormick (2022)
	:param B_i: 	[m s**-1] SH wave velocity in ice - this study
	:param B_b: 	[m s**-1] SH wave velocity in bedrock - Vp of 4790 from Stevens and others (2022) and Vp/Vs of 1.8 for Dolomite/fractured limestone (e.g. McCormick and others, 2023)
	:param int_kwargs: dictionary to pass key-word arguments to obspy.core.Trace.trace.integrate().
	:: OUTPUT ::
	:return M0: [N-m] seismic moment

	"""
	# Calculate numerator of equation
	numerator = 4.*RR*np.pi*(p_i*p_b*B_i*B_b**5)**0.5
	# Calculate denominator of equation
	denominator = FF*SF
	# integrate then trim displacement seismogram
	tr_IuSH = tr_u.copy().integrate(**int_kwargs).trim(starttime=iT1,endtime=iT2)
	# Calculate summed displacement
	# intval = np.nansum(np.abs(tr_IuSH.data))
	intval = np.nansum(tr_IuSH.data)
	# Calculate the moment
	M0 = (numerator/denominator)*intval
	if signed:
		return M0, intval
	else:
		return np.abs(M0),np.abs(intval)



def calc_M0_WSR(tr_u,RR,iT1,iT2,FF,vv,pp=910.,SF=2.,int_kwargs={'method':'cumtrapz'},signed=True):
	"""
	Calculate the seismic moment from recordings of the displacement coda integral based on the
	method from Boatwright (1980) and parameterizations/simplifications after: 
	Walter and others (2009) - formulation for a half-space with an upper free surface, parameterization for glaciers
	Stork and others (2014) - general form for a homogeneous whole-space
	Roeoesli and others (2016) - ignore the bed (i.e., Bi = Bx, pi = px) - thus sources are considered in a homogeneous half-space


	M0 = [[4*pi*RR*pp*vv**3/[FF*SSH]] int_{iT1}^{iT2}{tr_u dt}

	:: INPUTS ::
	:param tr_u: 	[m] Phase-appropriate displacement seismogram (obspy.Trace)
	:param RR: 		[m] Phase-appropriate pathlength (float)
	:param iT1: 	[timestamp] lower limit of integration (obspy.UTCDateTime) - should be at the SH wave onset time
	:param iT2: 	[timestamp] upper limit of integration (obspy.UTCDateTime) - Stork et al. (2014) suggest tS + (tS - tP)
	:param FF: 		[-] Phase-appropriate spreading coefficient (float) - see calc_FSH()
	:param vv: 		[m s**-1] Phase-appropriate velocity in ice (float) - this study
	:param pp: 		[kg m**-3] density of ice (float) - e.g., Cuffey & Patterson (2010)
	:param SF: 		[-] surface amplification factor (float) - see Roeoesli and others (2016) and p.130 in Aki & Richards (2002)
	:param int_kwargs: dictionary to pass key-word arguments to obspy.core.Trace.trace.integrate().
	:: OUTPUT ::
	:return M0: 	[N-m] seismic moment
	:return intval: [m sec] definite integral value

	"""
	# Calculate numerator of equation
	numerator = 4.*RR*np.pi*pp*vv**3
	# Calculate denominator of equation
	denominator = FF*SF
	# integrate then trim displacement seismogram
	tr_Iu = tr_u.copy().integrate(**int_kwargs).trim(starttime=iT1,endtime=iT2)
	# Calculate summed displacement to evaluate integral
	intval = np.nansum(tr_Iu.data)
	# Calculate the moment
	M0 = (numerator/denominator)*intval
	if signed:
		return M0, intval
	else:
		return np.abs(M0),np.abs(intval)

def calc_FSH(rake,dip,strike,takeoff_ang,SRAZ):
	"""
	Calculate the horizontal shear wave radiation coefficient using Eqn. 4-88 in Aki & Richards (2002)
	(Eqn. 4-91 in the 1st Edition)

	:: INPUTS ::
	:param rake: 		[rad] fault motion rake angle - angle between fault strike and slip angle
	:param dip: 		[rad] fault dip angle - right hand rule
	:param strike:		[rad] fault strike angle - right hand rule relative to East basis vector
	:param takeoff_ang: [rad] SH wave takeoff angle relative to downward/vertical (i.e., upward = pi rad)
	:param SRAZ: 		[rad] Source-Receiver-AZimuth - left hand rule relative to North basis vector

	:: OUTPUT ::
	:return FSH: 		[-] Horizontal shear wave radiation coefficient
	"""
	Dphi = SRAZ - strike
	part1 = np.cos(rake)*np.cos(dip)*np.cos(takeoff_ang)*np.sin(Dphi)
	part2 = np.cos(rake)*np.sin(dip)*np.cos(2.*takeoff_ang)*np.cos(2*Dphi)
	part3 = np.sin(rake)*np.cos(2.*dip)*np.cos(takeoff_ang)*np.cos(Dphi)
	part4 = 0.5*np.sin(rake)*np.sin(2.*dip)*np.sin(takeoff_ang)*np.sin(2.*Dphi)

	FSH = part1 + part2 + part3 - part4

	return FSH


def calc_FP(rake,dip,strike,takeoff_ang,SRAZ):
	"""
	Calculate the compressional wave radiation coefficient using Eqn. 4-86 in Aki & Richards (2002)
	(Eqn. 4-89 in the 1st edition)

	:: INPUTS ::
	:param rake: 		[rad] fault motion rake angle - angle between fault strike and slip angle
	:param dip: 		[rad] fault dip angle - right hand rule
	:param strike:		[rad] fault strike angle - right hand rule relative to East basis vector
	:param takeoff_ang: [rad] SH wave takeoff angle relative to downward/vertical (i.e., upward = pi rad)
	:param SRAZ: 		[rad] Source-Receiver-AZimuth - left hand rule relative to North basis vector

	:: OUTPUT ::
	:return FP: 		[-] Compressional wave radiation coefficient
	"""
	Dphi = SRAZ - strike
	part1 = np.cos(rake)*np.sin(dip)*np.sin(takeoff_ang)**2*np.sin(2.*Dphi)
	part2 = np.cos(rake)*np.cos(dip)*np.sin(2.*takeoff_ang)*np.cos(Dphi)
	part3 = np.sin(rake)*np.sin(2.*dip)*(np.cos(takeoff_ang)**2 - np.sin(takeoff_ang)**2*np.sin(Dphi))
	part4 = np.sin(rake)*np.cos(2.*dip)*np.sin(2.*takeoff_ang)*np.sin(2.*Dphi)

	FP = part1 - part2 + part3 + part4

	return FP

### GEOMETRY ROUTINES ###

def source_receivers_geom(x_src,x_rcv):
	"""
	Calculate source-receivers distances, takeoff angles, ESAZs
	from one source to many receivers

	:: INPUTS ::
	:param x_src: source location, [X,Y,Z] in meters
	:param x_rcv: receiver locations, np.c_[X,Y,Z] in meters

	:: OUTPUTS ::
	:return RR: source-receiver distance in meters
	:return takeoff_ang: [rad] takeoff angle from source to receiver
	:return SRAZ: [rad] source to recever azimuth
	"""
	# Get distance elements
	dele = x_rcv - x_src
	# Calculate linear distances
	RR = np.sqrt(np.sum(dele**2,axis=1))
	# Calculate takeoff angle
	takeoff_ang = np.arctan2(dele[:,2],np.sqrt(np.sum(dele[:,:1]**2),axis=1)) + np.pi/2
	# Calculate source-receivers azimuths
	SRAZ = 0.5*np.pi - np.arctan(dele[:,1],dele[:,0])
	# Calculate receivers-source azimuths
	RSAZ = 0.5*np.pi - np.arctan(-dele[:,1],-dele[:,0])
	return RR,takeoff_ang,SRAZ,RSAZ

def source_receiver_geom(x_src,x_rcv):
	"""
	Calculate source-receivers distances, takeoff angles, ESAZs
	from one source to many receivers

	:: INPUTS ::
	:param x_src: source location, [X,Y,Z] in meters
	:param x_rcv: receiver locations, [X,Y,Z] in meters

	:: OUTPUTS ::
	:return RR: source-receiver distance in meters
	:return takeoff_ang: [rad] takeoff angle from source to receiver
	:return SRAZ: [rad] source to recever azimuth
	"""
	# Get distance elements
	dele = x_rcv - x_src
	# Calculate linear distances
	RR = np.sqrt(np.sum(dele**2))
	# Calculate takeoff angle
	takeoff_ang = np.arctan2(dele[2],np.sqrt(np.sum(dele[:1]**2))) + np.pi/2
	# Calculate source-receivers azimuths
	SRAZ = 0.5*np.pi - np.arctan2(dele[1],dele[0])
	# Calculate receivers-source azimuths
	RSAZ = 0.5*np.pi - np.arctan2(-dele[1],-dele[0])
	return RR,takeoff_ang,SRAZ,RSAZ


def run_presets_model(x_src,x_rcv,AZr,strike,dip,rake,tr1,tr2,VpVs=1.93,t0_padsec=0.5):
	# Calculate Source-Receivers Geometries
	RR,takeoff_ang,SRAZ,RSAZ = source_receivers_geom(x_src,x_rcv)
	# Rotate data from 12 to NE
	dataN,dataE = rotate_ne_rt(tr1.data,tr2.data,AZr*(np.pi/180.))
	# Rotate data from NE to RT
	dataR,dataT = rotate_ne_rt(dataN,dataE,RSAZ)
	# Create traces for radial and transverse
	trR = tr1.copy()
	trR.stats.channel = tr1.stats.channel[:-1]+'R'
	trR.data = dataR
	trT = tr2.copy()
	trT.stats.channel = tr1.stats.channel[:-1]+'T'
	trT.data = dataR
	# Calculate spreading coefficient
	FSH = calc_FSH(rake,dip,strike,takeoff_ang,SRAZ)

	# Get origin time
	t0 = trT.stats.starttime + t0_padsec
	# Calculate SH wave travel time
	ttS = B_i*RR
	# Calculate P wave travel time
	ttP = VpVs*B_i*RR
	# Calculate tS - tP differential travel time
	dtSP = ttS - ttP
	# Set limits of integration
	T1 = t0 + ttS
	T2 = t0 + ttS + 2.*dtSP
	# Model moment
	M0 = calc_M0_IuSH(trT,RR,T1,T2,FSH)
	# Estimate moment magnitude
	mw = calc_mw(M0)

	return FSH,M0,mw


### TIME SERIES PROCESSING ROUTINES ###
def tr_SNR(tr,t0,t1,t2,minsamp=30):
	"""
	Calculate the SNR in dB for a trace with a noise window
	defined by data[t0:t1] and signal window devined by data[t1:t2]

	:: INPUTS ::
	:param tr: [obspy.trace] trace object with data
	:param t0: [obspy.UTCDateTime] start time of noise window
	:param t1: [obspy.UTCDateTime] end time of noise window/start time of signal window
	:param t2: [obspy.UTCDateTime] end time of signal window

	:: OUTPUT ::
	:return snr: [float] signal-to-noise ratio in decibels
	"""
	dat1 = tr.copy().trim(starttime=t0,endtime=t1).data
	# if t2 < tr.stats.endtime:
	dat2 = tr.copy().trim(starttime=t1,endtime=t2).data
	# elif t1 < tr.stats.endtime:
	# 	dat2 = tr.copy().trim(starttime=t1).data
	# 	print('ran into end of trace')
	# elif t1 > tr.stats.endtime:
	# 	print('cannot run - no signal data')
	if len(dat1[np.isfinite(dat1)]) >= minsamp and len(dat2[np.isfinite(dat2)]) >= minsamp:
		noise_rms = np.sqrt(np.nanmean(dat1**2))
		signal_rms = np.sqrt(np.nanmean(dat2**2))
		snr = 20.*np.log10(signal_rms/noise_rms)
	else:
		snr = np.nan

	return snr


def tr_aic_media(tr,ti,dt,edgesamp=2):
	"""
	Calculate the Akieki Information Criterion for a small
	window of data near a model pick to attempt to find the
	phase onset using the AIC definition from Media (1985)

	:: INPUTS ::
	:param tr: [obspy.Trace] Trace object with data
	:param ti: [obspy.UTCDateTime] modeled pick time
	:param dt: [float, seconds] window half-width surrounding the model pick time to calculate AIC
	:param edgesamp: [int] number of samples on each edge of AIC calculation window
	
	:: OUTPUTS ::
	:return taic: [obspy.UTCDateTime] time of the AIC minimum value

	:: TODO ::
	Allow options for outputting the AIC as a trace
	"""
	tr_W = tr.copy().trim(starttime=ti-dt,endtime=ti+dt)
	d_ = tr_W.data
	N_ = tr_W.stats.npts
	# if N_ >= minsamp:
	delta = tr_W.stats.delta
	if N_ > 4*edgesamp:
		aic = [k_*np.log(np.var(d_[:k_])) + (N_ - k_ - 1.)*np.log(np.var(d_[k_+1:N_])) for k_ in np.arange(edgesamp,N_-edgesamp)]
		I_min = np.nanargmin(aic)
		I_min += edgesamp # Correct for 1-indexed formulation above
		taic = tr_W.stats.starttime + delta*I_min
	else:
		taic = ti
	return taic


def rotate_waveforms(tr1,tr2,tr3,AZr,RSAZ,iTO):
	"""
	Conduct rotation of 3C raw waveforms with non-north orientation and
	assumed verticality into an LQT system with known Receiver-Source
	azimuth (back azimuth) and inclination angle, allowing for an additional
	instrument orientation correction azimuth

	:: INPUTS ::
	:param tr1: [obspy.Trace] Channel 1 trace "north-ish" oriented channel
	:param tr2: [obspy.Trace] Channel 2 trace "east-ish" oriented channel
	:param tr3: [obspy.Trace] Channel 3 trace "vertical-ish" oriented channel
	:param AZr: [float, rad] Surveyed azimuth of the station w.r.t. true north
	:param RSAZ: [float, rad] Receiver-to-Source AZimuth w.r.t. true north
	:param iTO: [float, rad] Inclination angle 

	"""
	# Calculate back-azimuth with instrument orientation
	BAZ = RSAZ - AZr
	BAZd = BAZ*(180./np.pi)

	if BAZd < 0.:
		while BAZd < 0.:
			BAZd += 360
	elif BAZd > 360.:
		while BAZd > 360.:
			BAZd -= 360.


	# Calculate the inclination angle at the instrument
	IANG = iTO - np.pi/2
	IANGd = IANG*(180./np.pi)


	try:
		datL,datQ,datT = rotate_zne_lqt(tr3.data,tr1.data,tr2.data,BAZd,IANGd)
	except:
		breakpoint()

	trL = tr3.copy()
	trL.stats.channel=tr3.stats.channel[:-1]+'L'
	trL.data = datL

	trQ = tr1.copy()
	trQ.stats.channel=tr1.stats.channel[:-1]+'Q'
	trQ.data = datQ

	trT = tr2.copy()
	trT.stats.channel=tr2.stats.channel[:-1]+'T'
	trT.data = datT

	return trL, trQ, trT


def trti2marker(tr,tmin=UTCDateTime(),tmax=None,phasename=None,kind=0):
	"""
	Use trace header information and DateTime(s) to create
	Pyrocko Snuffler phase markers
	"""
	if tr.stats.channel[1:] == 'PL':
		chan = tr.stats.channel[:2]+'3'
	elif tr.stats.channel[1:] == 'NL':
		chan = tr.stats.channel[:2]+'Z'
	elif tr.stats.channel[-1] == 'T':
		chan = tr.stats.channel[:2]+'1'
	elif tr.stats.channel[-1] == 'Q':
		chan = tr.stats.channel[:2]+'2'
	else:
		chan = tr.stats.channel

	nslc = (tr.stats.network,tr.stats.station,tr.stats.location[:2],chan)
	if isinstance(tmax,UTCDateTime):
		mark = pm.PhaseMarker((nslc,),tmin=tmin.timestamp,tmax=tmax.timestamp,phasename=phasename,kind=kind)
	else:
		mark = pm.PhaseMarker((nslc,),tmin=tmin.timestamp,tmax=None,phasename=phasename,kind=kind)
	return mark


###### PROCESSING SECTION #####
# Map root directory
ROOT = os.path.join('..','..','data')
# Map basal event wfdisc
F_BEWF = os.path.join(ROOT,'tables','BEQ_wfdisc.csv')
# Map catalog for source parameters
F_ORIG = os.path.join(ROOT,'tables','well_located_30m_basal_catalog_merr.csv')
# Map file for site orientation surveys
F_SURV = os.path.join(ROOT,'tables','Sorted_Passive_Servicing.csv')
SURV_TZC = pd.Timedelta(-6,unit='hour')
# Map file for site locations
F_SITE = os.path.join(ROOT,'tables','site.csv')

# Processing Control Parametes
save_traces = False
save_markers = True
model_plus_aic = True
aic_coef = 0.25    			# [-] tS - tP scaling for window half-width for Media (1985) AIC refinement of model pick
SP_SNR_coef = 0.5 			# [-] tS - tP scaling time for leading and lagging windows for SNR calculation

verb = 0
### CUTOFFS ###
max_age_sec = 10*24*3600 	# [sec] Maximum permitted age for a sensor installation
a_i = 3500.					# [m sec**-1] Compressional velocity in ice (Stevens and others, 2022)
B_i = a_i/1.932 			# [m sec**-1] Shear velocity in ice (This study)
a_r = 4750. 				# [m sec**-1] Compressional velocity of bed refraction (from data in Stevens and others, 2022)
B_r = a_r/1.85 			# [m sec**-1] Shear velocity of bed refraction (using dolomite/fractured limestone values)
p_i = 910 					# [kg m**-3] Density of glacier ice
p_r = 2550. 				# [kg m**-3] Density of local rock (field measurements/McCormick and others, 2023; references therein)

prefilt = {'type':'bandpass','freqmin':40,'freqmax':450}
# IPC = [-0.075,0.9] 			# [-] Integration limit Padding Coefficient (For M0 estimation)
# 							#     scaling of argmin{tS - tP} pads around the proposed phase arrival time
dtINT = [-0.005,0.030]			# [sec] integration limit bounds referenced to modeled (or AIC improved) pick time
CH_DICT = {'P':['GNZ','GP3'],'S1':['GN1','GP1'],'S2':['GN2','GP2']}
V_DICT = {'P':a_i,'S':B_i}
### FAULT GEOMETRY PARAMETERS ###
# Bedding Plane Geometries (used for fault plane orientation)
s_strike = np.pi*3./8. 		# [rad] Strike azimuth of bedding planes based on Iverson (1991) and Smart and Ford (1986)
s_dip = 5. * (np.pi/180.) 	# [rad] Dip angle of bedding plances based on Iverson (1991) and Smart and Ford (1986)

# Rake Calculation
flow_az = 44.5*(np.pi/180)  # [rad] Flow azimuth for study area (Stevens and others, 2022)
rake_ang = s_strike - flow_az # [rad] Rake angle based on bedding plane = fault orientation assumption


# Data calibration controls
declination = 15.68 		# [degE] Declination from EMM for 2019-08-10 at field site
t0_padsec = 0.5 			# [sec] front padding on waveform data from event t0



# Load event sources
df_SRC = pd.read_csv(F_ORIG,parse_dates=['t0'])
# Load WFDISC for events
df_WF = pd.read_csv(F_BEWF)
# Load SITE table
df_SITE = pd.read_csv(F_SITE)
# Load SURVEY table
df_SURV = pd.read_csv(F_SURV,parse_dates=['DateTime Mean'])

# Filter for basal event locations
df_SRC = df_SRC[df_SRC['G_filt']]

# Get mE, mN, mZ for SITE
mE,mN = Proj('epsg:32611')(df_SITE['lon'].values,df_SITE['lat'].values)
mZ = df_SITE['elev'].values*1e3
# Append metere-scaled results to dataframe
df_SITE = pd.concat([df_SITE,pd.DataFrame({'mE':mE,'mN':mN,'mZ':mZ},index=df_SITE.index)],axis=1,ignore_index=False)

df_MOD = pd.DataFrame()
cols = ['EVID','t0','sta','phase','refchan','age','RNAZ','dd m','iE rad','SRAZ','RSAZ','F*','tt sec','dtAIC','Iu_L','M0_L','SNR_L','Iu_Q','M0_Q','SNR_Q','Iu_T','M0_T','SNR_T']

### TODO: Consider turning this into a method to allow parallelization...


### OUTER LOOP ###
# Iterate across events
for EVID in tqdm(df_SRC.index[28325:]):
	if verb > 1:
		print('Processing EVID %06d'%(EVID))
	# Subset Event Features & Data
	iS_SRC = df_SRC.loc[EVID]
	# Subset WFDISC to EVID
	idf_WF = df_WF[(df_WF['EVID']==EVID)]
	# Get origin time of event
	t0 = iS_SRC['t0']
	# Get origin location
	x_src = np.array([k_ for k_ in iS_SRC[['mE','mN','mZ']].values])
	
	### STATION-LEVEL PROCESSING ###
	# Subset SITE for existing stations
	idf_SITE = df_SITE[df_SITE['sta'].isin(list(idf_WF['sta'].unique()))]


	# Create holders for output lines and markers for each event
	imarks = []
	OUTPUT_HOLDER = []
	### CONDUCT EVENT-SITE CALCULATIONS BY SITE ###
	for S_ in idf_SITE['sta']:
		if verb > 2:
			print('...station %s ...'%(str(S_)))
		# Filter for station entries that are younger than the source-time
		idf_SURV = df_SURV[(df_SURV['sta']==S_) & (df_SURV['DateTime Mean'] <= t0)]
		# Get the newest installation time
		i_t0 = idf_SURV['DateTime Mean'].max() + SURV_TZC
		# Calculate installation age
		age = (t0 - i_t0).total_seconds()/(24.*3600.)
		# Get surveyed azimuth & correct for declination
		AZr = (idf_SURV['Azimuth 2'].values[-1] + declination)*(np.pi/180.)

		# Get Site Location Coordinates in meters
		x_rcv = idf_SITE[idf_SITE['sta']==S_][['mE','mN','mZ']].values[0]
		# Calculate Source-Receiver Geometry
		RR,iTO,SRAZ,RSAZ = source_receiver_geom(x_src,x_rcv)
		# Estimate Travel Times
		ttP = RR/a_i; ttS = RR/B_i; dtSP = ttS - ttP
		# Place into dictionary
		tt_dict = {'P':ttP,'S':ttS}
		# Filter for station having a reported azimuth and takeoff angle must be greater than horizontal
		if np.isfinite(AZr) and iTO > np.pi/2:
			### LOAD WAVEFORMS ###
			st = Stream()

			jdf_WF = idf_WF[(idf_WF['sta']==S_)]
			tsmax = UTCDateTime(0)
			temin = UTCDateTime()
			for i_ in range(len(jdf_WF)):
				WF_ = os.path.join(ROOT,jdf_WF['path'].values[i_],jdf_WF['file'].values[i_])
				try:
					tr = read(WF_)[0]
					tr.stats.sampling_rate = round(tr.stats.sampling_rate)
					if tr.stats.starttime > tsmax:
						tsmax = tr.stats.starttime
					if tr.stats.endtime < temin:
						temin = tr.stats.endtime
					st += tr
				except:
					pass
			st = st.trim(starttime=tsmax,endtime=temin).merge(method=1,fill_value='interpolate').filter(**prefilt)
			npts_ref = st[0].stats.npts
			for tr____ in st:
				if tr____.stats.npts != npts_ref:
					breakpoint()

			# Only proceed if there are 3C data
			if len(st) == 3:
				tr1 = st.select(channel='G?1')[0]
				tr2 = st.select(channel='G?2')[0]
				tr3 = st.select(channel='G?[3Z]')[0]
				
				trL,trQ,trT = rotate_waveforms(tr1,tr2,tr3,AZr,RSAZ,iTO)

				for phz_type in tt_dict.keys():

					## CALCULATE RADIATION COEFFICIENT ##
					if phz_type == 'P':
						FF = calc_FP(rake_ang,s_dip,s_strike,iTO,SRAZ)
						_tr_ = trL
					elif phz_type == 'S':
						FF = calc_FSH(rake_ang,s_dip,s_strike,iTO,SRAZ)
						_tr_ = trT
		
						
					## ATTEMPT AIC ONSET-TIME REFINEMENT ON REFERENCE CHANNEL ##
					# Get relevant initial modeled travel time
					tt = tt_dict[phz_type]

					# Create initial modeled arrival time markers for QC
					imarks.append(trti2marker(_tr_,tmin=UTCDateTime(str(t0)) + tt,tmax=UTCDateTime(str(t0)) + tt,phasename='^%s'%(phz_type),kind=3))

					# Convert to DateTime
					that = UTCDateTime(str(t0)) + tt
					# Calculate SNR of initial modeled arrival time
					SNRo = tr_SNR(_tr_,that - SP_SNR_coef*dtSP,that,that + SP_SNR_coef*dtSP)
					# Calculate AIC for data surrounding the initial arrival time
					taic = tr_aic_media(_tr_,that,dtSP*aic_coef)
					# Calculate the SNR coninciding with the AIC minimum

					SNRi = tr_SNR(_tr_,taic - SP_SNR_coef*dtSP,taic,taic + SP_SNR_coef*dtSP)
					# Calculate the difference between the AIC and initial arrival time	
					dtaic = taic - that

					# Check if there is an improvement in SNR and the AIC doesnt peg the limits of the input AIC data
					if SNRi > SNRo and dtaic < np.abs(dtSP*aic_coef):
						tt = tt + dtaic
						SNR = SNRi
						dSNR = SNRi - SNRo
					# Failing this, stick with the original proposed arrival time
					else:
						SNR = SNRo
						dSNR = np.nan
						dtaic = np.nan

					# Append a new marker if there is an improvement via AIC
					TT = UTCDateTime(str(t0)) + tt
					if np.isfinite(dtaic):
						imarks.append(trti2marker(_tr_,tmin=TT,tmax=TT,phasename='^%saic'%(phz_type),kind=4))

					

					## CONDUCT MOMENT AND SNR CALCULATIONS ON ALL CHANNELS
					
					# Set limits of integration
					Ti1 = TT + dtINT[0]
					Ti2 = TT + dtINT[1]
					# Make a marker for the limits of integration
					imarks.append(trti2marker(_tr_,tmin=Ti1,tmax=Ti2,phasename='Iu%s'%(phz_type),kind=5))
					# Make line entry on reference channel
					line = [EVID,t0,S_,phz_type,_tr_.stats.channel,age,AZr,RR,iTO,SRAZ,RSAZ,FF,tt,dtaic]
					# Iterate across rotated traces
					for itr in [trL,trQ,trT]:
						# Get moment and evaluated integral
						M0,int_u = calc_M0_WSR(itr,RR,Ti1,Ti2,FF,vv=V_DICT[phz_type])
						ithat = UTCDateTime(str(t0))+tt
						SNR = tr_SNR(itr,ithat - SP_SNR_coef*dtSP,ithat,ithat + SP_SNR_coef*dtSP)
						line.append(int_u)
						line.append(M0)
						line.append(SNR)

					OUTPUT_HOLDER.append(line)
		
	if len(OUTPUT_HOLDER) > 1:
		idf_MOD = pd.DataFrame(OUTPUT_HOLDER,columns=cols)
		idf_MOD.to_csv(os.path.join(ROOT,jdf_WF['path'].values[0],'EVID%06d_mw_3C.csv'%(EVID)),header=True,index=False)

		df_MOD = pd.concat([df_MOD,idf_MOD],axis=0,ignore_index=True)
	if save_markers:
		pm.save_markers(imarks,os.path.join(ROOT,jdf_WF['path'].values[0],'EVID%06d_that_3C.dat'%(EVID)))
breakpoint()
df_MOD.to_csv(os.path.join(ROOT,'tables','mw_estimation_results_3C.csv'),header=True,index=False)

	# # OPTIONAL: Make modeled phase markers for QC
	# 		if save_markers:
	# 			# Create absolute DateTime times for modeled phases (and potentially updated S_wave arrival)
	# 			tP = UTCDateTime(str(t0)) + ttP
	# 			tS = UTCDateTime(str(t0)) + ttS
	# 			# S-wave onset marker for the ??1 channel (allows use even if rotated traces are not output)
	# 			imarks.append(trti2marker(st[0],tmin=tS,tmax=tS,phasename='S',kind=4))
	# 			_tr_z = tr_T.copy()
	# 			if _tr_z.stats.channel[:2] == 'GP':
	# 				_tr_z.stats.channel = _tr_z.stats.channel[:2]+'3'
	# 			else:
	# 				_tr_z.stats.channel = _tr_z.stats.channel[:2]+'Z'
	# 			imarks.append(trti2marker(_tr_z,tmin=tP,tmax=tP,phasename='P',kind=3))
	# 		# Set absolute times for limits of integration
	# 		Ti1 = UTCDateTime(str(t0)) + ttS + IPC[0]*dtSP
	# 		Ti2 = UTCDateTime(str(t0)) + ttS + IPC[1]*dtSP # Based on analysis by Stork and others (2014)
	# 		# Add integration time-range 
	# 		if save_markers:
	# 			imarks.append(trti2marker(st[0],tmin=Ti1,tmax=Ti2,phasename='uSH',kind=5))
	# 		# Calculate M0
	# 		# M0stork = calc_M0_IuSH(tr_T,RR,Ti1,Ti2,FSH,p_i=p_i,p_b=p_r,B_i=B_i,B_b=B_r,)
	# 		M0T,int_uT = calc_M0_IuSH(tr_T,RR,Ti1,Ti2,FSH,p_i=p_i,p_b=p_r,B_i=B_i,B_b=B_r)
	# 		if len(st) == 3:
	# 			M0R,int_uR = calc_M0_IuSH(tr_R,RR,Ti1,Ti2,FSH,p_i=p_i,p_b=p_r,B_i=B_i,B_b=B_r)
	# 		else:
	# 			M0R, int_uR = np.nan,np.nan
	# 		if not model_plus_aic:
	# 			SR_SNRo = tr_SNR(tr_T,Ti1 - SP_SNR_coef*dtSP,Ti1,Ti1 + SP_SNR_coef*dtSP)








	# 		# Calculate mw
	# 		if M0T > 0:
	# 			mw = calc_mw(M0T)
	# 		else:
	# 			mw = np.nan

	# 		# Save data on solution
	# 		cols = ['EVID','source time','dtaic','dSNR','sta','chan','age','d m','iTO','SRAZ','RSAZ','FSH','int uT','int uR','M0T','M0R','mwT','SH_SNR']
	# 		line = [EVID,t0,dtaic,dSNR,S_,tr_T.stats.channel,age,RR,iTO,SRAZ,RSAZ,FSH,int_uT,int_uR,M0T,M0R,mw,SH_SNR]
	# 		OUTPUT_HOLDER.append(line)

			# ## Option to save rotated traces to disk - CAN TAKE A LOT OF SPACE
			# if save_traces:
			# 	tr_R.write(os.path.join(ROOT,jdf_WF['path'].values[0],'%s.%s.%s.%s_%06d.sac'%\
			# 			   (tr_R.stats.network,tr_R.stats.station,tr_R.stats.location,tr_R.stats.channel,EVID)))
			# 	tr_T.write(os.path.join(ROOT,jdf_WF['path'].values[0],'%s.%s.%s.%s_%06d.sac'%\
			# 			   (tr_T.stats.network,tr_T.stats.station,tr_T.stats.location,tr_T.stats.channel,EVID)))
			
		# # If the station installation is younger than the cutoff age
		# ### CORE PROCESSING ###
		# if np.isfinite(AZr):

		# 	# Get waveform files for station
		# 	## GET HORIZONTAL
		# 	jdf_WF = idf_WF[(idf_WF['sta']==S_) & (idf_WF['chan'].isin(['GP1','GP2','GN1','GN2']))].sort_values('chan')
		# 	# Load waveforms
		# 	st = Stream()
		# 	for i_ in range(len(jdf_WF)):
		# 		WF_ = os.path.join(ROOT,jdf_WF['path'].values[i_],jdf_WF['file'].values[i_])
		# 		tr = read(WF_)[0]
		# 		tr.stats.sampling_rate = round(tr.stats.sampling_rate)
		# 		st += tr

		# 	## GET VERTICAL
		# 	jdf_WF = idf_WF[(idf_WF['sta']==S_) & (idf_WF['chan'].isin(['GP3','GNZ']))].sort_values('chan')
		# 	tr_Z = read(os.path.join(ROOT,jdf_WF['path'].values[0],jdf_WF['file'].values[0]))[0]
		# 	tr_Z.stats.sampling_rate = round(trZ.stats.sampling_rate)
		# 	st += tr_Z

# # #### UNDER DEVELOPMENT ####
# run_as_MP = False

# def run_MP_event_processing()

# if run_as_MP:
# 	if __name__ == "__main__":
# 		with mp.Pool(n_pool) as p:
# 			sout = p.map(run_event_processing,)


