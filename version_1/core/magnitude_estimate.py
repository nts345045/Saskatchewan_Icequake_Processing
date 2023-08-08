"""
:module: magnitude_estimate.py
:purpose: Estimate event magnitude from S-coda energy integral method (e.g., Roeoesli and others, 2016)
:author: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 
"""

import os
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace, UTCDateTime, read_inventory, read_event
from glob import glob
from scipy.optimize import curve_fit
from tqdm import tqdm

### SUPPORTING ROUTINES ###

def plane_fun(P,mx,my,dz):
    """
    Functional representation of a plane formatted for use in scipy.optimize.curve_fit()
    to fit a planar feature to bed elevation model points
    :: INPUTS ::
    :param P: [numpy.ndarray] Position matrix with column vectors of x & y data (n,2)
    :param mx: [float] x-axis slope
    :param my: [float] y-axis slope
    :param dz: [float] z-axis offset

    :: OUTPUT ::
    :return z(P): [numpy.ndarray] modeled elevations at positions P
    """
    x = P[:,0]
    y = P[:,1]
    return x*mx + y*my + dz

def XmYm2strdip(xm,ym):
    """
    Convert slope representation of a fault plane with arbitrary vertical position
    into strike & dip values.

    Equation set provided by interaction with ChatGPT and independently verified 
    by human QC.

    :: INPUTS ::
    :param xm: [float] x-axis slope value
    :param ym: [float] y-axis slope value

    :: OUTPUTS ::
    :return phi_f: [float, rad] strike angle
    :return az: [float, rad] strike azimuth
    :return dlt: [float, rad] dip angle

    """
    # Calculate slope magnitude
    S = (xm**2 + ym**2)**0.5
    # Calculate dip angle
    dlt = np.arctan(S/1.)
    # Calculate strike angle (clockwise from the north basis vector)
    phi_f = np.arctan2(ym,xm)
    # Calculate azimuth angle
    az = phi_f + 0.5*np.pi

    return phi_f,az,dlt

def az2rake(strike_az,slip_az):
    """
    Calculate the rake angle from a slip azimuth and fault-plane orientation
    :: INPUTS ::
    :param strike_az:   [radians] fault strike azimuth
    :param slip_az:     [radians] fault slip azimuth

    :: OUTPUT ::
    :return rake:       [radians] rake angle
    """
    rake = slip_az - strike_az


### CORE PROCESS ###

def moment_mag_SI(M0):
    """
    Calculate moment magnitude from a moment release
    Eq. 3 in Roeoesli and others (2016)
    :: INPUT ::
    :param M0:      [float] Moment in Newton-meters (Nm) (1e3 dyne-cm)
    :: OUTPUT ::
    :return mw:     [float] Moment magnitude
    """
    mw = np.log10(M0) - 6
    return mw

def Scoda_energy_moment(tr_ush,RR,Fsh,Ssh=2,p_i=917,p_b=2550,B_i=1836,B_b=2510,int_meth='obspy',int_kwargs={'method':'cumtrapz'}):
    """
    Estimate event energy from the SH-wave coda ground displacement observed
    on the transverse channel (Boatwright, 1980) Eq. 2 in Roeoesli and others (2016).

    Limits of integration are assumed as the starttime and endtime of the input trace

    :: INPUTS ::
    :param tr_ush:  [obspy.Trace] Horizontal shear wave displacement seismogram [m]
    :param RR:      [float] source-receiver distance radius [m]
    :param Fsh:     [float] Radiation coefficient
                            (Aki & Richards, 2002 eqns 4.89-4.91; after Roeoesli and others, 2016)
    :param Ssh:     [float] Free-space amplification factor - default Ssh
    :param p_i:     [float] Ice density [kg m**-3]
                        Default value = 917 - e.g., Cuffey & Patterson (2010)
    :param p_b:     [float] Near-interface bed density [kg m**-3]
                        Default value = 2550 - McCormick and others (2023); McCormick (2022; pers. comm. 2023)
    :param B_i:     [float] Shear-wave velocity in ice [m s**-1]
    :param B_b:     [float] Near-interface bed shear-wave velocity [m s**-1]
                        Default value = 2510 - Based on Stevens and others (2022) their Figure S2,
                                               and a typical Vp/Vs ratio for carbonates of 1.68
    :param int_meth:[str] Option for integral estimation method - defaults to obspy.trace.Trace.integrate
    :param int_kwargs: [dict] key-word arguments for integration method selected

    :: OUTPUT ::
    :return M0: [float] seismic moment [N m]
    """
    numerator = 4.*np.pi*(p_i*p_r*B_i*B_r**5)**0.5
    denominator = Fsh*Ssh
    if int_meth.lower() == 'obspy':
        integral = tr_ush.integrate(**int_kwargs)
    else:
        print('invalid integration method option')
        integral = np.nan


    return M0


def calc_Fsh(lbd,dlt,Dphi,iTO):
    """
    Calculate the radiation coefficient F^{SH} using equation 4.88 in Aki & Richards (2002)
    :: INPUTS ::
    :param lbd: [rad] slip rake
    :param dlt: [rad] fault dip
    :param Dphi: [rad] source-receiver azimuth
    :param iTO: [rad] ray take-off angle 
                note: we assume indicence==takeoff angles for near-source
    :: OUTPUT ::
    :return Fsh: [float] radiation coefficient
    """
    part1 = np.cos(lbd)*np.cos(dlt)*np.cos(iTO)*np.sin(Dphi)
    part2 = np.cos(lbd)*np.cos(2.*iTO)*np.cos(2.*Dphi)
    part3 = np.sin(lbd)*np.cos(2.*dlt)*np.cos(iTO)*np.cos(Dphi)
    part4 = 0.5*np.sin(lbd)*np.sin(2.*dlt)*np.sin(iTO)*np.sin(2.*Dphi)

    Fsh = part1 + part2 + part3 - part4

    return Fsh

def estimate_fault_geometry(X_bed,Y_bed,Z_bed,S_bed,M_bed,sXY,flow_ang,samp_rad=25,max_nan_frac=0.25,output='full'):
    """
    Estimate the strike, dip, and rake of the basal fault from bed DEM and
    a specified flow angle (right-handed, with "east" = 0 rad)

    :: INPUTS ::
    :param X_bed: [numpy.ndarray,float] X-coordinates of model gridpoints [m]
    :param Y_bed: [numpy.ndarray,float] Y-coordinates of model gridpoints [m]
    :param Z_bed: [numpy.ndarray,float] Z-coordinates of model gridpoints [m]
    :param S_bed: [numpy.ndarray,float] Standard deviations of model elevations [m]
    :param M_bed: [numpy.ndarray,Bool]  In-bounds mask for model gridpoints
    :param sXY: [array-like, float] X-Y coordinates for source location [m]
    :param flow_ang: [float] ice-flow angle relative to X-axis [rad]
    :param samp_rad: [float] sampling radius for fitting a fault-plane [m]
    :max_nan_frac: [float] maximum fraction of sampled values that are NaN for proceeding with plane fitting
    :output: [str] output method (see OUTPUTS below)

    :: OUTPUTS ::
    :return popt: [numpy.ndarray] model fit parameters [X-slope,Y-slope,Z-offset] from scipy.optimize.curve_fit
    :return pcov: [numpy.ndarray] model covariance matrix from scipy.optimize.curve_fit
    """
    # Calculate distances
    D_mat = ((X_bed - sXY[0])**2 + (Y_bed - sXY[1])**2)**0.5
    # Get sample point mask
    R_mask = D_mat <= samp_rad

    # Find amount of valid data points
    MR_frac = np.sum(M_bed[R_mask])/len(np.ravel(M_bed[R_mask]))
    # If union of radius mask and in-bounds mask yield enough valid data
    if MR_frac >= 1. - max_nan_frac:
        PP,ZZ = [],[]
        for i_ in range(X_bed.shape[0]):
            for j_ in range(X_bed.shape[1]):
                if M_bed[i_,j_] and R_mask[i_,j_]:
                    PP.append([X_bed[i_,j_],Y_bed[i_,j_]])
                    ZZ.append([Z_bed[i_,j_],S_bed[i_,j_]])
        PP = np.matrix(PP)
        ZZ = np.matrix(ZZ)
        popt,pcov = curve_fit(plane_fun,PP,ZZ[:,0],sigma=ZZ[:,1],absolute_sigma=True)

        return popt,pcov

    else:
        return np.nan,np.nan

def geometry_processing_SH(tr3,tr1,tr2,Istrike,Idip,Irake,Sstrike,Sdip,Srake,Ip,Sp):
    """
    For a defined source location, receiver location, bed geometry,
    rotate ZNE seismograms to LQT and calculate radiation coefficient
    for SH waves

    :: INPUTS ::
    :param tr3: [obspy.Trace] "vertical" trace
    :param tr1: [obspy.Trace] "magnetic north" trace
    :param tr2: [obspy.trace] "magnetic east" trace
    :param Istrike: [float,rad] Instrument strike angle
    :param Idip: [float,rad] Instrument dip angle, right hand rule
    :param Irake: [float,rad] Instrument rake angle (north arrow misalignment angle)
    :param Sstrike: [float,rad] Fault patch strike
    :param Sdip: [float,rad] Fault patch dip
    :param flow_az: [float,rad] flow-direction angle (used to calculate slip rake)
    :param EVENT: [obspy.Event] Event entry from catalog for specific event
    :param STATION: [obspy.Station] Station entry from inventory corresponding to traces

    :: OUTPUTS ::
    :return RR: [float] Source-receiver separation [m]
    :return Fsh: [float] 
    :return stLQT: [obspy.Stream] rotated 3-C seismogram
    """


    ## GEOMETRY/RADIATION COEFFICIENT MODELING ##
    # Pull source-receiver angles from NonLinLoc modeled values
    iTO = None      # take off angle
    seaz = None     # source-receiver azimuth
    Dphi = seaz - Fstrike
    RR = None

    # Calculate radiation factor
    Fsh = calc_Fsh(Frake,Fdip,Dphi,iTO)

    return RR,Fsh,stLQT

def preprocess_waveforms(df_wf,df_o,station,starttime,endtime,resp_file,chan_map={'GPZ':'GP3','GP1':'GP1','GP2':'GP2'}):
    """
    Conduct data fetch and pre-processing of waveform data with the following steps:
    1) Filter wfdisc for appropriate station & starttime & endtime
    2) Load waveform data
    3) Append instrument response metadata
    4) Estimate instrument orientation

    """
    IND = (df_wf['sta']==station)&\
          (df_wf['starttime']<=endtime)&\
          (df_wf['endtime']>=starttime)
    idf_wf = df_wf[IND]
    st = Stream()
    for i_ in range(len(idf_wf)):
        tr = read(os.path.join(idf_wf['path'].values[i_],idf_wf['file'].values[i_]))
        tr.stats.sampling_rate = np.round(tr.stats.sampling_rate)
    # merge traces
    st = st.merge(method=1,fill_value='interpolate')
    # trim traces
    st = st.trim(starttime=UTCDateTime(str(starttime)),endtime=UTCDateTime(str(endtime)))
    # Attach response file and update channel naming for rotation
    for tr in st:
        tr.stats.response = read_inventory(resp_file)
        tr.stats.channel = chan_map[tr.stats.channel]

    





    ## METDATA PREPARATION ##
    # Compose initial data stream
    st312 = Stream([tr3,tr1,tr2])
    # Compose single-station inventory for rotation
    INV = Inventory()
    # Append STATION to INVentory
    #TODO

    ## DATA ROTATION ##
    # Rotate stream to ZNE
    stZNE = st312.copy()._rotate_to_zne(INV)
    # Rotate to LQT
    stLQT = st.copy().rotate('ZNE->LQT',back_azimuth=,inclination=)

### DATA WRANGLING/PREPARATION ###

def process_solo_orientations(S_es,S_ed,S_ea,S_os,S_od,S_oa):
    """
    Estiamte instrument orientation time-series give time-indexed 
    pandas.Series of strike (S_?s), dip (S_?d), and azimuth (S_?a) 
    from SmartSolo DigiSolo.LOG information (?=e) 
    and field observations (?=)
    
    :: INPUTS ::
    
    """

    return None



### PROCESSING SECTION ###
ROOT = '/mnt/icy1/Zoet_Lab/UW_Projects/Saskatchewan_Glacier/data/seismic/2019'
cat_file = os.path.join(ROOT,'catalog.xml')
wfd_file = os.path.join(ROOT,'wfdisc.csv')

dt_front = pd.Timedelta(0.006,unit='sec')
dt_back = pd.Timedelta()
# Load catalog & inventory data
print('Loading catalog - this takes some time')
cat = read_events(cat_file)
# inv = read_inventory(inv_file)
dfw = pd.read_csv(wfdisc_file,parse_dates=['starttime','endtime'])
# Create data holder
holder = []
# Iterate across each event
for E_ in tqdm(cat.events):
    # Get origin
    ORIG = E_.origins[0]
    # Get origin uncertainties
    OU = ORIGIN.origin_uncertainty
    # Iterate across arrivals (picks used for source estimation)
    for i_,A_ in enumerate(ORIG.arrivals):
        # Process only S-wave arrivals
        if A_.phase in ['s','S']:
            # Pull associated pick
            if E_.picks[i_].resource_id == A_.pick_id:
                PICK = E_.picks[i_]
            # If there's a misordering, scan picks
            else:
                for P_ in E_.picks:
                    if P_.resource_id == A_.pick_id:
                        PICK = P_
            # Pull takeoff angle & SEAZ from model
            iTO = ARR.takeoff_angle
            seaz = ARR.azimuth
            # Get station name
            sta = PICK.waveform_id['station_code']
            # Get pick time
            tA = pd.Timestamp(PICK.time.isoformat())
            # Get start and endtimes for data pull
            tS = tA - dt_front
            tE = tA + dt_back
            flist = query_wf_files(dfw,)
            

            # Get station orientation from field data

            # 


