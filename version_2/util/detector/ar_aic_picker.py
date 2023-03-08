"""
:module: ar_aic_picker.py
:purpose: Methods for using an adapted version of Akazawa (2004)'s strong-motion detection algorithm used in Stevens (2022, Chapter 4)
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major update: 15. APR 2020
    Import paths updated and header added 8.MAR 2023 for GitHub upload
"""


import sys
import numpy as np
import pandas as pd
from warnings import warn
# sys.path.append('/usr1/ntstevens/SCRIPTS/PYTHON/SGGS_Workflows')
# from detectors import kurtosis as okt
# from detectors import f_dist_fit as fdf
# from dbtools.wf_query import db2st
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

sys.path.append(os.path.join('..'))
import util.detector.kurtosis as okt
import util.detector.f_dist_fit as fdf
from util.pisces_DB.data import db2st



def classic_sta_lta_diff_py(trace,tsta=0.01,tlta=0.1):
    """
    Calculate the difference of the LTA-STA for a classic STA/LTA scheme
    adapted from obspy.signal.trigger.classic_sta_lta_py
    """
    sr = trace.stats.sampling_rate
    nsta = int(tsta*sr)
    nlta = int(tlta*sr)
    a = trace.data
    sta = np.cumsum(a**2)
    sta = np.require(sta, dtype=np.float)
    lta = sta.copy()
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]

    sta[:nlta - 1] = 0

    # Overwrite trace data
    trace.data = sta-lta
    trace.stats.processing.append('extended trigger methods: classic STA-LTA, sta: %.4e, lta: %.4e'%(nsta,nlta))
    
    return trace


def classic_lta_sta_diff_py(trace,tsta=0.01,tlta=0.1):
    """
    Calculate the difference of the LTA-STA for a classic STA/LTA scheme
    adapted from obspy.signal.trigger.classic_sta_lta_py
    """
    sr = trace.stats.sampling_rate
    nsta = int(tsta*sr)
    nlta = int(tlta*sr)
    a = trace.data
    sta = np.cumsum(a**2)
    sta = np.require(sta, dtype=np.float)
    lta = sta.copy()
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]

    sta[:nlta - 1] = 0

    # Overwrite trace data
    trace.data = lta - sta
    trace.stats.processing.append('extended trigger methods: classic STA-LTA, sta: %.4e, lta: %.4e'%(nsta,nlta))
    
    return trace


def recursive_sta_lta_diff_py(trace,tsta,tlta):
    """
    Calculate the difference of the STA & LTA for a recursive STA/LTA scheme
    adapted from recursive_sta_lta_py
    """
    sr = trace.stats.sampling_rate
    a = trace.data
    try:
        a = a.tolist()
    except Exception:
        pass
    nsta = int(tsta*sr)
    nlta = int(tlta*sr)
    m = len(a)
    csta = 1./nsta
    clta = 1./nlta
    sta = 0.
    lta = 1e-99
    crf = [0.0] * m
    icsta = 1 - csta
    iclta = 1 - clta
    for i_ in range(1, m):
        sq = a[i_]**2
        sta = csta*sq + icsta*sta
        lta = clta*sq + iclta*lta
        crf[i_] = sta - lta
        if i_ < nlta:
            crf[i_] = 0.
        CRF = np.array(crf)
    trace.data = CRF
    trace.stats.processing.append('extended trigger methods: Recursive STA-LTA, sta: %.4e, lta: %.4e'%(nsta,nlta))
    return trace

def snrdB_tr(tr,t_pick,pad_sec,epsilon=1e-10):
    """
    Calculate the SNR of a input obspy trace given a UTCDataTime pick time and a 
    forward/backward padding time. Include an epsilon term to avoid 0-division

    """
    # Do time sanity checks
    if tr.stats.endtime > t_pick + pad_sec:
        tei = t_pick + pad_sec
    else:
        tei = tr.stats.endtime

    if tr.stats.starttime < (t_pick - pad_sec):
        tsi = t_pick - pad_sec
    else:
        tsi = tr.stats.starttime

    n_rms = np.sqrt(np.mean(tr.copy().trim(starttime=tsi,endtime=t_pick).data**2))
    s_rms = np.sqrt(np.mean(tr.copy().trim(starttime=t_pick,endtime=tei).data**2))
    snr = 10.*np.log10((s_rms/(n_rms + epsilon))**2)
    #snr = 10.*np.log10((s_rms / n_rms)**2)
    return snr

def calc_aic(data,epsilon=1e-10):
    """
    Calculate the AIC of a data vector directly from time-series data using the method
    of Maeda (1985). includes machine precision scale term epsilon that adds a small value
    to log() arguments in the event of very small variances in data.
    """
    N = len(data)
    aic = np.zeros((N))

    for k_ in range(N):
        if 0 < k_ < N:
            k1 = k_ + 1
            aic[k_] = (float(k_)*np.log(np.var(data[:k_]) + epsilon) + float(N - k1)*np.log(np.var(data[k1:]) + epsilon))
        elif k_ == 0:
            aic[k_] = float((N - 1.))*np.log(np.var(data[1:]) + epsilon) + 1.
        elif k_ == N:
            aic[k_] = float(k_)*np.log(np.var(data) + epsilon)
    # Replace infinite values with np.nan values
    aic[~np.isfinite(aic)] = np.nan
    minaic = np.nanargmin(aic)
    return aic, minaic


 
def timeininterval(time,ti,tf,code=None):
    """
    Subroutine to make sure that a given time falls within a time window
    """
    if ti <= time <= tf:
        return time
    elif ti > time:
        print('%s: input time before initial time, adjusted to initial time.'%(code))
        return ti
    elif time > tf:
        print('%s: input time after final time, adjusted to final time'%(code))
        return tf
    else:
        return None


def snr_max_scan_tr(tr,times,snrmin=3.,pad_samples = 15, dtmax=0.05):
    """
    Given a list of potential pick times, select an optimal "pick" time based upon the
    maximization of the signal-to-noise ratio of an input trace with equal signal and noise
    sampling windows.
    Outputs an estimate
    """
    sr = tr.stats.sampling_rate
    dt = max(times) - min(times)
    if dt <= dtmax:
        dn = dt*sr + 1 + pad_samples*2
        snr_v = np.zeros((dn,))
        for i_ in range(dn):
            ti = min(times) + i_/sr - pad_samples/sr
            snr_v[i_] = snrdB_tr(tr,ti)
        i_max = np.argmax(snr_v)
        if snr_v[i_max] >= snrmin:
            # Eventually do a PDF fit here
            t_peak = (i_max + pad_samples)*sr + min(times)
            snr_peak = snr_v[i_max]
#            n_std = 
            
    return t_peak, snr_peak

def ar_aic_pick_parser(times,snrs,snrmin=3.,snrtol=1.,dttol=0.003,dtmax=0.03,verb=0):
    """
    Compare the two pick estimates produced for each channel by the ar_aic_picker routine and select a preferred value
    based upon the relative signal-to-noise (snr) ratios and relative pick times
    
    :: INPUTS ::
    :type times: list
    :param times: times of picks
    :type snrs: list
    :param snrs: snr values calculated for picks --> MUST MATCH THE ORDER OF "times"
    :type snrmin: float
    :param snrmin: minimum snr value acceptable
    :type snrtol: float
    :param snrtol: tolerance level to say snr values are "identical"
    :type dttol: float
    :param dttol: tolerance level to say time values are "identical"
    :type dtmax: float
    :param dtmax: maximum difference in pick times permissable - both picks rejected if this dt value is exceeded
    :type verb: float OR int
    :param verb: level of verbosity for reporting processing steps to terminal

    """
    # Make list holders for accepted values based on times and SNR
    atimes = []
    asnrs = []
    # If both estimates have a large enough SNR...
    if snrs[0] > snrmin and snrs[1] >= snrmin:
        # If the SNR difference is below the tolerance level...
        if abs(snrs[0] - snrs[1]) <= snrtol:
            # If the time difference between the picks is reasonable, take the earlier pick
            if abs(times[0] - times[1]) <= dtmax:
                t_pref = min(times)
                snr_pref = snrs[times.index(t_pref)]
                # If the time difference is insignificant, assign a pick uncertianty scaled with the tolerance
                if abs(times[0] - times[1]) <= dttol:
                    t_err = dttol
                # Else if time difference is significant, but reasonable ...
                else:
                    t_err = abs(times[0])
            # If the time difference between the picks is unreasonable, reject both
            t_pref = None
            snr_pref = None
            t_err = None
        # Else if the SNR values are large enough and the time difference is small enough...
        elif abs(times[0] - times[1]) <= dtmax:
            # Take the value of the larger SNR valued pick.
            t_pref = times[snrs.index(max(snrs))]
            snr_pref = max(snrs)
            # If the time values are nearly identical....
            if abs(times[0] - times[1]) <= dttol:
                # Scale uncertainty to the dt tolerance level.
                t_err = dttol
            elif abs(times[0] - times[1]) > dttol:
                # Use the time difference as the uncertainty
                t_err = abs(times[0] - times[1])
        # Even if SNR values check out, if the time difference is above tolerance, reject both.
        else:
            t_pref = None
            snr_pref = None
            t_err = None
    # If only one pick has an acceptable SNR value
    elif max(snrs) >= snrmin:
        # Take the value of the only acceptable SNR valued pick.
        t_pref = times[snrs.index(max(snrs))]
        snr_pref = max(snrs)
        # and use the assumed dt tolerance as the pick uncertainty
        t_err = dttol

    # If SNR is insufficient for both, reject both.
    else:
        t_pref = None
        t_err = None
        snr_pref = None

    return t_pref, t_err, snr_pref




#

#### PRIMARY ROUTINE #### -- EVID removed from this version of the code...
def ar_aic_station_picker(Zstream,Hstream,sta,trigger,trig_pad_sec,Pmute,
                         Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                         Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                         pickprefKw,snr_wind,verb=0):
    """
    Core of the AR-AIC picking routine. Processes a single station with 3-C data

    """
    fps = 0.25
    picklist=[]
    Zst = Zstream.select(station=sta).copy()
    Hst = Hstream.select(station=sta).copy()
    # Get Trace sampling Rate & Station Name
    sr = Zst[0].stats.sampling_rate
    
    if verb > 5:
        print(sta)
        print('Picking %s (%s)',sta,str(okt.UTCDateTime()))

    # Get Time Indices 1 & 2 - By station on Z channel
    # If station has t2 value provided in input arguments, skip CRF calculation
    tr_i = Zst.copy()[0]
    if np.var(tr_i.data) < 1e-8:
        print('Very low variance (< 1e-8) in input data, likely uniform-valued, skipping picking attempt')
    else:
        tr_f = tr_i.copy().filter(Pfilt1,**PfKw1)
        
        # If the station is included in the trigger...
        if sta in trigger['stations']:
            # Calculate the front-padded time from the peak CRF value time...
            t1_i = trigger['peak_times'][trigger['stations'].index(sta)] - trig_pad_sec[0]
        # Else, assign a dummy value of 1970-1-1 that is definitely outside the data range
        else:
            t1_i = okt.UTCDateTime(0)

        # If this candidate t1 is in the data range
        if t1_i >= tr_i.stats.starttime and t1_i < tr_i.stats.endtime:
            # Extract peak CRF time for the current station if included in the t2_dict
            t2_i = t1_i + trig_pad_sec[0]
            if verb > 3:
                print('Using existing trigger for t1_i.')
        # Otherwise, calculate a new CRF and use the peak value for t2
        else:
            t1_i = trigger['time'] - trig_pad_sec[0]
            # Process trace & find peak value to determine station t2
    #            crf_i = Pcrf(tr_f.copy(),Pcrfkw):
            # crf_i = Pcrfn(tr_f,**PcKws) # TODO TODO: update inputs
            crf_i = tr_f.copy().trigger('zdetect',sta=0.01)
            max_i = np.nanargmax(crf_i.data)-crf_i.stats.delta
            t2_i = t1_i + max_i / sr
            if verb > 5:
                print('Picked new trigger max for t2_i.')
        if verb > 5:
            print('t1',t1_i)
            print('t2',t2_i)

        # If peak time falls after the start of the traces by at least half the padding distance, proceed
        if (t2_i - t1_i) > trig_pad_sec[0]*fps:
            # Calculate AIC for trimmed trace t = [t1_i,t2_i]
            aicP1,aic_i = calc_aic(tr_f.copy().trim(starttime=t1_i, endtime= t2_i).data)
            t3_i = t1_i + float(aic_i)/float(sr)

            # Calculate SNR for first P pick estimate using input of the full initial trace (after pre-processing)
            # it will be internally trimmed in the snrdB_tr() method.
            snrP1 = snrdB_tr(tr_f.copy(),t3_i,snr_wind)
            if verb > 3:
                print('t3',t3_i,'P candidate 1. snr',snrP1)        
            # Form window for second AIC attempt with a sanity check
            dt1 = abs(t2_i - t3_i)
            dt2 = trig_pad_sec[0]*fps 
            dt_test = max([dt1,dt2])
            # If there is a valid window to continue processing, proceed...
            if t3_i - dt_test > t1_i:
                if dt1 > dt2:
                    t4_i = t2_i - 2*dt1
                else:
                    t4_i = t2_i - dt2
                    if verb > 2:
                        print(tr_f.stats.station,'t2-t3 window too short, using',int((trig_pad_sec[0]*fps)*sr),'sample window instead')


                # Apply relaxed filtering for second AIC attempt - do overwrite of earlier tr_f as it is no longer necessary
                tr_f = tr_i.copy().filter(Pfilt2,**PfKw2)
                # Calculate AIC for trimmed trace
                aicP2,aic_i = calc_aic(tr_f.copy().trim(starttime=t4_i,endtime=t2_i).data)
                # Write minimum to t5_i
                t5_i = t4_i + aic_i / sr
                if verb > 3:
                    print(tr_f.stats.station,'ts',tr_f.stats.starttime,'te',tr_f.stats.endtime)
                if verb > 5:
                    print('t4',t4_i)

                # Get t5_i snr from trace
                snrP2 = snrdB_tr(tr_f.copy(),t5_i,snr_wind)
                if verb > 3:
                    print('t5',t5_i,'P candidate 2. snr',snrP2)

                # Get preferred P-Pick 
                t_pref,t_err,snr_pref = ar_aic_pick_parser([t3_i,t5_i],[snrP1,snrP2],**pickprefKw)
                if t_pref is not None and verb > 3:
                    print('Pick Accepted',t_pref,t_err,snr_pref)
                # So long as a preferred pick time is selected 
                if t_pref is not None:
                    # If t3_i = t_pref, write it as the preferred pick & use uncertainty as 0.5*dt_p
                    if t_pref == t3_i:
                        picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,t_err,1.])
                        picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,-999,0.])
                    # If t5_i = t_pref, write it as the preferred pick & use uncertainty as 0.5*dt_p
                    elif t_pref == t5_i:
                        picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,-999,0.])
                        picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,t_err,1.])
                    # If new t_pref, write as a 3rd entry -- not valid in the present framework for ar_aic_pick_parser
                    else:
                        picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,-999,0.])
                        picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,-999,0.])
                        picklist.append([sta,tr_f.stats.channel,'P',t_pref,snr_pref,t_err,1.])
                else:
                    picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,-999,0.])
                    picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,-999,0.])
            #            preflist.append([tr_f.stats.station,tr_f.stats.channel,'P',\
            #                             t_pref,snr_pref,dt_p])

            # IF the sanity check fails, take the first estimate as the only candidate and make sure it's SNR is good enough
            elif snrP1 >= pickprefKw['snrmin']:
                if verb > 3:
                    print('Sanity check on second scan for P-pick failed, taking first AIC minimum as pick:',sta)
                t_pref = t3_i
                picklist.append([sta,tr_f.stats.channel,'P',t_pref,snrP1,pickprefKw['dttol'],0.5])
            else:
                t_pref = None

            # If there is a valid P-pick, attempt an S-pick
            viable_S = 0
            if t_pref is not None:
                # breakpoint()
                if verb > 5:
                    print('tpref',t_pref)
                # Set start time of S pick analysis as the preferred P-pick time plus an optional padding
                if Hst[0].stats.endtime - t_pref > Pmute*0.5:
                    t5p_i = t_pref
                    if Hst[0].stats.endtime - t_pref >= Pmute*1.5:
                        t5p_i += Pmute
                    if verb > 5:
                        print('t5p',t5p_i)
                    # Extract traces for horizontal traces from the original horizontal channel stream
                    st_i = Hst.copy().trim(starttime=t5p_i)

                    # Create list holder for preferred pick solutions on each channel 
                    # Will be come 2-element nested list documenting the pick preference results for each channel
                    j_data = []
                    # Create indexing term for number of viable S picks given pickpref kwargs
                    # Iterate across both channels
                    for j_ , tr_j in enumerate(st_i):
                        sr = tr_j.stats.sampling_rate
                        tr_f = tr_j.copy().filter(Sfilt1,**SfKw1)
                        j_chan = tr_f.stats.channel
                        # Conduct characteristic function calculation on horizontal channel
                        crf_i = Scrfn(tr_f.copy(),**ScKws)
                        dt6 = float(np.nanargmax(crf_i.data)/sr)
                        if dt6 < trig_pad_sec[0]*fps:
                            t6_i = tr_f.stats.endtime
                            # breakpoint()
                        else:
                        # Get timing of peak crf value
                            t6_i = t5p_i + float(np.argmax(crf_i.data)/sr)
                        if verb > 5:
                            print('t6',t6_i)
                        # Do AIC on data between Pmute end-time and maximum CRF value
                        aicS1,aic_i = calc_aic(tr_f.copy().trim(endtime=t6_i).data)
                        # Get timinng of minimum AIC and use this as t7_i - first S pick estimate
                        t7_i = t5p_i + float(aic_i/sr)
                        if verb > 5:
                            print('t7',t7_i)
                        # Get t7_i snr from a trace filtered with 
                        snrS1= snrdB_tr(tr_f.copy(),t7_i,snr_wind)
                        if verb > 3: 
                            print('t7',t7_i,'S pick candidate 1. snr',snrS1)
                        # Get refined window & use less filtered WF for second S pick attempt
                        dt1 = t6_i - t7_i
                        dt2 = trig_pad_sec[0]*fps 
                        dt_test = max([dt1,dt2])
                        if dt1 > dt2:
                            t8_i = t6_i - 2*dt1
                        else:
                            t8_i = t6_i - dt2
                            if verb > 3:
                                print(tr_f.stats.station,'t6-t7 window too short, used',int((trig_pad_sec[0]*fps)*sr),'sample window instead')
                        if t8_i < tr_f.stats.starttime:
                            t8_i = tr_f.stats.starttime
                            if verb > 3:
                                print('t8 projected to before start of trace, resetting t8 to start of trace')
                        if verb > 5:
                            print('t8',t8_i)
                        # Do second round of AIC 
                        tr_f = tr_j.copy().filter(Sfilt2,**SfKw2)
                        aicS2,aic_i = calc_aic(tr_f.copy().trim(starttime=t8_i,endtime=t6_i).data)
                        # Assign AIC minimum as second S pick attempt 
                        t9_i = t8_i + float(aic_i/sr)

                        # Get t9_i snr from a trace filtered with 
                        snrS2 = snrdB_tr(tr_f.copy(),t9_i,snr_wind) 
                        if verb > 3:
                            print('t9',t9_i,'S pick candidate 2. snr',snrS2)
                        # Get preferred pick time and snr for given horizontal channel
                        t_pref,t_err,snr_pref = ar_aic_pick_parser([t7_i, t9_i],[snrS1,snrS2],**pickprefKw)
                        # Save preferred solution information for inter-channel comparison
                        if t_pref is not None:
                            j_data.append([t_pref,snr_pref,t_err,j_chan])
                            if t_pref == t7_i:
                                picklist.append([sta,j_chan,'S',t_pref,snr_pref,t_err,0.5])
                                picklist.append([sta,j_chan,'S',t9_i,snrS2,-999,0])
                                viable_S += 1
                            elif t_pref == t9_i:
                                picklist.append([sta,j_chan,'S',t7_i,snrS1,-999,0])
                                picklist.append([sta,j_chan,'S',t_pref,snr_pref,t_err,0.5])
                                viable_S += 1
                            else:
                                picklist.append([sta,j_chan,'S',t7_i,snrS1,-999,0])
                                picklist.append([sta,j_chan,'S',t9_i,snrS2,-999,0])
                                picklist.append([sta,j_chan,'S',t_pref,snr_pref,snr_pref,0.5])
                                viable_S += 1
                        else:
                            picklist.append([sta,j_chan,'S',t7_i,snrS1,-999,0])
                            picklist.append([sta,j_chan,'S',t9_i,snrS2,-999,0])
                else:
                    if verb > 2:
                        print('Too little data left in the trace, aborting S picking')

            else:
                if verb > 2:
                    print('No viable P-pick, skipping S-pick attempt')
            if viable_S == 2:
                if j_data[0][0] <= j_data[1][0]:
                    S_pref = j_data[0]
                elif j_data[0][0] > j_data[1][0]:
                    S_pref = j_data[1]
            elif viable_S == 1:
                S_pref = j_data[0]
            
            if viable_S > 0:
                picklist.append([sta,S_pref[3],'S',S_pref[0],S_pref[1],S_pref[2],1])
        else:
            if verb > 2:
                print('Too little time between t1 and t2',t2_i-t1_i,'skipping P-pick attempt')
    return picklist



def ar_aic_picker_par_db(session,tabdict,trigger,n_cores,\
                         Zchans,Hchans,trig_pad_sec,Pmute,\
                         Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                         Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                         pickprefKw,snr_wind,verb=0):
    """
    DEVNOTE: Having trouble understanding how do do the right combination of positional arguments and
    keyword arguments to correctly pass *args and **kwargs to a positional argument that is the crf method
    for now, this method will use the stk/ltk routine as a default for P detection and the windowed kurtosis
    (NTS 3.10.2020)

    DEVNOTE: All input characteristic response function methods (Pcrfn & Scrfn) need to have a single argument (trace) and 
    the remaining arguments MUST be KWARGS that are zipped into 
    """
    picklist = []

    trig_start = trigger['time'] - trig_pad_sec[0]
    trig_end = trigger['time'] + trigger['duration'] + trig_pad_sec[1]

    # Pull in trace data from database, but allow for option to pass a valid Zstream used in trigger processing
    # ... trim stream if start and end do not match the trigger (allows for input of longer continuous stream) - Makes deepcopy before trim
    # if not isinstance(Zstream,None):
    #     if Zstream[0].stats.starttime <= trig_start and Zstream[0].stats.endtime >= trig_end:
    #         Zstream = Zstream.copy().trim(starttime=trig_start,endtime=trig_end)
    # # ... if Zstream is None [DEFAULT] (or fails the above criterion) read in stream data from database query
    # else:
    Zstream = db2st(session,tabdict,starttime=trig_start,endtime=trig_end,channels=Zchans)
    Hstream = db2st(session,tabdict,starttime=trig_start,endtime=trig_end,channels=Hchans)
    stas = []
    # Assemble the list of relevant stations
    for tr in Zstream: 
        if tr.stats.station not in stas: 
            stas.append(tr.stats.station)
    

    # # Assign Time Index 1 - Universal to Event?
    # t1 = trig_start
    # # Get station list that is included in t2_dict (if any)
    # t2_stas = list(t2_dict.keys())
    # if verb > 0:
    #     print('Picking trigger for potential EVID:',EVID)
    #### ITERATE ACROSS ALL STATIONS AND PROCESS ACCORDINGLY - TODO: TURN THIS INTO A PARALLELIZED LOOP?
    par_output = Parallel(n_jobs=n_cores)(delayed(ar_aic_station_picker)(\
                         Zstream,Hstream,sta,trigger,trig_pad_sec,Pmute,
                         Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                         Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                         pickprefKw,snr_wind) for sta in stas)
    
    for output in par_output:
        for line in output:
            if len(line) == 7:
                picklist.append(line)
    # Convert picklist into a dataframe for export
    pickdf = pd.DataFrame(picklist,columns=['sta','chan','phz','time','snr','dt','qual'])

    return pickdf



def ar_aic_picker_for_db(session,tabdict,trigger,\
                         Zchans,Hchans,trig_pad_sec,Pmute,\
                         Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                         Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                         pickprefKw,snr_wind,verb=0):

    picklist = []

    Zstream = db2st(session,tabdict,starttime=trig_start,endtime=trig_end,channels=Zchans)
    Hstream = db2st(session,tabdict,starttime=trig_start,endtime=trig_end,channels=Hchans)
    stas = []
    # Assemble the list of relevant stations
    for tr in Zstream: 
        if tr.stats.station not in stas: 
            stas.append(tr.stats.station)
    # Process across all stations
    for sta in stas:
        pl = ar_aic_station_picker(Zstream,Hstream,sta,trigger,trig_pad_sec,Pmute,\
                                Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                                Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                                pickprefKw,snr_wind,verb=0)
        # Run internal sanity check to make sure pick-list returns have all fields
        for p in pl:
            if len(p) == 7:
                picklist.append(p)

    pickdf = pd.DataFrame(picklist,columns=['sta','chan','phz','time','snr','dt','qual'])

    return pickdf


def assemble_inpar_inputs(trigger,trig_pad_sec,Zstream,Hstream):
    trig_start = trigger['time'] - trig_pad_sec[0]
    trig_end = trigger['time'] + trigger['duration'] + trig_pad_sec[1]
    Zst = Zstream.copy().trim(starttime=trig_start,endtime=trig_end)
    Hst = Hstream.copy().trim(starttime=trig_start,endtime=trig_end)
    stas = []
    # Assemble the list of relevant stations
    for tr in Zstream: 
        if tr.stats.station not in stas: 
            stas.append(tr.stats.station)


    return {'Zstream':Zst,'Hstream':Hst,'trigger':trigger,'stas':stas}



def ar_aic_picker_inpar(inpar_dict,trig_pad_sec,Pmute,\
                        Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                        Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                        pickprefKw,snr_wind,verb=0):
    picklist = []
    # Unpack inpar_dict
    Zstream = inpar_dict['Zstream']
    Hstream = inpar_dict['Hstream']
    trigger = inpar_dict['trigger']
    stas = inpar_dict['stas']
    
    # Iterate across stations.
    for sta in stas:
        spl = ar_aic_station_picker(Zstream,Hstream,sta,trigger,trig_pad_sec,Pmute,\
                                         Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                                         Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                                         pickprefKw,snr_wind,verb=verb)
        # Check that outputs have the necessary fields
        for p in spl:
            if len(p) == 7:
                picklist.append(p)

    # if ther e
    if len(picklist) > 0:
        pickdf = pd.DataFrame(picklist,columns=['sta','chan','phz','time','snr','dt','qual'])
        return pickdf
    else:
        if verb > 3:
            print('Empty pick DataFrame')




































































### PRIMARY ROUTINE #### -- EVID removed from this version of the code...
def ar_aic_picker_db(session,tabdict,trigger,\
                     Zchans,Hchans,trig_pad_sec,Pmute,
                     Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                     Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                     pickprefKw,snr_wind,Zstream=None,verb=0):
    """
    DEVNOTE: Having trouble understanding how do do the right combination of positional arguments and
    keyword arguments to correctly pass *args and **kwargs to a positional argument that is the crf method
    for now, this method will use the stk/ltk routine as a default for P detection and the windowed kurtosis
    (NTS 3.10.2020)

    DEVNOTE: All input characteristic response function methods (Pcrfn & Scrfn) need to have a single argument (trace) and 
    the remaining arguments MUST be KWARGS that are zipped into 
    """
    picklist = []

    trig_start = trigger['time'] - trig_pad_sec[0]
    trig_end = trigger['time'] + trigger['duration'] + trig_pad_sec[1]

    # Pull in trace data from database, but allow for option to pass a valid Zstream used in trigger processing
    # ... trim stream if start and end do not match the trigger (allows for input of longer continuous stream) - Makes deepcopy before trim
    if Zstream[0].stats.starttime <= trig_start and Zstream[0].stats.endtime >= trig_end:
        Zstream = Zstream.copy().trim(starttime=trig_start,endtime=trig_end)
    # ... if Zstream is None [DEFAULT] (or fails the above criterion) read in stream data from database query
    else:
        Zstream = db2st(session,tabdict,starttime=trig_start,endtime=trig_stop,channels=Zchans)

    Hstream = db2st(session,tabdict,starttime=trig_start,endtime=trig_stop,channels=Hchans)
    stas = []
    # Assemble the list of relevant stations
    for tr in Zstream: 
        if tr.stats.station not in stas: 
            stas.append(tr.stats.station)
    

    # # Assign Time Index 1 - Universal to Event?
    # t1 = trig_start
    # # Get station list that is included in t2_dict (if any)
    # t2_stas = list(t2_dict.keys())
    # if verb > 0:
    #     print('Picking trigger for potential EVID:',EVID)
    #### ITERATE ACROSS ALL STATIONS AND PROCESS ACCORDINGLY - TODO: TURN THIS INTO A PARALLELIZED LOOP?
    for sta in stas:

        if verb > 0:
            print('Picking ',sta)
        # Get Trace sampling Rate
        sr = Zstream.select(station=sta)[0].stats.sampling_rate

        # Get Time Indices 1 & 2 - By station on Z channel
        # If station has t2 value provided in input arguments, skip CRF calculation
        tr_i = Zstream.select(station=sta).copy()[0]
        tr_f = tr_i.copy().filter(Pfilt1,PfKw1)
        
        if sta in trigger['stations']:
            t1_i = trigger['peak_times'][trigger['stations'].index(sta)] - trig_pad_sec[0]
            # Extract peak CRF time for the current station if included in the t2_dict
            t2_i = t1_i + trig_pad_sec[0]
        else:
            t1_i = trig_start
            # Process trace & find peak value to determine station t2
#            crf_i = Pcrf(tr_f.copy(),Pcrfkw):
            crf_i = Pcrfn(tr_f,**PcKws) # TODO TODO: update inputs
            max_i = np.argmax(crf_i.data)
            t2_i = t1 + max_i / sr
            # Write t2 value and station to running list & dictionary containing data
            t2_stas.append(sta)
#            t2_d[sta] = t2_i
        
        # Calculate AIC for trimmed trace t = [t1_i,t2_i]
        aic_i,snrP1,aicP1 = wfaic(tr_f.copy().trim(starttime=t1, endtime= t2_i).data)
        t3_i = aic_i/sr + t1

        # Calculate SNR for first P pick estimate using input of the full initial trace (after pre-processing)
        # it will be internally trimmed in the snrdB_tr() method.
        snrP1 = snrdB_tr(tr_f.copy(),t3_i,snr_wind)
        
        # Form window for second AIC attempt with a small sanity check
        if abs(t2_i - t3_i)*sr > 5: # Require at least 10 samples in this window
            t4_i = t2_i - 2*abs(t2_i - t3_i)
        else: # Failing this, take 10 samples in front of the first pick to do the second round of AIC calculation
            t4_i = t2_i - 10. / sr

        # Apply relaxed filtering for second AIC attempt - do overwrite of earlier tr_f as it is no longer necessary
        tr_f = tr_i.copy().filter(Pfilt2,**PfKw2)
        # Calculate AIC for trimmed trace
        aic_i,snrP2,aicP2 = wfaic(tr_f.copy().trim(starttime=t4_i,endtime=t2_i).data)
        # Write minimum to t5_i
        t5_i = t4_i + aic_i / sr
        # Get t5_i snr from a trace filtered with 
        snrP2 = snrdB_tr(tr_f.copy(),t5_i,snr_wind) 
        # Get preferred P-Pick 
        t_pref,t_err,snr_pref = ar_aic_pick_parser([t3_i,t5_i],[snrP1,snrP2],**pickprefKw)
        # So long as a preferred pick time is selected 
        if tP_pref is not None:
            # If t3_i = t_pref, write it as the preferred pick & use uncertainty as 0.5*dt_p
            if tP_pref == t3_i:
                picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,tP_err,1])
                picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,-999,0])
            # If t5_i = t_pref, write it as the preferred pick & use uncertainty as 0.5*dt_p
            elif tP_pref == t5_i:
                picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,-999,0])
                picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,tP_err,1])
            # If new t_pref, write as a 3rd entry -- not valid in the present framework for ar_aic_pick_parser
            else:
                picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,-999,0])
                picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,-999,0])
                picklist.append([sta,tr_f.stats.channel,'P',t_pref,snr_pref,t_err,1])
        else:
            picklist.append([sta,tr_f.stats.channel,'P',t3_i,snrP1,-999,0])
            picklist.append([sta,tr_f.stats.channel,'P',t5_i,snrP2,-999,0])
#            preflist.append([tr_f.stats.station,tr_f.stats.channel,'P',\
#                             t_pref,snr_pref,dt_p])

        # If there is a valid P-pick, attempt an S-pick
        if t_pref is not None:
            # Set start time of S pick analysis as the preferred P-pick time plus an optional padding
            t5p_i = t_pref + abs(Pmute)
            # Extract traces for horizontal traces from the original horizontal channel stream
            st_i = Hstream.select(station=sta).copy().trim(starttime=t5p_i)

            # Create list holder for preferred pick solutions on each channel 
            # Will be come 2-element nested list documenting the pick preference results for each channel
            j_data = []
            # Create indexing term for number of viable S picks given pickpref kwargs
            viable_S = 0

            # Iterate across both channels
            for j_ , tr_j in enumerate(st_f):
                sr = tr_j.stats.sampling_rate
                tr_f = tr_j.copy().filter(Sfilt1,**SfKw2)
                j_chan = tr_f.stats.channel
                # Conduct characteristic function calculation on horizontal channel
                crf_i = Scrfn(tr_f.copy(),**ScKws)
                # Get timing of peak crf value
                t6_i = np.argmax(crf_i.data)/sr + t5p_i
                # Do AIC on data between Pmute end-time and maximum CRF value
                aic_i,snrS1,aicS1 = wfaic(tr_f.copy().trim(endtime=t6_i).data)
                # Get timinng of minimum AIC and use this as t7_i - first S pick estimate
                t7_i = aic_i/sr + t5p_i
                # Get refined window & use less filtered WF for second S pick attempt
                t8_i = t6_i - 2*abs(t7_i - t6_i)
                # Do second round of AIC 
                aic_i,snrS2,aicS2 = wfaic(tr_j.trim(starttime=t8_i,endtime=t6_i).filter(Sfilt2,**SfKw2).data)
                # Assign AIC minimum as second S pick attempt and write to holding dictionary
                t9_i = t8_i + aic_i/sr            
                # Get preferred pick time and snr for given horizontal channel
                t_pref,t_err,snr_pref = ar_aic_pick_parser([t7_i, t9_i],[snrS1,snrS2],**pickprefKw)
                # Save preferred solution information for inter-channel comparison
                if t_pref is not None:
                    j_data.append([t_pref,snr_pref,t_err,j_chan])
                    if t_pref == t7_i:
                        picklist.append([sta,j_chan,'S',t_pref,snr_pref,t_err,0.5])
                        picklist.append([sta,j_chan,'S',t9_i,snrS2,-999,0])
                        viable_S += 1
                    elif t_pref == t9_i:
                        picklist.append([sta,j_chan,'S',t7_i,snrS1,-999,0])
                        picklist.append([sta,j_chan,'S',t_pref,snr_pref,t_err,0.5])
                        viable_S += 1
                    else:
                        picklist.append([sta,j_chan,'S',t7_i,snrS1,-999,0])
                        picklist.append([sta,j_chan,'S',t9_i,snrS2,-999,0])
                        picklist.append([sta,j_chan,'S',t_pref,snr_pref,snr_pref,0.5])
                        viable_S += 1
                else:
                    picklist.append([sta,j_chan,'S',t7_i,snrS1,-999,0])
                    picklist.append([sta,j_chan,'S',t9_i,snrS2,-999,0])

        if viable_S == 2:
            if j_data[0][0] <= j_data[1][0]:
                S_pref = j_data[0]
            elif j_data[0][0] > j_data[1][0]:
                S_pref = j_data[1]
        elif viable_S == 1:
            S_pref = j_data[0]
        
        if viable_S > 0:
            picklist.append([sta,st_f[0].stats.channel,'S',S_pref[0],S_pref[1],S_pref[2],1])

    # Convert picklist into a dataframe for export
    pickdf = pd.DataFrame(picklist,columns=['sta','chan','phz','time','snr','dt','qual'])

    return pickdf





## PROVIDE A LARGELY PARAMETERIZED EXAMPLE FROM THE SGGS DATASET ##
def ar_aic_shorthand(session,tabdict,trigger,SPexp,verb=1):

    # Lots O' Parameters!
    # Channel naming conventions
    Zchans = ['GNZ','GP3','p0']
    Hchans = ['GN1','GN2','GP1','GP2','p1','p2']
    # P-pick processing parameters
    Pfilt1 = 'bandpass'
    PfKw1 = {'freqmin':80.,'freqmax':430.}
    Pcrfn = okt.tr_wind_kurt_py
    PcKws = {'wind':0.05,'step':0.01}
    Pfilt2 = 'highpass'
    PfKw2 = {'freq':40.}
    Sfilt1 = 'bandpass'
    SfKw1 = {'freqmin':40.,'freqmax':430.}
    Scrfn = classic_sta_lta_diff_py
    ScKws = {'sta':0.01,'lta':0.1}
    Sfilt2 = 'highpass'
    SfKw2 = {'freq':40.}
    EVID = 1
    snrwind = SPexp/5.
    trig_pad_sec = np.array([0.5,3])*SPexp
    Pmute = SPexp*0.5
    # Set the rules for selecting preferable pick estimates
    pickprefKw = {'snrmin':3.,'snrtol':1.,'dttol':0.003,'dtmax':SPexp/5.,'verb':verb}

    # Actual method implemented
    pick_df = ar_aic_picker_db(session,tabdict,trigger,\
                               Zchans,Hchans,trig_pad_sec,Pmute,
                               Pfilt1,PfKw1,Pcrfn,PcKws,Pfilt2,PfKw2,\
                               Sfilt1,SfKw1,Scrfn,ScKws,Sfilt2,SfKw2,\
                               pickprefKw,snr_wind,Zstream=None,verb=verb)
    # Send standard out to interface
    return pick_df
























































####### GRAVEYARD #######


## This old version will be obsolited in the next iteration of this code
def snrdB(data,index,epsilon=1e-10):
    """
    Calculate the signal-to-noise ratio of an input data vector with the 
    index representing the onset of the signal.
    assumes time is rightward increasing and signal onsets after a pure
    noise segment
    """
    index = int(index)
    noiserms = np.sqrt(np.mean(data[:index-1]**2))
    signalrms = np.sqrt(np.mean(data[index:]**2))
    snr = 10.*np.log10((signalrms/(noiserms+epsilon))**2) # Small epsilon included to prevent 0-division
    return snr


def wfaic(data,eta=np.e,normalize=False):
    """
    Calculate the AIC of a data array directly from time-series data using the method
    of Maeda (1985)
    AIC[k] = k*log(var(x[0:k])) + (N - k - 1)*log(var(x[k+1:N]))

    :: INPUTS ::
    :type data: (N,) numpy.ndarray
    :param data: vector of equally sampled time series data
    :type eta: float
    :param eta: natural root value to prevent 0-valued arguments for log

    :: OUTPUT ::
    :rtype aic: numpy.ndarray
    :return aic: Vector of AIC values 
    :rtype aic_stats: numpy.ndarray
    :return aic_stats: Array containing statistics on AIC curve:
                        IND PARAMETER
                        0   minimum finite AIC value
                        1   index of minimum finite AIC value
                        2   kurtosis of AIC - higher values are more peaked
    """

    N = len(data)
    aic = np.zeros(data.shape)
    for k_ in range(N):
        if k_ > 0 and k_ < N:
            k_1 = k_ + 1
            aic[k_] = (float(k_)*np.log(np.var(data[:k_]+eta)) +
                       float(N - k_ - 1.)*np.log(np.var(data[k_ + 1:] + eta)))
        elif k_ == 0:
            aic[k_] = float((N - 1.))*np.log(np.var(data[1:]+eta)) + 1.
        elif k_ == N:
            aic[k_] = float(k_)*np.log(np.var(data+eta))

    # Replace infinite values with NaN
    aic[~np.isfinite(aic)] = np.nan
    if not normalize:
        aic_out = aic
    elif normalize:
        Amin = np.nanmin(aic)
        Amax = np.nanmax(aic)
        Arng = Amax - Amin
        aic_out = np.divide(aic - Amin,Arng)

    minval = np.nanmin(aic)
    try:
        minindex = np.where(aic == minval)[0][0]
    except:
        minindex = 0
    snr = snrdB(data,minindex)

    return minindex, snr, aic_out

def assess_2_ar_pick_estimates(times,snrs,snrmin=3.,snrtol=1.,dtident=0.002,dtmax=0.05,Ptime=None,SPmm=[0.03,0.12],verbose=True):
    """
    Somewhat hackey way to assess the quality of two possible picks for a given phase - #TODO: needs updating
    """
    atimes = []
    asnrs = []
    # Go over the time and snr pairs
    for i_ in range(2):
        # Accept picks only if SNR is above minimum SNR
        if snrs[i_] > snrmin:
            # If not an S-pick, just use SNR criteria
            if verbose:
                print('Accept 1')
            if Ptime is None:
                atimes.append(times[i_])
                asnrs.append(snrs[i_])
                if verbose:
                    print('Accept via SNR 1.A')
            # If an S-pick, accept only those with some physically reasonable S-P time
            elif Ptime is not None:
                if min(SPmm) <= times[i_] - Ptime <= max(SPmm):
                    atimes.append(times[i_])
                    asnrs.append(snrs[i_])
                    if verbose:
                        print('Accept via Ptime fit 1.B')
            else:
                if verbose:
                    print('REJECT via Ptime misfit: %s'%(str(Ptime)))
        else:
            if verbose:
                print('REJECT via SNR')

    # If both pick estimates are accepted in the above...
    if len(atimes) == 2:
        # If SNRs are close ...
        if abs(asnrs[0] - asnrs[1]) <= snrtol:
            # ... and differential time is less than the identical time threshold, use mean
            if dtident >= abs(atimes[0]-atimes[1]):
                if verbose:
                    print('Accept nearly identical picks. 2.A')
                dt = max(atimes) - min(atimes)
                t_pref = min(atimes)+dt/2.
                snr_pref = (asnrs[0]+asnrs[1])/2.
            # ... if between identical time tolerance and maximum differential time, use time
            # of pick with greater SNR
            elif dtident < abs(atimes[0]-atimes[1]) <= dtmax:
                if verbose:
                    print('Accept between dtident & dtmax. 2.B')
                t_pref = atimes[asnrs.index(max(asnrs))]
                snr_pref = asnrs[atimes.index(t_pref)]
            # ... if differential times are large, but SNRs are close, reject both.
            else:
                if verbose:
                    print('REJECT outside dtmax. 2.C')
                t_pref = None
                snr_pref = None
        # If SNRs are not close ...
        else:
            # ... preference to larger SNR
            if verbose:
                print('Accept maximum SNR. 2.D')
            t_pref = atimes[asnrs.index(max(asnrs))]
            snr_pref = max(asnrs)

    # If only one pick survives filtering, that's the preferred one!
    elif len(atimes) == 1:
        if verbose:
            print('Accept Single Entry. 3.A')
        t_pref = atimes[0]
        snr_pref = asnrs[0]
    # For any other result, return None!
    else:
        if verbose:
            print('REJECT via empty list. 4.A')
        t_pref = None
        snr_pref = None

    return t_pref, snr_pref






























