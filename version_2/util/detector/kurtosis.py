#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Filename: kurtosis.py
# Purpose:  Python trigger & picker routines using kurtosis for seismology (some with C++ core)
# Author:   Nathan T. Stevens
# Email:    ntstevens@wisc.edu
# Attribution:  Based upon the obspy.signal.trigger.py and associated scripts by
#               Moritz Beyreuther & Tobias Megies
# Copyright (C) 2019, 2020 Nathan T. Stevens
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from future.builtins import *
import ctypes as C
import warnings
import numpy as np
import scipy as sp
import pandas as pd
from obspy.realtime.signal import *
from obspy.signal.trigger import coincidence_trigger
from copy import deepcopy
from obspy.core import read, UTCDateTime, Stream, Trace
import matplotlib.pyplot as plt


#### Announcement subroutines -- used when parallel processing ####

def _announce_iter_start(verb,crftr):
    tic = UTCDateTime()
    if verb > 0:
        print('Starting: %s.%s -- %s to %s -- (%s)\n'%
             (crftr.stats.station,crftr.stats.channel,
              str(crftr.stats.starttime),str(crftr.stats.endtime),
              tic))
    return tic



def _announce_iter_complete(verb,tic,crftr):
    if verb > 0:
        toc = UTCDateTime()
        print('Complete: %s.%s -- Elapsed: %.2f min -- (%s)\n'%
              (crftr.stats.station,crftr.stats.channel,
               (toc-tic)/60.,str(tic)))


#### Window generation subroutine ####
def _get_window_indices(nwind,nstep,nsamp):
    wind_count = int(np.floor((int(nsamp) - int(nwind))/int(nstep)))
    starts = np.linspace(0,(wind_count - 1)*nstep,wind_count).astype('int')
    stops = starts + nwind - 1
    indices = []
    for i_ in range(wind_count):
        indices.append([starts[i_],stops[i_]])
    return indices


#### Parallelization-ready windowed kurtosis characteristic response function methods ####
def tr_wind_kurt_py(trace,win=0.05,step=0.01,verb=0,axis=0,fisher=False,bias=True,nan_policy='propagate'):
    """
    Conduct a moving-window kurtosis calculation for data inside a trace.
    Note that data are truncated by the number of samples in the window - 1

    :: INPUTS ::
    :type trace: obspy.core.trace.Trace
    :param trace: Trace object with data and stats
    :type win: float
    :param win: window length in samples
    :type step: float
    :param step: sample increment to advance window by in samples (sets output crf S/R)
    :type method: string
    :param method: location of sample point
    
    :: EXPLICIT KWARGS FOR scipy.stats.kurtosis :: - required for passing to joblib.Parallel
    axis
    fisher - Shifted to default "False" in this implementation
    bias
    nan_policy

    See scipy.stats.kurtosis for documentation.

    :: OUTPUT ::
    :rtype crftr: obspy.core.trace.Trace
    :return crftr: Characteristic response function trace:

    Based on equations in McBrearty et al., 2019
    """
    # Create copy of trace to preserve header information
    crftr = trace.copy()
    # Get data
    x_0 = crftr.data
    # Get number of samples in trace
    nsamp = trace.stats.npts
    # Get sampling rate for trace
    SR1 = trace.stats.sampling_rate
    # Calculate window length in samples
    nwind = int(round(win*SR1))
    # Calculate step length in samples
    nstep = int(round(step*SR1))
    # Get indices for moving windows
    indices = _get_window_indices(nwind,nstep,nsamp)
    # Create containers for crf calculated window and locations
    crf = np.zeros((len(indices),))

    tic = _announce_iter_start(verb,crftr)

    for i_ in range(len(indices)):
        wsi = indices[i_][0]
        wei = indices[i_][1]
        # Get Data Subset
        x_i = x_0[wsi:wei]
        # Calculate sample kurtosis, applying **kwargs 
        K_i = sp.stats.kurtosis(x_i,axis=axis,fisher=fisher,bias=bias,nan_policy=nan_policy)

        # Work around for nan & inf values coming out of sp.stats.kurtosis as "replace"
        # is not an option for nan_policy kwarg
        if np.isnan(K_i):
            K_i = 1e-8
            #print('nan value calculated in',crftr.stats.station,'replacing with 1e-8')
        if np.isinf(K_i):
            K_i = 1e8
        # Write sample kurtosis to crf container

        crf[i_] = K_i
    #if method == 'central':
    # This is now 'right'
    t_start = trace.stats.starttime + float(win) #/2.
    #elif method == 'left':
    #    t_start = trace.stats.starttime
    #elif method == 'right':
    #    t_start = trace.stats.starttime + win

    crftr.data = crf
    crftr.stats.starttime = t_start
    crftr.stats.delta = step

    _announce_iter_complete(verb,tic,crftr)
    
    procdat = "ObsPy 1.1.1: Windowed kurtosis(options{'win':%.4f,'step':%.4f,'fisher':%s})"%(win,step,fisher)
    crftr.stats.processing.append(procdat)

    return crftr

def tr_iqr_py(trace,win=1.0,step=0.5,axis=0,iqr_rng=[10., 90.],scale='raw',nan_policy='propagate',interpolation='linear',keepdims=True,verb=0):
    """
    Conduct a moving-window interquartline range calculation for data inside a trace.
    Note that data are truncated by the number of samples in the window - 1

    :: INPUTS ::
    :type trace: obspy.core.trace.Trace
    :param trace: Trace object with data and stats
    :type win: float
    :param win: window length in samples
    :type step: float
    :param step: sample increment to advance window by in samples (sets output crf S/R)
    :type method: string
    :param method: location of sample point
    
    :: EXPLICIT KWARGS FOR scipy.stats.kurtosis :: - required for passing to joblib.Parallel
    axis
    fisher - Shifted to default "False" in this implementation
    bias
    nan_policy

    See scipy.stats.kurtosis for documentation.

    :: OUTPUT ::
    :rtype crftr: obspy.core.trace.Trace
    :return crftr: Characteristic response function trace:

    Based on equations in McBrearty et al., 2019
    """
    # Create copy of trace to preserve header information
    crftr = trace.copy()
    # Get data
    x_0 = crftr.data
    # Get number of samples in trace
    nsamp = trace.stats.npts
    # Get sampling rate for trace
    SR1 = trace.stats.sampling_rate
    # Calculate window length in samples
    nwind = int(round(win*SR1))
    # Calculate step length in samples
    nstep = int(round(step*SR1))
    # Get indices for moving windows
    indices = _get_window_indices(nwind,nstep,nsamp)
    # Create containers for crf calculated window and locations
    crf = np.zeros((len(indices),))

    tic = _announce_iter_start(verb,crftr)

    for i_ in range(len(indices)):
        wsi = indices[i_][0]
        wei = indices[i_][1]
        # Get Data Subset
        x_i = x_0[wsi:wei]
        # Calculate sample kurtosis, applying **kwargs 
        iqr_i = sp.stats.iqr(x_i,axis=axis,rng=iqr_rng,scale=scale,nan_policy=nan_policy,\
                           interpolation=interpolation,keepdims=keepdims)
        # Write sample kurtosis to crf container
        crf[i_] = iqr_i
    #if method == 'central':
    t_start = trace.stats.starttime + float(win)/2.
    #elif method == 'left':
    #    t_start = trace.stats.starttime
    #elif method == 'right':
    #    t_start = trace.stats.starttime + win

    crftr.data = crf
    crftr.stats.starttime = t_start
    crftr.stats.delta = step

    _announce_iter_complete(verb,tic,crftr)
    
    procdat = "ObsPy 1.1.1: Windowed IQR(options{'win':%.4f,'step':%.4f,'iqr_rng':%.2f,%.2f})"%(win,step,iqr_rng[0],iqr_rng[1])
    crftr.stats.processing.append(procdat)

    return crftr



def tr_ltk_stk_py(trace,stk,ltk,step,verb=0,axis=0,fisher=False,bias=True,nan_policy='propagate'):
    """
    Conduct a moving-window short-term/long-term ratio kurtosis calculation for data inside a trace.
    Note that data are truncated by the number of samples in the window - 1

    :: INPUTS ::
    :type trace: obspy.core.trace.Trace
    :param trace: Trace object with data and stats
    :type stk: float
    :param stk: window length in samples for short-term window (leads ltk window)
    :type ltk: float
    :param ltk: window length in samples for long-term window (trails stk window)
    :type step: float
    :param step: sample increment to advance window by in samples (sets output crf S/R)
    :type method: string
    :param method: location of sample point
    
    :: EXPLICIT KWARGS FOR scipy.stats.kurtosis :: - required for passing to joblib.Parallel
    axis
    fisher - Shifted to default "False" in this implementation
    bias
    nan_policy

    See scipy.stats.kurtosis for documentation.

    :: OUTPUT ::
    :rtype crftr: obspy.core.trace.Trace
    :return crftr: Characteristic response function trace:

    Based on equations in McBrearty et al., 2019
    """

    crftr = trace.copy()
    x_0 = crftr.data
    nsamp = trace.stats.npts
    SR1 = trace.stats.sampling_rate
    nst = int(round(stk*SR1))
    nlt = int(round(ltk*SR1))
    nstep = int(round(step*SR1))
    indices = _get_window_indices(nst+nlt-1,nstep,nsamp)
    crf = np.zeros((len(indices),))

    tic = _announce_iter_start(verb,crftr)

    # if verb > 0:
    #     tic = UTCDateTime()
    #     print('Starting: %s.%s -- %s to %s -- (%s)\n'%
    #          (crftr.stats.station,crftr.stats.channel,
    #           str(crftr.stats.starttime),str(crftr.stats.endtime),
    #           tic))

    for i_ in range(len(indices)):
        wsil = indices[i_][0]
        wsis = wsil + nlt
        weil = wsis
        weis = indices[i_][1]
        x_lt = x_0[wsil:weil]
        x_st = x_0[wsis:weis]
        K_lt = sp.stats.kurtosis(x_lt,axis=axis,fisher=fisher,bias=bias,nan_policy=nan)
        K_st = sp.stats.kurtosis(x_st,axis=axis,fisher=fisher,bias=bias,nan_policy=nan)
        crf[i_] = K_st / K_lt
    crftr.data = crf
    crftr.stats.starttime = trace.stats.starttime + stk + ltk
    crftr.stats.delta = step
    crftr.stats['stk'] = stk
    crftr.stats['ltk'] = ltk

    # if verb > 0:
    #     toc = UTCDateTime()
    #     print('Complete: %s.%s -- Elapsed: %.2f min -- (%s)\n'%
    #           (crftr.stats.station,crftr.stats.channel,
    #            (toc-tic)/60.,str(tic)))

    # Append processing data to the trace header information
    procdat = "ObsPy 1.1.1: S/L kurtosis(options{'stk':%.4f,'ltk':%.4f}::type='recursive')"%(stk,ltk)
    crftr.stats.processing.append(procdat)

    # Announce iteration completion for acceptable verb values
    if isinstance(verb,float) or isintance(verb,int):
        _announce_iter_complete(verb,tic,crftr)

    return crftr


#### END OF NEXT-GEN SCRIPTS ####

def tr_diff(trace):
    """
    Differentiate trace data and return a trace with updated metadata

    :: INPUT ::
    :type trace: obspy.core.trace
    :param trace: Seismic trace object

    :: OUTPUT ::
    :rtype tr: obspy.core.trace
    :return tr: Seismic trace object with updated "processing" stats and data
    """
    tr = deepcopy(trace)
    dtdata = differentiate(tr)
    tr.data = dtdata
    procdat = "ObsPy 1.1.1: differentiate"
    tr.stats.processing.append(procdat)

    return tr

def tr_boxcar(trace,nswin):
    """
    Convolve trace data with a boxcar function of specified sample length
    and return an updated trace

    :: INPUTS ::
    :type trace: obspy.core.trace
    :param trace: Seismic trace object
    :type nswin: int
    :param nswin: Length of boxcar operator in samples

    :: OUTPUT ::
    :rtype tr: obspy.core.trace
    :return tr: Seismic trace with convolved data and updated stats
    """
    tr = deepcopy(trace)
    boxdata = boxcar(tr,nswin)
    procdat = "ObsPy 1.1.1: boxcar(width=%d)"%(nswin)
    tr.stats.processing.append(procdat)
    tr.data = boxdata

    return tr


def tr_ls_wind_kurt_thresh_py(trace,win,step,method='right',sig_coef=1.96,**kwargs):
    full_mean = np.mean(trace.data)
    full_stdev = np.std(trace.data)
    t_data = trace.data[abs(trace.data - full_mean) < full_stdev*sig_coef]
    trace_kurtosis = sp.stats.kurtosis(t_data,**kwargs)
    tr_lswfk_crf = tr_wind_kurt_py(trace,win,step,method=method,**kwargs)
    tr_lswfk_crf.data /= trace_kurtosis
    tr_lskfw_crf.stats['k_bar'] = trace_kurtosis
    tr_lskfw_crf.stats['stats'] = [full_mean, full_stdev]
    tr_lskfw_crf.stats.station += '_Ku'
    return tr_lskfw_crf


def tr_ls_wind_kurt_py(trace,win,step,method='right',**kwargs):
    """
    Do the windowed kurtosks CRF calculation and then divide values of the output
    CRF by the kurtosis value calculated for the entire input trace data-series
    
    BE CERTAIN THAT THE DATA HAVE BEEN DETRENDED!!!
    """
    trace_kurtosis = sp.stats.kurtosis(trace.data,**kwargs)
    tr_lswk_crf = tr_wind_kurt_py(trace,win,step,method=method,**kwargs)
    tr_lswk_crf.data /= trace_kurtosis
    tr_lswk_crf.stats['k_bar'] = trace_kurtosis
    tr_lswk_crf.stats.station += '_Ksl'
    return tr_lswk_crf

def tr_kurt_py(trace,win,chanpre='RK'):
    """
    Conduct recrusive kurtosis on a trace and return the trace with update
    stats. 

    Wrapper for the obspy.realtime.kurtosis function
    :type trace: obspy.core.trace.Trace
    :param trace: Seismic Trace
    :type win: float
    :param win: kurtosis calculation window [sec]
    :type chanpre: optional, string
    :param chanpre: Prefix to attach to the updated "channel" name (saving last 2 characters from original)

    :rtype tr: obspy.core.trace.Trace
    :return tr: trace containing processed trace (characteristic response function) and updated "stats"
    """
    tr = deepcopy(trace)
    data = kurtosis(tr,win)
    tr.stats.channel=chanpre+trace.stats.channel[-2:]
    tr.data = data
    procdat = "ObsPy 1.1.1: kurtosis(options{'win':%.4f}::type='recursive'::oldchancode=%s)"%(win,trace.stats.channel)
    tr.stats.processing.append(procdat)

    return tr


def tr_slkurt_py(trace, stwin, ltwin, eta=1e-8, chanpre='SLK'):
    """
    Calculate short-time/long-time recrusive kurtosis on a trace 
    and return the trace with updated stats. 

    Wrapper for the obspy.realtime.kurtosis function
    :type trace: obspy.core.trace.Trace
    :param trace: Seismic Trace
    :type stwin: float
    :param stwin: short-time kurtosis calculation window [seconds]
    :type ltwin: float
    :param ltwin: long-time kurtosis calculation window [seconds]
    :type chanpre: optional, string
    :param chanpre: Prefix to attach to the updated "channel" name (saving last 2 characters from original)

    :rtype tr: obspy.core.trace.Trace
    :return tr: trace containing processed trace (characteristic response function) and updated "stats"
    """   
    tr = deepcopy(trace)
    stk = kurtosis(tr, stwin)
    ltk = kurtosis(tr, ltwin)
    slkurt = stk/(ltk+eta)
    tr.data = slkurt
    tr.stats.channel = chanpre + trace.stats.channel[-2:]
    procdat = "ObsPy 1.1.1: S/L kurtosis(options{'stwin':%.4f,'ltwin':%.4f}::type='recursive'::oldchancode=%s)"%(stwin,ltwin,trace.stats.channel)
    tr.stats.processing.append(procdat)

    return tr


def tr_dtkurt_py(trace, win, boxorder=0, nswin=10, chanpre='DK',signed=True):
    """
    Calculate the time derivative of the kurtosis of an input seismic trace
    return a trace of equal length with updated stats

    :type trace: obspy.core.trace.Trace
    :param trace: Seismic Trace
    :type win: float
    :param win: kurtosis calculation window [sec]
    :type boxorder: optional, int
    :param boxorder: Number of boxcar smoothing iterations to apply
    :type nswin: optional, int
    :param nswin: Length of boxcar operator in samples
    :type chanpre: optional, string
    :param chanpre: Prefix to attach to the updated "channel" name (saving last 2 characters from original)

    :rtype tr: obspy.core.trace.Trace
    :return tr: trace containing processed trace (characteristic response function) and updated "stats"
    """
    tr = deepcopy(trace)
    tr = tr_kurt_py(tr, win)
    tr = tr_diff(tr)
    i_ = 0
    # If applying n-order boxcar smoothing, iterate
    if boxorder>0:
        while i_ < boxorder:
            tr = tr_boxcar(tr,nswin)
            i_ += 1
    tr.stats.channel = chanpre + trace.stats.channel[-2:]
    if not signed:
        tr.data = np.abs(tr.data)
    
    return tr


def con_trig_dtkurt_py(stream, thr1, thr2, swin, coin_trig_thresh, boxord=0, nswin=1, sign=True, isplot=False, **kwargs):
    """
    Conduct a coincidence trigger on dtkurt characteristic functions of input stream

    """
    TIC = UTCDateTime()
    st = stream.copy()
    for i_,tr in enumerate(st):
        stmp = Stream()
        stmp += tr.copy()
        chan = tr.stats.channel
        tic = UTCDateTime()
        crftr = tr_dtkurt_py(tr, swin, boxorder=boxord, nswin=nswin, chanpre=chan[:1], signed=sign)
        stmp += crftr
        if isplot:
            stmp.plot(equal_scale=False)
        toc = UTCDateTime()
        rtmin = (toc - tic)/60.
        print("%s processed (%d/%d). Runtime: %.2f min"%(crftr.stats.station,i_+1,len(stream),rtmin))
        st += crftr
    TOC = UTCDateTime()
    print('CRF Processing Concluded. Elapsed Time: %.1f sec'%(TOC-TIC))
    trig = coincidence_trigger(None,thr1,thr2,st,coin_trig_thresh, **kwargs)
    TOC = UTCDateTime()
    print('Coincidence Trigger Calculation Concluded. Elapsed Time: %.1f sec'%(TOC-TIC))
    return trig,st

### SECTION OF PARALLELIZED CODE

from joblib import Parallel, delayed
import time

def tr_kurt_methods_py_parallel(trace,method='windkurt',**kwargs):
    """
    A wrapper for different types of pure-python (_py_) kurtosis based CRF calculators
    for input ObsPy trace objects.

    """
    # Hack-ey sanity check index to make sure processing occurred
    processed = 0
    tr = deepcopy(trace)
    # Run selected method
    TIC = UTCDateTime()
    # Central windowed kurtosis of McBrearty et al. (2019)
    if method == 'windkurt':
        try:
            tr = tr_wind_kurt_py(tr,**kwargs)
            processed = 1
        except Exception as e:
            print(e)
    # Time differentiated recursive kurtosis (REF)
    elif method == 'dtrkurt':
        try:
            tr = tr_kurt_py(tr,**kwargs)
            processed = 1
        except Exception as e:
            print(e)
    # Short Term vs Long Term Windowed Kurtosis (McBrearty et al., 2019; Akram & Eaton, 2015, and refs. therein)
    elif method == 'slkurt':
        try:
            tr = tr_kurt_py(tr,**kwargs)
            processed = 1
        except Exception as e:
            print(e)
    # STK/LTK on recursive Kurtosis (REFS)
    elif method == 'slrkurt':
        try:
            tr = tr_kurt_py(tr,**kwargs)
            processed = 1
        except Exception as e:
            print(e)
    else:
        warn('Invalid "method" selected')
    
    if processed == 1:
        TOC = UTCDateTime()
        print('%s Processed. Elapsed Time: %.1f sec'%(tr.stats.station, TOC-TIC))
        return tr

def coincidence_trigger_parallel(stream,thr1,thr2,swin,nstamin,ncores,**kwargs):
    iTIC = UTCDateTime()
    print('Parallel Processing Starting at %s'%(str(iTIC)))
    crflist = Parallel(n_jobs = ncores)(delayed(tr_kurt_methods_py_parallel)(trace=tr_i,method=method,kwargs=kwargs) for tr_i in stream)


### OLD CODES BELOW HERE, STILL VIABLE, JUST LESS VERSITILE ###


def tr_dtkurt_py_parallel(trace, win=0.1, boxorder=5, nswin=3):
    """
    Calculate the time derivative of the kurtosis of an input seismic trace
    return a trace of equal length with updated stats

    :type trace: obspy.core.trace.Trace
    :param trace: Seismic Trace
    :type win: float
    :param win: kurtosis calculation window [sec]
    :type boxorder: optional, int
    :param boxorder: Number of boxcar smoothing iterations to apply
    :type nswin: optional, int
    :param nswin: Length of boxcar operator in samples
    :type chanpre: optional, string
    :param chanpre: Prefix to attach to the updated "channel" name (saving last 2 characters from original)

    :rtype tr: obspy.core.trace.Trace
    :return tr: trace containing processed trace (characteristic response function) and updated "stats"
    """
    tr = deepcopy(trace)
    tr = tr_kurt_py(tr, win)
    tr = tr_diff(tr)
    i_ = 0
    tic = UTCDateTime()
    print('Processing %s. Started %s'%(tr.stats.station,str(tic)))
    # If applying n-order boxcar smoothing, iterate
    if boxorder>0:
        while i_ < boxorder:
            tr = tr_boxcar(tr,nswin)
            i_ += 1

    tr.data = np.abs(tr.data)
    toc = UTCDateTime()
    print('%s Processed. Elapsed Time: %.1f sec'%(tr.stats.station,toc-tic)) 
    return tr


def con_trig_dtkurt_py_parallel(stream, thr1, thr2, swin, coin_trig_thresh, ncores, **kwargs):
    """
    Conduct a coincidence trigger on dtkurt characteristic functions of input stream

    """
    iTIC = UTCDateTime()
    print('Parallel Processing Starting at %s'%(str(iTIC)))
    crflist = Parallel(n_jobs = ncores)(delayed(tr_dtkurt_py_parallel)(trace=tr_i) for tr_i in stream)
    st = Stream()
    for tr in crflist:
        st += tr
#    breakpoint()
    iTOC = UTCDateTime()
    print('CRF Processing Concluded. Elapsed Time: %.1f sec'%(iTOC-iTIC))
    trig = coincidence_trigger(None,thr1,thr2,st,coin_trig_thresh, **kwargs)
    iTOC = UTCDateTime()
    print('Coincidence Trigger Calculation Concluded. Elapsed Time: %.1f sec'%(iTOC-iTIC))
    # Bake in some time for computer cool-down
    if iTOC - iTIC < 180:
        print('Waiting 5 sec')
        time.sleep(5)
    elif iTOC-iTIC >= 180 and iTOC-iTIC < 360:
        print('Waiting 15 sec')
        time.sleep(15)
    elif iTOC-iTIC >= 360 and iTOC-iTIC < 720:
        print('Waiting 30 sec')
        time.sleep(30)
    elif iTOC - iTIC > 720:
        print('Processing time exceeded 12 min, waiting 1 min.')
        time.sleep(60)
    return trig

#def contrig_dtkurt_recpick_py(stream, thr1, trh2, swin, coin_trig_thresh, boxord=0, nswin=1, sign=True, pick_type='onset', phase_label='?', **kwargs):
 #   st = Stream()
  #  for tr in stream:
   #     chan = tr.stats.channel
    #    crftr = tr_dtkurt_py(tr, swin, boxorder=boxord, nswin=nswin, chanpre=chan[:1], signed=sign)
     #   print(str(crftr.stats.station)+' processed.')

#    trig = coincidence_trigger(None, thr1, thr2, st, coin_trig_thresh, **kwargs)
 #   # For stations included in coincidence trigger, conduct phase picking on input traces
  #  for itrig in trig:
   #     if pick_type.lower() == 'max':
    #        
     #   elif pick_type.lower() == 'onset':
#
 #       elif not pick_type:

#def kurt_ar_pick(
#    """
#    Adapted version of Akazawa's Method using other detection methods
#    besides STA/LTA (kurtosis characteristic methods)
#
#    """
#    # ??Z
#    a = scipy.signal.detrend(a, type='linear')
#    # ??N
#    b = scipy.signal.detrend(b, type='linear')
#    # ??E
#    c = scipy.signal.detrend(c, type='linear')
#
#    a = np.require(a, dtype=np.float32, requirements=['C_CONTIGUOUS'])
#    b = np.require(b, dtype=np.float32, requirements=['C_CONTIGUOUS'])
#    c = np.require(c, dtype=np.float32, requirements=['C_CONTIGUOUS'])






#def ckurtosis(a, nwin, kbari=0., mu1i=0., mu2i=1.):
#    """
#    Recursive Kurtosis calculation
#
#    Fast version written in C via ctypes
#
#    Based on obspy.realtime.kurtosis
#
#    :type a: numpy.ndarray
#    :param a: Time-series
#    :type nwin: int
#    :param nwin: length of kurtosis calculation window (recursive) in samples
#
#    :rtype charfct: numpy.ndarray
#    :return charfct: Characteristic function of recursive Kurtosis
#    """
#    a = np.ascontiguousarray(a, np.float64)
#    ndat = len(a)
#    charfct = np.empty(ndat, dtype=np.float64)
#
#    clibsignal.rkurt(a, charfct, kbari, mu1i, mu2i, ndat, nwin)
#
#    return charfct



##
