#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Filename: kurtosis.py
# Purpose:  Python trigger & picker routines using kurtosis for seismology (some with C++ core) <-- 7. MAR 2023 retrospective (C++ core [cython] not implemented here)
# Author:   Nathan T. Stevens
# Email:    ntstevens@wisc.edu
# Attribution:  Based upon the obspy.signal.trigger.py and associated scripts by
#               Moritz Beyreuther & Tobias Megies
# Copyright (C) 2019 Nathan T. Stevens

:last major revision: 25. MAR 2020
    Header timestamped on 7. MAR 2023
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

#from obspytools.detectors.headers import clibsignal
def _get_window_indices(nwind,nstep,nsamp):
    wind_count = int(np.floor((int(nsamp) - int(nwind))/int(nstep)))
    starts = np.linspace(0,(wind_count - 1)*nstep,wind_count).astype('int')
    stops = starts + nwind - 1
    indices = []
    for i_ in range(wind_count):
        indices.append([starts[i_],stops[i_]])
    return indices


def f_dist_fit_model(data,dfn=None,ppfb=0.995,dpdfsize=10000,*args,**kwargs):
    """
    Conduct a f-distribution fitting to the data vector using 

    Based upon the noise adaptive algorithm by Carmichael (2013) Carmichael et al. (2015)
    """
    # Conduct fit to the data
    if dfn is not None:
        params = sp.stats.f.fit(data,dfn,*args,**kwargs)
    else:
        params = sp.stats.f.fit(data,*args,**kwargs)
    # Extract parameters for getting PDF value at specified ppf bound
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    # Get ppf bound
    ppf_lower = sp.stats.f.ppf((1-ppfb)/2,*arg,loc=loc,scale=scale)
    ppf_upper = sp.stats.f.ppf((1-ppfb)/2 + ppfb,*arg,loc=loc,scale=scale)
    ppf_bound = sp.stats.f.ppf(ppfb,*arg,loc=loc,scale=scale)
    # Convert to a discretized version of generated f PDF
    x = np.linspace(ppf_lower, ppf_upper, dpdfsize)
    y = sp.stats.f.pdf(x,loc=loc,scale=scale,*arg)
    pdf = pd.Series(y,x)

    return pdf

def f_dist_fit_bound(data,dfn=2,ppfb=0.995,*args,**kwargs):
    """
    Conduct a f-distribution fitting to the data vector using 

    Based upon the noise adaptive algorithm by Carmichael (2013) Carmichael et al. (2015)
    """
    # Conduct fit to the data
    params = sp.stats.f.fit(data,dfn,*args,**kwargs)
    # Extract parameters for getting PDF value at specified ppf bound
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    # Get ppf bound
    ppf_bound = sp.stats.f.ppf(ppfb,*arg,loc=loc,scale=scale)

    return ppf_bound


def display_f_dist_results(dst,cst,f_bnds=[0.95,0.99,0.95],g_bnds=5):
    for ctr in cst:
        npts_mag = np.floor(np.log10(ctr.npts))
        g_mu = np.mean(ctr.data)
        g_sig = np.std(ctr.data)
        f_fits = []
        g_fits = []
        for f_bnd in f_bnds:
            if npts_mag - 4 > 0:
                f_fits.append(f_dist_fit_bound(ctr.data[::10**npts_mag-4],ppfb=f_bnd))
                f_model = f_dist_fit_model(ctr.data[::10**npts_mag-4])
            else:
                f_fits.append(f_dist_fit_bound(ctr.data,ppfb=f_bnd))
                f_model = f_dsit_fit_model(ctr.data)
        for g_bnd in range(g_bnds):
            g_fits.append(g_mu + g_bnd*g_sig)
        plt.figure()
        plt.subplot(311)
        plt.hist(crf.data,1000,density=True)
        plt.plot(f_model)
        plt.legend('CRF Data','F-model fit - 2 dof')
        ax1 = plt.subplot(312)
        dc_scale = dst.select(station=ctr.stats.station)[0].stats.npts/ctr.stats.npts
        plt.plot(np.arange(ctr.stats.npts)*dc_scale,ctr.data)
        for g_ in g_fits:
            plt.plot(np.ones((2,))*g_,np.array([0,1]))
        for f_ in f_fits:
            plt.plot(np.ones((2,))*f_,np.array([0,1]))
        plt.subplot(313,sharex=ax1)
        plt.plot(dst.select(station=ctr.stats.station)[0].data)
        for g_ in g_fits:
            plt.plot(np.array([0,ctr.stats.npts*dc_scale]),np.ones((2,))*g_)
        for f_ in f_fits:
            plt.plot(np.array([0,ctr.stats.npts*dc_scale]),np.ones((2,))*f_)
    plt.show()
    return f_fits, g_fits   



def naive_3sigma_bound(data):
    mu = np.mean(data)
    sig = np.std(data)
    return mu,sig


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

def tr_wind_kurt_py(trace,win,step,fisher=False):
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
    :type **kwargs: Key-word argments 
    :param **kwargs: Key-word arguments for scipy.stats.kurtosis()

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
    
    for i_ in range(len(indices)):
        wsi = indices[i_][0]
        wei = indices[i_][1]
        # Get Data Subset
        x_i = x_0[wsi:wei]
        # Calculate sample kurtosis, applying **kwargs 
        K_i = sp.stats.kurtosis(x_i,fisher=fisher)
        # Write sample kurtosis to crf container
        crf[i_] = K_i
    #if method == 'central':
    t_start = trace.stats.starttime + float(win)/2.
    #elif method == 'left':
    #    t_start = trace.stats.starttime
    #elif method == 'right':
    #    t_start = trace.stats.starttime + win

    crftr.data = crf
    crftr.stats.starttime = t_start
    crftr.stats.delta = step

    return crftr

def tr_ltk_stk_py(trace,stk,ltk,step,fisher=False,verb=0,**kwargs):
    crftr = trace.copy()
    x_0 = crftr.data
    nsamp = trace.stats.npts
    SR1 = trace.stats.sampling_rate
    nst = int(round(stk*SR1))
    nlt = int(round(ltk*SR1))
    nstep = int(round(step*SR1))
    indices = _get_window_indices(nst+nlt-1,nstep,nsamp)
    crf = np.zeros((len(indices),))
    if verb > 0:
        tic = UTCDateTime()
        print('Starting: %s.%s -- %s to %s -- (%s)\n'%
             (crftr.stats.station,crftr.stats.channel,
              str(crftr.stats.starttime),str(crftr.stats.endtime),
              tic))

    for i_ in range(len(indices)):
        wsil = indices[i_][0]
        wsis = wsil + nlt
        weil = wsis
        weis = indices[i_][1]
        x_lt = x_0[wsil:weil]
        x_st = x_0[wsis:weis]
        K_lt = sp.stats.kurtosis(x_lt,fisher=fisher,**kwargs)
        K_st = sp.stats.kurtosis(x_st,fisher=fisher,**kwargs)
        crf[i_] = K_st / K_lt
    crftr.data = crf
    crftr.stats.starttime = trace.stats.starttime + stk + ltk
    crftr.stats.delta = step
    crftr.stats['stk'] = stk
    crftr.stats['ltk'] = ltk
    crftr
    if verb > 0:
        toc = UTCDateTime()
        print('Complete: %s.%s -- Elapsed: %.2f min -- (%s)\n'%
              (crftr.stats.station,crftr.stats.channel,
               (toc-tic)/60.,str(tic)))
    return crftr

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
