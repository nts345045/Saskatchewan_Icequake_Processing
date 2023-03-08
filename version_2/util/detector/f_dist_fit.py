#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: f_dist_fit.py
Purpose:  Python implementation of F-distribution fitting to seismic characteristic
            response function (CRF) data
Author:   Nathan T. Stevens
Email:    ntstevens@wisc.edu

Attribution:  Implementation of the statistical core of the noise-adaptive detection threshold
            algorithm developed in Arrowsmith et al. (2009), Carmichael (2013), and 
            Carmichael et al. (2015)

Copyright (C) 2020 Nathan T. Stevens

:last major revision: 10. APR 2020
    Header updated with last revision date on 8. MAR 2023 for upload to GitHub
"""
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from warnings import warn
from obspy.signal.trigger import trigger_onset


def f_dist_fit_model(data,dfn=2,ppfb=0.995,dpdfsize=10000,*args,**kwargs):
    """
    Conduct a F-distribution fitting to an input data vector and return the 
    fit F-distribution probability density function (PDF)

    :: INPUTS ::
    :type data: numpy.ndarray (n,)
    :param
    
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
    
    ::INPUTS::
    :type data: numpy.ndarray (n,)
    :param data: data vector to fit an F-distribution to
    :type dfn: [optional] int
    :param dfn: [optional] number of degrees of freedom to include in 
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
    """
    Display results of fitting an F-distribution to a stream of characteristic response functions
    and compare to the initial input data stream
    INPUTS
    :type dst: obspy.core.Stream
    :param dst: Stream containing pre-crf transform data
    :type cst: obspy.core.Strea
    :param cst: Stream containing characteristic response functions developed from dst
    :type f_bnds: [optional] list-like
    :param f_bdns: [optional] confidence bounds to highlight for F-distribution
    :type g_bnds: [optional] integer
    :param g_bnds: [optional] maximum gaussian uncertainty bound to plot (plots 1-g_bnds)

    OUTPUTS
    :rtype f_fits: list
    :return f_fits: list of f_bnds level values from each CRF input via cst and defined by f_bnds
    :rtype g_fits: list
    :return g_fits: list of g_bnds level values from each CRF input via cst and defined by g_bnds
    TODO: Need to allow for some type of station and channel labeling in these lists...
    """

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

def f_dist_trigger(tr,ppfb_on,ppfb_off,mlvl=4,max_trigger_length=2.,delete_long_trigger=True):
    """
    Use F-distribution confidence intervals on the input trace to produce lists of triggers
    """
    triggers = []
    # prepare kwargs for trigger_onset
    kwargs = {'max_len_delete': delete_long_trigger}
    npts_mag = int(np.floor(np.log10(tr.stats.npts)))

    # If number of samples exceeds 10**4, down-sample by 10**(m - 4) -- keeps estimation of f-distribution based on ~10**4 data
    if npts_mag > mlvl:
        mag_downsample = int(10**(npts_mag - mlvl))
        f_data = tr.data[::mag_downsample]
        thr_on = f_dist_fit_bound(f_data,ppfb=ppfb_on)
        thr_off = f_dist_fit_bound(f_data,ppfb=ppfb_off)
    else:
        mag_downsample = tr.data.npts
        f_data = tr.data
        thr_on = f_dist_fit_bound(f_data,ppfb=ppfb_on)
        thr_off = f_dist_fit_bound(f_data,ppfb=ppfb_off)
    print('F-Distribution Fit for',tr.stats.station)
    print('On: %.4f  Off: %.4f   Nsamp: %d'%(thr_on,thr_off,len(f_data)))
    threshold = [tr.stats.station,tr.stats.channel,tr.stats.starttime,\
                  tr.stats.endtime,len(f_data),ppfb_on,ppfb_off,thr_on,thr_off]
#     if tr.id not in trace_ids:

#         msg = "At least one trace's ID was not found in the " + \
#               "trace ID list and was disregarded (%s)" % tr.id
#         warnings.warn(msg, UserWarning)
# #        if trigger_type is not None:
# #            tr.trigger(trigger_type, **options)
    kwargs['max_len'] = int(max_trigger_length * tr.stats.sampling_rate + 0.5)

    tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
    for on, off in tmp_triggers:
        if off - on > tr.stats.delta:
            try:
                cft_peak = tr.data[on:off].max()
                cft_peak_idx = np.argmax(tr.data[on:off])
                cft_std = tr.data[on:off].std()
            except ValueError:
                cft_peak = tr.data[on]
                cft_peak_idx = on
                cft_std = 0
            on = tr.stats.starttime + float(on) / tr.stats.sampling_rate
            off = tr.stats.starttime + float(off) / tr.stats.sampling_rate
            peak = on + float(cft_peak_idx) / tr.stats.sampling_rate
            triggers.append((on.timestamp, off.timestamp, peak.timestamp, 
                             tr.id, cft_peak,
                             cft_std,thr_on,thr_off))


    return [triggers,threshold]

