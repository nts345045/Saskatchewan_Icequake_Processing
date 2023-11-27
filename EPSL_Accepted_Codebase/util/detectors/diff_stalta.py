"""
:module: diff_stalta.py
:purpose: Methods for various pure-python Short Term Average - Long Term Average window characteristic functions not included in standard ObsPy distributions
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major update: 21. NOV 2019
    Header updated 7. MAR 2023 for upload to GitHub

"""


from obspy.core import Stream, Trace, UTCDateTime
import numpy as np


def classic_STA_LTA_diff_py(trace,tsta,tlta):
    """
    Calculate the difference of the LTA-STA for a classic STA/LTA scheme
    adapted from classic_sta_lta_py
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
    trace.stats.processing.append('extended trigger methods: STA-LTA, sta: %.4e, lta: %.4e'%(nsta,nlta))
    return trace

def classic_LTA_STA_diff_py(trace,tsta,tlta):
    """
    Calculate the difference LTA-STA for a classic STA/LTA scheme
    adapted from classic_sta_lta_py
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
    trace.stats.processing.append('extended trigger methods: LTA-STA, sta: %.4e, lta: %.4e'%(nsta,nlta))
    return trace


def delayed_STA_LTA_diff_py(trace,tsta,tlta):
    """
    Calculate of the difference of the STA & LTA for a delayed STA/LTA scheme
    adapted from delayed_sta_lta
    """
    sr = trace.stats.sampling_rate
    a = trace.data
    nsta = int(tsta*sr)
    nlta = int(tlta*sr)
    m = len(trace.data)
    sta = np.zeros(m, dtype=np.float64)
    lta = np.zeros(m, dtype=np.float64)
    for i_ in range(m):
        sta[i_] = (a[i_]**2 + a[i_ - nsta]**2) / nsta + sta[i_ - 1]
        lta[i_] = (a[i_ - nsta - 1]**2 + a[i_ - nsta - nlta - 1]**2) /nlta + lta[i_ - 1]
    sta[0:nlta + nsta + 50] = 0
    lta[0:nlta + nsta + 50] = 1
    
    # Overwrite trace data
    trace.data = sta - lta
    trace.stats.processing.append('extended trigger methods: Delayed STA-LTA, sta: %.4e, lta: %.4e'%(nsta,nlta))
    return trace


def recursive_STA_LTA_diff_py(trace,tsta,tlta):
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

def recursive_LTA_STA_diff_py(trace,tsta,tlta):
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
        crf[i_] = lta - sta
        if i_ < nlta:
            crf[i_] = 0.
        CRF = np.array(crf)
    trace.data = CRF
    trace.stats.processing.append('extended trigger methods: Recursive LTA-STA, sta: %.4e, lta: %.4e'%(nsta,nlta))
    return trace



