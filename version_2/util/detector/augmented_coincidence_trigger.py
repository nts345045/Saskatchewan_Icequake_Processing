#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: augmented_coincidence_trigger.py
# Purpose:  Python network coincidence trigger routine using F-distribution noise 
#           adaptive trigger thresholding.
#
# Author:   Nathan T. Stevens
# Email:    ntstevens@wisc.edu
# Attribution:  Based upon the obspy.signal.trigger.py and associated scripts by
#               Moritz Beyreuther & Tobias Megies (Megies et al., 2010; 2015), 
#               and methods from Arrowsmith et al. (2009), Carmichael (2013), 
#               and Carmichael et al. (2015).
#
# Copyright (C) 2019, 2020 Nathan T. Stevens


from obspy.signal.trigger import *
import sys
sys.path.append('/usr1/ntstevens/SCRIPTS/PYTHON/SGGS_Workflows')
import detectors.kurtosis as okt
import detectors.f_dist_fit as fdf
from joblib import Parallel, delayed
import pandas as pd

def cftrigger(crf_stream, ppfb_on,  ppfb_off, thr_coincidence_sum,\
              mlvl=4, trace_ids = None, max_trigger_length = 2.,\
              delete_long_trigger = True, trigger_off_extension = 0,\
              details = True, event_templates = {},\
              similarity_threshold = 0.8, **options):
    """
    Perform a network coincidence trigger using the noise-adaptive detection
    threshold algorithm from Carmichael (2013); Charmichael et al. (2015)
    where detection thresholds are estimated from confidence bounds
    of an f-distribution fit to continuous data input from 
    characteristic functions. This particular formulation has been tested
    with the windowed kurtosis CRF used in McBrearty et al., (in review)
    using the Pearson formula of kurtosis (sp.stats.kurtosis(..., fisher=False))


    The routine works in the following steps:  *=Original version, **=Adapted here
      * take every single trace in the stream
      ** calculate the F-distribution fit to (potentially down-sampled) CRF
        data from each trace in the stream
      ** evaluate all single station triggering results using specified F-distribution
        confidence bounds.
      * compile chronological overall list of all single station triggers
      * find overlapping single station triggers
      * calculate coincidence sum of every individual overlapping trigger
      * add to coincidence trigger list if it exceeds the given threshold
      * optional: if master event templates are provided, also check single
        station triggers individually and include any single station trigger if
        it exceeds the specified similarity threshold even if no other stations
        coincide with the trigger
      ** return list of network coincidence triggers & information on the F-distribution
        based trigger levels for each input trace.


    :: UPDATED COINCIDENCE TRIGGER INPUTS :: -  adaptations in this script
    :type crf_stream: obspy.core.Stream
    :param crf_stream: Stream containing characteristic response functions that contain
                    real, non-negative values only.
    :type ppfb_on: float
    :param ppfb_on: Value in (0,1) indicating the data-fit F-distribution confidence bound to use
                    to determine the trigger ON value for a given input crf trace in crf_stream
    :type ppfb_off: float
    :param ppfb_off: Value in (0,1) indicating the data-fit F-distribution confidence bound to use
                    to determine the trigger OFF value for a given input crf trace in crf_stream
    :type thr_coincidence_sum: float
    :param thr_coincidence_sum: Essentially the minimum number of stations that need to be
                    triggered in order to accept a network trigger
    :type mlvl: float
    :param mlvl: Magnitude threshold level for down-sampling input crf trace data for estimating
                    F-distribution fits. Data will be sampled every int(10**(log10(nsamp)- mlvl)))
                    if nsamp > mlvl. Default is 4, resulting in O~10,000 evenly spaced samples being
                    used to calculate F-distributions. This aids in coputational speed-up for processing 
                    a batch of triggers.

    :: STANDARD COINCIDENCE TRIGGER INPUTS :: - from obspy.signal.trigger.py
    :type trace_ids: list or dict, optional
    :param trace_ids: Trace IDs to be used in the network coincidence sum. A
        dictionary with trace IDs as keys and weights as values can
        be provided. If a list of trace IDs is provided, all
        weights are set to 1. The default of ``None`` uses all traces present
        in the provided stream. Waveform data with trace IDs not
        present in this list/dict are disregarded in the analysis.
    :type max_trigger_length: int or float
    :param max_trigger_length: Maximum single station trigger length (in
        seconds). ``delete_long_trigger`` controls what happens to single
        station triggers longer than this value.
    :type delete_long_trigger: bool, optional
    :param delete_long_trigger: If ``False`` (default), single station
        triggers are manually released at ``max_trigger_length``, although the
        characteristic function has not dropped below ``thr_off``. If set to
        ``True``, all single station triggers longer than
        ``max_trigger_length`` will be removed and are excluded from
        coincidence sum computation.
    :type trigger_off_extension: int or float, optional
    :param trigger_off_extension: Extends search window for next trigger
        on-time after last trigger off-time in coincidence sum computation.
    :type details: bool, optional
    :param details: If set to ``True`` the output coincidence triggers contain
        more detailed information: A list with the trace IDs (in addition to
        only the station names), as well as lists with single station
        characteristic function peak values and standard deviations in the
        triggering interval and mean values of both, relatively weighted like
        in the coincidence sum. These values can help to judge the reliability
        of the trigger.
    :param options: Necessary keyword arguments for the respective trigger
        that will be passed on. For example ``sta`` and ``lta`` for any STA/LTA
        variant (e.g. ``sta=3``, ``lta=10``).
        Arguments ``sta`` and ``lta`` (seconds) will be mapped to ``nsta``
        and ``nlta`` (samples) by multiplying with sampling rate of trace.
        (e.g. ``sta=3``, ``lta=10`` would call the trigger with 3 and 10
        seconds average, respectively)
    :param event_templates: Event templates to use in checking similarity of
        single station triggers against known events. Expected are streams with
        three traces for Z, N, E component. A dictionary is expected where for
        each station used in the trigger, a list of streams can be provided as
        the value to the network/station key (e.g. {"GR.FUR": [stream1,
        stream2]}). Templates are compared against the provided `stream`
        without the specified triggering routine (`trigger_type`) applied.
    :type event_templates: dict
    :param similarity_threshold: similarity threshold (0.0-1.0) at which a
        single station trigger gets included in the output network event
        trigger list. A common threshold can be set for all stations (float) or
        a dictionary mapping station names to float values for each station.
    :type similarity_threshold: float or dict
    :rtype: list
    :returns: List of event triggers sorted chronologically.

    :: OBSOLITED INPUTS :: - not used here, but used in obspy.signal.trigger.py

    :param trigger_type: String that specifies which trigger is applied (e.g.
        ``'recstalta'``). See e.g. :meth:`obspy.core.trace.Trace.trigger` for
        further details. If set to `None` no triggering routine is applied,
        i.e.  data in traces is supposed to be a precomputed characteristic
        function on which the trigger thresholds are evaluated.
    :type trigger_type: str or None
    :type thr_on: float
    :param thr_on: threshold for switching single station trigger on
    :type thr_off: float
    :param thr_off: threshold for switching single station trigger off
    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: Stream containing waveform data for all stations. These
        data are changed inplace, make a copy to keep the raw waveform data.
    :type thr_coincidence_sum: int or float
    :param thr_coincidence_sum: Threshold for coincidence sum. The network
        coincidence sum has to be at least equal to this value for a trigger to
        be included in the returned trigger list.

    """
    st = crf_stream.copy()
    # if no trace ids are specified use all traces ids found in stream
    if trace_ids is None:
        trace_ids = [tr.id for tr in st]
    # we always work with a dictionary with trace ids and their weights later
    if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
        trace_ids = dict.fromkeys(trace_ids, 1)
    # set up similarity thresholds as a dictionary if necessary
    if not isinstance(similarity_threshold, dict):
        similarity_threshold = dict.fromkeys([tr.stats.station for tr in st],
                                             similarity_threshold)

    # the single station triggering
    triggers = []
    # trigger level reporting
    thresholds = []
    # prepare kwargs for trigger_onset
    kwargs = {'max_len_delete': delete_long_trigger}
    for tr in st:
        # NTS added the single "if" block below

        mu = np.mean(tr.data)
        sig = np.std(tr.data)
#       thr_on = mu + sig*crf_on_coef
#       thr_off = mu + sig*crf_off_coef
        npts_mag = int(np.floor(np.log10(tr.stats.npts)))

        # If number of samples exceeds 10**4, down-sample by 10**(m - 4) -- keeps estimation of f-distribution based on ~10**4 data
        print('Processing f-dist for ',tr.stats.station)
        if npts_mag > mlvl:
            mag_downsample = int(10**(npts_mag - mlvl))
            f_data = tr.data[::mag_downsample]
            thr_on = fdf.f_dist_fit_bound(f_data,ppfb=ppfb_on)
            thr_off = fdf.f_dist_fit_bound(f_data,ppfb=ppfb_off)
        else:
            mag_downsample = tr.data.npts
            f_data = tr.data
            thr_on = fdf.f_dist_fit_bound(f_data,ppfb=ppfb_on)
            thr_off = fdf.f_dist_fit_bound(f_data,ppfb=ppfb_off)
        print('On Threshold',thr_on)
        print('Off Threshold',thr_off)
        thresholds.append([tr.stats.station,tr.stats.channel,tr.stats.starttime,\
                            tr.stats.endtime,len(f_data),\
                            ppfb_on,ppfb_off,thr_on,thr_off])
        if tr.id not in trace_ids:
            msg = "At least one trace's ID was not found in the " + \
                  "trace ID list and was disregarded (%s)" % tr.id
            warnings.warn(msg, UserWarning)
            continue
#        if trigger_type is not None:
#            tr.trigger(trigger_type, **options)
        kwargs['max_len'] = int(max_trigger_length * tr.stats.sampling_rate + 0.5)
        tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
        for on, off in tmp_triggers:
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
    triggers.sort()

    # the coincidence triggering and coincidence sum computation
    coincidence_triggers = []
    last_off_time = 0.0
    while triggers != []:
        # remove first trigger from list and look for overlaps
        on, off, peak, tr_id, cft_peak, cft_std, thr_on, thr_off = triggers.pop(0)
        sta = tr_id.split(".")[1]
        event = {}
        event['time'] = UTCDateTime(on)
        event['on_times'] = [UTCDateTime(on)]
        event['peak_times'] = [UTCDateTime(peak)]
        event['trig_dur'] = [off - on]
        event['stations'] = [tr_id.split(".")[1]]
        event['trace_ids'] = [tr_id]
        event['coincidence_sum'] = float(trace_ids[tr_id])
        event['similarity'] = {}
        if details:
            event['cft_peaks'] = [cft_peak]
            event['cft_stds'] = [cft_std]
            event['thr_on'] = [thr_on]
            event['thr_off'] = [thr_off]
        # evaluate maximum similarity for station if event templates were
        # provided
        templates = event_templates.get(sta)
        if templates:
            event['similarity'][sta] = \
                templates_max_similarity(stream, event['time'], templates)
        # compile the list of stations that overlap with the current trigger
        for trigger in triggers:
            tmp_on, tmp_off, tmp_peak, tmp_tr_id, tmp_cft_peak, tmp_cft_std, tmp_thr_on, tmp_thr_off = trigger
            tmp_sta = tmp_tr_id.split(".")[1]
            # skip retriggering of already present station in current
            # coincidence trigger
            if tmp_tr_id in event['trace_ids']:
                continue
            # check for overlapping trigger,
            # break if there is a gap in between the two triggers
            if tmp_on > off + trigger_off_extension: # TODO: Use this to prevent overlapping full triggers??
                break
            event['on_times'].append(UTCDateTime(tmp_on))
            event['peak_times'].append(UTCDateTime(tmp_peak))
            event['trig_dur'].append(tmp_off - tmp_on)
            event['stations'].append(tmp_sta)
            event['trace_ids'].append(tmp_tr_id)
            event['coincidence_sum'] += trace_ids[tmp_tr_id]
            if details:
                event['cft_peaks'].append(tmp_cft_peak)
                event['cft_stds'].append(tmp_cft_std)
                event['thr_on'].append(tmp_thr_on)
                event['thr_off'].append(tmp_thr_off)
            # allow sets of triggers that overlap only on subsets of all
            # stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
            off = max(off, tmp_off)
            # evaluate maximum similarity for station if event templates were
            # provided
            templates = event_templates.get(tmp_sta)
            if templates:
                event['similarity'][tmp_sta] = \
                    templates_max_similarity(stream, event['time'], templates)
        # skip if both coincidence sum and similarity thresholds are not met
        if event['coincidence_sum'] < thr_coincidence_sum:
            if not event['similarity']:
                continue
            elif not any([val > similarity_threshold[_s]
                          for _s, val in event['similarity'].items()]):
                continue
        # skip coincidence trigger if it is just a subset of the previous
        # (determined by a shared off-time, this is a bit sloppy)
        if off <= last_off_time:
            continue
        event['duration'] = off - on
        if details:
            weights = np.array([trace_ids[i] for i in event['trace_ids']])
            weighted_values = np.array(event['cft_peaks']) * weights
            event['cft_peak_wmean'] = weighted_values.sum() / weights.sum()
            weighted_values = np.array(event['cft_stds']) * weights
            event['cft_std_wmean'] = \
                (np.array(event['cft_stds']) * weights).sum() / weights.sum()
        coincidence_triggers.append(event)
        last_off_time = off
        thresholds = pd.DataFrame(thresholds,columns=['sta','chan','starttime','endtime','F_samp','Fcb_on','Fcb_off','trig_on','trig_off'])
    return coincidence_triggers, thresholds



def cftrigger_par(crf_stream, ppfb_on,  ppfb_off, thr_coincidence_sum,\
                  mlvl=4, n_cores = 4, trace_ids = None, max_trigger_length = 2.,\
                  delete_long_trigger = True, trigger_off_extension = 0,\
                  details = True, event_templates = {},\
                  similarity_threshold = 0.8, **options):
    """
    Perform a network coincidence trigger using the noise-adaptive detection
    threshold algorithm from Carmichael (2013); Charmichael et al. (2015)
    where detection thresholds are estimated from confidence bounds
    of an f-distribution fit to continuous data input from 
    characteristic functions. This particular formulation has been tested
    with the windowed kurtosis CRF used in McBrearty et al., (in review)
    using the Pearson formula of kurtosis (sp.stats.kurtosis(..., fisher=False))


    The routine works in the following steps:  *=Original version, **=Adapted here
      * take every single trace in the stream
      ** calculate the F-distribution fit to (potentially down-sampled) CRF
        data from each trace in the stream
      ** evaluate all single station triggering results using specified F-distribution
        confidence bounds.
      * compile chronological overall list of all single station triggers
      * find overlapping single station triggers
      * calculate coincidence sum of every individual overlapping trigger
      * add to coincidence trigger list if it exceeds the given threshold
      * optional: if master event templates are provided, also check single
        station triggers individually and include any single station trigger if
        it exceeds the specified similarity threshold even if no other stations
        coincide with the trigger
      ** return list of network coincidence triggers & information on the F-distribution
        based trigger levels for each input trace.


    :: UPDATED COINCIDENCE TRIGGER INPUTS :: -  adaptations in this script
    :type crf_stream: obspy.core.Stream
    :param crf_stream: Stream containing characteristic response functions that contain
                    real, non-negative values only.
    :type ppfb_on: float
    :param ppfb_on: Value in (0,1) indicating the data-fit F-distribution confidence bound to use
                    to determine the trigger ON value for a given input crf trace in crf_stream
    :type ppfb_off: float
    :param ppfb_off: Value in (0,1) indicating the data-fit F-distribution confidence bound to use
                    to determine the trigger OFF value for a given input crf trace in crf_stream
    :type thr_coincidence_sum: float
    :param thr_coincidence_sum: Essentially the minimum number of stations that need to be
                    triggered in order to accept a network trigger
    :type mlvl: float
    :param mlvl: Magnitude threshold level for down-sampling input crf trace data for estimating
                    F-distribution fits. Data will be sampled every int(10**(log10(nsamp)- mlvl)))
                    if nsamp > mlvl. Default is 4, resulting in O~10,000 evenly spaced samples being
                    used to calculate F-distributions. This aids in coputational speed-up for processing 
                    a batch of triggers.

    :: STANDARD COINCIDENCE TRIGGER INPUTS :: - from obspy.signal.trigger.py
    :type trace_ids: list or dict, optional
    :param trace_ids: Trace IDs to be used in the network coincidence sum. A
        dictionary with trace IDs as keys and weights as values can
        be provided. If a list of trace IDs is provided, all
        weights are set to 1. The default of ``None`` uses all traces present
        in the provided stream. Waveform data with trace IDs not
        present in this list/dict are disregarded in the analysis.
    :type max_trigger_length: int or float
    :param max_trigger_length: Maximum single station trigger length (in
        seconds). ``delete_long_trigger`` controls what happens to single
        station triggers longer than this value.
    :type delete_long_trigger: bool, optional
    :param delete_long_trigger: If ``False`` (default), single station
        triggers are manually released at ``max_trigger_length``, although the
        characteristic function has not dropped below ``thr_off``. If set to
        ``True``, all single station triggers longer than
        ``max_trigger_length`` will be removed and are excluded from
        coincidence sum computation.
    :type trigger_off_extension: int or float, optional
    :param trigger_off_extension: Extends search window for next trigger
        on-time after last trigger off-time in coincidence sum computation.
    :type details: bool, optional
    :param details: If set to ``True`` the output coincidence triggers contain
        more detailed information: A list with the trace IDs (in addition to
        only the station names), as well as lists with single station
        characteristic function peak values and standard deviations in the
        triggering interval and mean values of both, relatively weighted like
        in the coincidence sum. These values can help to judge the reliability
        of the trigger.
    :param options: Necessary keyword arguments for the respective trigger
        that will be passed on. For example ``sta`` and ``lta`` for any STA/LTA
        variant (e.g. ``sta=3``, ``lta=10``).
        Arguments ``sta`` and ``lta`` (seconds) will be mapped to ``nsta``
        and ``nlta`` (samples) by multiplying with sampling rate of trace.
        (e.g. ``sta=3``, ``lta=10`` would call the trigger with 3 and 10
        seconds average, respectively)
    :param event_templates: Event templates to use in checking similarity of
        single station triggers against known events. Expected are streams with
        three traces for Z, N, E component. A dictionary is expected where for
        each station used in the trigger, a list of streams can be provided as
        the value to the network/station key (e.g. {"GR.FUR": [stream1,
        stream2]}). Templates are compared against the provided `stream`
        without the specified triggering routine (`trigger_type`) applied.
    :type event_templates: dict
    :param similarity_threshold: similarity threshold (0.0-1.0) at which a
        single station trigger gets included in the output network event
        trigger list. A common threshold can be set for all stations (float) or
        a dictionary mapping station names to float values for each station.
    :type similarity_threshold: float or dict
    :rtype: list
    :returns: List of event triggers sorted chronologically.

    :: OBSOLITED INPUTS :: - not used here, but used in obspy.signal.trigger.py

    :param trigger_type: String that specifies which trigger is applied (e.g.
        ``'recstalta'``). See e.g. :meth:`obspy.core.trace.Trace.trigger` for
        further details. If set to `None` no triggering routine is applied,
        i.e.  data in traces is supposed to be a precomputed characteristic
        function on which the trigger thresholds are evaluated.
    :type trigger_type: str or None
    :type thr_on: float
    :param thr_on: threshold for switching single station trigger on
    :type thr_off: float
    :param thr_off: threshold for switching single station trigger off
    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: Stream containing waveform data for all stations. These
        data are changed inplace, make a copy to keep the raw waveform data.
    :type thr_coincidence_sum: int or float
    :param thr_coincidence_sum: Threshold for coincidence sum. The network
        coincidence sum has to be at least equal to this value for a trigger to
        be included in the returned trigger list.

    """
    st = crf_stream.copy()
    # if no trace ids are specified use all traces ids found in stream
    if trace_ids is None:
        trace_ids = [tr.id for tr in st]
    # we always work with a dictionary with trace ids and their weights later
    if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
        trace_ids = dict.fromkeys(trace_ids, 1)
    # set up similarity thresholds as a dictionary if necessary
    if not isinstance(similarity_threshold, dict):
        similarity_threshold = dict.fromkeys([tr.stats.station for tr in st],
                                             similarity_threshold)

    # the single station triggering
    triggers = []
    # trigger level reporting
    thresholds = []

    # PARALLELIZED VERSION OF TRIGGER GENERATION
    par_outputs = Parallel(n_jobs=n_cores)(delayed(fdf.f_dist_trigger)(\
                                tr=tr,ppfb_on=ppfb_on,ppfb_off=ppfb_off,mlvl=mlvl,\
                                max_trigger_length = max_trigger_length,\
                                delete_long_trigger=delete_long_trigger) for tr in st)

    # Unpack parallel processing outputs to continue with detection threshold as normal.
    for output in par_outputs:
        for trigger in output[0]:
            triggers.append(trigger)
        for threshold in output[1]:
            thresholds.append(output[1])


    triggers.sort()

    # the coincidence triggering and coincidence sum computation
    coincidence_triggers = []
    last_off_time = 0.0
    while triggers != []:
        # remove first trigger from list and look for overlaps
        on, off, peak, tr_id, cft_peak, cft_std, thr_on, thr_off = triggers.pop(0)
        sta = tr_id.split(".")[1]
        event = {}
        event['time'] = UTCDateTime(on)
        event['on_times'] = [UTCDateTime(on)]
        event['peak_times'] = [UTCDateTime(peak)]
        event['trig_dur'] = [off - on]
        event['stations'] = [tr_id.split(".")[1]]
        event['trace_ids'] = [tr_id]
        event['coincidence_sum'] = float(trace_ids[tr_id])
        event['similarity'] = {}
        if details:
            event['cft_peaks'] = [cft_peak]
            event['cft_stds'] = [cft_std]
            event['thr_on'] = [thr_on]
            event['thr_off'] = [thr_off]
        # evaluate maximum similarity for station if event templates were
        # provided
        templates = event_templates.get(sta)
        if templates:
            event['similarity'][sta] = \
                templates_max_similarity(stream, event['time'], templates)
        # compile the list of stations that overlap with the current trigger
        for trigger in triggers:
            tmp_on, tmp_off, tmp_peak, tmp_tr_id, tmp_cft_peak, tmp_cft_std, tmp_thr_on, tmp_thr_off = trigger
            tmp_sta = tmp_tr_id.split(".")[1]
            # skip retriggering of already present station in current
            # coincidence trigger
            if tmp_tr_id in event['trace_ids']:
                continue
            # check for overlapping trigger,
            # break if there is a gap in between the two triggers
            if tmp_on > off + trigger_off_extension: # TODO: Use this to prevent overlapping full triggers??
                break
            event['on_times'].append(UTCDateTime(tmp_on))
            event['peak_times'].append(UTCDateTime(tmp_peak))
            event['trig_dur'].append(tmp_off - tmp_on)
            event['stations'].append(tmp_sta)
            event['trace_ids'].append(tmp_tr_id)
            event['coincidence_sum'] += trace_ids[tmp_tr_id]
            if details:
                event['cft_peaks'].append(tmp_cft_peak)
                event['cft_stds'].append(tmp_cft_std)
                event['thr_on'].append(tmp_thr_on)
                event['thr_off'].append(tmp_thr_off)
            # allow sets of triggers that overlap only on subsets of all
            # stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
            off = max(off, tmp_off)
            # evaluate maximum similarity for station if event templates were
            # provided
            templates = event_templates.get(tmp_sta)
            if templates:
                event['similarity'][tmp_sta] = \
                    templates_max_similarity(stream, event['time'], templates)
        # skip if both coincidence sum and similarity thresholds are not met
        if event['coincidence_sum'] < thr_coincidence_sum:
            if not event['similarity']:
                continue
            elif not any([val > similarity_threshold[_s]
                          for _s, val in event['similarity'].items()]):
                continue
        # skip coincidence trigger if it is just a subset of the previous
        # (determined by a shared off-time, this is a bit sloppy)
        if off <= last_off_time:
            continue
        event['duration'] = off - on
        if details:
            weights = np.array([trace_ids[i] for i in event['trace_ids']])
            weighted_values = np.array(event['cft_peaks']) * weights
            event['cft_peak_wmean'] = weighted_values.sum() / weights.sum()
            weighted_values = np.array(event['cft_stds']) * weights
            event['cft_std_wmean'] = \
                (np.array(event['cft_stds']) * weights).sum() / weights.sum()
        coincidence_triggers.append(event)
        last_off_time = off
        thresholds = pd.DataFrame(thresholds,columns=['sta','chan','starttime','endtime','F_samp','Fcb_on','Fcb_off','trig_on','trig_off'])
    return coincidence_triggers, thresholds





# def ctrigger(crf_stream, crf_on_coef,crf_off_coef, thr_coincidence_sum, trace_ids=None,
#                         max_trigger_length=2., delete_long_trigger=True,
#                         trigger_off_extension=0, details=True,
#                         event_templates={}, similarity_threshold=0.8,
#                         **options):
#     """
#     Perform a network coincidence trigger.

#     The routine works in the following steps:
#       * take every single trace in the stream
#       * apply specified triggering routine (can be skipped to work on
#         precomputed custom characteristic functions)
#       * evaluate all single station triggering results
#       * compile chronological overall list of all single station triggers
#       * find overlapping single station triggers
#       * calculate coincidence sum of every individual overlapping trigger
#       * add to coincidence trigger list if it exceeds the given threshold
#       * optional: if master event templates are provided, also check single
#         station triggers individually and include any single station trigger if
#         it exceeds the specified similarity threshold even if no other stations
#         coincide with the trigger
#       * return list of network coincidence triggers

#     .. note::
#         An example can be found in the
#         `Trigger/Picker Tutorial
#         <https://tutorial.obspy.org/code_snippets/trigger_tutorial.html>`_.

#     .. note::
#         Setting `trigger_type=None` precomputed characteristic functions can
#         be provided.

#     .. seealso:: [Withers1998]_ (p. 98) and [Trnkoczy2012]_

#     :param trigger_type: String that specifies which trigger is applied (e.g.
#         ``'recstalta'``). See e.g. :meth:`obspy.core.trace.Trace.trigger` for
#         further details. If set to `None` no triggering routine is applied,
#         i.e.  data in traces is supposed to be a precomputed characteristic
#         function on which the trigger thresholds are evaluated.
#     :type trigger_type: str or None
#     :type thr_on: float
#     :param thr_on: threshold for switching single station trigger on
#     :type thr_off: float
#     :param thr_off: threshold for switching single station trigger off
#     :type stream: :class:`~obspy.core.stream.Stream`
#     :param stream: Stream containing waveform data for all stations. These
#         data are changed inplace, make a copy to keep the raw waveform data.
#     :type thr_coincidence_sum: int or float
#     :param thr_coincidence_sum: Threshold for coincidence sum. The network
#         coincidence sum has to be at least equal to this value for a trigger to
#         be included in the returned trigger list.
#     :type trace_ids: list or dict, optional
#     :param trace_ids: Trace IDs to be used in the network coincidence sum. A
#         dictionary with trace IDs as keys and weights as values can
#         be provided. If a list of trace IDs is provided, all
#         weights are set to 1. The default of ``None`` uses all traces present
#         in the provided stream. Waveform data with trace IDs not
#         present in this list/dict are disregarded in the analysis.
#     :type max_trigger_length: int or float
#     :param max_trigger_length: Maximum single station trigger length (in
#         seconds). ``delete_long_trigger`` controls what happens to single
#         station triggers longer than this value.
#     :type delete_long_trigger: bool, optional
#     :param delete_long_trigger: If ``False`` (default), single station
#         triggers are manually released at ``max_trigger_length``, although the
#         characteristic function has not dropped below ``thr_off``. If set to
#         ``True``, all single station triggers longer than
#         ``max_trigger_length`` will be removed and are excluded from
#         coincidence sum computation.
#     :type trigger_off_extension: int or float, optional
#     :param trigger_off_extension: Extends search window for next trigger
#         on-time after last trigger off-time in coincidence sum computation.
#     :type details: bool, optional
#     :param details: If set to ``True`` the output coincidence triggers contain
#         more detailed information: A list with the trace IDs (in addition to
#         only the station names), as well as lists with single station
#         characteristic function peak values and standard deviations in the
#         triggering interval and mean values of both, relatively weighted like
#         in the coincidence sum. These values can help to judge the reliability
#         of the trigger.
#     :param options: Necessary keyword arguments for the respective trigger
#         that will be passed on. For example ``sta`` and ``lta`` for any STA/LTA
#         variant (e.g. ``sta=3``, ``lta=10``).
#         Arguments ``sta`` and ``lta`` (seconds) will be mapped to ``nsta``
#         and ``nlta`` (samples) by multiplying with sampling rate of trace.
#         (e.g. ``sta=3``, ``lta=10`` would call the trigger with 3 and 10
#         seconds average, respectively)
#     :param event_templates: Event templates to use in checking similarity of
#         single station triggers against known events. Expected are streams with
#         three traces for Z, N, E component. A dictionary is expected where for
#         each station used in the trigger, a list of streams can be provided as
#         the value to the network/station key (e.g. {"GR.FUR": [stream1,
#         stream2]}). Templates are compared against the provided `stream`
#         without the specified triggering routine (`trigger_type`) applied.
#     :type event_templates: dict
#     :param similarity_threshold: similarity threshold (0.0-1.0) at which a
#         single station trigger gets included in the output network event
#         trigger list. A common threshold can be set for all stations (float) or
#         a dictionary mapping station names to float values for each station.
#     :type similarity_threshold: float or dict
#     :rtype: list
#     :returns: List of event triggers sorted chronologically.
#     """
#     st = crf_stream.copy()
#     # if no trace ids are specified use all traces ids found in stream
#     if trace_ids is None:
#         trace_ids = [tr.id for tr in st]
#     # we always work with a dictionary with trace ids and their weights later
#     if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
#         trace_ids = dict.fromkeys(trace_ids, 1)
#     # set up similarity thresholds as a dictionary if necessary
#     if not isinstance(similarity_threshold, dict):
#         similarity_threshold = dict.fromkeys([tr.stats.station for tr in st],
#                                              similarity_threshold)

#     # the single station triggering
#     triggers = []
#     # prepare kwargs for trigger_onset
#     kwargs = {'max_len_delete': delete_long_trigger}
#     for tr in st:
#         # NTS added the single "if" block below

#         mu = np.mean(tr.data)
#         sig = np.std(tr.data)
#         thr_on = mu + sig*crf_on_coef
#         thr_off = mu + sig*crf_off_coef

#         if tr.id not in trace_ids:
#             msg = "At least one trace's ID was not found in the " + \
#                   "trace ID list and was disregarded (%s)" % tr.id
#             warnings.warn(msg, UserWarning)
#             continue
# #        if trigger_type is not None:
# #            tr.trigger(trigger_type, **options)
#         kwargs['max_len'] = int(max_trigger_length * tr.stats.sampling_rate + 0.5)
#         tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
#         for on, off in tmp_triggers:
#             try:
#                 cft_peak = tr.data[on:off].max()
#                 cft_peak_idx = np.argmax(tr.data[on:off])
#                 cft_std = tr.data[on:off].std()
#             except ValueError:
#                 cft_peak = tr.data[on]
#                 cft_peak_idx = on
#                 cft_std = 0
#             on = tr.stats.starttime + float(on) / tr.stats.sampling_rate
#             off = tr.stats.starttime + float(off) / tr.stats.sampling_rate
#             peak = on + float(cft_peak_idx) / tr.stats.sampling_rate
#             triggers.append((on.timestamp, off.timestamp, peak.timestamp, 
#                              tr.id, cft_peak,
#                              cft_std,thr_on,thr_off))
#     triggers.sort()

#     # the coincidence triggering and coincidence sum computation
#     coincidence_triggers = []
#     last_off_time = 0.0
#     while triggers != []:
#         # remove first trigger from list and look for overlaps
#         on, off, peak, tr_id, cft_peak, cft_std, thr_on, thr_off = triggers.pop(0)
#         sta = tr_id.split(".")[1]
#         event = {}
#         event['time'] = UTCDateTime(on)
#         event['on_times'] = [UTCDateTime(on)]
#         event['peak_times'] = [UTCDateTime(peak)]
#         event['trig_dur'] = [off - on]
#         event['stations'] = [tr_id.split(".")[1]]
#         event['trace_ids'] = [tr_id]
#         event['coincidence_sum'] = float(trace_ids[tr_id])
#         event['similarity'] = {}
#         if details:
#             event['cft_peaks'] = [cft_peak]
#             event['cft_stds'] = [cft_std]
#             event['thr_on'] = [thr_on]
#             event['thr_off'] = [thr_off]
#         # evaluate maximum similarity for station if event templates were
#         # provided
#         templates = event_templates.get(sta)
#         if templates:
#             event['similarity'][sta] = \
#                 templates_max_similarity(stream, event['time'], templates)
#         # compile the list of stations that overlap with the current trigger
#         for trigger in triggers:
#             tmp_on, tmp_off, tmp_peak, tmp_tr_id, tmp_cft_peak, tmp_cft_std, tmp_thr_on, tmp_thr_off = trigger
#             tmp_sta = tmp_tr_id.split(".")[1]
#             # skip retriggering of already present station in current
#             # coincidence trigger
#             if tmp_tr_id in event['trace_ids']:
#                 continue
#             # check for overlapping trigger,
#             # break if there is a gap in between the two triggers
#             if tmp_on > off + trigger_off_extension: # TODO: Use this to prevent overlapping full triggers??
#                 break
#             event['on_times'].append(UTCDateTime(tmp_on))
#             event['peak_times'].append(UTCDateTime(tmp_peak))
#             event['trig_dur'].append(tmp_off - tmp_on)
#             event['stations'].append(tmp_sta)
#             event['trace_ids'].append(tmp_tr_id)
#             event['coincidence_sum'] += trace_ids[tmp_tr_id]
#             if details:
#                 event['cft_peaks'].append(tmp_cft_peak)
#                 event['cft_stds'].append(tmp_cft_std)
#                 event['thr_on'].append(tmp_thr_on)
#                 event['thr_off'].append(tmp_thr_off)
#             # allow sets of triggers that overlap only on subsets of all
#             # stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
#             off = max(off, tmp_off)
#             # evaluate maximum similarity for station if event templates were
#             # provided
#             templates = event_templates.get(tmp_sta)
#             if templates:
#                 event['similarity'][tmp_sta] = \
#                     templates_max_similarity(stream, event['time'], templates)
#         # skip if both coincidence sum and similarity thresholds are not met
#         if event['coincidence_sum'] < thr_coincidence_sum:
#             if not event['similarity']:
#                 continue
#             elif not any([val > similarity_threshold[_s]
#                           for _s, val in event['similarity'].items()]):
#                 continue
#         # skip coincidence trigger if it is just a subset of the previous
#         # (determined by a shared off-time, this is a bit sloppy)
#         if off <= last_off_time:
#             continue
#         event['duration'] = off - on
#         if details:
#             weights = np.array([trace_ids[i] for i in event['trace_ids']])
#             weighted_values = np.array(event['cft_peaks']) * weights
#             event['cft_peak_wmean'] = weighted_values.sum() / weights.sum()
#             weighted_values = np.array(event['cft_stds']) * weights
#             event['cft_std_wmean'] = \
#                 (np.array(event['cft_stds']) * weights).sum() / weights.sum()
#         coincidence_triggers.append(event)
#         last_off_time = off
#     return coincidence_triggers
