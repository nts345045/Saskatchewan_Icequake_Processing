"""
:module: triggerwrite.py
:purpose: Methods for writing outputs from obspy's coincidence_trigger() into pyrocko.snuffler marker files
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 5. DEC 2019
    Header updated 7. MAR 2023 for upload to GitHub

"""


import os
import numpy as np

def open_fobj(fname,fpath='.'):
    filepath = os.path,join(fpath,fname)
    fobj = open(filepath,'w')

def close_fobj(fobj):
    fobj.close()

def writeheader(fobj):
    fobj.write('# Snuffler Markers File Version 0.2\n')

def writewindow(trig_entry,fobj,tpad1=-0.25,tpad2=0.25):
    dur = tpad2 - tpad1
    idt1 = trig_entry['time'] + tpad1
    idt2 = trig_entry['time'] + tpad2
    (idate1,itime1) = idt1.isoformat().split('T')
    (idate2,itime2) = idt2.isoformat().split('T')
    grade = 0

    writestr = '%s %s %s %s %.17f  %d None\n' % (idate1, itime1, idate2, itime2, dur, grade)
    fobj.write(writestr)

    return writestr

def writephase(pick_time, fobj, trace_id, phase='?', eventhash=None, eventdate=None, eventtime=None):
#    sta = tr_stats.station
#    net = tr_stats.network
#    loc = tr_stats.location[:1]
#    chan = tr_stats.channel
    net = trace_id.split('.')[0]
    sta = trace_id.split('.')[1]
    loc = trace_id.split('.')[2][:1]
    chan = trace_id.split('.')[-1]

    (pdate,ptime) = pick_time.isoformat().split('T')

    if not eventhash:
        writeentries = ['phase:',pdate,ptime,grade,net,sta,loc,chan,'None','None','None',phase,'None','False']
        writestr = '%s %s %s  %s %s.%s.%2s.%s  %-15s%-13s%-13s%-9s%s %s\n'%(writeentries)
#        (phase:',pdate,ptime,grade,net,sta,loc,chan,'None','None','None',phase,'None','False')
    elif eventhash:
        writeentries = ['phase:',pdate,ptime,grade,net,sta,loc,chan,eventhash,'=',eventdate,eventime,phase,'None','False']
        writestr = '%s %s %s  %s %s.%s.%2s.%s  %s%s %s   %s %-9s%s %s\n'%(writeentries)
#        ('phase:',pdate,ptime,grade,net,sta,loc,chan,eventhash,eventdate,eventime,phase,'None','False')
    fobj.write(writestr)

    return writestr

def triggerwrite(trigger,fname,fpath='.',tpad1= -0.1,tpad2=0.25,mute=0.,gradelist=None):
    filepath = os.path.join(fpath,fname)
    fobj = open(filepath,'w')
    writeheader(fobj)

    for i_, itrig in enumerate(trigger):
        if i_ == 0:
            lasttime = itrig['time']
            dowrite = True
        # If the absolute time offset between
        elif np.abs(itrig['time'] - lasttime) > mute:
            lasttime = itrig['time']
            dowrite = True
        else:
            dowrite = False

        if dowrite:
#           writewindow(itrig,fobj,tpad1=tpad1,tpad2=tpad2)
            dur = tpad2 - tpad1
            idt1 = itrig['time']+tpad1
            idt2 = itrig['time']+tpad2
            idate1 = idt1.isoformat().split('T')[0]
            itime1 = idt1.isoformat().split('T')[1]
            idate2 = idt2.isoformat().split('T')[0]
            itime2 = idt2.isoformat().split('T')[1]
            if gradelist:
                nstas = len(itrig['stations'])
                grade = 0
                for gt in gradelist:
                    if nstas > gt:
                        grade += 1
                if grade > 5:
                    grade = 5
            else:
                grade = 0

            fobj.write('%s %s %s %s %.17f  %d None\n' % (idate1, itime1, idate2, itime2, dur,grade))
        
    fobj.close()


def trig_pick_writer(trigger,fname,fpath='.',tpad1=-0.1, tpad2 = 0.25, mute=0.):
    filepath = os.path.join(fpath,fname)
    fobj = open(filepath,'w')
    fobj.write('# Snuffler Markers File Version 0.2\n')
    for i_ in itrig in enumerate(trigger):
        if i_ == 0:
            lasttime = itrig['time']
            dowrite = True
        elif np.abs(itrig['time'] - lasttime) > mute:
            lasttime = itrig['time']
            dowrite = True
        else:
            dowrite = False
        if dowrite:
            writewindow(fobj,itrig, tpad1 = tpad1, tpad2 = tpad2)
            phases = []
            for j_,p_ in enumerate(itrig['pick_time']):
                phases.append((p_.timestamp,itrig['trace_ids'][j_],itrig['pick_type'][j_]))
            phases.sort()
            for p_ in phases:
                writephase(UTCDateTime(p_[0]),p_[1],phase=p_[2])

    fobj.close()
            



def pick_dict(trace_id,phase_datetime,phase_uncertainty,phase_name='?'):
    pick = {'ID':trace_id,'Ptype':phase_name,'Time':UTCDateTime,'Err':phase_uncertainty}
    return pick

#def contrig_recpick(trigger_type, thr1, thr2, stream,
#                    thr_coincidence_sum, trace_ids=None,
#                    max_trigger_length=1., delete_long_trigger=True,
#                    trigger_off_extension=0, details=True,
#                    event_templates={}, similarity_threshold=0.8,
#                    pick_type='onset',pick_label='?',
#                    **options):
#    """
#    Modified version of the coincidence_trigger routine from Obspy
#    that produces an augmented trigger dictionary containing estimates
#    of phase arrival times using a variety of methods
#
#    Most inputs are identical to coincidence_trigger, with added kwargs
#    pick_type: Method for selecting the onset time from the crf
#    pick_label: Label to include for the phase marker
#
#    """
#    st = deepcopy(stream)
#    kwargs = {'max_len_delete': delete_long_trigger}
#    # Iterate through traces in stream to produce triggers
#    for tr in st:
#        if tr.id not in trace_ids:
#            msg = "At least one trace's ID was not found in the " + \
#                  "trace ID list and was disregarded (%s)" % tr.id
#            warnings.warn(msg, UserWarning)
#            continue
#        if trigger_type is not None:
#            tr.trigger(trigger_type, **options)
#        kwargs['max_len'] = int(
#            max_trigger_length * tr.stats.sampling_rate + 0.5)
#        tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
#        # Forall on and off times in temporary triggers, get statistics and convert times to UTCDateTimes
#        for on, off in tmp_triggers:
#            try:
#                cft_peak = tr.data[on:off].max()
#                cft_std = tr.data[on:off].std()
#            except ValueError:
#                cft_peak = tr.data[on]
#                cft_std = 0
#            on = tr.stats.starttime + float(on) / tr.stats.sampling_rate
#            off = tr.stats.starttime + float(off) / tr.stats.sampling_rate
#            # Generate trigger list with times, ids, and statistics
#            triggers.append((on.timestamp, off.timestamp, tr.id, cft_peak,
#                             cft_std))
#    # When all traces have been processed, sort the triggers by on-time
#    triggers.sort()
#
#    # the coincidence triggering and coincidence sum computation
#    coincidence_triggers = []
#    last_off_time = 0.0
#    while triggers != []:
#        # remove first trigger from list and look for overlaps
#        on, off, tr_id, cft_peak, cft_std = triggers.pop(0)
#        sta = tr_id.split(".")[1]
#        event = {}
#        event['time'] = UTCDateTime(on)
#        event['stations'] = [tr_id.split(".")[1]]
#        event['trace_ids'] = [tr_id]
#        event['coincidence_sum'] = float(trace_ids[tr_id])
#        event['similarity'] = {}
#        if details:
#            event['cft_peaks'] = [cft_peak]
#            event['cft_stds'] = [cft_std]
#        # NTS added these lines
#        if pick_type == 'onset':
#            event['pick_time'] = UTCDateTime(on)
#        if phase_label:
#            event['pick_phase'] = phase_label
#        # End of NTS added lines for this segment
#
#        # evaluate maximum similarity for station if event templates were
#        # provided
#        templates = event_templates.get(sta)
#        if templates:
#            event['similarity'][sta] = \
#                templates_max_similarity(stream, event['time'], templates)
#        # compile the list of stations that overlap with the current trigger
#        # Go through each trigger left in the trigger list 
#        # NTS: This could be made more efficient with a time-range boundary to subset triggers...
#        for trigger in triggers:
#            # Get information from trigger
#            tmp_on, tmp_off, tmp_tr_id, tmp_cft_peak, tmp_cft_std = trigger
#            tmp_sta = tmp_tr_id.split(".")[1]
#            # skip retriggering of already present station in current
#            # coincidence trigger
#            if tmp_tr_id in event['trace_ids']:
#                continue
#            # check for overlapping trigger,
#            # break if there is a gap in between the two triggers
#            if tmp_on > off + trigger_off_extension:
#                break
#            event['stations'].append(tmp_sta)
#            event['trace_ids'].append(tmp_tr_id)
#            event['coincidence_sum'] += trace_ids[tmp_tr_id]
#            if details:
#                event['cft_peaks'].append(tmp_cft_peak)
#                event['cft_stds'].append(tmp_cft_std)
#            # NTS added these lines
#            if pick_type == 'onset'
#                event['pick_time'].append(UTCDateTime(tmp_on))
#            # End of NTS additions
#
#            # allow sets of triggers that overlap only on subsets of all
#            # stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
#            off = max(off, tmp_off)
#            # evaluate maximum similarity for station if event templates were
#            # provided
#            templates = event_templates.get(tmp_sta)
#            if templates:
#                event['similarity'][tmp_sta] = \
#                    templates_max_similarity(stream, event['time'], templates)
#        # skip if both coincidence sum and similarity thresholds are not met
#        if event['coincidence_sum'] < thr_coincidence_sum:
#            if not event['similarity']:
#                continue
#            elif not any([val > similarity_threshold[_s]
#                          for _s, val in event['similarity'].items()]):
#                continue
#        # skip coincidence trigger if it is just a subset of the previous
#        # (determined by a shared off-time, this is a bit sloppy)
#        if off <= last_off_time:
#            continue
#        event['duration'] = off - on
#        if details:
#            weights = np.array([trace_ids[i] for i in event['trace_ids']])
#            weighted_values = np.array(event['cft_peaks']) * weights
#            event['cft_peak_wmean'] = weighted_values.sum() / weights.sum()
#            weighted_values = np.array(event['cft_stds']) * weights
#            event['cft_std_wmean'] = \
#                (np.array(event['cft_stds']) * weights).sum() / weights.sum()
#        coincidence_triggers.append(event)
#        last_off_time = off
#    return coincidence_trigger


#        rcoinmag = np.log10(cointhresh)
#        icoinmag = np.log10(itrig['coincidence_sum'])
#        coinrat = icoinmag/rcoinmag
#        if coinrat > 1 and coinrat < 2:
#            grade = 4
#        elif coinrat < 3 and coinrat >= 2:
#            grade = 3
#        elif coinrat < 4 and coinrat >= 3#:
#            grade = 2
#        else:
#            grade = 0
#


