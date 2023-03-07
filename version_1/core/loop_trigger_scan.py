"""
:module: loop_trigger_scan.py
:purpose: Driver (with shared-memory parallelization options) for event detection using kurtosis 
            characteristic response functions based on McBrearty et al. (2020) [a preprint at the time]
             a noise adaptive trigger threshold algorithm based on conversations with Josh Carmichael (LANL), 
            Ian McBrearty (Stanford U.), and their writings/research, and an adaptation of the coincidence_trigger()
            network detection method in standard ObsPy distributions (e.g., Megies et al., 2015)
:author: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 22. JAN 2020
    Header added and repository module imports updated 7. MAR 2023 for upload to GitHub

"""


import sys
import os
from obspy.core import read, UTCDateTime, Stream, Trace
from copy import deepcopy
import numpy as np

#sys.path.append('/home/nates/Scripts/Python/PKG_DEV')
# sys.path.append('/usr1/ntstevens/SCRIPTS/PYTHON/PKG_DEV')
# import dbscripts.wf_query as wfq
# import dbscripts.connect as conn
# import obspytools.detectors.kurtosis as kdet
# import obspytools.detectors.triggerwrite as trgw
sys.path.append(os.path.join('..','..'))
import util.pisces_DB.wf_query as wfq
import util.pisces_DB.connect as conn
import util.detectors.kurtosis as kdet
import util.detectors.triggerwrite as trgw


#zchans = ['GNZ','GP1']
zchans = ['GNZ','GP3'] # GP1 was a horizontal channel (1 ~ N, 2 ~ E, 3 = Z)
SPexp = 0.05
# Define Time Parameters
year = 2019
jday0 = 213
for j_ in range(2):
    jday = jday0 + j_
    jdstr = str(year)+str(jday)
    #date = UTCDateTime(year,mo,day)
    date = UTCDateTime(jdstr)
    mo = date.month
    day = date.day
    #folder = './dtkurt_'+str(date.year)+str(date.julday)
    folder = './dtkurt_v3_'+jdstr

    TICD = UTCDateTime()
    print('Runtime Start: %s'%(str(TICD)))
    try:
        os.mkdirs(folder)
    except:
        print('%s  already exists'%(folder))

    # Connect to sqlite database
    session, meta, tabdict, tablist, collist = conn.connect4query()
    wfdisc = tabdict['wfdisc']
    plotting = False
    parallel = True
    if parallel:
        ncores = 6

    for hr_ in range(24):
        TICH = UTCDateTime()
        stime = UTCDateTime(year,mo,day,hr_)
        etime = stime + 3605.
        print('Starting time window '+str(stime)+' to '+str(etime))

        # Query waveforms for specified time window
      #  breakpoint()
        wf_q = session.query(wfdisc).filter(wfdisc.time < etime.timestamp)
        wf_q = wf_q.filter(wfdisc.endtime > stime.timestamp)
        wf_q = wf_q.filter(wfdisc.chan.in_(zchans))
        st = Stream()
        for wf in wf_q:
            wfile = os.path.join(wf.dir,wf.dfile)
            print('Loading file: %s'%(wfile))
            st += read(wfile,starttime=stime,endtime=etime)
            
        st.merge(method=1,fill_value='interpolate')

        if len(st) > 0:
            # Preprocess Data
            try:
                st.filter('highpass',freq=80)
            except NotImplementedError:
                for tr in st:
                    if isinstance(tr.data,np.ma.masked_array):
                        print('Trace: %s is a masked array')
                        tr.data = tr.data._cleanup(misalignment_threshold = 1e-3)
                st.filter('highpass',freq=80)
            print('Waveforms Filtered: %d'%(len(st)))
            # Conduct coincidence trigger analysis using dt recursive kurtosis crf
            ctr  = 6    # Coincidence Trigger Threshold
            print('Implementing Grading Scheme')
            gtl  = [ctr, 10, 15, 20, 25] # List of thresholds to exceed to increase grade
            trg1 = 500 # Trace Trigger ON Threshold
            trg2 = 500 # Trace Trigger OFF Threshold
            dtkw = 0.1  # [sec] dt recursive kurtosis evaluation window
            bo   = 5    # Number of times to apply boxcar smoothing algorithm
            nbw  = 3    # [samples] length of boxcar operator
            mtl  = 2    # [sec] Maximum trigger length
            simt = 0.9  # Similarity threshold
            signed = False
            if not parallel:
                trig,dtkurt_st = kdet.con_trig_dtkurt_py(st,trg1,trg2,dtkw,ctr,boxord=bo,nswin=nbw,sign=signed, isplot=plotting, max_trigger_length=mtl,delete_long_trigger=True,details=True,similarity_threshold=simt) 
            if parallel:
                trig = kdet.con_trig_dtkurt_py_parallel(st,trg1,trg2,dtkw,ctr,ncores,max_trigger_length=mtl,delete_long_trigger=True,details=True,similarity_threshold=simt) 

            print('Number of triggers: '+str(len(trig)))
            TOC = UTCDateTime()
            print('Day Elapsed Time: %.2f min'%((TOC-TICD)/60.))
            print('Hour Elapsed Time: %.2f min'%((TOC-TICH)/60.))
            fname = '%s%d%d%02d'%('dtkurt_trigs_v4_',stime.year,stime.julday,stime.hour)
            trgw.triggerwrite(trig,fname,fpath=folder,tpad1=-2.*SPexp,tpad2=7*SPexp,\
                              mute=3*SPexp,gradelist = gtl)
