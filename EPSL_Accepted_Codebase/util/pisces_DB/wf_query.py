"""
:module: wf_query.py
:purpose: Convenience/documentation for queries using an established SQLAlchemy session with specific formatting from PISCES and custom table reference
        dictionary (tabdict) generated with the connection routines in util.connect.
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 6. DEC 2019
    Header updated 7. MAR 2023 for upload to GitHub
"""

from obspy.core import UTCDateTime, read, Stream, Trace
import os


def wf_timefilter(session,tabdict,starttime=None,endtime=None):
    """
    Pass UTCDateTime arguments for start and end times for 
    :: INPUTS ::
    :type session: sqlalchemy.orm.session.Session
    :param session: session connection to sqlite database
    :type tabdict: Dictionary
    :param tabdict: Dictionary containing entries of table definitions pulled from the sqlite session
    :type starttime: obspy.core.UTCDateTime
    :param starttime: starttime to query waveforms from (Optional)
    :type endtime: obspy.core.UTCDateTime
    :param endtime: endtime to query waveforms from (Optional)

    :: OUTPUT ::
    :rtype wf_q: sqlalchemy.orm.query.Query
    :return wf_q: Query object containing connections to 

    """
    # Make initial session instance
    wf_q = session.query(tabdict['wfdisc'])
    if starttime and not endtime:
        # If starttime is specified, query all records that end after the starttime
        wf_q = wf_q.filter(tabdict['wfdisc'].endtime > starttime.timestamp)
     
    elif endtime and not starttime:
        # If endtime is specified, query all records that start before the endtime
        wf_q = wf_q.filter(tabdict['wfdisc'].time < endtime.timestamp)
    
    elif starttime and endtime:
        wf_q = wf_q.filter(tabdict['wfdisc'].time < endtime.timestamp).filter(tabdict['wfdisc'].endtime > starttime.timestamp)
    
    # If neither are specified, return the full wfdisc query without filtering

    return wf_q



def db2st(session,tabdict,starttime=None,endtime=None,channels=None,verb=0,tol=10):
    """
    Load waveforms within a specified timeframe as documented in a pisces seismic database

    :: INPUTS ::
    :type session: sqlalchemy.orm.session.Session
    :param session: session connection to sqlite database
    :type tabdict: Dictionary
    :param tabdict: Dict containing entries of table defs pulled from the sqlite session
    :type starttime: obspy.core.UTCDateTime
    :param starttime: starting time for window pull
    :type endtime: obspy.core.UTCDateTime
    :param endtime: ending time for window pull
    :type channels: OPTIONAL None or list
    :param channels: list of strings exactly describing desired channel names (case sensitive)
    :type verb: int
    :param verb: level of verbosity

    :: OUTPUT ::
    :rtype st: obspy.core.Stream
    :return st: Stream containing trimmed waveforms.
    """
    wf_q = wf_timefilter(session,tabdict,starttime=starttime,endtime=endtime)
    # Update, now include option to filter by channels
    if isinstance(channels,list):
        wf_q = wf_q.filter(tabdict['wfdisc'].chan.in_(channels))

    st = Stream()
    st_tmp = Stream()
    for wf in wf_q:
        fpath = os.path.join(wf.dir,wf.dfile)
        if verb > 1:
            print(fpath)
        tr = read(fpath,starttime=starttime,endtime=endtime)[0]
        if tr.stats.sampling_rate != round(tr.stats.sampling_rate):
            tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        st_tmp += tr
        if verb > 0:
            print('LOADED: '+fpath)
        # Merge traces if crossing over from one file to another
    st_tmp.merge(method=1,fill_value='interpolate')
    for tr in st_tmp:
        if tr.stats.npts == 0:
            print('Skipping trace: %s. Zero data'%(str(tr)))
        elif tr.stats.npts < (endtime - starttime)*tr.stats.sampling_rate - tol:
            exp_npts = (endtime - starttime)*tr.stats.sampling_rate
            print('Skipping trace: %s. --> %d required (tolerance: %d)'\
                   %(str(tr),exp_npts,tol))
        else:
            st += tr
    return st




