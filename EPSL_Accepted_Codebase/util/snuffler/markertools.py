"""
:module: markertools.py
:purpose: Methods for working with pyrocko.snuffler marker file data
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 24. NOV 2019
    Header added 7. MAR 2023 for upload to GitHub
"""


#from pyrocko import utils
from pyrocko.gui import marker as pm

# Initial read routine wrappers (mostly for references)
def _read(markerfile):
    markerlist = pm.load_markers(markerfile)
    return markerlist

def _getevents(markerlist):
    eventlist = []
    for marker in markerlist:
        if isinstance(marker, pm.EventMarker):
            eventlist.append(marker)

    return eventlist


def _getphases(markerlist):
    phaselist = []
    for marker in markerlist:
        if isinstance(marker, pm.PhaseMarker):
            phaselist.append(marker)
    return phaselist


def _getothers(markerlist):
    otherlist = []
    for marker in markerlist:
        if not isinstance(marker, (pm.EventMarker, pm.PhaseMarker)):
            otherlist.append(marker)
    return otherlist


def getmarkertype(marker,tol=1e-6):
    """ 
    Actually useful here, I think 
    Get numeric 'grade' label for marker
    and return the marker type as a string
    "Event"
    "PhaseInstant"
    "PhaseWindow"
    "OtherInstant"
    "OtherWindow"
    """
    # If not a windowed time
    if abs(marker.get_tmax() - marker.get_tmin()) <=tol:
        # If event or phase
        if isinstance(marker,pm.EventMarker):
            mtype = 'Event'
        elif isinstance(marker,pm.PhaseMarker):
            mtype = 'PhaseInstant'
        # If uncategorized, non-window
        else:
            mtype = 'OtherInstant'
    # If it is a windowed time
    elif abs(marker.get_tmax() - marker.get_tmin()) > tol:
        if isinstance(marker,pm.PhaseMarker):
            mtype = 'PhaseWindow'
        else:
            mtype = 'OtherWindow'
    else:
        print('Something went wrong')

    grade = marker.kind
    return grade, mtype

def newassocmarker(nslctuple,phzname,tmin,tmax,eventmarker,autopick,**kwargs):
    hashkey = eventmarker.get_event_hash()
    evtime = eventmarker.tmin()
    phznew = pm.PhaseMarker(nslctuple,tmin,tmax,event_hash=hashkey,event_time=evtime,\
                            phasename=phzname,automatic=autopick,**kwargs)
    return phznew

def splitbytype(markerlist,gradelist=[0,1,2,3,4,5],SplitWinds=False,tol=1e-6):
    """
    Separate out events, phases, and unflagged markers
    give the option to filter by grade and whether the marker is an
    instant of time or window of time

    Dependency on subroutine: getgradeandtype - in this module
    """
    eventlist = []
    phaselist = []
    otherlist = []
    rejectlist = []
    if SplitWinds:
        phinstlist = []
        phwindlist = []
        otinstlist = []
        otwindlist = []

    for m_, marker in enumerate(markerlist):
#        print(m_)
        grade,mtype = getmarkertype(marker,tol=tol)
        if 'Phase' in mtype and grade in gradelist:
            if not SplitWinds:
                phaselist.append(marker)
            elif SplitWinds and 'Window' in mtype:
                phwindlist.append(marker)
            elif SplitWinds and 'Instant' in mtype:
                phinstlist.append(marker)
            else:
                print('Something went wrong parsing:')
                print(marker)
                print('as a Phase. Index %d'%d(m_))
        elif 'Event' in mtype and grade in gradelist:
            try:
                eventlist.append(marker)
            except:
                print('Something went wrong appending:')
                print(marker)
                print('as an Event. Index %d'%(m_))
        elif 'Other' in mtype and grade in gradelist:
            if not SplitWinds:
                otherlist.append(marker)
            elif SplitWinds and 'Window' in mtype:
                otwindlist.append(marker)
            elif SplitWinds and 'Instant' in mtype:
                otinstlist.append(marker)
            else:
                print('Something went wrong parsing:')
                print(marker)
                print('as an event. Index %d'% (m_))

        else:
            rejectlist.append(marker)

        if SplitWinds:
            phaselist = [phinstlist, phwindlist]
            otherlist = [otinstlist, otwindlist]

    return eventlist, phaselist, otherlist, rejectlist


