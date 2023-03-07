"""
:module: markers2NLLPhase.py
:purpose: Methods for converting pyrocko.Snuffler marker files with associating events into NonLinLoc (Lomax et al., 2000) *.obs files
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last revision: 13. DEC 2019
    Headers updated 7. NOV 2023 for upload to GitHub

"""


import pandas as pd
from pyrocko.gui import marker as pm
from obspy.core import UTCDateTime
import os

NLLocPhz=['Station','Instrument','Component','PhaseOnset','iPhase','FMotion','Date','Time','Seconds','ErrorType','ErrorMag','CodaDuration','Amplitude','Period','PriorWt']

onset = 'i'
fm = '?'
defnum = '-1.00e+00'

def parsemarkertimes(marker,precision=1e-5):
    t1 = UTCDateTime(marker.tmin)
    if marker.tmin == marker.tmax:
        tsig = precision
        tmu = t1
    elif marker.tmin+precision < marker.tmax:
        t2 = UTCDateTime(marker.tmax)
        tsig = abs(t2 - t1)/2.
        tmu = t1 + tsig
    return tmu, tsig
       

def marker2dataframes(markerfile):
    NLLocPhz=['EventHash','Station','Instrument','Component','PhaseOnset','iPhase',\
              'FMotion','Year','Month','Day','Hour','Minute','Seconds','ErrorType',\
              'ErrorMag','CodaDuration','Amplitude','Period','PriorWt']

    onset = 'i'
    fm = '?'
    defnum = -1.0

    markers = pm.load_markers(markerfile)
    hashlist = []
    phaselist = []
    for marker in markers:
        if isinstance(marker,pm.PhaseMarker) and marker.get_event_hash() is not None:
            tmu,tsig = parsemarkertimes(marker)
            il = [marker.get_event_hash(),marker.nslc_ids[0][1],'?',marker.nslc_ids[0][3],\
                  onset,marker.get_label(),fm,tmu.year,tmu.month,tmu.day,tmu.hour,tmu.minute,\
                  tmu.second+tmu.microsecond*1e-6, 'GAU', tsig,defnum,defnum,defnum,1]
            phaselist.append(il)
        elif isinstance(marker,pm.EventMarker) and marker.get_event_hash() is not None:
            hashlist.append(marker.get_event_hash())

    phzdf = pd.DataFrame(phaselist,columns=NLLocPhz)
    return phzdf, hashlist

def dfhashmatch(dataframe,hashkey):
    return dataframe[dataframe['EventHash'].isin([hashkey])]

def writeNLLocPhase(filename,dataframe,hashlist,filepath='./'):
    """
    Write phase entries of a maker dataframe that have matching hashkeys to eventmarker-sourced hashkeys (i.e. associated phases)
    to the NonLinLoc phase file format.
    NLLoc Format                    phaselist#  Null Value
    0)  %-6s Station Name           [0].[1]     REQUIRED
    1)  %-4s Instrument                         ?
    2)  %-4s Component              [0].[3]     ?
    3)  %1s  P phase onset                      ?
    4)  %-6s Phase Descriptor       [3]         REQUIRED
    5)  %1s  First Motion                       ?
    6)  %04d Year Component         [1].year    REQUIRED
    7)  %02d Month Component        [1].month   REQUIRED
    8)  %02d Day Component          [1].day     REQUIRED
    9)  %02d Hour Component         [1].hour    REQUIRED
    10) %02d Minute Component       [1].minute  REQUIRED
    11) %7.4f Seconds               [1].seconds REQUIRED
    12) %3s  Error Type                         REQUIRED (GAU Default)
    13) %9.2e Error magnitude       [2]         REQUIRED (< dominant period)
    14) %9.2e Coda duration                     -1.00
    15) %9.2e Amplitude                         -1.00
    16) %9.2e Period of amplitude               -1.00
    17) %1d   Prior Weight          [4]         REQUIRED (0 or 1 default)
    """

    #           0     1     2     3    4    5    6   7   8   
    fstring1 = '%-6s\t%-4s\t%-4s\t%1s\t%6s\t%1s\t%04d%02d%02d\t'
    #           9   10    11     12   13     14     15     16     17
    fstring2 = '%02d%02d\t%7.4f\t%3s\t%9.2e\t%9.2e\t%9.2e\t%9.2e\t%1d\n'
    fstring = fstring1+fstring2

    fobj = open(os.path.join(filepath,filename),'w')
    for ihash in hashlist:
        dfi = dfhashmatch(dataframe,ihash)
        nlines = len(dfi)
        mat = dfi.values[:,1:]
        for i_ in range(nlines):
            l_ = mat[i_]
            dl = fstring%tuple(l_)
            fobj.write(dl)
        fobj.write('\n')
    fobj.close()

