"""
:module: adapted_akazawa_loop_driver.py
:purpose: Driver for phase picking analysis using an adaptation for the strong-motion phase detection from Akazawa (2004) developed by Stevens (2022, Chapter 4)
:author: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 9. APR 2020
    Header added and repository module imports updated 7. MAR 2023 for upload to GitHub

"""




from obspy.core import UTCDateTime
import os
from copy import deepcopy
import pyrocko.gui.marker as pm
import pandas as pd
import numpy as np

import sys
#sys.path.append('/home/nates/Scripts/Python/PKG_DEV')
# sys.path.append('/usr1/ntstevens/SCRIPTS/PYTHON/PKG_DEV')
sys.path.append(os.path.join('..','..'))
# import dbscripts.connect as conn
# from dbscripts.wf_query import db2st
# import obspytools.snuffler.markertools as mtools
# import obspytools.detectors.DEVELOP.augmented_akazawa as AAM
import util.pisces_DB.connect as conn
import util.snuffler.markertools as mtools
import util.detectors.augmented_akazawa as AAM
from util.pisces_DB.wf_query import db2st


####################################################################################################
############################## USER DEFINED PARAMETERS #############################################
####################################################################################################
projdir = '/t31/ntstevens/SGGS'
## Processing reporting level
verbosity = 0
# [0] - Run in silent mode
# [1] - Print reports from this driver script
# [2] - Print reports in in [1] and primary reports from main subroutines
# [3] - Print reports as in [2] and secondary reports in main subroutines
# [4] - 
# [>10] - Print everything
isplot = False

# File Paths


### Maximum Time Windows ###
TMIN = UTCDateTime(2019,7,31)
TMAX = UTCDateTime(2019,9,1)
minimum_window = 0.1        # [sec] minimum window length. shorter windows dilated to this length
tol = 1e-4                  # [sec] time tolerance when detecting windows

YEAR = '2019'
JDAY0 = 220
JDAY_RANGE = 2
gradelist = [2]
Zchans = ['GP3','GNZ','p0']
Hchans = ['GP1','GP2','GN1','GN2','p1','p2']
#HOUR = '05'
for JD_ in range(JDAY_RANGE):#'223','224','225','226','227','228','229','230']:
    JDAY = str(JDAY0 + JD_)
    print('######## STARTING TO PICK %s %s ########### '%(YEAR,JDAY))
    for HOUR in range(2):
        HOUR = HOUR + 22
        print('########## PICKING %s IN %s %s ############'%(HOUR,YEAR,JDAY))
        ### Station/Channel Subset ###
        # Example # stnsubset={'channel':'??Z','station'='R10[12345]'}
        stnsubset = None
        ### Marker Data Inputs
        #markerdir = '/home/nates/ActiveProjects/SGGS/ANALYSIS/EVENT_DETECTION/Visual/%s%s'%(YEAR,JDAY)
        markerdir = '%s/ANALYSIS/EVENT_DETECTION/Automated/dtkurt_v3_%s%s'%(projdir,YEAR,JDAY)
        #mfile = 'nts-%s-%s-%s-review'%(YEAR,JDAY,HOUR)
        mfile = 'dtkurt_trigs_v4_%s%s%02d'%(YEAR,JDAY,HOUR)
        mfpath = os.path.join(markerdir,mfile)

        ### Marker & Analysis Data Outputs
        marker_output_dir = '%s/ANALYSIS/PHASE_PICKING'%(projdir)
        PICKERDIR = 'ARPICK_full_array'
        JDDIR = YEAR+JDAY
        pref_markers_file = 'ar_pick_assoc_pref-%s-%s-%02d_G%d'%(YEAR,JDAY,HOUR,gradelist[0])
#        full_results_file = 'ar_pick_assoc_full-%s-%s-%02d_G%s.dat'%(YEAR,JDAY,HOUR,gradelist[0])
        saveevery = 100     # Save/overwrite marker file every "n" markers processed

        # Pisces/sqlite database containing wfdisc for project (DEV: & write target for phases)
        dbID = 'sqlite:////%s/DB/sasyDBpt.sqlite'%(projdir)
        
        ### AR PICKER SETTINGS
        # P-pick settings
        Pcrf = 'zdetect'
        Pcrfk = {'sta':1e-2}
        #crfkwargs = {'max_len':1e1,'max_len_delete':True}
        Pf1 = 'bandpass'
        Pf1k = {'freqmin':80.,'freqmax':450}
        Pf2 = 'bandpass'
        Pf2k = {'freqmin':30.,'freqmax':450.}


        # S-pick settings
        SPexp = 0.06        # Anticipated minimum S-P time
        SPpads = [0.5,3]
        Scrf = 'classic_lta-sta'
        Scrfk = {'sta':5e-3,'lta':1e-2}
        Sf1 = 'bandpass'
        Sf1k = {'freqmin':30.,'freqmax':200.}
        Sf2 = 'bandpass'
        Sf2k = {'freqmin':30.,'freqmax':450.}

        # Preferred AR pick settings (assess_2_ar_pick_estimates.py)
        if verbosity > 2:
            pverbo = True
        else:
            pverbo = False
        pickwargs={'snrmin':5.,'snrtol':1.,\
                   'dtident':2e-3,'dtmax':5e-2,\
                   'SPmm':[SPexp/2.,SPexp*2],\
                   'verbose':pverbo}

        # Capture input contols and write to *.LOG file
        indatainfo = {'windowed_markers':mfpath,'grades':gradelist,'wf_db':dbID,'stn_subset':stnsubset}
        outdatainfo = {'pref_file':os.path.join(marker_output_dir,PICKERDIR,JDDIR,pref_markers_file)}
        Ppickinfo = {'Pcrf':Pcrf,'Pcrfkwargs':Pcrfk,'filter1':Pf1,'filt1params':Pf1k,\
                     'filter2':Pf2,'filt2params':Pf2k}
        Spickinfo = {'Expected S-P':SPexp,'S-P Padding Coefficients':SPpads,'Scrf':Scrf,'Scrfkwargs':Scrfk,\
                     'filter1':Sf1,'filt1params':Sf1k,'filter2':Sf2,'filt2params':Sf2k}
                     
        paraminfo = {'TMIN':TMIN,'TMAX':TMAX,'min_wind':minimum_window,'wind_tol':tol,'saveevery':saveevery}
        try:
            os.mkdirs(os.path.join(marker_output_dir,PICKERDIR,JDDIR))
        except:
            print('Already exists: %s'%(os.path.join(marker_output_dir,PICKERDIR,JDDIR)))
        fobj2 = open(os.path.join(marker_output_dir,PICKERDIR,JDDIR,pref_markers_file+'.LOG'),'w+')
        fobj2.write('LOGFILE STARTED: %s\n'%(str(UTCDateTime())))
        fobj2.write('General timing parameters\n')
        fobj2.write(str(paraminfo))
        fobj2.write('Input Data Parameters\n')
        fobj2.write(str(indatainfo))
        fobj2.write('\nOutput Data Parameters\n')
        fobj2.write(str(outdatainfo))
        fobj2.write('\nP-picking Parameters\n')
        fobj2.write(str(Ppickinfo))
        fobj2.write('\nS-picking Parameters\n')
        fobj2.write(str(Spickinfo))
        fobj2.write('\nPick Comparison Parameters\n')
        fobj2.write(str(pickwargs))
        fobj2.write('\nBEGINNING OF EVENT TIME, P COUNT, S COUNT SECTION\n')
#        fobj2.close()
        ####################################################################################################
        ################################## END OF USER DEFINED PARAMETERS ##################################
        ####################################################################################################

        ### MAKE FILE DIRECTORY STRUCTURE IF NOT ALREADY PRESENT
        try:
            os.mkdirs(marker_output_dir)
        except:
            print('PHASES directory already exists... continuing')
        try:
            os.mkdirs(os.path.join(marker_output_dir,PICKERDIR))
        except:
            print('ARPICK directory already exists ... continuing')
        marker_output_sdir = os.path.join(marker_output_dir,PICKERDIR,JDDIR)
        try:
            os.mkdirs(marker_output_sdir)
        except:
            print('%s already exists ... continuing'%(marker_output_sdir))

#        FRF = os.path.join(marker_output_sdir,full_results_file)
        PMF = os.path.join(marker_output_sdir,pref_markers_file)
        #event_output_sdir = os.path.join(marker_output_sdir,'ASSOCsingle')
        #try:
        #    os.mkdirs(event_output_sdir)
        #except:
        #    print('%s already exists ... continuing'%(event_output_sdir))
        #shutil.copy(thisfile
        try:
            if verbosity > 0:
                print('Attempting to create %s'%(os.path.join(marker_output_dir,marker_output_sdir)))
            os.mkdir(os.path.join(marker_output_dir,marker_output_sdir))
        except FileExistsError:
            if verbosity > 0:
                print('Already exists: %s'%(marker_output_sdir))


        ### Connect to Database
        session, meta, tabdict, tablist, collist = conn.connect4query(dbstr=dbID)
        ### Load marker file
        rawmarkers = pm.load_markers(mfpath)
        ### Split marker file into event, phase, other, and reject lists
        (evm,phm,otm,rjm) = mtools.splitbytype(rawmarkers,gradelist=gradelist,SplitWinds=True,tol=tol)
        # Select only markers that DO NOT have a type assigned and DO represent a window of time
        if len(otm) > 1:
            otherwindows = otm[1]
        else:
            otherwindows = None
        outputlist = []
#        fobj = open(FRF,"w+")
        if otherwindows and len(otherwindows) > 0:
            # Iterate over window markers (mwind)
    #        outputlist = []
    #        fobj = open(FRF,"w+")
            itersave = 0
            #singleevent = []
            for W_, mwind in enumerate(otherwindows):
                if mwind.kind in gradelist:
                    #singlelist = []
#                    if mwind.kind == 4:
#                        SPswitch = None
#                    else:
#                        SPswitch = SPexp
                    if verbosity > 1:
                        print('Beginning processing of marker window, index: %d'%(W_))
                    tstart = UTCDateTime(mwind.tmin)
                    tend = UTCDateTime(mwind.tmax)
                    emark = deepcopy(mwind)
                    emark.convert_to_event_marker()
                    # Make event name for single preferred event phase outputs
                    evfname='event_prefpick_%s'%(UTCDateTime(emark.tmin))
                    #singlelist.append(emark)
                    ehash = emark.get_event_hash()
                    outputlist.append(emark)
                    # Do sanity check on minimum window length
                    if tend - tstart < minimum_window:
                        if verbosity > 1:
                            print('Window for ingested marker was too short (dt < %.5f sec)'%(tend-tstart))
                        tmean = tstart + (tend-tstart)/2.
                        tstart = tmean - minimum_window/2.
                        tend = tmean + minimum_window/2.
                
                # Final time sanity-check for hard-set values
                if tstart > TMIN and tend < TMAX:
                    if verbosity > 1:
                        print("Window passed last time-frame sanity check. Passing to AR picker")
            ######## LOAD SEISMIC DATA FOR WINDOW
                    print('Loading data for %s to %s (%d of %d)'%(str(tstart),str(tend),W_+1,len(otherwindows)))
                    Zst = db2st(session,tabdict,starttime=tstart,endtime=tend,channels=Zchans)
                    print('Vertical Trace Count in Stream: %d'%(len(Zst)))
                    Hst = db2st(session,tabdict,starttime=tstart,endtime=tend,channels=Hchans)
                    print('Horizontal Trace Count in Stream: %d'%(len(Hst)))
                    if stnsubset:
                        st = st.select(**stnsubset)
            ######## PRIMARY PHASE PICKING ROUTINE

                    phzdf,prefdf=AAM.ar_pick_window_py(Zst,Hst,Pf1,Pf1k,Pcrf,Pcrfk,\
                                                       Pfilt2=Pf2,Pf2kwargs=Pf2k,\
                                                       EVID=ehash,SPexp=SPexp,SPpads=SPpads,\
                                                       Sfilt=Sf1,Sfkwargs=Sf1k,\
                                                       Scrf=Scrf,Scrfkwargs=Scrfk,\
                                                       Sfilt2=Sf2,Sf2kwargs=Sf2k,\
                                                       prefpickwargs=pickwargs,isplot=isplot,
                                                       verbosity=verbosity)
                    # Get number of preferred phases
                    nphase = len(prefdf)
                   # TODO: Sloppy work-around, fix this later 
                    st = Zst.copy()
                    st += Hst
            #        breakpoint()
                    if nphase >= 1:
                        FA = prefdf.sort_values('time').values[0]
                        FA_st = st.copy().select(station=FA[1],channel=FA[3])[0]
            #            lat = FA_st.stats.sac['stla']
            #            lon = FA_st.stats.sac['stlo']
                        # Iterate through preferred solutions (1 P and 0-1 S and write to files)
                        Pcount = 0
                        Scount = 0
                        for ip_ in range(nphase):
                            ipz = prefdf.values[ip_,:]
                            if ipz[5] == 'P':
                                Pcount += 1
                            elif ipz[5] == 'S':
                                Scount += 1
                            if verbosity > 2:
                                print(ipz)
                            nslc = ((str(ipz[0]),str(ipz[1]),'10',str(ipz[3])),)
                            ihash = ipz[4]
                            iphase = ipz[5]
                            tmin = tmax = ipz[6].timestamp
                            auto = 'ar_picker'
                            imarker = pm.PhaseMarker(nslc,tmin,tmax,event_hash=ihash,\
                                                 event_time=tstart,phasename=iphase,\
                                                 automatic=auto)
                            outputlist.append(imarker)
                            #singlelist.append(imarker)
#                            fobj.write('%s, %s, %.5f, %.3f\n'%(nslc,str(ihash),tmin,ipz[7]))
                            itersave += 1
                        print('Processed event has P count: %d and S count: %d'%(Pcount,Scount))
                        fobj2.write('%s %d %d \n'%(str(tstart),Pcount,Scount))
            #            pm.save_markers(singlelist,os.path.join(event_output_sdir,evfname),fdigits=4)
                    if itersave >= saveevery:
                        pm.save_markers(outputlist,PMF,fdigits=4)
                        itersave = 0

            ### SAVE FILE OUTPUTS
            ### Save marker file
        print('######### SAVING %d RESULTS FOR %s %s HOUR %s ##########'%(len(outputlist),YEAR,JDAY,HOUR))
        print(gradelist)
        print('%d += range(%d)'%(JDAY0,JDAY_RANGE))
        pm.save_markers(outputlist,PMF,fdigits=4)
        fobj2.close()
#        fobj.close()


