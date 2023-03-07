

### Import Environmental Modules ### - seispy Environment
import sys
import os
import numpy as np
import pandas as pd
from obspy.core import read, Stream, UTCDateTime

from joblib import Parallel, delayed

### Import Home-Brewed Modules ###
sys.path.append('/usr1/ntstevens/SCRIPTS/PYTHON/SGGS_Workflows')
import detectors.kurtosis as okt
import detectors.augmented_coincidence_trigger as act
import detectors.f_dist_fit as fdf
import detectors.ar_aic_picker as arp
import dbtools.connect as conn
import translate.phase_translator as pt
sys.path.append('/usr1/ntstevens/SCRIPTS/PYTHON/SGGS_Workflows/rect_reprocess')
from rect_reprocess.util.data import db2st
# from dbtools.wf_query import db2st


##################################################################################################################
########## END OF MODULE IMPORT SECTION ##########################################################################
##################################################################################################################
########## START OF USER CONTROL BLOCK ###########################################################################
##################################################################################################################

# Data Load Parameters - Needs to be an explicit list of channel names for db2st routine
Zchans = ['GNZ','GP3'] #,'p0']
Hchans = ['GN1','GP1','GN2','GP2'] #,'p2']

#### Pisces / CSS3.0 Formatted SQLite Database with 'wfdisc' table that maps input waveform data
wfdb = 'sqlite:////t31/ntstevens/SGGS/DB/sasyDBpt.sqlite'
# savedb = 'sqlite:////t31/ntstevens/SGGS/DB/sggs_wk_araic_db_f95.sqlite'
savedb= 'sqlite:////t31/ntstevens/SGGS/DB/sggs_wk_araic_db_f95.sqlite'
## Writing / Saving Controls ##
qls = 0.5   # Quality level of picks to allow into savedb - recommend either 0.5 or 1.0
            # NOTE: Only qual = 1 entries will be included
verb = 2    # Level of print-output reporting during processing
write_mode = 'a+'   # Writing mode for F-distribution threshold outputs --- DO NOT CHANGE
auth = 'ntstevens'  # Author to write to savedb

ZCRF_SAVE = False   # THIS WILL CREATE A VERY LARGE DATASET --- KEEP AS False -- breakpoint failsafe also embedded below
IQR_PROCESS = False  # If True, while already processing the whole dataset, calculate & save IQR time-series of vertical traces
IQR_S_FORMAT = 'SAC'
iqr2db = False

# OUTPUT DIRECTORIES
logfile = '/t31/ntstevens/SGGS/PROCESSING/stdout_%s'%(str(okt.UTCDateTime()))
phase_save_file_path = '/t31/ntstevens/SGGS/PROCESSING/PHASES_f95'
iqr_save_file_path = '/t31/ntstevens/SGGS/PROCESSING/IQR'
crf_save_file_path = '/t31/ntstevens/SGGS/PROCESSING/ZCRF'
thresh_save_file_path = '/t31/ntstevens/SGGS/PROCESSING/F95_LEVELS'


#### Parallelization Controls ####
n_cores = 10            # [#] Number of CPU's to use in joblib parallelization
trigsinbatch = n_cores*2.5 
#### Overall Run Data Timeframe ####
run_start = UTCDateTime(2019,8,2,5,30) # [DateTime] Starting time 
run_length = 60.*60.*(24.*19.)    # [sec] Length of processing run

# Individual Processing Windows (IPW's)
twin_load = 15.*60.       # [sec] Length of continuous data load segments
                  # NOTE: This parameter dictates the sampling window for trigger threshold calculation
tover_load = 2.          # [sec] Length of overlapping data between load segments


#### Physical Parameters Section & Prior Knowledge Insertion #####
# VpVs = 1.95
VpVs = 1.932         # [-] Anticipated Vp/Vs value for velocity structure
# Vp = 3700
Vp = 3451.          # [m/s] Anticipated Vp value for velocity structure
d_exp = 300.        # [m] Anticipated mean source-receiver distance
# d_exp = 200.        # [m] Anticipated mean source-receiver distance
## Calculate physically based time estimates to guide AR-AIC picking ##
# [sec] Anticipated mean S-P differential arrival time for stations
SP_exp = d_exp*(1./Vp)*(VpVs - 1);

# Use scalars of tS-tP expected value to add time padding around P-detection triggers 
front_pad = SP_exp      # Lead trigger times to include some "noise" segment
Pmute = SP_exp/3.       # Time following P-pick to mute trace (P-coda) - start of S-picking window
back_pad = SP_exp*3.    # Tail trigger times to include likely S-arrivals & coda
snr_wind = SP_exp/2.    # Window for SNR calculation
tps = [front_pad, back_pad] # Trigger Padding [Seconds]


#### Processing Key-Word-Arguments for the IQR Noise Proxy Detection
iqrKw = {'win':1.0,'step':0.5,'iqr_rng':[10.,90.],'verb':verb}
# See scipy.stats.iqr for further information

#### Preprocessing Filter for Vertical Channels ####
# operates on an obspy.core.Stream object, so formatting is different from calling the filter functions directly.
# See obspy.core.Stream documentation
ppfilt = 'bandpass'
ppfKw = {'freqmin':80.,'freqmax':430.}


#### Vertical Trace CRF Parameters (Initial Detection) ####
zcrfn = okt.tr_wind_kurt_py # [method] method to use for 
# input keyword arguments
zcrfKw = {'win':0.1,'step':0.01,'verb':verb}

crf_code = 'WK'
#t_win = 0.05              # [sec] Length of window for kurtosis calculation (get at least 100 samples)
#t_step = 0.01              # [sec] Length of step for kurtosis processing window


#### Augmented Network Coincidence Trigger Parameters ####
# f_bnd_on = 0.99
f_bnd_on = 0.95                   # [%] Confidence bound for f-distribution fit Trigger ON
# f_bnd_off = 0.99
f_bnd_off = 0.95                  # [%] Confidence bound for f-distribution fit Trigger OFF
mlvl = 3                          # [pow] Order of magnitude of samples to use for F-distribution estimation
                                  #    NOTE: This is a point where compute time can be saved with
                                  #          smart selection of a diminished sample population.
# min_sta_count = 6                                  
min_sta_count = 4                  # [count] Minimum number of triggers
sim_thresh = 0.8                  # [fract] Correlation coefficient threshold (similarity threshold)
max_trig_len = 3.*SP_exp          # [sec] maximum trigger length before trigger is discarded from consideration


#### Adapted Akazawa Phase Picking Trigger Parameters ####
# P-Pick Parameters
# Use STK/LTK peak for the t_2 selection

# Trace Pre-Processing for vertical channels
Pcrfn = okt.tr_wind_kurt_py
PcKw = {'win':0.05,'step':0.01,'verb':0}
#Pcrfn = classic_sta_lta
Pf1 = 'bandpass'
PfKw1 = {'freqmin':80.,'freqmax':430} # This first freqmax tends to remove HF alias artifacts from FIR filter
Pf2 = 'highpass'
PfKw2 = {'freq':30.} # Keep a highpass in the second assessment to keep out long-period oscillations

# S-Pick Parameters
Scrfn = arp.classic_lta_sta_diff_py
ScKw = {'tsta':0.01,'tlta':0.03}
Sf1 = 'bandpass'
SfKw1 = {'freqmin':80.,'freqmax':430.}
Sf2 = 'highpass'
SfKw2 = {'freq':30.}

#
phaseprefKw = {'snrmin':3.,'snrtol':1.,'dttol':0.001,'dtmax':SP_exp/2.,'verb':verb}




# detection_label = 'wk_trigger_araic_markers_'
# trig_marker_dir = os.path.join(marker_save_file_path,'wkurt_trigger_markers')
# araic_marker_dir = os.path.join(marker_save_file_path,'araic_pick_markers')
# fdigits = 4

##################################################################################################################
########## END OF USER CONTROL BLOCK #############################################################################
##################################################################################################################
########## START OF PROJECT DIRECTORY & USER INPUT PRE-PROCESSING SECTION ########################################
##################################################################################################################


######### MAKE PROJECT DIRECTORIES EXIST #######
## SAVE PATH FOR RAW PHASE PICKING OUTPUT
try:
    os.makedirs(phase_save_file_path)
    print('New file save location generated: %s'%(phase_save_file_path))
except FileExistsError:
    print('%s Save Subdirectory Already Exists, Continuing'%(phase_save_file_path))    

## SAVE PATH FOR IQR TRACE OUTPUTS
try:
    os.makedirs(iqr_save_file_path)
    print('New file save location generated: %s'%(iqr_save_file_path))
except FileExistsError:
    print('%s Save Subdirectory Already Exists, Continuing'%(iqr_save_file_path))

## SAVE PATH FOR CRF TRACE OUTPUTS
## TODO - This should be implemented using the hdf5 (ph5, most likely sub-class) data format
# to save on space -- realizes ~ 3x better compression compared to SAC.


## SAVE PATH FOR F-DISTRIBUTION THESHOLD DATA
try:
    os.makedirs(thresh_save_file_path)
    print('New file save location generated: %s'%(thresh_save_file_path))
except FileExistsError:
    print('%s Save Subdirectory Already Exists, Continuing'%(thresh_save_file_path))   

# sys.stdout = open(logfile,'w')

######### CONNECT TO DATABASE ##########
print('Connecting to Database(s)')
session,meta,tabdict,tablist,collist = conn.connect4query(dbstr = wfdb)
# if wfdb != savedb:
save_session,save_meta,save_td,save_tl,save_cl = conn.connect4query(dbstr = savedb)
# else:
#     save_session = session
#     save_td = tabdict

print ('Connected to Database(s)')

WFD = tabdict['wfdisc']
levid = pt._get_lastid(save_session,save_td,'evid')
lorid = pt._get_lastid(save_session,save_td,'orid')
larid = pt._get_lastid(save_session,save_td,'arid')


print('Last EVID',levid)


######### SET UP LARGE TIME STEPPING PROCESSING LOOP #########
# iterated start time
it_start = run_start
it_end = run_start + twin_load + tover_load

thresh_csv = 'f95_trigger_levels_%4d%03d_%02d%02d_%4d%03d_%02d%02d.csv'%\
              (run_start.year,run_start.julday,run_start.hour,run_start.minute,\
               (run_start + run_length).year,(run_start + run_length).julday,\
               (run_start + run_length).hour,(run_start + run_length).minute)

thresh_csv_fp = os.path.join(thresh_save_file_path,thresh_csv)

# master time-check - loop start time
mTIC = UTCDateTime()


##################################################################################################################
########## END OF USER INPUT INTERPRETATION & PROCESS SETUP SECTION ##############################################
##################################################################################################################
########## START OF TIME STEPPING PROCESSING LOOP ################################################################
##################################################################################################################
while it_start < run_start + run_length:

    # Generate window ending time
    phase_csv = 'phase_%4d%03d_%02d%02d_%dmin.csv'%\
                (it_start.year,it_start.julday,it_start.hour,it_start.minute,int((it_end - it_start)/60.))
    phase_csv_fp = os.path.join(phase_save_file_path,phase_csv)
    if verb > 0:
        print('Phase files will be saved to %s'%(phase_csv_fp))


######### LOAD TRACE DATA ########
    # Time check on loading traces
    tltic = UTCDateTime()
    # for wf in wf_qz:
    if verb > 1:
        print('Loading  Data for %s to %s (%s)'%(str(it_start),str(it_end),str(UTCDateTime())))
    # Get data using wfdisc housed in pisces sqlite database
    istz = db2st(session,tabdict,starttime=it_start,endtime=it_end,channels=Zchans)
    isth = db2st(session,tabdict,starttime=it_start,endtime=it_end,channels=Hchans)
    if len(istz) >=min_sta_count:

    ######### PREPROCESS STREAM DATA ########
        # Create a copy of the data to do preprocessing on so the initial traces 
        # are already in memory for additional processing later? 
        #TODO: Set load routine in AR-AIC picker to pull raw trace segments from db - less memory intensive approach
        istz_pp = istz.copy()
        istz_pp.filter(ppfilt,**ppfKw)


    ######### CONDUCT PARALLELIZED PROCESSING TO MAKE CRFs #######
        print('===== Parallelized processing starting (%s) (n_cores: %d) ====='%(str(UTCDateTime()),n_cores))
        # Timecheck - start of parallelized inner loop
        pTIC = UTCDateTime()
        
        # PARALLELIZED PROCESSING FOR RAW VERTICAL CHANNEL NOISE LEVEL FROM IQR

        if IQR_PROCESS:
            iqr_list = Parallel(n_jobs = n_cores)(delayed(okt.tr_iqr_py)(trace=tr_i,**iqrKw) for tr_i in istz)
            for iqr_tr in iqr_list:
                sta_iqr_sfp = os.path.join(iqr_save_file_path,iqr_tr.stats.station)
                try: 
                    os.mkdir(sta_iqr_sfp)
                except FileExistsError:
                    if verb > 5:
                        print('IQR save directory already exists for station',iqr_tr.stats.station)
                stime = iqr_tr.stats.starttime
                savename = '%s.%s.%s.%s.IQR.%4d-%02d-%02dT%02d.%02d.%02d.%s'%(\
                            iqr_tr.stats.network,iqr_tr.stats.station,iqr_tr.stats.location,iqr_tr.stats.channel,\
                            stime.year,stime.month,stime.day,stime.hour,stime.minute,stime.second,\
                            IQR_S_FORMAT.lower())
                if not os.path.isfile(os.path.join(sta_iqr_sfp,savename)):
                    iqr_tr.write(os.path.join(sta_iqr_sfp,savename),format=IQR_S_FORMAT)
                else:
                    if verb > 5:
                        print('IQR file already exists, did not save')

                # if iqr2db:
                #     pt.add_wfdisc_entry(save_session,save_td,os.path.join(sta_iqr_sfp,savename),iqr_tr)
            pTOC = UTCDateTime()
            # Report to terminal
            print('Parallelized IQR Processing Complete! (ET: %.2f min) (%s)'%((pTOC-mTIC)/60.,str(pTOC)))

        # PARALLELIZED GENERATION OF THE SPECIFIED Z-CHANNEL CHARACTERISTIC FUNCTION FOR EVENT DETECTION
        crf_list = Parallel(n_jobs = n_cores)(delayed(zcrfn)(trace=tr_i,**zcrfKw) for tr_i in istz_pp)

                            #stk=t_stk,ltk=t_ltk,step=t_step,verb=verb) for tr_i in istz_pp)
        # Convert parallelized processing output list into an obspy tream
        crf_st = Stream()
        for crf in crf_list:
            crf_st += crf

        if ZCRF_SAVE:
            breakpoint()
            for crf_tr in crf_st:
                sta_iqr_sfp = os.path.join(crf_save_file_path,crf_tr.stats.station)
                try: 
                    os.mkdir(sta_crf_sfp)
                except FileExistsError:
                    if verb > 5:
                        print('IQR save directory already exists for station',crf_tr.stats.station)   
                stime = crf_tr.stats.starttime
                savename = '%s.%s.%s.%s.%s.%4d-%02d-%02dT%02d.%02d.%02d.%s'%(\
                            crf_tr.stats.network,crf_tr.stats.station,crf_tr.stats.location,crf_tr.stats.channel,\
                            crf_code,stime.year,stime.month,stime.day,stime.hour,stime.minute,stime.second,\
                            IQR_S_FORMAT.lower())
                crf_tr.write(os.path.join(sta_crf_sfp,savename),format=CRF_S_FORMAT)

        # Timecheck
        #breakpoint()
        pTOC = UTCDateTime()
        # Report to terminal
        print('Parallelized CRF Processing Complete! (ET: %.2f min) (%s)'%((pTOC-mTIC)/60.,str(pTOC)))


    #### END OF PARALLELIZED DETECTION ####


    ######### CONDUCT CONICIDENCE TRIGGER #######
    # This form of the code (art.ctrigger) uses an adapted (stupid) version of the F-test noise-adaptive trigger
    # threshold algorithm of Carmichael, 2013 and Arrowsmith et al., 2009. 
    # Assumes CRF data are normally distributed and triggers on values beyond the sig_on bound 
        print('Beginning Coincdence Trigger on CRFs (%s)'%(str(UTCDateTime())))
    #   trig = act.ctrigger(crf_st,sig_on,sig_off,min_sta_count,maximum_trigger_length=max_trig_len,similarity_threshold=sim_thresh)
        # triggers,thresholds = act.cftrigger(crf_st,f_bnd_on,f_bnd_off,min_sta_count,\
        #                                     mlvl = mlvl,maximum_trigger_length=max_trig_len,\
        #                                     similarity_threshold=sim_thresh)

        triggers, thresholds = act.cftrigger_par(crf_st,f_bnd_on,f_bnd_off,min_sta_count,\
                                                mlvl = mlvl, n_cores=n_cores,maximum_triger_length=max_trig_len,\
                                                similarity_threshold=sim_thresh)

        pTOC = UTCDateTime()
        print('Parallelized Coincidence Triggering Complete! (ET: %.2f min) (%s)'%((pTOC-mTIC)/60.,str(pTOC)))
        print('Triggers Selected: %d'%(len(triggers)))

        # Write thresholds to CSV file for run
        if not os.path.isfile(thresh_csv_fp):
            thresholds.to_csv(thresh_csv_fp,header='column_names',index=False)
        else:
            thresholds.to_csv(thresh_csv_fp,mode=write_mode,index=False)


    # TODO: Write thresholds to a CRF processing log file

    # TODO: This is about where QuakeMigrate routines could be spliced in using guidance on 
    # analysis windows from the triggers. Would need to 
            

    ######### CONDUCT ADAPTED AKAZAWA PICKER ########
        # assoc_event2window
        # use triggers for initial picking leads
        # TODO: Eventually use these as seeds for kpick?
        # TODO: Update argument list to match with ar.ar_aic_picker_db in current form
        # TODO: Update Parallel key-word arguments as **araic_par_kwargs
        if len(triggers) == 0:
            print('No triggers to process --- Continuing with next time iteration')
        else:
            print('Beginning parallel processing for phase picking for triggers (%s)'%(str(UTCDateTime())))
            
            # Compose a list of dictionaries that can be passed into the parallelized AR-AIC picker
            iTIC = UTCDateTime()
            print('Assembling picking parallelization input dictionaries (%s)'%(str(iTIC)))
            nbatches = int(np.ceil(len(triggers)/trigsinbatch))
            for b_ in range(nbatches):
                print('=== Processing batch (%d/%d) ==='%(b_+1,nbatches))
                b0i = int(b_*trigsinbatch)
                b1i = int((b_ + 1)*trigsinbatch)
                tbatch = triggers[b0i:b1i]

                inpar_dicts = []
                for i_,trigger in enumerate(tbatch):
                    trig_dict = arp.assemble_inpar_inputs(trigger,tps,istz,isth)
                    inpar_dicts.append(trig_dict)
                iTOC = UTCDateTime()
                print('===== Input dictionaries assembled. Took %.2f min ====='%((iTOC - iTIC)/60.))

                par_outputs = Parallel(n_jobs=n_cores)(delayed(arp.ar_aic_picker_inpar)(\
                                                inpar_dict,tps,Pmute,\
                                                Pf1,PfKw1,Pcrfn,PcKw,Pf2,PfKw2,\
                                                Sf1,SfKw1,Scrfn,ScKw,Sf2,SfKw2,\
                                                phaseprefKw,snr_wind,verb=verb)
                                                for inpar_dict in inpar_dicts)

                for pick_df in par_outputs:
                    try:
                        n_pref = len(pick_df[pick_df['qual'] == 1])
                    except:
                        n_pref = 0
                    if n_pref >= min_sta_count:
                        pt.df2db(pick_df,save_session,save_td,quality_level=qls,auth=auth)
                        with open(phase_csv_fp,write_mode) as f:
                            pick_df.to_csv(f,header=False,index=False)
                    else:
                        print('Insufficient number of phases passed quality criteria. Event submission to DB skipped')

                iiTOC = UTCDateTime()        
                print('===== Batch (%d/%d) Parallel Processing Concluded (%d Entries). Took %.2f min (ET: %.2f min) ====='%\
                        (b_+1,nbatches,len(par_outputs),(iiTOC - tltic)/60.,(iiTOC - mTIC)/60.))


            #     print('...processing trigger starting at ',trigger['time'],'(%d/%d) --- (%s)'%(i_+1,len(triggers),str(UTCDateTime())))
            #     pick_df = arp.ar_aic_picker_par_db(session,tabdict,trigger,n_cores,Zchans,Hchans,tps,Pmute,\
            #                                         Pf1,PfKw1,Pcrfn,PcKw,Pf2,PfKw2,\
            #                                         Sf1,SfKw1,Scrfn,ScKw,Sf2,SfKw2,\
            #                                         phaseprefKw,snr_wind,verb=verb)



            # # for pick_df in pick_df_list:
            #     n_pref = len(pick_df[pick_df['qual'] == 1])
            #     if n_pref >= min_sta_count:
            #         pt.df2db(pick_df,save_session,save_td,quality_level=qls,auth=auth)
            #         # Write each pick data_frame to CSV with a blank line between each entry
            #         # https://stackoverflow.com/questions/56937679/pandas-dataframe-to-csv-writes-blank-line-after-each-line
            #         with open(phase_csv_fp,write_mode) as f:
            #             pick_df.to_csv(f, header=False, index=False)
            #     else:
            #         print('Insufficient number of phases passed quality criteria. Event submission to DB skipped')

            pTOC = UTCDateTime()
            print('Parallelized Phase Picking Complete! (ET: %.2f min) (%s)'%((pTOC-mTIC)/60.,str(pTOC)))
            # Close file object.
            f.close()

        ## SAVE POINT ## File Containing Picker Products ## SAVE POINT ##
            # TODO: Save outputs from ar_aic_picker to csv, pyrock phase-event & NLLoc phase file formats 
        ## SAVE LOGS ##
    else:
        print('Insufficient stations present in this time window to meet "min_sta_count"',len(istz),'<',min_sta_count)
        print('Advancing to next time window')

        # TODO: Build up log output segment of code to save the "stdout" from all the imbedded "print" functions.
    ######### ADVANCE TIME VALUES ########
    it_start += twin_load
    it_end += twin_load
#--->## END OF TIME ITERATION LOOP ##<---#
##################################################################################################################
########## END OF TIME WINDOW ITERATION LOOP ####################################################################
##################################################################################################################
print('!!!!! ===== RUN COMPLETE (Elapsed Time: %.2f min) ===== !!!!!'%((UTCDateTime() - mTIC)/60.))
session.close()
save_session.close()

##################################################################################################################
##################################################################################################################
########## END OF SCRIPT #########################################################################################
##################################################################################################################
##################################################################################################################
