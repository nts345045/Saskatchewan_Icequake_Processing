"""
:module: augmented_akazawa.py
:purpose: Methods for using an adapted version of Akazawa (2004)'s strong-motion detection algorithm used in Stevens (2022, Chapter 4)
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major update: 9. DEC 2019
    Minor updates for version_1 module structure on 7. MAR 2023
    Header added 7.MAR 2023 for GitHub upload
"""


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import warn
# sys.path.append('/home/nates/Scripts/Python/PKG_DEV')
# from obspytools.detectors.STABLE.diff_stalta import classic_LTA_STA_diff_py
sys.path.append('..','..')
from util.detectors.diff_stalta import classic_LTA_STA_diff_py


def snrdB(data,index,epsilon=1e-10):
    """
    Calculate signal-to-noise ratio of an input data vector the
    index of the signal onset in the data

    Assumes time is rightward increasing and signal onsets after
    some segment of noise
    :param data: numpy.ndarray ((n,)) containing data

    """
    index = int(index)
    noiserms = np.sqrt(np.mean(data[:index-1]**2))
    signalrms = np.sqrt(np.mean(data[index:]**2))
    snr = 10.*np.log10((signalrms/(noiserms+epsilon))**2) # small epsilon term included to prevent 0-division

    return snr


def wfaic(data,eta=np.e,normalize=False):
    """
    Calculate the AIC of a data array directly from time-series data using the
    method from Maeda 1985:

    AIC[k] = k*log(var(x[0:k])) + (N - k - 1)*log(var(x[k+1:N]))

    :: INPUTS ::
    :type data: (N,) numpy.ndarray
    :param data: vector of equally sampled time series data
    :type eta: float
    :param eta: natural root value to prevent 0-valued arguments for log

    :: OUTPUT ::
    :rtype aic: numpy.ndarray
    :return aic: Vector of AIC values 
    :rtype aic_stats: numpy.ndarray
    :return aic_stats: Array containing statistics on AIC curve:
                        IND PARAMETER
                        0   minimum finite AIC value
                        1   index of minimum finite AIC value
                        2   kurtosis of AIC - higher values are more peaked
    """

    N = len(data)
    aic = np.zeros(data.shape)
    for k_ in range(N):
        if k_ > 0 and k_ < N:
            k_1 = k_ + 1
            aic[k_] = (float(k_)*np.log(np.var(data[:k_]+eta)) +
                       float(N - k_ - 1.)*np.log(np.var(data[k_ + 1:] + eta)))
        elif k_ == 0:
            aic[k_] = float((N - 1.))*np.log(np.var(data[1:]+eta)) + 1.
        elif k_ == N:
            aic[k_] = float(k_)*np.log(np.var(data+eta))

        # Replace infinite values with NaN
        aic[~np.isfinite(aic)] = np.nan
        if not normalize:
            aic_out = aic
        elif normalize:
            Amin = np.nanmin(aic)
            Amax = np.nanmax(aic)
            Arng = Amax - Amin
            aic_out = np.divide(aic - Amin,Arng)

        minval = np.nanmin(aic)
        try:
            minindex = np.where(aic == minval)[0][0]
        except:
            minindex = 0
        snr = snrdB(data,minindex)

    return minindex, snr, aic_out


def timeininterval(time,ti,tf,code=None):
    if ti <= time <= tf:
        return time
    elif ti > time:
        print('%s: input time before initial time, adjusted to initial time.'%(code))
        return ti
    elif time > tf:
        print('%s: input time after final time, adjusted to final time'%(code))
        return tf
    else:
        return None


def assess_2_ar_pick_estimates(times,snrs,snrmin=3.,snrtol=1.,dtident=0.002,dtmax=0.05,Ptime=None,SPmm=[0.03,0.12],verbose=True):
    atimes = []
    asnrs = []
    # Go over the time and snr pairs
    for i_ in range(2):
        # Accept picks only if SNR is above minimum SNR
        if snrs[i_] > snrmin:
            # If not an S-pick, just use SNR criteria
            if verbose:
                print('Accept 1')
            if Ptime is None:
                atimes.append(times[i_])
                asnrs.append(snrs[i_])
                if verbose:
                    print('Accept via SNR 1.A')
            # If an S-pick, accept only those with some physically reasonable S-P time
            elif Ptime is not None:
                if min(SPmm) <= times[i_] - Ptime <= max(SPmm):
                    atimes.append(times[i_])
                    asnrs.append(snrs[i_])
                    if verbose:
                        print('Accept via Ptime fit 1.B')
            else:
                if verbose:
                    print('REJECT via Ptime misfit: %s'%(str(Ptime)))
        else:
            if verbose: 
                print('REJECT via SNR')

    # If both pick estimates are accepted in the above...
    if len(atimes) == 2:
        # If SNRs are close ...
        if abs(asnrs[0] - asnrs[1]) <= snrtol:        
            # ... and differential time is less than the identical time threshold, use mean
            if dtident >= abs(atimes[0]-atimes[1]):
                if verbose:
                    print('Accept nearly identical picks. 2.A')
                dt = max(atimes) - min(atimes)
                t_pref = min(atimes)+dt/2.
                snr_pref = (asnrs[0]+asnrs[1])/2.
            # ... if between identical time tolerance and maximum differential time, use time
            # of pick with greater SNR
            elif dtident < abs(atimes[0]-atimes[1]) <= dtmax:
                if verbose:
                    print('Accept between dtident & dtmax. 2.B')
                t_pref = atimes[asnrs.index(max(asnrs))]
                snr_pref = asnrs[atimes.index(t_pref)]
            # ... if differential times are large, but SNRs are close, reject both.
            else:
                if verbose:
                    print('REJECT outside dtmax. 2.C')
                t_pref = None
                snr_pref = None
        # If SNRs are not close ...
        else:
            # ... preference to larger SNR
            if verbose:
                print('Accept maximum SNR. 2.D')
            t_pref = atimes[asnrs.index(max(asnrs))]
            snr_pref = max(asnrs)
        
    # If only one pick survives filtering, that's the preferred one!
    elif len(atimes) == 1:
        if verbose:
            print('Accept Single Entry. 3.A')
        t_pref = atimes[0]
        snr_pref = asnrs[0]
    # For any other result, return None!
    else:
        if verbose:
            print('REJECT via empty list. 4.A')
        t_pref = None
        snr_pref = None
    
    return t_pref, snr_pref


def ar_pick_window_py(Zstream,Hstream,Pfilt,Pfkwargs,Pcrf,Pcrfkwargs,\
                      Pfilt2=None,Pf2kwargs=None,EVID=None,SPexp=None,\
                      SPpads=[0.5,3.0],Sfilt=None,Sfkwargs=None,\
                      Scrf='classic_lta-sta',Scrfkwargs={'sta':1e-3,'lta':1e-2},\
                      Sfilt2=None,Sf2kwargs=None,prefpickwargs={},isplot=False,\
                      verbosity=0):
    """
    Adapted version of Akazawa's (2004) method for strong motion event phase picking
    Requires user input to pre-window events in a reasonable manner to isolate events
    Suggest use of a coincidence trigger with selection of Pre Event Time (PEM) and 
    Post Event Time (PET) (Trnkoczy, 1999).

    Algorithm: * denotes adaptations
    1) Filter input vertical trace data
    2) Calculate characteristic function of of filtered trace *(anything not classic STA/LTA is adaptation)
    3) Get the trigger onset (t2) and trigger peak (t3) times
    3) Calculate the AIC using the method from *Maeda (1985) for t1:t3 window on filtered data and pick the minimum value (t4)
    4) Calculate the *Maeda (1985) AIC for the t4-t3:t2 window on unfiltered data and pick the minimum value (t5)
    *5) Report the P-pick as the average of these two estimates (mean(t4,t5)) with error u - range/2
    *6) Calculate and report other metrics on the pick (SNRdB, 



    S Picking - sorta based on Akazawa's but working on very short timescales 
    (S_STA = 10**-3, S_LTA = 
    ...a lot of shit is flipped compared to the results in Akazawa (2004).
    NTS 20. 11. 2019

    Algorithm (More-or-less all updated from Akazawa...)
    7) Conduct classic LTA-STA from 2-3x(S-P_expected) following P pick
    8) Pick maximum value from entire window (if parameterized to match ~2-3xTdomS* STA & 5-10x* LTA)
    9) Conduct AIC on interval between P-pick time and peak of LTA-STA
    10)Window same as in step 4) for P-pick and conduct AIC
    11)Take S-pick and error from the minima of the two AIC calculations

    time codes
    P Picking Times

    ti0 = Start of Current Trace
    ti1 = End of Current Trace
    ti2 = Time of peak CRF on Z channel
    ti3 = Time of minimum AIC value on filtered Z in (ti0:ti2)
    ti4 = Lead time of ti2-ti3 ahead of ti3 for second AIC
    ti5 = Time of minimum AIC value on unfilted Z in (ti4:ti2)
    
    S Picking Times
    
    ti7 = Time of peak CRF on H channel after ti5
    ti8 = Time of AIC minimum on filtered H in (ti5:ti7)
    ti9 = Lead time of ti7-ti8 ahead of ti8 for fourth AIC
    ti10 = Time of minimum AIC value on unfiltered H in (ti9:ti7)

    v1.2 Updates TODO
DONE1) Convert outputs to a pandas DataFrame
DONE2) Output index directly from aic
DONE3) Output snr directly from aic
DONE4) Introduce a minimum lag-time following P-pick to start S-pick processing (1/2 S-P expected)
DONE    4.A) plot full horizontal traces (not truncated at ti5)
DONE5) Introduce an SNR-based preference for large AIC1-AIC2 spacings for vertical
DONE6) Introduce an SNR-based preference for comparison of horizontal channels
PUNT7) Give option for other sCRF methods
        7.A) Give some sort of re-assess trigger
DONE8) Need to introduce a time correction criterion if ti0 > tiX < ti1
    For v1.3
    0) Delete obsolited code from v1.1 and earlier to clean up readability - also purge coeval header comments.
        0.A) To future readers, if there are any - go digging back into earlier versions of this code to see
             earlier development comments. They will be preserved for posterity and I hope that it may serve
             some use... NTS 2019-11-23
    1) Introduce option to impose a cosine taper on data to suppress edge effects on AIC & CRF.
        1.A) Should probably break out the steps in the below code in future version to be
             a string of called subroutines to boost readability & user friendlyness

    2) Give option for other sCRF methods
        2.A) Give some sort of re-assess trigger
            2.A.i) Introduce a full trigger threshold routine to get ON:OFF times 
                   and then impose a SPexp(ected) weighting for 
            2.A.ii) OR, you could "just" do Akazawa's second step and take the cumulative envelope and
                   locate the small up-step in the CumEnv function for the approximate S arrival
                    Thought: This is likely going to get messy with the lower amplitude arrivals in 
                    cryoseismic data, so this may be a fool's errand...
    3) Output
        3.A) Either give the option to output directly to a pyrocko/snuffler markerfile OR NonLinLoc phasefile
             OR create a 
    """
#    stas = optu.get_stations(stream)
    stas = []
    for tr in Zstream:
        if tr.stats.station not in stas:
            stas.append(tr.stats.station)

    datalist = []   
    preflist = []
    for s_,sta in enumerate(stas):
#        print('AR Picking station %s, (%d/%d)'%(str(sta),s_,len(stas)))
        trZ = Zstream.copy().select(station=sta)[0]
        # Get sampling rate
        sr = trZ.stats.sampling_rate
        # Filter station Z trace
        trZf = trZ.copy().filter(Pfilt,**Pfkwargs)
        # Define the initial and final times of window
        ti0 = trZ.stats.starttime
        i0 = 0
        ti1 = trZ.stats.endtime
        i1 = trZ.stats.npts-1
        # Calculate the crf
        try:
            crf = trZf.copy().trigger(Pcrf,**Pcrfkwargs)
        except:
            crf = trZf.copy().trigger('zdetect',sta=0.001)
        # Calculate ti2 from maximum of crf in window
        try:
            i2 = np.where(crf.data==np.max(crf.data))[0][0]
        except IndexError:
            print('Bad result for following trace')
            print(crf)
        ti2 = ti0 + i2/sr
        # Confirm that ti2 is in trace
        ti2 = timeininterval(ti2,ti0,ti1)
        # Trim Filtered Trace (ti0 to ti2)
        trAIC1 = trZf.copy().trim(endtime=ti2)
        # Calculate the first AIC on the vertical
        i3, snr1, aic1 = wfaic(trAIC1.data,normalize=True)
#        i3 = aic1stats[1]
        ti3 = ti0 + i3/sr
        ti3 = timeininterval(ti3,ti0,ti1,code='ti3.'+str(trAIC1.stats.station))
        if SPexp is not None:
            snr1b = snrdB(trZf.copy().trim(starttime=ti3-SPexp,endtime=ti3 + SPexp/2.).data,int(SPexp*sr))
        elif SPexp is None:
            snr1b = snr1
        # Form window for second AIC calculation
        ti4 = ti2 - 2.*abs(ti2-ti3)
        ti4 = timeininterval(ti4,ti0,ti1,code='ti4.'+str(trAIC1.stats.station))
        trAIC2 = trZ.copy().trim(starttime=ti4,endtime=ti2)
        if Pfilt2 is not None:
            trAIC2.filter(Pfilt2,**Pf2kwargs)
        # Calculate second AIC
        i5, snr2, aic2 = wfaic(trAIC2.data,normalize=True)
        # Get AIC minimum index and absolute time
#        i5 = aic2stats[1]
        ti5 = ti4 + i5/sr

        if SPexp is not None:
            trZf2 = trZ.copy().trim(starttime=ti5-SPexp,endtime=ti5+SPexp/2.)
            if Pfilt2 is not None:
                trZf2.filter(Pfilt2,**Pf2kwargs)
            snr2b = snrdB(trZf2.data,int(SPexp*sr))
        elif SPexp is None:
            snr2b = snr2
        # Save data to dictionary - eventually just have this as more of a trigger format
#        iPdata = {'Sta':trZ.stats.station,
#                  'Chan':trZ.stats.channel,
#                  'tAIC2':ti5,'AIC2stats':aic2stats,
#                  'tAIC1':ti3,'AIC1stats':aic1stats}
        if verbosity > 2:
            print('P pick inputs: %s (%.2e) and %s (%.2e)'%(str(ti3),snr1b, str(ti5),snr2b))
        ptpref,psnrpref = assess_2_ar_pick_estimates([ti3,ti5],[snr1b,snr2b],**prefpickwargs)
#        if verbosity > 2:
#            print(str(ptpref)+"is type: "+str(type(ptpref)))
        if ptpref is not None:
            if verbosity > 2:
                print('P pick preference: %s snr: %.2e'%(ptpref,psnrpref))
            preflist.append([trZ.stats.network,trZ.stats.station,trZ.stats.location,\
                             trZ.stats.channel,EVID,'P',ptpref,psnrpref])

        datalist.append([trZ.stats.station,trZ.stats.channel,EVID,ti3,ti5,snr1,snr2,trAIC1.stats.npts,trAIC2.stats.npts])

        if isplot:
            plt.figure()
            if SPexp is not None:
                plt.subplot(3,1,1)
            plt.title('%s.%s'%(trZ.stats.station,trZ.stats.channel))
            gain = 1./(np.max(trZ.data) - np.min(trZ.data))
            plt.plot(trZ.data*gain-2,label='Raw Data %s.%s gain: %.2f'%(trZ.stats.station,trZ.stats.channel,gain))
            gain = 1./(np.max(trZf.data) - np.min(trZ.data))
            plt.plot(trZf.data*gain-1,label='Filt Data %s gain: %.2f'%(Pfilt,gain))
            plt.plot((crf.data/np.max(crf.data)),label='CRF %s gain: %.2e'%(Pcrf,np.max(crf.data)))
            plt.plot(aic1-1,label='First AIC')
            plt.plot(np.arange(len(aic2))+(ti4-ti0)*sr,aic2-2,label='Second AIC')
            plt.plot(np.array([ti3-ti0,ti3-ti0])*sr,np.array([-2.5,0.5]),'b-',label='First AIC Pick SNR: %.2e'%(snr2b))
            plt.plot(np.array([ti5-ti0,ti5-ti0])*sr,np.array([-3.5,-1.5]),'k-',label='Second AIC Pick SNR: %.2e'%(snr1b))
            plt.legend()
#            plt.xlabel('Elapsed Samples Since %s [SR: %d Hz]'%(str(ti0),sr))
            plt.show()

#        Pdata.append(iPdata)
        if SPexp is not None:
            st_i = Hstream.copy().select(station=sta)
            tmpst = []
            tmpssnr = []
            tmpchan = []
            for j_,tr_j in enumerate(st_i):
                # Advance by 1/2 expected S-P time to avoid AIC minimum near P-pick
                ti6 = ti5 + SPexp*SPpads[0]
                ti6 = timeininterval(ti6,ti0,ti1,code='ti6.'+str(tr_j.stats.station))
                trH = tr_j.copy().trim(starttime=ti6,endtime=ti6+SPpads[1]*SPexp) # Trim to 3.5 SPexp past P pick
                if Sfilt:
                    trH.filter(Sfilt,**Sfkwargs)
                sr = trH.stats.sampling_rate
#                breakpoint()
                # Conduct classic LTA - STA on filtered data
#                breakpoint()
                if Scrf == 'classic_lta-sta':
                    crfS = classic_LTA_STA_diff_py(trH.copy(),Scrfkwargs['sta'],Scrfkwargs['lta']) # Eventually update this to **Scrfkwargs
                else:
                    warn('Invalid S-phase CRF method selected')
                i7 = np.where(crfS.data==np.max(crfS.data))[0][0]
#                breakpoint()
                ti7 = ti6 + i7/sr
                ti7 = timeininterval(ti7,ti0,ti1,code='ti7.'+str(tr_j.stats.station))
                # Set window for sAIC
                trAIC3 = trH.copy().trim(endtime=ti7)
                # Conduct first sAIC
                i8,snr3,aic3 = wfaic(trAIC3.data,normalize=True)
                # Get AIC minimum index and absolute time
#                i8 = aic3stats[1]
                ti8 = ti6 + i8/sr
                SPcalc = abs(ti8 - ti5)
                snr3b = snrdB(trH.copy().trim(starttime=(ti8-SPcalc/2.),endtime=(ti8+SPcalc/2.)).data,\
                              int(SPcalc*sr/2.))
                # Get window for second sAIC
                ti9 = ti7 - 2.*abs(ti7 - ti8)
                ti9 = timeininterval(ti9,ti0,ti1,code='ti9.'+str(tr_j.stats.station))
                trAIC4 = tr_j.copy().trim(starttime=ti9,endtime=ti7)
                # Conduct second sAIC
                i10, snr4, aic4 = wfaic(trAIC4.data,normalize=True)
               # Get AIC minimum index and absolute time
#                i10 = aic4stats[1]
                ti10 = ti9 + i10/sr
                SPcalc = abs(ti10 - ti5)
                trHf2 = tr_j.copy().trim(starttime=(ti10-SPcalc/2.),endtime=(ti10+SPcalc/2.))
                if Sfilt2 is not None:
                    trHf2.filter(Sfilt2,**Sf2kwargs)
                snr4b = snrdB(trHf2.data,int(SPcalc*sr/2.))
#                breakpoint()
#                iSdata = {'Sta':trH.stats.station,
#                          'Chan':trH.stats.channel,
#                          'tAIC4':ti10,'AIC4stats':aic4stats,
#                          'tAIC3':ti8,'AIC3stats':aic3stats}
#                Sdata.append(iSdata)
                if isplot:
                    plt.subplot(3,1,2+j_)
                    plt.title('%s.%s'%(trH.stats.station,trH.stats.channel))
                    gain = 1./(np.max(tr_j.data)-np.min(tr_j.data))
                    plt.plot(tr_j.data*gain-2,label='Raw Data %s.%s gain: %.2f'%(tr_j.stats.station,tr_j.stats.channel,gain))
                    if Sfilt:
                        trHf = tr_j.copy().filter(Sfilt,**Sfkwargs)
                        gain = 1./(np.max(trHf.data) - np.min(trHf.data))
                        plt.plot(trHf.data*gain-1,label='Filt Data %s gain:%.2f'%(Sfilt,gain))
                    gain = 1./(np.max(crfS.data)-np.min(crfS.data))
                    plt.plot(np.arange(len(crfS))+(ti6-ti0)*sr,crfS.data*gain,label='CRF %s gain: %.2f'%(Scrf,gain))
                    plt.plot(np.arange(len(aic3))+(ti6-ti0)*sr,aic3-1,label='First S AIC')
                    plt.plot(np.arange(len(aic4))+(ti9-ti0)*sr,aic4-2,label='Second S AIC')
                    plt.plot(np.array([ti8-ti0,ti8-ti0])*sr,np.array([-2,1]),'b-',label='First S AIC Pick SNR:%.2e'%(snr3b))
                    plt.plot(np.array([ti10-ti0,ti10-ti0])*sr,np.array([-3,-1]),'k-',label='Second S AIC Pick SNR:%.2e'%(snr4b))
                    plt.plot(np.array([ti5-ti0,ti5-ti0])*sr, np.array([-3,2]),'r--',label='Second P AIC Pick')
                    if j_ == 2:
                        plt.xlabel('Elapsed Samples Since Second P AIC Pick %s [SR: %d Hz]'%(str(tr_j.stats.starttime),sr))
                    plt.legend()
                    plt.show()
                if ptpref is not None:
                    if verbosity > 2:
                        print('S pick inputs: %s (%.2e) and %s (%.2e)'%(str(ti8),snr3b, str(ti10),snr4b))
                    stpref,ssnrpref = assess_2_ar_pick_estimates([ti8,ti10],[snr3b,snr4b],Ptime=ptpref,\
                                                                 **prefpickwargs)
                    if verbosity > 2:
                        print('%s.%s S pick preference: %s'%(str(trH.stats.station),str(trH.stats.channel),str(stpref)))

                    if stpref is not None:
                        tmpst.append(stpref)
                        tmpssnr.append(ssnrpref)
                        tmpchan.append(trH.stats.channel)
                        if j_ == 1:
                            if len(tmpst) == 2:
                                if verbosity > 1:
                                    print('Pick solutions for both horizontal channels, selecting "best" estimate')
                                ppk = prefpickwargs
                                ppk['dtmax']=1e3 # Hard set to avoid dtmax exit
                                if verbosity > 5:
                                    print(ppk)
                                stpref2,ssnrpref2 = assess_2_ar_pick_estimates(tmpst,tmpssnr,Ptime=ptpref,**ppk)
                                if stpref2 in tmpst:
                                    sprefchan = tmpchan[tmpst.index(stpref2)]
                                else:
                                    sprefchan = tmpchan[0]
                            else:
                                stpref2 = tmpst[0]
                                ssnrpref2 = tmpssnr[0]
                                sprefchan = tmpchan[0]
                            preflist.append([trH.stats.network,trH.stats.station,trH.stats.location,\
                                             trH.stats.channel,EVID,'S',stpref2,ssnrpref2])
           #breakpoint()
                datalist.append([trH.stats.station,trH.stats.channel,EVID,ti8,ti10,snr3,snr4,\
                                 trAIC3.stats.npts,trAIC4.stats.npts])

    phzdf = pd.DataFrame(datalist,columns=['sta','chan','evid','utc1','utc2','snr1','snr2','nw1','nw2'])
    prefdf = pd.DataFrame(preflist,columns=['net','sta','loc','chan','evid','iphase','time','snr'])
    return phzdf, prefdf

