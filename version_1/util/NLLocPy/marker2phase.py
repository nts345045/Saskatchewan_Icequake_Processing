"""
:module: marker2phase.py
:purpose: Wrapper utility for converting pyrocko Snuffler marker files with associating events into NonLinLoc *.obs files
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 27. NOV 2019
    header information updated 7. MAR 2023 for upload to GitHub
    Paths updated for relative path imports 7. MAR 2023
"""


import sys
import os
# sys.path.append('/home/nates/Scripts/Python/PKG_DEV')
sys.path.append(os.path.join('..','..'))
# from obspytools.snuffler.markers2NLLphase import *
from util.NLLocPy.markers2NLLphase import *

YEAR = 2019
JDAY0 = 213

for J_ in range(19):
    JDAY = JDAY0 + J_
    indir = '/home/nates/ActiveProjects/SGGS/ANALYSIS/PHASE_PICKING/ARPICK_full_array/%d%d'%(YEAR,JDAY)
    outdir = '/home/nates/ActiveProjects/SGGS/ANALYSIS/LOCATION/NLLoc/OctTree_FA_v1/obs/%d%d'%(YEAR,JDAY)
    tag = 'G2'
    for i_ in range(24):
        ifile = 'ar_pick_assoc_pref-%d-%d-%02d_%s'%(YEAR,JDAY,i_,tag)
        infile = os.path.join(indir,ifile)
        print('Fetching: %s'%(os.path.join(indir,ifile)))
        outfile = 'ar_picks_%4d-%3d-%02d_%s.obs'%(YEAR,JDAY,i_,tag)
        try:
            df, hl = marker2dataframes(infile)
            writeNLLocPhase(outfile,df,hl,outdir)
        except:
            print('Failed to extract')
            continue
