#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Filename: phase_translator.py
# Purpose:  Routines for converting between several different data formats for phase picker routines
# Author:   Nathan T. Stevens
# Email:    ntstevens@wisc.edu
# Attribution: Based upon scripts from the Pyrocko Project and the Pisces Project
#
# Copyright (C) 2020 Nathan T. Stevens
# Last major revision: 10. DEC 2020
# 	Header datestamp added 8. MAR 2023 for upload to Git Hub


import pyrocko.gui.marker as pm
import sqlalchemy.exc as exc
from sqlalchemy import func, update
from obspy.core import UTCDateTime
from datetime import datetime

def _get_lastid(session,tabdict,keyname):
	if keyname == 'evid':
		try:
			table_ref = session.query(func.max(tabdict['event'].evid))[0][0]
		except:
			table_ref = None
		if table_ref is None:
			table_ref = 0
	elif keyname == 'orid':
		try:
			table_ref = session.query(func.max(tabdict['origin'].orid))[0][0]
		except:
			table_ref = None
		if table_ref is None:
			table_ref = 0
	elif keyname == 'arid':
		try:
			table_ref = session.query(func.max(tabdict['arrival'].arid))[0][0]
		except:
			table_ref = None
		if table_ref is None:
			table_ref = 0
	elif keyname == 'wfid':
		try:
			table_ref = session.query(func.max(tabdict['wfdisc'].wfid))[0][0]
		except:
			table_ref = None
		if table_ref is None:
			table_ref = 0
	else:
		table_ref = 0

	if table_ref is None:
		table_ref = 0  
		print('keyname not tied to a standard CSS3.0 table, deferring to lastid table value, if present')
	lid_q = session.query(tabdict['lastid']).filter(tabdict['lastid'].keyname == keyname)
	lastid = lid_q[0].keyvalue
	# Return the maximum value between the table cross reference and the lastid table value
	return max([table_ref, lastid])


def _attempt_commit(session,comment=None):
	try:
		session.commit()
		success =  True
	except exc.IntegrityError:
		session.rollback()
		print('rollback - Duplicate value or missing primary key')
		if comment is not None:
			print('Associated with %s'%(str(comment)))
		success = False

	except exc.OperationalError:
		session.rollback()
		print('rollback - No such table or database locked')
		if comment is not None:
			print('Associated with %s'%(str(comment)))
		success = False
	return success

### LASTIDS SQLite Databse Update Routines - imported from hyp2db_v6.py developed on Larsbreen (NTS 4.8.2020)
def get_lastids(session,tabdict):
    lid_q = session.query(tabdict['lastid'])
    id_dict = dict()
    for q in lid_q:
        id_dict[q.keyname]=q.keyvalue
    return id_dict


def update_lastid(session,tabdict,keyname,keyvalue):
    lid = tabdict['lastid']
    stmt = update(lid).where(lid.keyname==keyname).values(keyvalue=keyvalue,lddate=datetime.now())
    session.execute(stmt)


    #lastid = session.query(lid).filter(lid.keyname==keyname)[0]
    #lastid.keyvalue=keyvalue
    #lastid.lddate = datetime.now()
#    lastid_new = lid(keyname=keyname,keyvalue=keyvalue,lddate=datetime.now())
    # try:
    #session.add(lastid)

    # Commit "try" section moved out to a separate supporting function.
    #     session.commit()
    #     print('lastid Updated: %s= %d'%(keyname,keyvalue))
    # except exc.IntegrityError:
    #     session.rollback()
    #     print('rollback LASTID - Duplicate value or missing pimary keys')
    # except exc.OperationalError:
    #     session.rollback()
    #     print('rollback on LASTID - No such table or database locked')

# def trigger2df(trigger,)

# def add_wfdisc_entry(session,tabdict,fileandpath,trace):




def df2markers(pick_df,session,tabdict,frontpad=0,quality_level=1,pick_error=False,net='',loc=''):
	"""
	Convert pick DataFrames output by ar_aic_picker_db into Pyrocko marker lists.
	Event timing is based upon the first pick in the event characterized with the
	option to add a positive-valued "frontpad" to place the event earlier in time

	::INPUTS::
	:type pick_df: pandas.DataFrame
	:param pick_df: output dataframe containing phase-pick information from ar_aic_picker_db.py
	:type session: sqlalchemy.Session
	:param session: connection to SQLite database with 'site' table containing station locations
	:type tabdict: dict
	:param tabdict: dictionary listing the methods describing the SQLite database schema 
					Must have 'site' as an option
	:type frontpad: float
	:param frontpad: number of seconds to subtract from the earliest valid phase time in the input
					pick_df DataFrame to use as the initial assumed source time
	:type quality_level: float
	:param quality_level: minimum quality level to use as a filter for intermediate picks saved during
					the ar_aic_picker_db.py processing. Falls into the range [0,1]
	:type pick_error: Bool
	:param pick_error: Decision to include picking error in the conversion. Default False
					if True - a second marker is made that spans the pick time +/- the error value

	::OUTPUT::
	:rtype markerlist: list of pyrocko.gui.PhaseMarker and one pyrocko.gui.EventMarker
	"""

	markerlist = []
	# Filter picks by quality (qual)
	pdff = pick_df[pick_df['qual']>=quality_level]
	nphz = len(pick_df_filt)

	# Get minimum time & station location
	min_time = pdff['time'].min()
	min_sta = pdff[pdff['time'] == min_time]['sta'].values
	# Apply minimum time adjustment
	min_time -= frontpad
	# Get station location
	sta_q = session.query(tabdict['site']).filter(tabdict['site'].sta==min_sta)
	lat = sta_q.lat
	lon = sta_q.lon

	# Handle instance where min_time is an epoch time	
	if not isinstance(min_time,UTCDateTime):
		min_time = UTCDateTime(min_time)

	# Create EventMarker
	em = pm.Marker(tmin=min_time.timestamp)
	em.convert_to_event_marker(lat=lat,lon=lon)
	ehash = em.get_event_hash()

	# Append marker to list
	markerlist.append(em)
	# Iterate through accepted phases
	for i_ in range(nphz):
		im = pm.PhaseMarker(nslc_ids=[net,pdff['sta'].values[i_],loc,pdff['chan'].values[i_],],\
							tmin = pdff['time'].values[i_].timestamp,\
							event_hash = ehash, event_time=min_time.timestamp,\
							phasename = pdff['phz'].values[i_])
		markerlist.append(im)
		# If pick error is to be included, make second marker for the phase
		if pick_error:
			ime = pm.PhaseMarker(nslc_ids=[net,pdff['sta'].values[i_],loc,pdff['chan'].values[i_],],\
							tmin = pdff['time'].values[i_].timestamp - pdff['dt'].values[i_],\
							tmax = pdff['time'].values[i_].timestamp + pdff['dt'].values[i_],\
							event_hash = ehash, event_time=min_time.timestamp,\
							phasename = pdff['phz'].values[i_])
			markerlist.append(ime)

	return markerlist


def df2db(pick_df,session,tabdict,quality_level=1,auth='ntstevens',orig_method='First Arriving'):
	"""
	Write data from the dataframe to a Pisces/CSS3.0 formatted SQLite database
	"""
	larid = _get_lastid(session,tabdict,'arid')
	lorid = _get_lastid(session,tabdict,'orid')
	levid = _get_lastid(session,tabdict,'evid')


	# Define Table Methods for Shorthand Access
	ARRI = tabdict['arrival']
	ASSO = tabdict['assoc']
	ORIG = tabdict['origin']
	EVEN = tabdict['event']
	LAST = tabdict['lastid']
	SITE = tabdict['site']

	# Subset input pick dataframe based on quality level
	pdff = pick_df[pick_df['qual']>=quality_level]
	pdff1 = pick_df[pick_df['qual']==1.0]
	nass = len(pdff1)
	Pnass = len(pdff1[pdff1['phz']=='P'])
	#breakpoint()
	# Get minimum time & station location
	min_time = pdff['time'].min()
	min_sta = pdff[pdff['time'] == min_time]['sta'].values[0]
	print('Setting Origin Location to: ',min_sta)
	# Get station location
	sta_q = session.query(tabdict['site']).filter(tabdict['site'].sta==min_sta)
	lat = sta_q[0].lat
	lon = sta_q[0].lon
	elev = sta_q[0].elev

	
	# Create new origin entry
	lorid += 1

	# Create new event entry
	levid += 1

	print('Using ORID',lorid)
	norig = ORIG(lat=lat,lon=lon,depth = -1000.*elev, time=min_time.timestamp,\
		         orid=lorid,evid=levid,jdate='%4d%03d'%(min_time.year,min_time.julday),\
		         nass=nass,Pnass=Pnass,algorithm=orig_method,\
		         auth=auth,lddate=datetime.now())
	session.add(norig)
	update_lastid(session,tabdict,'orid',lorid)

	success = _attempt_commit(session,comment=' origin entry error')

	print('Using EVID',levid)
	neven = EVEN(evid=levid,prefor=lorid,auth=auth,lddate=datetime.now())
	session.add(neven)
	update_lastid(session,tabdict,'evid',levid)
	
	success = _attempt_commit(session,comment=' event entry error')
	# if not success:
		# breakpoint()
	if success:
		print('========== New event and origin committed successfully to savedb! ==========')
		# breakpoint()
		for i_ in range(len(pdff)):
			larid += 1
			ista = pdff['sta'].values[i_]
			ichan = pdff['chan'].values[i_]
			# Get Channel ID
			ichanid_q = session.query(tabdict['sitechan'])
			ichanid_q = ichanid_q.filter(tabdict['sitechan'].chan == ichan).filter(tabdict['sitechan'].sta == ista)

			itime = pdff['time'].values[i_]
			iphz = pdff['phz'].values[i_]
			ideltim = pdff['dt'].values[i_]
			iqual = pdff['qual'].values[i_]
			isnr  =pdff['snr'].values[i_]
			try:
				ichanid = ichanid_q[0].chanid

				iarri = ARRI(sta = ista,time = itime.timestamp,arid = larid,\
					         jdate = '%4d%03d'%(itime.year,itime.julday),\
					         chanid = ichanid, chan = ichan, iphase = iphz,\
					         deltim = ideltim, snr= isnr, qual = iqual,\
					         auth = auth, lddate = datetime.now())
			except IndexError:
				iarri = ARRI(sta = ista,time = itime.timestamp,arid = larid,\
					         jdate = '%4d%03d'%(itime.year,itime.julday),\
					         chan = ichan, iphase = iphz,\
					         deltim = ideltim, snr= isnr, qual = iqual,\
					         auth = auth, lddate = datetime.now())
			# Que add command in buffer
			session.add(iarri)
			# Use the wgt to map the quality of picks in the assoc - allows for one fewer 
			iasso = ASSO(arid = larid, orid = lorid, sta = ista, phase = iphz, wgt = iqual,\
						 vmodel = orig_method, timeres = ideltim, lddate = datetime.now())

			session.add(iasso)

		success = _attempt_commit(session,comment='phase and association entries')

	if success:
		update_lastid(session,tabdict,'arid',larid)
		success = _attempt_commit(session,comment='LASTID arid update')

	if success:
		print('Origin Time:',itime)
		print('Phase Count Saved to DB: ',len(pdff))
		print('P count:',len(pdff[pdff['phz']=='P']),'S count: ',len(pdff[pdff['phz']=='S']))
		print('Qual 1.0 Phase Count:',len(pdff1))
		print('P count:',len(pdff1[pdff1['phz']=='P']),'S count: ',len(pdff1[pdff1['phz']=='S']))
		print('========== Successful submission complete! =========== (',datetime.now(),')')
	else:
		print('XXXXXXXXXX Submission failed XXXXXXXXXX')
		



def db2nllphase(orid_query, session, tabdict, output_file, err_scale_min=None, dt_max=None, write_mode='w+',pwdict={'P':1,'S':1},qual=1.0):
	"""
	Generate NonLinLoc Formatted phase data file from a SQLite CSS3.0 formatted database

	:: INPUTS ::

	Lots of values placed as default values in this version of the script

	NLLoc Format                    phaselist#  Null Value
	0)  %-6s  Station Name          [0].[1]     REQUIRED
	1)  %-4s  Instrument                        ?
	2)  %-4s  Component             [0].[3]     ?
	3)  %1s   P phase onset                     ?
	4)  %-6s  Phase Descriptor      [3]         REQUIRED
	5)  %1s   First Motion                      ?
	6)  %04d  Year Component        [1].year    REQUIRED
	7)  %02d  Month Component       [1].month   REQUIRED
	8)  %02d  Day Component         [1].day     REQUIRED
	9)  %02d  Hour Component        [1].hour    REQUIRED
	10) %02d  Minute Component      [1].minute  REQUIRED
	11) %7.4f Seconds               [1].seconds REQUIRED
	12) %3s   Error Type                        REQUIRED (GAU Default)
	13) %9.2e Error magnitude       [2]         REQUIRED (< dominant period)
	14) %9.2e Coda duration                     -1.00e+00
	15) %9.2e Amplitude                         -1.00e+00
	16) %9.2e Period of amplitude               -1.00e+00
	17) %1d   Prior Weight          [4]         REQUIRED (in [0,1], default 1)


	DEVNOTE (4.13.2020) - This routine will likely benefit from an "eager" query or some sort of 
	table join when calling events -- data pull should be pushed onto the SQL side of things
	as much as possible to speed up I/O. Still need to figure these routines out -- NTS

	"""
	#       0)   1)   2)   3)  4)   5)  6)   7)   8)   9)   10)  11)   12) 13)   14)   15)   16)   17)
	fstr = '%-6s %-4s %-4s %1s %-6s %1s %04d %02d %02d %02d %02d %7.4f %3s %9.2e %9.2e %9.2e %9.2e %1d\n'
	fobj = open(output_file,write_mode)
	# Check that orid exists
	# o_q = session.query(tabdict['origin']).filter(tabdict['origin'].orid.in_(orid_list))
	if orid_query.count() > 0:
		for o_ in orid_query:
			i_orid = o_.orid
			# Get arrival id's
			assoc_q = session.query(tabdict['assoc']).filter(tabdict['assoc'].orid == i_orid)
			for a_q in assoc_q:
				# Pull associated phase from database
				phz_q = session.query(tabdict['arrival']).filter(tabdict['arrival'].arid == a_q.arid)
				# Make sure quality is sufficient
				phz_q = phz_q.filter(tabdict['arrival'].qual >= qual)
				# Check max uncertainty time, if applicable
				if isinstance(dt_max,int) or isinstance(dt_max,float):
					phz_q = phz_q.filter(tabdict['arrival'].deltim <= dt_max)
				if phz_q.count() == 1:
					phz_q = phz_q[0]				
					sta = phz_q.sta 					# 0)
					inst = '?' 							# 1)
					ch = phz_q.chan 					# 2)
					po = '?' 							# 3)
					pn = phz_q.iphase 					# 4)
					fm = '?'							# 5)
					time = UTCDateTime(phz_q.time)	
					yr = time.year 						# 6)
					mo = time.month 					# 7)
					da = time.day 						# 8)
					hr = time.hour 						# 9)
					mn = time.minute 					# 10)
					ds = time.second  					# 11)
					et = 'GAU' 							# 12)
					# Forked decision on implementing an alternative minimum uncertainty
					if err_scale_min is None:
						em = phz_q.deltim 				# 13)
					elif err_scale_min > phz_q.deltim:
						em = err_scale_min 				# 13) 
					co = -1.0 							# 14)
					am = -1.0 							# 15)
					pa = -1.0 							# 16)
					pw = pwdict[phz_q.iphase] 			# 17)
					# Write phase entry
					fobj.write(fstr%(sta, inst, ch, po, pn, fm, yr, mo, da, hr, mn, ds, et, em, co, am, pa, pw))
			# Insert a blank line between origin groupings
			fobj.write('\n')
	# Close file when done
	fobj.close()


