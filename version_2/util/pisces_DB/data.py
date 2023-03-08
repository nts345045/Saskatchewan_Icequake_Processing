"""
:module: pisces_DB/data.py
:purpose: A range of methods of connecting to a PISCES SQLite database and querying waveform files and
	instrument response files, along with data array conversions and metadata bookkeeping for phase
	(re)picking.
:auth: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 30. NOV 2020
	Header added 8. MAR 2023 for upload to GitHub


"""


import numpy as np
from obspy import UTCDateTime, read, Stream, Trace, read_inventory
import os
import pisces as ps
import sqlalchemy as sa

# Database Interface Routines
def db_connect_SM(dbstr='sqlite:////home/nates/ActiveProjects/SGGS/DB/sasyDB.sqlite'):
	"""
	Create session and metadata bound objects with a pices database
	:: INPUT ::
	:type dbstr: String
	:param dbstr: Fully formed sqlalchemy connection string

	:rtype session: sqlalchemy.orm.session object
	:return session: session connection for target sqlite db

	DEV NOTE: This was added into 
	"""
	engine = sa.create_engine(dbstr)
	session = sa.orm.Session(engine)
	meta = sa.MetaData()
	meta.reflect(engine)
		
	return session, meta

def meta2tablenames(meta):
	"""
	Create a list of table names from target sql database metadata

	:: INPUT ::
	:type meta: sqlalchemy.MetaData bound object
	:param meta: metadata from connected sql database

	:rtype tablist: Nested List of Strings
	:return tablist: List of table and associated names from "meta"
	"""
	tablist = []
	collist = []
	for table in meta.tables.values():
		icolnames= []
		for column in table.c:
			icolnames.append(column.name)
		tablist.append(table.name)
		collist.append(icolnames)
	return tablist, collist
	
def maketablehandledict(tablist,session):
	"""
	Create a dictionary with paired table names and their
	database reference handles to be used in subsequent queries

	:: INPUTS ::
	:type tablist: Nested List from meta2tablenames
	:param tablist: Nested list containing table and column names
	:type session: sqlalchemy.orm.Session
	:param session: session handle for active connection to SQL database

	:: OUTPUT ::
	:rtype tabdict: Dictionary of SQLAlchemy table handles
	:return tabdict: paired string-formatted names and table handles for DB

	"""
	tabids = []
	tabhand = []
	for tabentry in tablist:
		itabname = tabentry
		tabids.append(itabname)
		x = ps.get_tables(session.bind,[itabname])
		tabhand.append(x)

	tabdict = dict(zip(tabids,tabhand))
	for tabid in tabids:
		tabdict[tabid] = tabdict[tabid][0]

	return tabdict

def connect4query(dbstr = 'sqlite:////home/nates/ActiveProjects/SGGS/DB/sasyDB.sqlite'):
	"""
	String together the above to produce "session", "meta", "tabdict", "tablist" 
	and "collist" outputs with 1 commend
	
	:: INPUT ::
	:type dbstr: String
	:param dbstr: Fully formed sqlalchemy connection string
	
	:: OUTPUTS ::
	:rtype session: sqlalchemy.orm.Session
	:rtype meta: sqlalchemy.MetaData
	:return tabdict: Dictionary of pices tables associated with table names
	:return tablist: Nested list of table names and their column names
	:return collist: Nested list of columns for each table entry
	"""
	session, meta = db_connect_SM(dbstr=dbstr)
	tablist, collist = meta2tablenames(meta)
	tabdict = maketablehandledict(tablist,session)

	return session, meta, tabdict, tablist, collist

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

def db2wfrf(session,tabdict,starttime=None,endtime=None,reject_str=None,stations=None,channels=None,attach_response=True):
	"""
	Get file-names of waveforms within a specified timeframe as documented in a pisces seismic database

	:: INPUTS ::
	:type session: sqlalchemy.orm.session.Session
	:param session: session connection to sqlite database
	:type tabdict: Dictionary
	:param tabdict: Dict containing entries of table defs pulled from the sqlite session
	:type starttime: obspy.core.UTCDateTime
	:param starttime: starting time for window pull
	:type endtime: obspy.core.UTCDateTime
	:param endtime: ending time for window pull
	:type reject_str: string
	:param reject_str: argument for filter(~table.field.ilike(reject_str)) in a sqlalchemy query
	:type stations: OPTIONAL None or list
	:param stations: list of strings exactly matching station names
	:type channels: OPTIONAL None or list
	:param channels: list of strings exactly describing desired channel names (case sensitive)
	:type filt: OPTIONAL None or string
	:param filt: Name of filter to use in trace.filter()
	:type fkwargs: OPTIONAL None or dict
	:param fkwargs: Dictionary containing key word arguments for trace.filter()
	:type filt_pad: float
	:param filt_pad: padding to add to initially loaded data to capture filter edge effects.
					 data are trimmed to starttime and endtime after filtering has been applied
	:type verb: int
	:param verb: level of verbosity
	:type tol: int
	:param tol: Number of tolerated missing samples for trace - a detector for excessive missing data

	:: OUTPUT ::
	:rtype st: obspy.core.Stream
	:return st: Stream containing trimmed waveforms.
	"""
	wf_list = []
	rf_list = []

	wf_q = wf_timefilter(session,tabdict,starttime=starttime,endtime=endtime)
	# Update, now include option to filter by channels
	if isinstance(channels,list):
		wf_q = wf_q.filter(tabdict['wfdisc'].chan.in_(channels))
	elif isinstance(channels,str):
		wf_q = wf_q.filter(tabdict['wfdisc'].chan.in_([channels]))
	if isinstance(stations,list):
		wf_q = wf_q.filter(tabdict['wfdisc'].sta.in_(stations))
	elif isinstance(channels,str):
		wf_q = wf_q.filter(tabdict['wfdisc'].sta.in_([stations]))
	if reject_str is not None and isinstance(reject_str,str):
		wf_q = wf_q.filter(~tabdict['wfdisc'].sta.ilike(reject_str))

	wf_count = wf_q.count()
	for wf in wf_q:
		fpath = os.path.join(wf.dir,wf.dfile)
		wf_list.append(fpath)
		if attach_response:
			try:
				rq = session.query(tabdict['site'],tabdict['instrument']).\
								filter(tabdict['site'].statype == tabdict['instrument'].insname).\
								filter(tabdict['site'].sta==wf.sta)							
				try:
					resp_file = os.path.join(rq[0][1].dir,rq[0][1].dfile)
				except ValueError:
					resp_file = ''
			except:
				resp_file = ''
		rf_list.append(resp_file)
	return wf_list, rf_list



def db2wfrfrs(session,tabdict,starttime=None,endtime=None,reject_str=None,stations=None,channels=None,attach_response=True,rec_sign_dict=None):
	"""
	Get file-names of waveforms within a specified timeframe as documented in a pisces seismic database

	:: INPUTS ::
	:type session: sqlalchemy.orm.session.Session
	:param session: session connection to sqlite database
	:type tabdict: Dictionary
	:param tabdict: Dict containing entries of table defs pulled from the sqlite session
	:type starttime: obspy.core.UTCDateTime
	:param starttime: starting time for window pull
	:type endtime: obspy.core.UTCDateTime
	:param endtime: ending time for window pull
	:type reject_str: string
	:param reject_str: argument for filter(~table.field.ilike(reject_str)) in a sqlalchemy query
	:type stations: OPTIONAL None or list
	:param stations: list of strings exactly matching station names
	:type channels: OPTIONAL None or list
	:param channels: list of strings exactly describing desired channel names (case sensitive)
	:type filt: OPTIONAL None or string
	:param filt: Name of filter to use in trace.filter()
	:type fkwargs: OPTIONAL None or dict
	:param fkwargs: Dictionary containing key word arguments for trace.filter()
	:type filt_pad: float
	:param filt_pad: padding to add to initially loaded data to capture filter edge effects.
					 data are trimmed to starttime and endtime after filtering has been applied
	:type verb: int
	:param verb: level of verbosity
	:type tol: int
	:param tol: Number of tolerated missing samples for trace - a detector for excessive missing data

	:: OUTPUT ::
	:rtype st: obspy.core.Stream
	:return st: Stream containing trimmed waveforms.
	"""
	wf_list = []
	rf_list = []
	rs_list = []

	wf_q = wf_timefilter(session,tabdict,starttime=starttime,endtime=endtime)
	# Update, now include option to filter by channels
	if isinstance(channels,list):
		wf_q = wf_q.filter(tabdict['wfdisc'].chan.in_(channels))
	elif isinstance(channels,str):
		wf_q = wf_q.filter(tabdict['wfdisc'].chan.in_([channels]))
	if isinstance(stations,list):
		wf_q = wf_q.filter(tabdict['wfdisc'].sta.in_(stations))
	elif isinstance(channels,str):
		wf_q = wf_q.filter(tabdict['wfdisc'].sta.in_([stations]))
	if reject_str is not None and isinstance(reject_str,str):
		wf_q = wf_q.filter(~tabdict['wfdisc'].sta.ilike(reject_str))

	wf_count = wf_q.count()
	for wf in wf_q:
		fpath = os.path.join(wf.dir,wf.dfile)
		wf_list.append(fpath)
		if attach_response:
			try:
				rq = session.query(tabdict['site'],tabdict['instrument']).\
								filter(tabdict['site'].statype == tabdict['instrument'].insname).\
								filter(tabdict['site'].sta==wf.sta)							
				try:
					resp_file = os.path.join(rq[0][1].dir,rq[0][1].dfile)
					if rec_sign_dict is not None:
						rec_sign = rec_sign_dict[rq[0][1].insname]
					else:
						rec_sign = 1
				except ValueError:
					resp_file = ''
					rec_sign = 1
			except:
				resp_file = ''
				rec_sign = 1
		rf_list.append(resp_file)
		rs_list.append(rec_sign)
	return wf_list, rf_list, rs_list

def wfrf2st(wf_list,rf_list,starttime=None,endtime=None,merge=True,mkwargs={'method':1,'fill_value':'interpolate'},z_remap=['GP3'],suppress_loc=True,verb=0,tol=10,pad=True,pad_value=0):
	st = Stream()
	st_tmp = Stream()
	if len(wf_list) == len(rf_list):
		for i_,wff in enumerate(wf_list):
			tr = read(wff,starttime=starttime,endtime=endtime)[0]
			if rf_list[i_] != '':
				try:
					inv = read_inventory(rf_list[i_])
				except:
					inv = None
			else:
				inv = None
			# Ensure sampling rate is read correctly
			tr.stats.sampling_rate = np.round(tr.stats.sampling_rate)
			# Attach Response, if a valid inventory is present
			if inv is not None:
				tr.stats.response = inv.networks[0].stations[0].channels[0].response
			st_tmp += tr
	if merge:
		st_tmp.merge(**mkwargs)

	for tr in st_tmp:
		# Handle location code suppression option
		if not suppress_loc:
			trsplit = tr.id.split('.')
			if len(trsplit) == 5:
				tr.stats.location = trsplit[-2]
				if verb > 1:
					print('Shortening location string to 4th "." delimited element')
					print(tr.stats)
		else:
			tr.stats.location = ''
		# Handle Z channel naming reassignment
		if tr.stats.channel in z_remap:
			tr.stats.channel = tr.stats.channel[:-1]+'Z'

		if tr.stats.npts == 0:
			print('Skipping trace: %s. Zero data'%(str(tr)))
		elif tr.stats.npts < (endtime - starttime)*tr.stats.sampling_rate - tol:
			exp_npts = (endtime - starttime)*tr.stats.sampling_rate
			print('Skipping trace: %s. --> %d required (tolerance: %d)'\
				   %(str(tr),exp_npts,tol))
		else:
			st += tr
	# Ensure Traces are Trimmed to the exact correct length! - Occasionally an extra sample would slip through...
	st = st.trim(starttime=starttime,endtime=endtime,pad=pad,fill_value=pad_value)
	return st


def wfrfrs2st(wf_list,rf_list,rs_list,starttime=None,endtime=None,merge=True,mkwargs={'method':1,'fill_value':'interpolate'},z_remap=['GP3'],suppress_loc=True,verb=0,tol=10,pad=True,pad_value=0):
	st = Stream()
	st_tmp = Stream()
	if len(wf_list) == len(rf_list) == len(rs_list):
		for i_,wff in enumerate(wf_list):
			tr = read(wff,starttime=starttime,endtime=endtime)[0]
			if rf_list[i_] != '':
				try:
					inv = read_inventory(rf_list[i_])
				except:
					inv = None
			else:
				inv = None
			# Ensure sampling rate is read correctly
			tr.stats.sampling_rate = np.round(tr.stats.sampling_rate)
			# Attach Response, if a valid inventory is present
			if inv is not None:
				tr.stats.response = inv.networks[0].stations[0].channels[0].response
			# Apply Recording Notation correction, if applicable (default is multiplying by 1)
			tr.data = tr.data*float(rs_list[i_])
			# Append trace to temporary stream
			st_tmp += tr

	if merge:
		st_tmp.merge(**mkwargs)

	for tr in st_tmp:
		# Handle location code suppression option
		if not suppress_loc:
			trsplit = tr.id.split('.')
			if len(trsplit) == 5:
				tr.stats.location = trsplit[-2]
				if verb > 1:
					print('Shortening location string to 4th "." delimited element')
					print(tr.stats)
		else:
			tr.stats.location = ''
		# Handle Z channel naming reassignment
		if tr.stats.channel in z_remap:
			tr.stats.channel = tr.stats.channel[:-1]+'Z'
		# If there is no data in a given trace, skip it
		if tr.stats.npts == 0:
			print('Skipping trace: %s. Zero data'%(str(tr)))
		# If there is a very small amount of data in the trace, skip it
		elif tr.stats.npts < (endtime - starttime)*tr.stats.sampling_rate - tol:
			exp_npts = (endtime - starttime)*tr.stats.sampling_rate
			print('Skipping trace: %s. --> %d required (tolerance: %d)'\
				   %(str(tr),exp_npts,tol))
		else:
			st += tr
	# Ensure Traces are Trimmed to the exact correct length! - Occasionally an extra sample would slip through...
	st = st.trim(starttime=starttime,endtime=endtime,pad=pad,fill_value=pad_value)
	return st


def db2st(session,tabdict,starttime=None,endtime=None,reject_str=None,stations=None,channels=None,filt=None,fkwargs=None,filt_pad=0.1,verb=0,tol=10,merge=True,mkwargs={'method':1,'fill_value':'interpolate'},z_remap=['GP3'],suppress_loc=True,attach_response=True,pad=True,pad_value=0):
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
	:type reject_str: string
	:param reject_str: argument for filter(~table.field.ilike(reject_str)) in a sqlalchemy query
	:type stations: OPTIONAL None or list
	:param stations: list of strings exactly matching station names
	:type channels: OPTIONAL None or list
	:param channels: list of strings exactly describing desired channel names (case sensitive)
	:type filt: OPTIONAL None or string
	:param filt: Name of filter to use in trace.filter()
	:type fkwargs: OPTIONAL None or dict
	:param fkwargs: Dictionary containing key word arguments for trace.filter()
	:type filt_pad: float
	:param filt_pad: padding to add to initially loaded data to capture filter edge effects.
					 data are trimmed to starttime and endtime after filtering has been applied
	:type verb: int
	:param verb: level of verbosity
	:type tol: int
	:param tol: Number of tolerated missing samples for trace - a detector for excessive missing data

	:: OUTPUT ::
	:rtype st: obspy.core.Stream
	:return st: Stream containing trimmed waveforms.
	"""
	wf_q = wf_timefilter(session,tabdict,starttime=starttime,endtime=endtime)
	# Update, now include option to filter by channels
	if isinstance(channels,list):
		wf_q = wf_q.filter(tabdict['wfdisc'].chan.in_(channels))
	elif isinstance(channels,str):
		wf_q = wf_q.filter(tabdict['wfdisc'].chan.in_([channels]))
	if isinstance(stations,list):
		wf_q = wf_q.filter(tabdict['wfdisc'].sta.in_(stations))
	elif isinstance(channels,str):
		wf_q = wf_q.filter(tabdict['wfdisc'].sta.in_([stations]))
	if reject_str is not None and isinstance(reject_str,str):
		wf_q = wf_q.filter(~tabdict['wfdisc'].sta.ilike(reject_str))

	wf_count = wf_q.count()
	st = Stream()
	st_tmp = Stream()
	for wf in wf_q:
		fpath = os.path.join(wf.dir,wf.dfile)
		if verb > 1:
			print(fpath)
### FILTERING SECTION ###
		if filt is None:
			tr = read(fpath,starttime=starttime,endtime=endtime)[0]
		elif filt is not None:
			tr = read(fpath,starttime=starttime-filt_pad,endtime=endtime+filt_pad)[0]
		if tr.stats.sampling_rate != round(tr.stats.sampling_rate):
			tr.stats.sampling_rate = round(tr.stats.sampling_rate)
		if filt is not None:
			tr = tr.filter(filt,**fkwargs)
			tr.trim(starttime=starttime,endtime=endtime)
### INSTRUMENT RESPONSE ATTACHMENT ###
		if attach_response:
			try:
				rq = session.query(tabdict['site'],tabdict['instrument']).\
								filter(tabdict['site'].statype == tabdict['instrument'].insname).\
								filter(tabdict['site'].sta==tr.stats.station)
				try:
					inv = read_inventory(os.path.join(rq[0][1].dir,rq[0][1].dfile))
				except ValueError:
					breakpoint()
				tr.stats.response = inv.networks[0].stations[0].channels[0].response
			except IndexError:
				tr = tr
### OUTPUT STREAM ASSEMBLY ###
		st_tmp += tr
		if verb <= 1 and verb >0:
			print('LOADED %d/%d'%(i_+1,wf_count))
		if verb > 1:
			print('LOADED: '+fpath)
### MERGE TRACES ###
	# Merge traces if crossing over from one file to another
	if merge:
		st_tmp.merge(**mkwargs)

	for tr in st_tmp:
		# Handle location code suppression option
		if not suppress_loc:
			trsplit = tr.id.split('.')
			if len(trsplit) == 5:
				tr.stats.location = trsplit[-2]
				if verb > 1:
					print('Shortening location string to 4th "." delimited element')
					print(tr.stats)
		else:
			tr.stats.location = ''
		# Handle Z channel naming reassignment
		if tr.stats.channel in z_remap:
			tr.stats.channel = tr.stats.channel[:-1]+'Z'

		if tr.stats.npts == 0:
			print('Skipping trace: %s. Zero data'%(str(tr)))
		elif tr.stats.npts < (endtime - starttime)*tr.stats.sampling_rate - tol:
			exp_npts = (endtime - starttime)*tr.stats.sampling_rate
			print('Skipping trace: %s. --> %d required (tolerance: %d)'\
				   %(str(tr),exp_npts,tol))
		else:
			st += tr
		# Reapply trim with padding to ensure equal sample count in all traces
		st = st.trim(starttime=starttime,endtime=endtime,pad=pad,fill_value=pad_value)
	return st

# Trace to DataFrame routines
def ZEN2colmat(trZ,trE,trN):
	"""
	Convert 3 input obspy.trace.Trace() data vectors into an n-by-3 column vector
	doing a series of sanity checks to make sure data vectors are of equal length.
	"""
	# Try doing things the easy way first!
	try:
		out = np.concatenate([trZ.data[:,np.newaxis],\
							  trE.data[:,np.newaxis],\
							  trN.data[:,np.newaxis]],\
							  axis=1)
	# If this kicks a ValueError, start doing sanity checks
	except ValueError:
		tsZ = trZ.stats.starttime; 
		teZ = trZ.stats.endtime;
		tsE = trE.stats.starttime; 
		teE = trE.stats.endtime;
		tsN = trN.stats.starttime; 
		teN = trN.stats.endtime;
		tsV = np.array([tsZ,tsE,tsN]); 
		teV = np.array([teZ,teE,teN]);
		tsMAX = UTCDateTimeArrayFun(tsV,np.max)
		tsMIN = UTCDateTimeArrayFun(tsV,np.min)
		teMAX = UTCDateTimeArrayFun(teV,np.max)
		teMIN = UTCDateTimeArrayFun(teV,np.min)
		trZ = trZ.trim(starttime=tsMAX,endtime=teMIN,pad=True,fill_value=0)
		trE = trE.trim(starttime=tsMAX,endtime=teMIN,pad=True,fill_value=0)
		trN = trN.trim(starttime=tsMAX,endtime=teMIN,pad=True,fill_value=0)
		out = np.concatenate([trZ.data[:,np.newaxis],\
							  trE.data[:,np.newaxis],\
							  trN.data[:,np.newaxis]],\
							  axis=1)
		# dZ = trZ.data; dE = trE.data; dN = trN.data;
		# tsZ = trZ.stats.starttime; teZ = trZ.stats.endtime;
		# tsE = trE.stats.starttime; teE = trE.stats.endtime;
		# tsN = trN.stats.starttime; teN = trN.stats.endtime;
		# tsV = np.array([tsZ,tsE,tsN]); teV = np.array([teZ,teE,teN]);
		# ZENsamps = np.array([len(dZ),len(dE),len(dN)])
		# MAXsamps = np.max(ZENsamps)
		# MINsamps = np.min(ZENsamps)			
		# tsMAX = UTCDateTimeArrayFun(tsV,np.max())
		# tsMIN = UTCDateTimeArrayFun(tsV,np.min())
		# teMAX = UTCDateTimeArrayFun(teV,np.max())
		# teMIN = UTCDateTimeArrayFun(teV,np.min())
		# # If a single trace has extra samples, trim
		# if len(np.where(ZENsamps == MINsamps)) == 2:
		# 	idx = np.where(ZENsamps == MAXsamps)


		# # If a single trace has fewer samples, pad
		# elif len(np.where(ZENsamps == MAXsamps)) == 2:
		# 	idx = np.where(ZENsamps == MINsamps)
		# # If each trace has a different length, trim the long one, pad the short one
		# elif len(np.where(ZENsamps == MAXsamps)) == 1 and len(np.where(ZENsamps == MINsamps)) == 1:
		# 	MEDsamps = np.median(ZENsamps)
	return out

# Stream to Array
def stream2colmat(stream,order,key='channel'):
	"""
	Convert an obspy.stream.Stream() object into a numpy.ndarray with data vectors
	converted into column vectors in the order specified
	"""
	out = np.array(stream.copy().select(**{key:order[0]}))[:,np.newaxis]
	for o_ in order[1:]:
		iarray = np.array(stream.copy().select(**{key:o_}))[:,np.newaxis]
		out = np.concatenate([out,iarray],axis=1)

	return out[0]

# Apply numpy operations on UTCDateTime arrays
def UTCDateTimeArrayFun(UTCDateTimeArray,array_function,afkwargs={}):
	dt_array = UTCDateTimeArray - UTCDateTime(0)
	out = array_function(dt_array,**afkwargs)
	out_array = UTCDateTime(0) + out
	return out_array

def UTCweighted_average(UTCDateTimeArray,weights):
	# breakpoint()
	dt_array = np.array(UTCDateTimeArray) - UTCDateTime(0)
	out = np.sum(dt_array*np.array(weights))/np.sum(np.array(weights))
	UTCout = UTCDateTime(0) + out
	return UTCout

def weighted_pick_std(UTCDT_array,E_UTCDT,weights=None):
	# Conduct the weighted standard deviation on an input array-like set of UTCDateTimes and the
	# Mean pick UTCDateTime
	# breakpoint()
	values = np.array(UTCDT_array) - E_UTCDT
	var = np.average((values.astype(float))**2, weights=weights)
	return np.sqrt(var)

# Travel Time Modelling
def calc_tS_dSP(tP,dist,Vp,VpVs):
	tScal = dist*((VpVs - 1)/Vp) + tP
	dSPcal = tScal - tP
	return tScal, dSPcal


def evnt2nllobs(evnt,nll_obs_file):
	evnt.write(nll_obs_file,format='NLLoc')

def evnt2quakeml(evnt,quakeml_file,**kwargs):
	cat = Catalog(**kwargs)
	cat.events.append(evnt)
	cat.write(quakeml_file,format='quakeml')

def df_event2csv(df_event,csv_filename,header=True,index=False,**kwargs):
	df_event.to_csv(csv_filename,header=header,index=index,**kwargs)

def df_event2nllobs(df_event,obs_file):
	fmt_str = '%6s %4s %4s %1s %6s %1s %04d%02d%02d %02d%02d %7.4f %3s %9.2e %9.2e %9.2e %9.2e %9.2e\n'
	fobj = open(obs_file,'w')

	for i_ in range(len(df_event)):
		if 'time' in df_event.columns:
			if not df_event.iloc[i_].isna()['time']:
				sta = df_event.iloc[i_]['sta']
				inst = '?'
				comp = '?'
				onset = df_event.iloc[i_]['onset']
				if onset == 'impulsive':
					oset = 'i'
				elif onset == 'emergent':
					oset = 'e'
				else:
					oset = '?'

				phz = df_event.iloc[i_]['phz']
				sfm = df_event.iloc[i_]['polarity']
				if phz == 'P':
					fm = sfm
					# if sfm == 'positive':
					# 	fm = 'U'
					# elif sfm == 'negative':
					# 	fm = 'D'
					# elif sfm == 'undecidable':
					# 	fm = '.'
					# else:
					# 	fm = '?'
				else:
					fm = '?'
				UTCDT = df_event.iloc[i_]['time']
				yr = UTCDT.year
				mo = UTCDT.month
				day = UTCDT.day
				hr = UTCDT.hour
				minute = UTCDT.minute
				seconds = UTCDT.second
				usec = UTCDT.microsecond
				sec = seconds + usec*1e-6
				errtype = 'GAU'
				errmag = df_event.iloc[i_]['terr']
				if errmag is None or errmag == 'None':
					errmag = -1
				codadur = df_event.iloc[i_]['coda']
				if codadur is None or codadur == 'None':
					codadur = -1
				P2Pamp = df_event.iloc[i_]['Amax']
				if P2Pamp is None or P2Pamp == 'None':
					P2Pamp = -1
				P2Pdur = df_event.iloc[i_]['Tmax']
				if P2Pdur is None or P2Pdur == 'None':
					P2Pdur = -1
				# rect = df_event.iloc[i_]['rect']
				wgt = 1.
				try:
					fobj.write(fmt_str%(sta,inst,comp,oset,phz,fm,yr,mo,day,hr,minute,sec,errtype,errmag,codadur,P2Pamp,P2Pdur,wgt))
				except TypeError:
					print(df_event.iloc[i_])
	fobj.write('\n')
	fobj.close()







