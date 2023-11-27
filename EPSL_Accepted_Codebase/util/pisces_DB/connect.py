"""
:module: connect.py
:purpose: Methods for connecting with SQLite databases formatted in CSS3.0 schema via the PISCES project (Los Alamos National Lab)
:author: Nathan T. Stevens
:email: ntstevens@wisc.edu
:last major revision: 9. APR 2020
    Header updated 7. MAR 2023 for upload to GitHub
"""


import pisces as ps
import sqlalchemy as sa

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

#def connect4query(dbstr = 'sqlite:////home/nates/ActiveProjects/SGGS/DB/sasyDB.sqlite'):
def connect4query(dbstr = 'sqlite:////t31/ntstevens/SGGS/DB/sasyDBpt.sqlite'):
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



#def fetchtraces(session,filters,tabdict):
#    """
#    Fetch traces with applied filters. Filters is a nested list with the following sub-lists
#    [tablename,column,comparator,argument]
#    e.g.
#    ['wfdisc','jdate','lt','2019218']   --> Get entries before 2019218
#    ['wfdisc','sta','eq','R*08']        --> Get entries with matching station with wildcard
#    
#    SPECIAL HANDLING of 'time' and 'endtime' columns
#    ['wfdisc','time','le',UTCDateTime(.....)] --> automatically converts
#
#    """
#    wf_q = session.query(tabdict['wfdisc'])
#    for filt in filters:
#        if 
#
#
#        wf_q.filter(tabdict[filt[0]

#def read_wf_query(db_query,**kwargs)
#    
#
#    st = Stream()
#    for wf in db_query:
#        iwf_path = os.path.join(wf.dir,wf.dfile)
 #       st += read(iwf_path, **kwargs)

#    return st

#def characterize_tables(session):
#    Wfdisc, Site, Sitechan, Affil = ps.get_tables(session.bind, ['wfdisc','site','sitechan','affiliation'])
#    Origin, Event, Arrival, Assoc = ps.get_tables(session.bind, ['origin','event','arrival','assoc'])


#
