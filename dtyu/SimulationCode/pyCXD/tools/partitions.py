import numpy as np
''' This is meant for making partitions for SAXS measurement (2D coordinates,
ignring Ewald sphere curvaturve.  For making q phi coordites, see tools.coordinates:
mkpolar 
Note : This only depends on dimension (2d coordinates), but not the actual basis.

Credit : The idea of usings lists and partitioning them with coordinate and
dimension independent id's etc goes to Mark Sutton who originally implemented
this in yorick. Quite a bit of thought over years of research had been invested
into getting something that was more intuitive/encompassing (even though
simplistic) so he must be credited. Although I've written this on my own
without referring to other code, I have been strongly influenced my him.
Julien Lhermitte Tuesday November 3rd, 2015
'''

def linplist(a,b,n,width=None):
    ''' Make a linear partition list. Partition lists are tuples of
        start,finish.
        Meant to aid the partition making.
        a - start
        b - finish
        n - number of elements in interval
        The array is purposely set up so that the dimension of length 2 is the
        fastest varying (righter most in python). This means, collapsed to 1d, it becomes:
        [start1, finish1, start2, finish2, ....]
        NOTE : This only works for increasing values as of now. Tweaking for decreasing
            values is probably not difficult.
    '''
    plist = np.zeros((n,2))
    # width of interval
    widthba = (b-a)/n
    if(width is None):
        ''' If no width set, assume const width. '''
        width = widthba
    plist[:,0] = a + np.arange(n)*widthba
    plist[:,1] = a + np.arange(n)*widthba + width
    return plist.ravel()

def partition2D(qlist,philist,qvals,phivals):
    ''' Partition a list of pixels with values qvals and phivals
        according to the partition lists qlist and philist.
        Lists are a list of widths [start1,finish1, start2,finish2] etc.
        Note this can be generalized to any 2D coordinate system.
    '''
    qid = np.searchsorted(qlist.ravel(),qvals,"right")
    pid = np.searchsorted(philist.ravel(),phivals,"right")
    pselect = np.where((qid % 2 == 1)*(pid %2 == 1))[0]
    qid = qid[pselect]//2
    pid = pid[pselect]//2
    return qid, pid, pselect

def partition1D(qlist,qvals):
    ''' Partition a list of pixels with values qvals
        according to the partition lists qlist.
        Lists are a list of widths [start1,finish1, start2,finish2] etc.
        Note this can be generalized to any 2D coordinate system.
    '''
    qid = np.searchsorted(qlist.ravel(),qvals,"right")
    pselect = np.where((qid % 2 == 1))[0]
    qid = qid[pselect]//2
    return qid, pselect

def partition_sum(pxdata, bid):
    ''' Sum over various bins of the data with bin id # bid'''
    return np.bincount(bid,weights=pxdata)

def partition_avg(pxdata, bid):
    ''' Avg over various bins of the data with bin id # bid'''
    num = np.bincount(bid).astype(float)
    res = partition_sum(pxdata, bid)
    # res will be zero where num = 0
    w = np.where(num > 0)[0]
    if(len(w) > 0):
        res[w] /= num[w]
    return res
    
def partition_tsum(IMGS,pxlst,bid):
    ''' Sum over various bind of the data from a time series IMGS
        indexed by pxlst with bin id's bid.
        Note: assumes the len(IMGS) returns the length of time series (num images) 
            and that IMGS[i] returns a 2D image indexed by pxlst.
    '''

    ipix = np.zeros((len(IMGS),len(pxlst)))
    for i in range(len(IMGS)):
        ipix[i] = partition_sum(IMGS[i].ravel()[pxlst],bid)
    return ipix

def partition_tavg(IMGS,pxlst,bid):
    ''' Sum over various bind of the data from a time series IMGS
        indexed by pxlst with bin id's bid.
        Note: assumes the len(IMGS) returns the length of time series (num images) 
            and that IMGS[i] returns a 2D image indexed by pxlst.
    '''

    ipix = np.zeros((len(IMGS),len(noperbin)))
    noperbin = np.bincount(bid).astype(float)
    wnum = np.where(num != 0)
    ipix = partition_tsum(IMGS,pxlst,bid)
    ipix /= num[np.newaxis,:]
    return ipix
