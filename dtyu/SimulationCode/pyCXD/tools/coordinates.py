import numpy as np
from tools.partitions import partition2D, partition_avg, linplist
''' This module deals with 2D coordinate transformations. So far, we only have:
    - x,y (in pixels)
    - q, phi (in pixels, rads)
'''

''' Make the polar coordinate systems.'''
def mkpolar(IMG=None,X=None,Y=None,x0=None,y0=None):
    ''' Make polar coordinates for a 2D image.'''
    if(IMG is not None):
        dimy,dimx = IMG.shape
        if(x0 is None):
            x0 = dimx/2.
        if(y0 is None):
            y0 = dimy/2.
        x = np.arange(dimx)-x0
        y = np.arange(dimy)-y0
        X,Y = np.meshgrid(x,y)
    else:
        if(X is None or Y is None):
            print("Error, need to supply either IMG or X,Y")
            return -1
    QS = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y,X)
    return QS, PHI

def mkcartesian(IMG=None,Q=None,PHI=None,x0=None,y0=None):
    ''' Make polar coordinates for a 2D image.'''
    if(IMG is not None):
        dimy,dimx = IMG.shape
        if(x0 is None):
            x0 = dimx/2.
        if(y0 is None):
            y0 = dimy/2.
        x = np.arange(dimx)-x0
        y = np.arange(dimy)-y0
        X,Y = np.meshgrid(x,y)
    else:
        if(Q is None or PHI is None):
            print("Error, need to supply either IMG or Q,PHI");
            return -1
        X = Q*np.cos(PHI)
        Y = Q*np.sin(PHI)


    return X, Y

''' Coordinate mappings in 2D.
    These are meant to map images from one coordinate system to another.
    These are more complex wrappers that use the base functions: mkpolar, 
        mkcartesian, linplist, partition2D, partition_avg
    xy2qphi - transform from x,y to q,phi
        returns : sqtot, mask_new, Q_new, PHI_new
    qphi2xy - transform from q, phi to x, y
        returns : xytot, mask_new, X_new, Y_new
    xyregrid(img,mask,X,Y,nx=None,ny=None) : regrid with (nx,ny) points
    qphiregrid(img,mask,QS,PHIS,nq=None,nphi=None) : regrid with (nq, nphi) points
'''
def xy2qphi(img,mask,X,Y,noqs=None,nophis=None):
    ''' Transform image from coordinates x,y to q,phi.
        Uses a binning scheme.
    '''
    if(mask is None):
        pxlst = np.arange(img.shape[0]*img.shape[1])
    else:
        pxlst = np.where(mask.ravel() == 1)[0]

    # make q coordinate system
    QS,PHIS = mkpolar(X=X,Y=Y)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]

    data = img.ravel()[pxlst]
    
    if(noqs is None):
        noqs = (np.max(qs)-np.min(qs))

    if(nophis is None):
        nophis = 12

    # make the lists for selection
    nophis = int(nophis);noqs = int(noqs);
    qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    philist_m1 = linplist(np.min(phis),np.max(phis),nophis)

    # partition the grid (digitize)
    qid,pid,pselect = partition2D(qlist_m1,philist_m1,qs,phis)
    bid = (qid*nophis + pid).astype(int)

    # average the partitions
    sq = partition_avg(data[pselect],bid)
    q_new = partition_avg(qs[pselect],bid)
    phi_new = partition_avg(phis[pselect],bid)
    mask_new = partition_avg(pselect*0 + 1,bid)

    sqtot = np.zeros((noqs,nophis))
    Q_new = np.zeros((noqs,nophis))
    PHI_new = np.zeros((noqs,nophis))
    MASK_new = np.zeros((noqs,nophis))

    maxid = np.max(bid)
    sqtot.ravel()[:maxid+1] = sq
    Q_new.ravel()[:maxid+1] = q_new
    PHI_new.ravel()[:maxid+1] = phi_new
    MASK_new.ravel()[:maxid+1] = mask_new

    return sqtot, MASK_new, Q_new, PHI_new

def qphi2xy(img,mask,Q,PHI,noxs=None,noys=None):
    ''' Transform image from coordinates x,y to q,phi.
        Uses a binning scheme.
    '''
    if(mask is None):
        pxlst = np.arange(img.shape[0]*img.shape[1])
    else:
        pxlst = np.where(mask.ravel() == 1)[0]

    X,Y = mkcartesian(Q=Q,PHI=PHI)
    x,y = X.ravel()[pxlst], Y.ravel()[pxlst]

    data = img.ravel()[pxlst]
    
    if(noxs is None):
        noxs = 400

    if(noys is None):
        noys = 400

    noxs = int(noxs);noys = int(noys);
    xlist_m1 = linplist(np.min(x),np.max(x),noxs)
    ylist_m1 = linplist(np.min(y),np.max(y),noys)

    xid,yid,pselect = partition2D(xlist_m1,ylist_m1,x,y)
    bid = (yid*noxs + xid).astype(int)

    xy = partition_avg(data[pselect],bid)
    x_new = partition_avg(x[pselect],bid)
    y_new = partition_avg(y[pselect],bid)
    mask_new = partition_avg(pselect*0 + 1,bid)

    xytot = np.zeros((noys,noxs))
    X_new = np.zeros((noys,noxs))
    Y_new = np.zeros((noys,noxs))
    MASK_new = np.zeros((noys,noxs))

    maxid = np.max(bid)
    xytot.ravel()[:maxid+1] = xy
    X_new.ravel()[:maxid+1] = x_new
    Y_new.ravel()[:maxid+1] = y_new
    MASK_new.ravel()[:maxid+1] = mask_new

    return xytot, MASK_new, X_new, Y_new

def xyregrid(img,mask,X,Y,nx=None,ny=None):
    ''' regrid with (nx,ny) points. 
        This can also be used on QPHI coordinates
        any 2D rectangular grid.
    '''

    pixellist = np.where(mask.ravel() == 1)
    data = img.ravel()[pixellist]

    x = X.ravel()[pixellist]
    y = Y.ravel()[pixellist]
    xlist = linplist(np.min(X),np.max(X),nx)
    ylist = linplist(np.min(Y),np.max(Y),ny)
    xid, yid, pselect = partition2D(xlist, ylist, x, y)
    bid = (yid*nx + xid)   

    xy = partition_avg(data[pselect], bid)
    x_new = partition_avg(x[pselect], bid)
    y_new = partition_avg(y[pselect], bid)
    mask_new = partition_avg(pselect*0 + 1,bid)

    xytot = np.zeros((ny,nx))
    X_new = np.zeros((ny,nx))
    Y_new = np.zeros((ny,nx))
    MASK_new = np.zeros((ny,nx))

    maxid = np.max(bid)
    xytot.ravel()[:maxid+1] = xy
    X_new.ravel()[:maxid+1] = x_new
    Y_new.ravel()[:maxid+1] = y_new
    MASK_new.ravel()[:maxid+1] = mask_new

    return xytot, MASK_new, X_new, Y_new
