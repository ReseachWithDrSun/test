''' The new delta phi correlations.'''
from numpy.fft import ifft, fft, fftshift
#nneded for qphi correlations
from tools.coordinates import mkpolar
from tools.partitions import linplist, partition2D
from tools.partitions import partition_sum, partition_avg
from tools.circavg import qphiavg
import numpy as np
from tools.convol import convol1d
from tools.interpolations import fillin1d

def deltaphicorr(img1,img2,mask=None, qlist=None, philist=None,dq=0, noqs=None, nophis=None,x0=None,y0=None,interp=0):
    ''' The new delta phi correlations.
        img1, img2 : used for the correlations (set to equal if same image)
            this is a 2nd moment calc so you need two quantities
        mask : the mask for the data set
        It uses the binning scheme only in one step. It transforms the image to
            a qphi square grid. It then uses FFT's to transform to a qphi correlation.
        interp : interpolation method. 
            0 - none (default)
            1 - linear interpolation
            2 - linear interpolation and reflect about 180 degrees 
        dq : shift by q in number of bins before correlating (q-q correlations)
        '''
    if(x0 is None):
        x0 = img1.shape[1]/2
    if(y0 is None):
        y0 = img1.shape[0]/2
    if(mask is None):
        mask = np.ones(img1.shape)
    if(qlist is None):
        if(noqs is None):
            noqs = 800
    else:
        noqs = len(qlist)//2
    if(philist is None):
        if(nophis is None):
            nophis = 360
    else:
        nophis = len(philist)//2

    #   0. get pixellist and relevant data
    pxlst = np.where(mask.ravel() == 1)
    data1 = img1.ravel()[pxlst]
    data2 = img2.ravel()[pxlst]
    
    #   1. Make coord system and grab selected pixels (only if qlist or philist not specified)
    if(qlist is None or philist is None):
        QS,PHIS = mkpolar(img1,x0=x0,y0=y0)
        qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]
    
    #   2.  make the lists for selection (only if qlist of philist not specified)
    if(qlist is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    else:
        qlist_m1 = qlist

    if(philist is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)
    else:
        philist_m1 = philist

    qs_m1 = (qlist_m1[0::2] + qlist_m1[1::2])/2.
    phis_m1 = (philist_m1[0::2] + philist_m1[1::2])/2.
    
    # 3. transform image into a qphi map (and mask)
    sq1 = qphiavg(img1, mask=mask, x0=x0, y0=y0, qlist=qlist_m1, philist=philist_m1, interp=interp)
    sq2 = qphiavg(img2, mask=mask, x0=x0, y0=y0, qlist=qlist_m1, philist=philist_m1, interp=interp)
    sqmask = qphiavg(img1*0+1, mask=mask, x0=x0, y0=y0, qlist=qlist_m1, philist=philist_m1, interp=interp)

    # do a q-q delta phi correlation if requested
    if(dq != 0):
        sq2 = np.roll(sq2,-int(dq),axis=0)

    # perform convolution
    sqcphi = convol1d(sq1,sq2,axis=1) #ifft(fft(sq2,axis=1)*np.conj(fft(sq1,axis=1)),axis=1).real
    sqcphimask = convol1d(sqmask,axis=1) #ifft(fft(sqmask,axis=1)*np.conj(fft(sqmask,axis=1)),axis=1).real

    # say .1 not zero to account for finite errors
    if(interp == 0):
        w = np.where(sqcphimask > .1)
        sqcphi[w] /= sqcphimask[w]
    elif interp == 1 or interp == 2:
        sqcphi /= nophis
    elif interp == 3:
        # No normalization (useful for counting correlated pixels in mask)
        pass
    else:
        print("Warning: reached an unexpected interp flag in qphicorr, ignoring...")
        pass

    return sqcphi

def deltaphicorr_qphivals(img1,img2,mask=None, qlist=None, philist=None,dq=0, noqs=None, nophis=None,x0=None,y0=None,interp=0):
    ''' return the qs and phis for the delta phicorrelation parameters
    '''

    if(x0 is None):
        x0 = img1.shape[1]/2
    if(y0 is None):
        y0 = img1.shape[0]/2
    if(mask is None):
        mask = np.ones(img1.shape)
    if(qlist is None):
        if(noqs is None):
            noqs = 800
    else:
        noqs = len(qlist)//2
    if(philist is None):
        if(nophis is None):
            nophis = 360
    else:
        nophis = len(philist)//2

    #   0. get pixellist and relevant data
    pxlst = np.where(mask.ravel() == 1)
    data1 = img1.ravel()[pxlst]
    data2 = img2.ravel()[pxlst]
    
    #   1. Make coord system and grab selected pixels (only if qlist or philist not specified)
    if(qlist is None or philist is None):
        QS,PHIS = mkpolar(img1,x0=x0,y0=y0)
        qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]
    
    #   2.  make the lists for selection (only if qlist of philist not specified)
    if(qlist is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    else:
        qlist_m1 = qlist

    if(philist is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)
    else:
        philist_m1 = philist

    qs_m1 = (qlist_m1[0::2] + qlist_m1[1::2])/2.
    phis_m1 = (philist_m1[0::2] + philist_m1[1::2])/2.

    return qs_m1, phis_m1


def deltaphidiff(img1,img2,mask=None, qlist=None, philist=None,x0=None,y0=None,PF=False,interp=0):
    ''' The new delta phi difference correlations.
        img1, img2 : used for the correlations (set to equal if same image)
            this is a 2nd moment calc so you need two quantities
        mask : the mask for the data set
        It uses the binning scheme only in one step. It transforms the image to
            a qphi square grid. It then uses FFT's to transform to a qphi correlation.
        interp : interpolation method. 
            0 - none (default)
            1 - linear interpolation
            2 - linear interpolation and reflect about 180 degrees (if possible)
        '''
    if(x0 is None):
        x0 = img1.shape[1]/2
    if(y0 is None):
        y0 = img1.shape[0]/2
    if(mask is None):
        mask = np.ones(IMG.shape)
    if(qlist is None):
        noqs = 800
        qlist_m1 = None
    elif(len(qlist) == 1):
        noqs = qlist
        qlist_m1 = None
    else:
        qlist_m1 = qlist

    if(philist is None):
        nophis = 360 
        philist_m1 = None
    elif(len(philist) == 1):
        noqs = philist
        philist_m1 = None
    else:
        philist_m1 = qlist
        
    #   0. get pixellist and relevant data
    pxlst = np.where(mask.ravel() == 1)
    data1 = img1.ravel()[pxlst]
    data2 = img2.ravel()[pxlst]
    
    #   1. Make coord system and grab selected pixels
    QS,PHIS = mkpolar(img1,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]
    
    #   2.  make the lists for selection
    # m1 means 1st moment
    if(qlist_m1 is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    if(philist_m1 is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)
    
    #   3.  partition according to the lists, 2D here, make bin id 1 dimensional
    qid_m1,pid_m1,pselect_m1 = partition2D(qlist_m1,philist_m1,qs,phis)
    bid_m1 = (qid_m1*nophis + pid_m1).astype(int)
    
    # the sqphi here
    sq1 = np.zeros((noqs,nophis))
    sq2 = np.zeros((noqs,nophis))
    sq_qs = np.zeros((noqs,nophis))
    sq_phis = np.zeros((noqs,nophis))
    sqmask = np.zeros((noqs,nophis))
    sqcphi = np.zeros((noqs, nophis))
    sqcphimask = np.zeros((noqs, nophis))

    #transform image into a qphi map (1st moment)
    maxid = np.max(bid_m1)
    sq1.ravel()[:maxid+1] = partition_avg(data1[pselect_m1],bid_m1)
    sq2.ravel()[:maxid+1] = partition_avg(data2[pselect_m1],bid_m1)
    sqmask.ravel()[:maxid+1] = partition_avg(pselect_m1*0 + 1,bid_m1)
    
    # calculate the correlation (2nd moment, keep same binning as first moment)
    nx = np.arange(nophis)
    NX = np.tile(nx,(nophis,1))
    NY = np.copy(NX.T)
    # in the case where img1 != img2 there should be an asymmetry,
    # but I ignore the direction and just take absolute value. It should not matter
    # as the order in which I supply img1 and img2 are not relevant
    NYX = NY - NX
    w = np.where(NYX < 0)
    NYX[w] = NYX[w] + nophis
    TAUID_m2 = NYX[np.newaxis,:,:]
    #import matplotlib.pyplot as plt
    #plt.figure(10);plt.imshow(TAUID[0])
    # m2 means 2d moment
    QID_m2 = np.arange(noqs)[:,np.newaxis,np.newaxis]
    BID_m2 = TAUID_m2*noqs + QID_m2
    bid_m2 = BID_m2.ravel()

    # indexing tricks, expand phi dimension to a (phi, phi) dimension
    sqd1 = sq1[:,NX].ravel()
    sqd2 = sq2[:,NY].ravel()
    sqmask1 = sqmask[:,NX].ravel()
    sqmask2 = sqmask[:,NY].ravel()
    # cross multiply masks
    #sqd1 *= sqmask2
    #sqd2 *= sqmask1
    sqdiff = np.abs(sqd2-sqd1)**2
    # test first try convolution and see that it matches with current qphi corr
    sqdiff = sqd2*sqd1
    sqdiffmask = sqmask1*sqmask2
    sqdiff *= sqdiffmask

    res = partition_avg(sqdiff,bid_m2)
    resmask = partition_avg(sqdiffmask,bid_m2)
    w = np.where(resmask > 1e-6)
    res[w] /= resmask[w]
    bins = np.arange(np.max(bid_m2)+1)
    sqcphi[bins%noqs,bins//noqs] = res

#    for i in range(dimy):
#        for j in range(dimx):
#            for k in range(dimx):
#                if(PF is True):
#                    print("iteration {}, {}, {} of {}, {}, {}".format(i,j,k,dimy,dimx,dimx))
#                sqcphi[i,k] += np.abs(sq1[i,j] - sq2[i,(j + k)%dimx])**2*sqmask[i,j]*sqmask[i,(j+k)%dimx]
#                sqcphimask[i,j] += sqmask[i,j]*sqmask[i,(j+k)%dimx]
    #sqcphi = ifft(fft(sq2,axis=1)*np.conj(fft(sq1,axis=1)),axis=1).real
    #sqcphimask = ifft(fft(sqmask,axis=1)*np.conj(fft(sqmask,axis=1)),axis=1).real
    
    return sq_qs, sq_phis, sqcphi

def deltaphiqqdiff(img1,img2,mask=None, noqs=None, nophis=None,x0=None,y0=None,PF=False):
    ''' A delta phi difference correlation. This one returns a:
        q1,q2,dphi matrix where each element is <(I(q1,phi)-I(q2,phi+dphi))^2>_phi
        img1, img2 : used for the correlations (set to equal if same image)
            this is a 2nd moment calc so you need two quantities
        mask : the mask for the data set
        It uses the binning scheme only in one step. It transforms the image to
            a qphi square grid. It then uses FFT's to transform to a qphi correlation.
        '''
    if(x0 is None):
        x0 = img1.shape[1]/2
    if(y0 is None):
        y0 = img1.shape[0]/2
    if(mask is None):
        mask = np.ones(IMG.shape)
    if(noqs is None):
        noqs = 800
    if(nophis is None):
        nophis = 360

    #   0. get pixellist and relevant data
    pxlst = np.where(mask.ravel() == 1)
    data1 = img1.ravel()[pxlst]
    data2 = img2.ravel()[pxlst]
    
    #   1. Make coord system and grab selected pixels
    QS,PHIS = mkpolar(img1,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]
    
    #   2.  make the lists for selection
    qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    philist_m1 = linplist(np.min(phis),np.max(phis),nophis)
    
    #   3.  partition according to the lists, 2D here, make bin id 1 dimensional
    qid,pid,pselect = partition2D(qlist_m1,philist_m1,qs,phis)
    bid = (qid*nophis + pid).astype(int)
    
    # the sqphi here
    sq1 = np.zeros((noqs,nophis))
    sq2 = np.zeros((noqs,nophis))
    sq_qs = np.zeros((noqs,nophis))
    sq_phis = np.zeros((noqs,nophis))
    sqmask = np.zeros((noqs,nophis))

    maxid = np.max(bid)
    sq1.ravel()[:maxid+1] = partition_avg(data1[pselect],bid)
    sq2.ravel()[:maxid+1] = partition_avg(data2[pselect],bid)
    sq_qs.ravel()[:maxid+1] = partition_avg(qs[pselect],bid)
    sq_phis.ravel()[:maxid+1] = partition_avg(phis[pselect],bid)
    sqmask.ravel()[:maxid+1] = partition_avg(pselect*0 + 1,bid)
    
    sq_qs = np.sum(sq_qs,axis=1)/np.sum(sqmask,axis=1)
    sq_phis = np.sum(sq_phis,axis=1)/np.sum(sqmask,axis=1)
    
    sqcphitot = np.zeros((noqs,noqs,nophis))
    tx = np.arange(noqs)
    ty = np.arange(noqs)
    TX,TY = np.meshgrid(tx,ty)
    
    # perform convolution
    for i in np.arange(-noqs+1,noqs):
        if (PF is True):
            print("Iterating over slice {} of {}".format(noqs+i,2*noqs+1))
        # grab the slice in q
        w = np.where(TX-TY == i)
        rngbeg = np.maximum(i,0)
        rngend = np.minimum(noqs,noqs+i)
        sq1tmp = np.roll(sq1,i,axis=0)[rngbeg:rngend,:]
        sq2tmp = sq2[rngbeg:rngend]
        sqmask1 = np.roll(sqmask,i,axis=0)[rngbeg:rngend,:]
        sqmask2 = sqmask[rngbeg:rngend,:]
        sqcphi = ifft(fft(sq2tmp,axis=1)*np.conj(fft(sq1tmp,axis=1)),axis=1).real
        sqcphimask = ifft(fft(sqmask1,axis=1)*np.conj(fft(sqmask2,axis=1)),axis=1).real
        sqcphi /= sqcphimask
        sqcphitot[w[0],w[1],:] = sqcphi
    
    return sq_qs, sq_phis, sqcphitot

#def deltaphiqqcorr(img1,img2,mask=None, qlist=None, philist=None, noqs=None, nophis=None,x0=None,y0=None,PF=False):
def deltaphiqqcorr(img1,img2,mask=None, qlist=None, philist=None, noqs=None, nophis=None,x0=None,y0=None,interp=0,PF=False):
    ''' The new delta phi correlations. This one returns a:
        q1,q2,dphi matrix where each element is <I(q1,phi)*I(q2,phi+dphi)>_phi
        img1, img2 : used for the correlations (set to equal if same image)
            this is a 2nd moment calc so you need two quantities
        mask : the mask for the data set
        It uses the binning scheme only in one step. It transforms the image to
            a qphi square grid. It then uses FFT's to transform to a qphi correlation.
        NOTE: This will take quite a bit of memory depending on how big your qlist is.
        '''
    if(x0 is None):
        x0 = img1.shape[1]/2
    if(y0 is None):
        y0 = img1.shape[0]/2
    if(mask is None):
        mask = np.ones(img1.shape)
    if(qlist is None):
        if(noqs is None):
            noqs = 800
    else:
        noqs = len(qlist)//2
    if(philist is None):
        if(nophis is None):
            nophis = 360
    else:
        nophis = len(philist)//2

    #   0. get pixellist and relevant data
    pxlst = np.where(mask.ravel() == 1)
    data1 = img1.ravel()[pxlst]
    data2 = img2.ravel()[pxlst]
    
    #   1. Make coord system and grab selected pixels (only if qlist or philist not specified)
    if(qlist is None or philist is None):
        QS,PHIS = mkpolar(img1,x0=x0,y0=y0)
        qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]
    
    #   2.  make the lists for selection (only if qlist of philist not specified)
    if(qlist is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    else:
        qlist_m1 = qlist

    if(philist is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)
    else:
        philist_m1 = philist

    qs_m1 = (qlist_m1[0::2] + qlist_m1[1::2])/2.
    phis_m1 = (philist_m1[0::2] + philist_m1[1::2])/2.
    
    # 3. transform image into a qphi map (and mask)
    sq1 = qphiavg(img1, mask=mask, x0=x0, y0=y0, qlist=qlist_m1, philist=philist_m1, interp=interp)
    sq2 = qphiavg(img2, mask=mask, x0=x0, y0=y0, qlist=qlist_m1, philist=philist_m1, interp=interp)
    sqmask = qphiavg(img1*0+1, mask=mask, x0=x0, y0=y0, qlist=qlist_m1, philist=philist_m1, interp=interp)

    # calculate the correlation (2nd moment, keep same binning as first moment)
    nx = np.arange(noqs)
    NX = np.tile(nx, (noqs,1))
    NY = np.copy(NX.T)

    # indexing tricks, expand s(q,phi) into s(q1,q2, phi)
    sqd1 = sq1[NX,:]
    sqd2 = sq2[NY,:]
    sqmask1 = sqmask[NX,:]
    sqmask2 = sqmask[NY,:]

    # perform convolution in phi
    sqqphi = convol1d(sqd1, sqd2, axis=2)
    sqmask = convol1d(sqmask1, sqmask2, axis=2)
    w = np.where(sqmask != 0)
    sqqphi[w] /= sqmask[w]
    

#    # perform convolution
#    for i in np.arange(-noqs+1,noqs):
#        if (PF is True):
#            print("Iterating over slice {} of {}".format(noqs+i,2*noqs+1))
#        # grab the slice in q
#        w = np.where(TX-TY == i)
#        rngbeg = np.maximum(i,0)
#        rngend = np.minimum(noqs,noqs+i)
#        sq1tmp = np.roll(sq1,i,axis=0)[rngbeg:rngend,:]
#        sq2tmp = sq2[rngbeg:rngend]
#        sqmask1 = np.roll(sqmask,i,axis=0)[rngbeg:rngend,:]
#        sqmask2 = sqmask[rngbeg:rngend,:]
#        sqcphi = ifft(fft(sq2tmp,axis=1)*np.conj(fft(sq1tmp,axis=1)),axis=1).real
#        sqcphimask = ifft(fft(sqmask1,axis=1)*np.conj(fft(sqmask2,axis=1)),axis=1).real
#        sqcphi /= sqcphimask
#        sqcphitot[w[0],w[1],:] = sqcphi
    
    return sqqphi
