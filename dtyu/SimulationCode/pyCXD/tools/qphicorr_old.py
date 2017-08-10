'''Q-phi correlation routines.'''
from tools.Xtools import SAXSObj, ExpPara
#import matplotlib.animation as animation
#import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt

#for the animated images
from plugins.rotate._rotate import rotate

#plt.ion()

'''non object version.'''
def compute_sqcphi_old(img,mask,qd,qs, sq_q, phis,plot=0, x0=None, y0=None, sqcphicnt=None,PF=False,sqphiout=None):
    ''' Compute the qphi correlation on an image img
        imgs - numframes by x by y
        Need: pxlist, idpxlist, ids, img, numperid, numids
        Make: phis - go from 0 to pi for now
        plot (default 1) : plot the calculation as it comes along
            (note: assumes pyqtgraph)
        IMG : average image, subtract if set
        x0, y0 : center of image
    '''
    # fix the data types for the rotate
    mask = mask.astype(np.float64)
    img = img.astype(np.float64)

    dims = img.shape
    pixellist = np.where(mask.ravel()==1)
    QD = np.zeros(dims,dtype=int)*0
    QD.ravel()[pixellist] = qd
    QS = np.copy(mask)*0
    QS.ravel()[pixellist] = qs

    maskr = np.copy(mask)*0

    nphis = len(phis)

    sqcphi = np.zeros((len(sq_q),nphis))
    imgr = np.copy(img)*0

    if(y0 is None):
        ceny = dims[0]/2.
    else:
        ceny = y0

    if(x0 is None):
        cenx = dims[1]/2.
    else:
        cenx = x0

    #predefined parameters for rotation
#    x = np.arange(dims[1])
#    y = np.arange(dims[0])
#    X,Y = np.meshgrid(x,y)
    for i in np.arange(nphis):
        if(PF is True):
            print("Computing delta phi correlation, iteration {} of {}".format(i,len(phis)));
        imgr *= 0.
        #Should use my plugin for rotation when possible (10x speedup)
        rotate(mask,maskr,phis[i],cenx,ceny)
        #check rotated pixels that overlap with existing pixels
        wr = np.where((mask*maskr) != 0)
        nr = np.bincount(QD[wr])
        w = np.where(nr != 0)

        rotate(img, imgr,phis[i], cenx, ceny)
        if(plot == 1):
            plt.cla()
            plt.imshow(imgr)
            plt.draw()
            plt.clim(0,10)
            plt.pause(.0001)

        sqc = np.bincount(QD[wr],weights=img[wr]*imgr[wr])[w]/nr[w]
        sqc_q = np.bincount(QD[wr],weights=QS[wr])[w]/nr[w]

        if(len(sqc_q) > 0):
            sqcphi[:,i] = np.interp(sq_q,sqc_q,sqc)
        #print("Iteration {} of {}".format(i,phis.shape[0]))
    return sqcphi

def compute_sqcphi2(img,mask,qd,qs, sq_q, phis,plot=0):
    ''' Compute the qphi correlation on an image img
        imgs - numframes by x by y
        Need: pxlist, idpxlist, ids, img, numperid, numids
        Make: phis - go from 0 to pi for now
        plot (default 1) : plot the calculation as it comes along
            (note: assumes pyqtgraph)
    '''
    dims = img.shape
    pixellist = np.where(mask.ravel()==1)[0]
    QD = np.zeros(dims,dtype=int)*0
    QD.ravel()[pixellist] = qd
    QS = np.copy(mask)*0
    QS.ravel()[pixellist] = qs

    mask2 = np.copy(mask)*0
    mask3 = np.copy(mask)*0
    mask4 = np.copy(mask)*0

    img2 = np.zeros(mask.shape)
    img3 = np.zeros(mask.shape)
    img4 = np.zeros(mask.shape)

    nphis = len(phis)

    sqcphi = np.zeros((len(sq_q),nphis,nphis))
    imgr = np.copy(img)*0

    ceny = dims[0]/2.
    cenx = dims[1]/2.

    #predefined parameters for rotation
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    X,Y = np.meshgrid(x,y)
    for i in np.arange(nphis):
        #sample 2 will be rotated
        rotate(mask,mask2,phis[i],ceny,cenx)
        rotate(img,img2,phis[i],ceny,cenx)
        #now q phi 4 corr
        for j in np.arange(nphis):
            #rotate both samples by dphi
            rotate(mask, mask3,phis[j], ceny,cenx)
            rotate(mask2,mask4,phis[i] + phis[j], ceny,cenx)
            imgr *= 0.
            #Should use my plugin for rotation when possible (10x speedup)
            #check rotated pixels that overlap with existing pixels
            wr = np.where((mask*mask2*mask3*mask4) != 0)
            nr = np.bincount(QD[wr])
            w = np.where(nr != 0)
    
            rotate(img, img3,phis[j], ceny, cenx)
            rotate(img2, img4,phis[i] + phis[j], ceny, cenx)

            if(plot == 1):
                plt.cla()
                plt.imshow(imgr)
                plt.draw()
                plt.clim(0,1e6)
                plt.pause(.0001)
    
            sqc = np.bincount(QD[wr],weights=img[wr]*img2[wr]*img3[wr]*img4[wr])[w]/nr[w]
            sqc_q = np.bincount(QD[wr],weights=QS[wr])[w]/nr[w]
            
            sqcphi[:,i,j] = np.interp(sq_q,sqc_q,sqc)
            #print("Iteration {0},{1} of {2},{2}".format(i,j,nphis))
    return sqcphi
