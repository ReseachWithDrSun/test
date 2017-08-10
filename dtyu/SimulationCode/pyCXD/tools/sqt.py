#sq versus time stuff
from tools.sumimgs import avgimgs
from tools.circavg import circavg
from detector.eiger import EigerImages
import numpy as np

def sqtfromlist(filelist,imgthresh=None,x0=None,y0=None,mask=None,DDIR='.'):
    ''' Get sq versus item for a list.'''
    for i in range(len(filelist)):
        IMGS = EigerImages(DDIR + "/" + filelist[i])
        IMG,ivfrm = avgimgs(IMGS,imgthresh=imgthresh)
        sqx, sqy = circavg(IMG,x0=x0,y0=y0,mask=mask)
        if(i == 0):
            sqt = np.zeros((len(filelist),len(sqx)))
        sqt[i] = sqy

    return sqx, sqt
