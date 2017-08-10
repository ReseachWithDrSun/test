from tools.circavg import circavg,circavg2
from fit.linfit import linfit,linfitfunc
import numpy as np
import matplotlib.pyplot as plt

def compute_contrast(IMG,mask=None,x0=None,y0=None,plotwin=None,ptskip=None):
    ''' Quickly compute the contrast for an image. Note this only works in the
        case of a static image. If there are dynamics, the contrast will change with
    q.'''
    # get a quick measure of contrast versus q (no binning)
    Ints, Ints2 = compute_contrast_stats(IMG,mask=mask,x0=x0,y0=y0)

    #now fit for the contrast
    xdata = 1/Ints
    ydata = Ints2/Ints**2
    yfit,pars = linfit(xdata,ydata,a=1.,b=1.,plotwin=plotwin)
    if(plotwin is not None):
        plt.xlabel("$1/<I>$")
        plt.ylabel("$<I^2>/<I>^2$")

    print("Fit pars: a={} ; b={}".format(pars['a'].value,pars['b'].value))
    print("Contrast is beta = {}".format(pars['a'].value-1))
    print("Your deviation from shot noise statistics is = {}% (how far b is from 1)".format(100*np.abs((pars['b'].value-1))))
    return [pars['a'].value, pars['b'].value]

def compute_contrast_stats(IMG,mask=None,x0=None,y0=None,ptskip=None):
    ''' get the stats needed for the computation of an image contrast.
        currently I'm skipping a lot of data to limit the number of data points plotted.
        This is meant more to be qualitative.
        Set ptskip = 0 to get all points
    '''
    if(ptskip is None):
        ptskip = 1000
    SIMG = IMG*0
    SIMG2 = IMG*0
    sqx,sqy = circavg(IMG,x0=x0,y0=y0,mask=mask,SIMG=SIMG)
    sq2x,sq2y = circavg2(IMG,IMG,x0=x0,y0=y0,mask=mask,SIMG=SIMG2)
    
    pxlst = np.where(mask.ravel() != 0)[0][::ptskip]
    Ints2 = SIMG2.ravel()[pxlst]
    Ints = SIMG.ravel()[pxlst]
    return Ints, Ints2
