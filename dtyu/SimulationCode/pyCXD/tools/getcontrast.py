#tool to look at counting stats and get the contrast
import numpy as np

# for the circular averaging
from tools.circavg import circavg

# for the coordinate system
from tools.polar import mkpolar


from fit.FitObject import FitObject, Parameters
from fit.fns.distributions import negbinomialdist, poissondist

def histqrings(IMG, x0=None,y0=None,mask=None,PF=False,DYN_RANGE=None):
    ''' Take an image with integer number of counts and obtain
        a histogram of counts per q ring.
        PF : the print flag
        x0, y0 : the beam x and y center
        mask : the mask
        DYN_RANGE : change dynamic range of detector
    '''
    # the dynamic range of the detector
    if(DYN_RANGE is None):
        DYN_RANGE = 10000
    if(mask is None):
        print("Warning, mask not set.")
        mask = np.ones(IMG.shape)
    if(x0 is None):
        x0 = IMG.shape[1]/2.
    if(y0 is None):
        y0 = IMG.shape[0]/2.

    # look for speckle
    QS, PHI = mkpolar(IMG,x0=x0,y0=y0)
    QD = QS.astype('int64')
    # only get pixels in mask and remove the Q=0 pieces near center
    pxlst = np.where(mask.ravel()*(QD.ravel() > 0))
    # get the q val per bin
    qx = np.bincount(QD.ravel()[pxlst],weights=QD.ravel()[pxlst])/(np.maximum(1,np.bincount(QD.ravel()[pxlst])))

    histstot = np.zeros((len(qx),DYN_RANGE))

    for j in range(len(qx)):
        if(PF):
            print("Counting rings: iteration {} of {}".format(j,len(qx)))
        subpxlst = np.where(mask.ravel()*(QD.ravel() > 0)*(QD.ravel()==j))
        hists = np.bincount(IMG.ravel()[subpxlst]);
        mnind = np.min([len(hists),DYN_RANGE])
        histstot[j,0:mnind] += hists[:mnind]

    histstot /= np.sum(histstot,axis=1)[:,np.newaxis];

    return qx, histstot


def histfitqrings(histstot,qx,Ix=None,PF=False):
    ''' fit the hists'''
    histspar = np.zeros((histstot.shape[0],2))
    histsparpois = histspar*0
    histsfit = histstot*0
    histsfitpois = histstot*0

    for i in range(histstot.shape[0]):
        if(PF):
            print("iteration {} or {}".format(i,histstot.shape[0]))
        #kbar = np.interp(i,sqx,sqy)
        pars = Parameters()
        pars.add('M',value=4,vary=True)
        if(Ix is not None):
            pars.add('kbar',value=Ix[i],vary=False)
        else:
            pars.add('kbar',value=2.,vary=True)
        xdata = np.arange(histstot.shape[1])
        ydata = histstot[i];
        fitfn = FitObject(negbinomialdist,pars)
        fitfnpois = FitObject(poissondist,pars)
        fitfn.fit(xdata,ydata)
        fitfnpois.fit(xdata,ydata)
        pars = fitfn.mn.params
        parspois = fitfnpois.mn.params
        histsfit[i] = fitfn.fn(xdata,pars)
        histsfitpois[i] = fitfnpois.fn(xdata,pars)
        histspar[i] = np.array([pars['kbar'].value,pars['M'].value])
        histsparpois[i] = np.array([parspois['kbar'].value,parspois['M'].value])
    return histsfit, histspar, histsfitpois, histsparpois
