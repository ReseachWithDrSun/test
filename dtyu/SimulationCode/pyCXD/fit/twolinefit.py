from fit.fitfns import twolinefitfunc
from fit.FitObject import FitObject, Parameters
import matplotlib.pyplot as plt
import numpy as np

def twolinefit(xdata,ydata,m1=None,m2=None,xc=None,yc=None,varyslopes=True,plotwin=None): 
    ''' Two line fit func
        fits to two lines with slopes m1 and m2 intersecting at point (xc, yc)
        m1= ,m2= : set m1 and m2 initial parameters (not necessary)
            Default is 1
        varyslopes= True : set to false to fix slopes (need to set m1 and m2 then)
        set plotwin to a window number to plot to that window number
    '''

    if(m1 is None):
        m1 = 1
    if(m2 is None):
        m2 = 1

    if(xc is None):
        xc = 0
    if(yc is None):
        yc = 0
        
    pars = Parameters()
    pars.add('m1', value = m1, vary= varyslopes)
    pars.add('m2', value = m2, vary= varyslopes)
    pars.add('xc', value = xc, vary= True)
    pars.add('yc', value = yc, vary= True)

    fitfn = FitObject(fn=twolinefitfunc,pars=pars)
    fitfn.fit(xdata,ydata)

    yfit = twolinefitfunc(xdata,fitfn.pars)
    if(plotwin is not None):
        plt.figure(plotwin)
        plt.plot(xdata,ydata,'ko')
        #compute with finer spacing
        xfit = np.linspace(np.min(xdata),np.max(xdata));
        yfit2 = twolinefitfunc(xfit,fitfn.pars)
        plt.plot(xfit,yfit2,'r')
    pars = fitfn.pars
    return yfit, pars
