from fit.fitfns import linfitfunc
from fit.FitObject import FitObject, Parameters
import matplotlib.pyplot as plt
import numpy as np

def linfit(xdata,ydata,a=None,b=None,plotwin=None): 
    ''' quick lin fit func
        linear fit y = a + b*x
    
        If plot win set, plot results in a window'''
    if a is None:
        a = 1
    if b is None:
        b = 1.

    pars = Parameters()
    pars.add('a' , value= a, vary= True)
    pars.add('b' , value = b, vary= True)
    fitfn = FitObject(fn=linfitfunc,pars=pars)
    fitfn.fit(xdata,ydata)
    yfit = linfitfunc(xdata,fitfn.pars)
    if(plotwin is not None):
        plt.figure(plotwin)
        plt.plot(xdata,ydata,'ko')
        #compute with finer spacing
        xfit = np.linspace(np.min(xdata),np.max(xdata));
        yfit2 = linfitfunc(xfit,fitfn.pars)
        plt.plot(xfit,yfit2,'r')
    return yfit, fitfn.pars
