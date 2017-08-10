from fit.fitfns import polyfitfunc
from fit.FitObject import FitObject, Parameters
import matplotlib.pyplot as plt
import numpy as np

def polyfit(xdata,ydata,a=None,b=None,varyexps=False,plotwin=None,positive=False): 
    ''' quick lin fit func
        poly fit y = sum(a[i]*x**b[i])

        note : assumes that the exponents are not varied. If that's ever necessary,
            set varyexps to True.
            Also, it is recommended you always set a and b, don't rely on default vals
                as they may change.

        If plot win set, plot results in a window'''
    if(a is None):
        a = [0, 1, 1]
    if(b is None):
        b = [0, 1, 2]
        
    if(positive is True):
        minval = 0
    else:
        minval = None

    pars = Parameters()
    for i in range(len(a)):
        pars.add('a{}'.format(i) , value=a[i], vary= True,min=minval)
        pars.add('b{}'.format(i) , value = b[i], vary= varyexps)
    fitfn = FitObject(fn=polyfitfunc,pars=pars)
    fitfn.fit(xdata,ydata)
    yfit = polyfitfunc(xdata,fitfn.pars)
    pars = fitfn.pars
    if(plotwin is not None):
        plt.figure(plotwin)
        plt.plot(xdata,ydata,'ko')
        #compute with finer spacing
        xfit = np.linspace(np.min(xdata),np.max(xdata));
        yfit2 = polyfitfunc(xfit,fitfn.pars)
        plt.plot(xfit,yfit2,'r')
    return yfit, pars
