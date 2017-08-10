#sample fit to the fit object
#include a fit function
from fit.fitfns import spheresqfunc,sspheresqfunc,savitzky_golay,spherepolygauss
import numpy as np
from fit.FitObject import FitObject,Parameters
import matplotlib.pyplot as plt

plt.figure(0)
#initialize the parameters
pars = Parameters()
pars.add('amp',value=1,vary=False)
pars.add('radius',value=100,vary=True)
pars.add('window',value=21,vary=True)
ndata = 400
xdata = np.linspace(1e-4, .1, ndata)
ydata = spheresqfunc(xdata,pars) + np.random.random(ndata)*.01
erbs = xdata*0 + .01
fitfn = FitObject(fn=sspheresqfunc,pars=pars)
fitfn.fit(xdata,ydata,erbs)
fitfn.plot(xdata,ydata,logxy=[1,1])
fitfn.printpars()

plt.figure(1)
#Try a different distribution
pars = Parameters()
pars.add('amp',value=1,vary=True)
pars.add('radius',value=100,vary=True)
pars.add('FWHM',value=10,vary=True)
pars.add('parasitic',value=1e-3,vary=True)
pars.add('bg',value=0,vary=False)
ndata = 400
xdata = np.linspace(1e-4, .1, ndata)
ydata = spheresqfunc(xdata,pars) + np.random.random(ndata)*.01
erbs = xdata*0 + .01
fitfn = FitObject(fn=spherepolygauss,pars=pars)
fitfn.fit(xdata,ydata,erbs)
fitfn.fit(xdata,ydata,erbs)
fitfn.plot(xdata,ydata,logxy=[1,1])
fitfn.printpars()
print(fitfn.mn.success)
