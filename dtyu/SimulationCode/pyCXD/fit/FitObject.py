#This fit object uses the Marquardt non linear least square fitting algorithm
#from scipy.optimize.leastsq
import matplotlib.pyplot as plt
import numpy as np
#This lib uses scipy.optimize.leastsq, Marquardt algorithm
#Only difference is it easily allows you to add/remove parameters
from lmfit import minimize,Parameters

from scipy.optimize import leastsq
class FitObject:
    def __init__(self,fn=None,pars=None,logxy=[0,0]):
        '''Initialize the fit object. 
        fn - the fit function
        pars - the parameters
        '''
        self.fn = fn
        self.pars = pars
        self.mn = None
        self.oldpars = None
        self.logxy = logxy

    def residual(self,pars,xdata,ydata,erbs=None):
        '''The residual.'''
        if(erbs is not None):
            return (self.fn(xdata,pars)-ydata)/erbs
        else:
            return (self.fn(xdata,pars)-ydata)

    def fit(self,xdata,ydata,erbs=None):
        ''' Fit the function using the error bars.'''
        if(self.mn is not None):
            self.oldpars = self.mn.params
        if(erbs is None):
            self.mn = minimize(self.residual,self.pars,args=(xdata,ydata))
        else:
            self.mn = minimize(self.residual,self.pars,args=(xdata,ydata,erbs))
        self.pars = self.mn.params

    def plot(self,xdata,ydata,logxy=None,winnum=None,color=None):
        '''Plot the fit.'''
        if(logxy is None):
            logxy = self.logxy
        plt.plot(xdata, ydata,'k')
        x = np.linspace(np.min(xdata),np.max(xdata),1000)
        plt.plot(x,self.fn(x,self.pars),color=color)
        ax = plt.gca()
        if(logxy[0] == 1):
            ax.set_xscale('log')
        else:
            ax.set_xscale('linear')
        if(logxy[1] == 1):
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')

    def printpars(self):
        '''Print the parameters. The parameters object has its own nice way to do this.'''
        print("Old parameters:")
        for key in self.pars:
            print(self.pars[key])
        if(self.mn is not None):
            print("New parameters:")
            for key in self.mn.params:
                print(self.mn.params[key])
