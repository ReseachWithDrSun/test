from fit.FitObject import FitObject, Parameters
from fit.fns.formfactors import sqsphereschultz

class SchultzFitObject(FitObject):
    ''' Initiate a Schultz fit object. Sets parameters to
        default parameters that tend to make sense in a SAXS setup.
        Units : 
            amp : amplitude
            rbar: avg radius in angstroms
            Z: schultz parameter
    '''
    def __init__(self,pars=None,xdata=None,ydata=None):
        #default is to plot in logxy 1,1
        self.logxy = [1,1]
        self.xdata = xdata
        self.ydata = ydata
        self.wdata = None
        if(pars is None):
            pars = Parameters()
            pars.add("amp",value=1.,vary=True)
            pars.add("rbar",value=500.,vary=True)
            pars.add("z",value=100.,vary=True)
            pars.add("parasitic",value=0.,vary=False)
            pars.add("bg",value=0.,vary=False)
        else:
            #If it's not in the Parameters object format, convert it
            if(not hasattr(pars["amp"],"value")):
                pars1 = pars
                pars = Parameters()
                for key in pars1:
                    pars.add(key,value=pars1[key])

        super(SchultzFitObject, self).__init__(fn=sqsphereschultz,pars=pars)
        
    def fit(self,xdata=None, ydata=None, wdata=None):
        ''' Fit the function. If not data is given use previous data.'''
        if(xdata is None):
            xdata = self.xdata
            if(self.xdata is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.xdata = xdata
        if(ydata is None):
            ydata = self.ydata
            if(self.ydata is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.ydata = ydata
        if(wdata is None):
            wdata = self.wdata
        else:
            self.wdata = wdata
        super(SchultzFitObject,self).fit(xdata,ydata,erbs=wdata)

    def plot(self,xdata=None,ydata=None,wdata=None,logxy=None,winnum=None,color=None):
        if(logxy is None):
            logxy = self.logxy
        else:
            self.logxy = logxy
        if(xdata is None):
            xdata = self.xdata
            if(self.xdata is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.xdata = xdata
        if(ydata is None):
            ydata = self.ydata
            if(self.ydata is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.ydata = ydata
        if(wdata is None):
            wdata = self.wdata
        else:
            self.wdata = wdata
        super(SchultzFitObject, self).plot(xdata, ydata, logxy=logxy,winnum=winnum,color=color)
