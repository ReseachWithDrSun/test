from fit.fitfns import gauss2Dfitfunc
from fit.FitObject import FitObject,minimize,Parameters

class Gauss2DFitter(FitObject):
    '''A class to fit a Gaussian peak.
    '''
    def __init__(self,fn=None,pars=None,IMG=None,mask=None):
        self.mask = mask
        self.IMG = mask
        if(pars is None):
            pars = {    'amp'   : 1,
                        'x0'    : 0.,
                        'y0'    : 0,
                        'sigmax': 1.,
                        'sigmay': 1.,
                        'const' : 0
            }

        if(hasattr(pars['amp'],"value")):
            self.pars = pars
        else:
            self.pars = Parameters()
            for key in pars:
                self.pars.add(key, pars[key])

        super(Gauss2DFitter,self).__init__(fn=gauss2Dfitfunc,pars=pars)

    def fit(self,IMG=None,mask=None):
        ''' Fit the function. If not data is given use previous data.'''
        if(IMG is None):
            IMG = self.IMG
            if(self.IMG is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.IMG = IMG

        if(mask is None):
            mask = self.mask
            if(self.mask is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.mask = mask

        pixellist = self.pixellist
        x = np.arange(IMG.shape[1])
        y = np.arange(IMG.shape[0])
        X,Y = np.meshgrid(x,y)
        xy = np.array([X.ravel()[pixellist],Y.ravel()[pixellist]])
        
        erbs = np.ones(len(pixellist[0]))*1
        
        
        fitobj.fit(xy,data,erbs=erbs)
        
        
        super(Gauss2DFitter,self).fit(xdata,ydata,erbs=wdata)

    def setmask(self,mask):
        self.mask = mask
        self.pixellist = np.where(mask.ravel() == 1)

    def setIMG(self,IMG):
        self.IMG = IMG

    def fit(self,IMG=None,mask=None):
        ''' Fit the function. If not data is given use previous data.'''
        if(IMG is None):
            IMG = self.IMG
            if(self.IMG is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.IMG = IMG

        if(mask is None):
            mask = self.mask
            if(self.mask is None):
                print("Cannot fit, no data in fit object (please supply data)")
                return
        else:
            self.mask = mask

        pixellist = self.pixellist
        x = np.arange(IMG.shape[1])
        y = np.arange(IMG.shape[0])
        X,Y = np.meshgrid(x,y)
        xy = np.array([X.ravel()[pixellist],Y.ravel()[pixellist]])
        
        erbs = np.ones(len(pixellist[0]))*1
        
        
        fitobj.fit(xy,data,erbs=erbs)
        
        
        super(RingFit,self).fit(xdata,ydata,erbs=wdata)

    def getfit(self):
        ''' Get the current fitted image.'''
        pars = self.pars
        data = np.zeros(self.IMG.shape)
        data.ravel()[pixellist] = self.fn(xy,pars)
        data.ravel()[pixellist] = self.fn(xy,fitobj.pars)
        return data

    def getfitparams(self):
        return self.pars
