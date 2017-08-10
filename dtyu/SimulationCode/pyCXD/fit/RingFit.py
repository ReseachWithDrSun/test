from fit.fitfns import ring2Dfitfunc
from fit.FitObject import FitObject,minimize,Parameters

class RingFitObject(FitObject):
    '''A class to fit a ring for detector calibration.'''
    def __init__(self,fn=None,pars=None,x0=None,y0=None,r0=None,sigma=None,IMG=None,mask=None):
        self.mask = mask
        self.IMG = mask

        self.pars = Parameters()
        self.pars.add('amp', 2,min=0,max=10)
        self.pars.add('bg', 1)
        self.pars.add('bgexp', 4.,min=1.,max=4.)
        self.pars.add('r0', 1500,min=1000,max=2000)#,vary=False)
        self.pars.add('sigma', 2,min=.1, max=100)#,vary=False)
        self.pars.add('x0',XCEN,min=XCEN-10,max=XCEN+10)
        self.pars.add('y0',YCEN,min=YCEN-10,max=YCEN+10)
        super(RingFit,self).__init__(fn=fn,pars=pars)

    #setting parameters (can type rngfitobj.set and hit tab to see them)
    def setr0(self,r0,min=None,max=None,vary=None):
        self.pars['r0'].value = r0
        if(min is not None):
            self.pars['r0'].min = min
        if(max is not None):
            self.pars['r0'].max = max
        if(vary is not None):
            self.pars['r0'].vary = vary

    def setamp(self,amp,min=None,max=None,vary=None):
        self.pars['amp'].value = amp
        if(min is not None):
            self.pars['amp'].min = min
        if(max is not None):
            self.pars['amp'].max = max
        if(vary is not None):
            self.pars['amp'].vary = vary

    def setx0(self,x0,min=None,max=None,vary=None):
        self.pars['x0'].value = x0
        if(min is not None):
            self.pars['x0'].min = min
        if(max is not None):
            self.pars['x0'].max = max
        if(vary is not None):
            self.pars['x0'].vary = vary

    def sety0(self,y0,min=None,max=None,vary=None):
        self.pars['y0'].value = y0
        if(min is not None):
            self.pars['y0'].min = min
        if(max is not None):
            self.pars['y0'].max = max
        if(vary is not None):
            self.pars['y0'].vary = vary

    def setsigma(self,sigma,min=None,max=None,vary=None):
        self.pars['sigma'].value = sigma
        if(min is not None):
            self.pars['sigma'].min = min
        if(max is not None):
            self.pars['sigma'].max = max
        if(vary is not None):
            self.pars['sigma'].vary = vary

    def setbg(self,bg,min=None,max=None,vary=None):
        self.pars['bg'].value = bg
        if(min is not None):
            self.pars['bg'].min = min
        if(max is not None):
            self.pars['bg'].max = max
        if(vary is not None):
            self.pars['bg'].vary = vary

    def setbgexp(self,bgexp,min=None,max=None,vary=None):
        self.pars['bgexp'].value = bgexp
        if(min is not None):
            self.pars['bgexp'].min = min
        if(max is not None):
            self.pars['bgexp'].max = max
        if(vary is not None):
            self.pars['bgexp'].vary = vary

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
