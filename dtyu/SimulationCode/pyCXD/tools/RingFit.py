from fit.FitObject import *
from fit.fitfns import ring2Dfitfunc
import matplotlib.pyplot as plt
import time

class RingFit:
    '''This class is meant to attempt to fit a 2D image to a ring.
    '''
    def __init__(self,img,mask=None,winnum=None):
        '''Set up the ring fit routine.'''
        if(winnum is None):
            winnum = 0

        self.img = img 
        x = np.arange(img.shape[1])
        y = np.arange(img.shape[0])
        X,Y = np.meshgrid(x,y)

        if(mask is None):
            self.mask = np.ones(img.shape)
        else:
            self.mask = mask

        pixels          = np.where(self.mask.ravel() == 1)
        pixelx, pixely  = X.ravel()[pixels],Y.ravel()[pixels]
        self.xy         = np.array([pixelx,pixely])

        data            = img.ravel()[pixels]
        pars            = Parameters()
        pars.add('amp', 1000, min=300, max=2000)
        pars.add('r0', 100., min=10, max=500)
        pars.add('sigma', 2., min=.1, max=10)
        pars.add('x0', img.shape[0]/2., min=0, max=img.shape[1]-1)
        pars.add('y0', img.shape[1]/2., min=0, max=img.shape[0]-1)
        fitobj = FitObject(fn=ring2Dfitfunc,pars=pars)
        self.done = True

        self.fig = plt.figure(winnum)
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(img*self.mask)

    def start(self):
        '''Routine to start a fit. Asks for user to input clicks.'''
        print("Starting the centering routine")
        print("Shortcuts: +/- (increase decrease brightness)")
        print("Please choose the approximate beam center")
        self.done = False
        #connect clicks to stuff
        self.cid = self.fig.canvas.mpl_connect("button_press_event",self.clickpoint)
        plt.draw()
        plt.pause(.001)

    def clickbeam(self,ev):
        '''get beam center upon click.'''
        res = clickpoint(ev)
        if(res == True):

    def clickrad(self):
        '''get radius upon click.'''

    def clickpoint(self,ev):
        '''This will be used to interact with matplotlib upon click. Return a
        value and disconnect cid'''
        print(ev.button)
        if(ev.button == 1):
            self.xclick,self.yclick = ev.xdata,ev.ydata
            self.done = True
            print("Got the beam cen: ({},{})",self.xclick,self.yclick)
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = self.fig.canvas.mpl_connect('button_press_event',self.clickrad)
