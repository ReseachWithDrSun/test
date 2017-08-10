import numpy as np
import h5py
from tools.MaskCreator import MaskCreator
#import matplotlib.pyplot as plt
'''Xray scattering tools.'''

class ExpPara:
    '''Prototpype class to add experimental parameters for xray scattering.'''
    pass

class CircAvgObj:
    '''A circular average object for 2D images. Do simple things like look at
    circular average etc. Basically makes a partition to compute the circular
    average on.
    (x0,y0) beam center
    res - resolution (qperpixel for example)
    NOTE: Still work in progress. First developing SAXS object, then will move
    bits and pieces into SAXS object.
    '''
    def __init__(self,img,x0=None,y0=None,resolution=1.,dp=1):
        self.img = img
        self.dims = img.shape
        if(x0 is None):
            x0 = self.dims[0]/2
        if(y0 is None):
            y0 = self.dims[0]/2

        dimx,dimy = self.dims
        x = np.arange(dimx)
        y = np.arange(dimy)
        X,Y = np.meshgrid(x,y)
        self.R = np.sqrt((X-x0)**2 + (Y-y0)**2).ravel()
        # by default we have pixel resolution qlists
        # us 
        if(dp==1):
            self.Rd = (self.R+.5).astype(int).ravel()
        else:
            self.Rd = np.searchsorted(self.rlist,self.R,side="right")

    def partition(self,qlist):
        ''' Partition according to the qlist in pixels.
        '''


    def circavg(self):
        '''Perform circular average.'''
        self.sq_q,self.sq = self.binavg(self.Rd,self.R,self.img.ravel())

    def binavg(self,xd,x,y,noperxd=None):
        ''' Average over digitized values xd with values x and weights y.
            If noperxd not given, it is calculated again. Note
            The code will be faster if this is specified.'''
        noperxd = np.bincount(xd).astype(float)
        w = np.where(noperxd != 0)
        xvals = np.bincount(xd,weights=x)[w]/noperxd[w]
        yvals = np.bincount(xd,weights=y)[w]/noperxd[w]
        return xvals,yvals

class SAXSObj:
    '''This is a SAXS object. The idea is to have all 
    components to manipulate SAXS data here:
    1. image reading
    2. coordinate mapping (small angle approximation)
    3. Data plotting
    4. Circular averaging
        Expected from the detector:
        det.pxdimx
        det.pxdimy
        sdet.beamx0
        sdet.beamy0
        sdet.wavelength
        sdet.det_distance #sample det distance
        sdet.exposuretime
        sdet.timeperframe
        sdet.nframes
        sdet.dimx
        sdet.dimy
        If things are behaving strangely, set PF=1 first to have a more verbose
        output.
    '''
    def __init__(self,expt,det,PF=0):
        '''Takes an experiment and detector object to create a saxs object
        '''
        #customization parameters
        self.PF = PF

        #Grab experimental parameters
        self.mask_filename = expt.mask_filename
        if(self.mask_filename is not None):
            self.loadmask(expt.mask_filename)
        else:
            self.mask = np.ones(det.dims)
        self.dsetname = expt.dsetname

        #Add reference to detector object (See doc for expected variables)
        self.det                = det
        self.xcen               = det.beamx0
        self.ycen               = det.beamy0

        #parameters for the object
        self.qlist              = None# define to none first
        self.calcqphi()
        self.sq                 = None
        self.sq_q               = None
        #self.get_avg_img()

    def calcqphi(self):
        '''Calc qphi coordinates the the image. Need to supply a mask with the
            dimensions of the image.
           asym : the asymmetry between x and y pixels. Normally, we'll have square pixels but this 
            should take asymmetric pixels into account.
            '''
        if(self.PF==1):
            print("Calculating the qphi coordinates.")
        #Calculate the pixels selected from the mask
        self.pixellist = np.where(self.mask.ravel() == 1)[0]
        #now calculate q and phi (Assumeing small angle)
        #4*pi/lambda *sin(2th/2.) where 2th = pxdim/det_distance
        self.qperpixel = 4*np.pi/self.det.wavelength*(self.det.pxdimx/2./self.det.det_distance)
        self.asym = self.det.pxdimy/self.det.pxdimx

        x = np.arange(self.det.dims[0])
        y = np.arange(self.det.dims[1])
        X,Y = np.meshgrid(x,y)
        xs,ys = X.ravel()[self.pixellist],Y.ravel()[self.pixellist]
        self.qs = np.sqrt((xs-self.xcen)**2 + (ys-self.ycen)**2)
        #avoid divide by zero
        quotient = xs-self.xcen
        w = np.where(quotient == 0)
        quotient[w] = 1
        #compute phi and correct for the 0 parts
        self.phis = np.arctan((ys-self.ycen)/quotient)
        self.phis[w] = np.pi/2.

    def get_avg_img(self):
        ''' Get average image from detector.'''
        if(self.PF==1):
            print("Obtaining the average image.")
        self.avg_img = np.zeros(self.det.dims)
        cnt = 0.
        for img in self.det:
            self.avg_img += img
            cnt += 1
        self.avg_img /= cnt

    def reflectimg(self,img):
        ''' Take regions that are masked and swap them out with their
            reflected counterpart, if available.
            Prototype, not finished.'''
        w = where(self.mask.ravel() == 0)
        reflectmask = np.copy(self.mask)
        #rotate image coords by pi around x0, y0
        #also x,y
        #Xr,Yr
        for i in np.arange(w):
            if(self.mask[yr[w],xr[w]] == 1):
                #exists so dd it in
                reflectmask[y[w],x[w]] = 1

    def mkqlist(self, dq=None, minq=None, maxq=None):
        '''Make a qlist from qs in terms of number of qs of width dq.
        Optional arguments: minq and maxq the min and max q to go over.
        Defaults to min(qs) and max(qs). The default is to use a 1 pixel
        qring.'''
        if(dq is None):
            dq = 1.
        if(minq is None):
            minq = np.min(self.qs)
        if(maxq is None):
            maxq = np.max(self.qs)
        Nq = int((maxq-minq)/dq)
        if(self.PF==1):
            print("Making qlist from {:3.1f} ({:3.1e}) to {:3.1f} ({:3.1e}) pixels (inv angs) with delta q {:3.1f}({:3.1e}) pixels (inv angs).".format(minq,\
                            minq*self.qperpixel,maxq,maxq*self.qperpixel,dq,dq*self.qperpixel))
        self.qlist = np.linspace(minq,maxq,Nq+1)
    
    def qpartition(self):
        ''' Partition the qs indices in terms of digitized indexes
            using qlist as the bins.
            Note: You must set qlist first.
            '''
        if(self.PF==1):
            print("Partitioning into q rings")
        if(self.qlist is None):
            print("Sorry, cannot qpartition, qlist is empty")
        else:
            self.qd = np.searchsorted(self.qlist,self.qs,side="right")
            self.noperq = np.bincount(self.qd).astype(float)
            self.noqs = self.noperq.shape[0]

    def qbinavg(self):
        '''Perform circular average.'''
        self.sq_q,self.sq = self.binavg(self.qd,self.qs,self.avg_img.ravel()[self.pixellist],noperxd=self.noperq)

    def binavg(self,xd,x,y,noperxd=None):
        ''' Average over digitized values xd with values x and weights y.
            If noperxd not given, it is calculated again. Note
            The code will be faster if this is specified.'''
        if(self.noperq is None):
            noperxd = np.bincount(xd).astype(float)
        w = np.where(noperxd != 0)
        xvals = np.bincount(xd,weights=x)[w]/noperxd[w]
        yvals = np.bincount(xd,weights=y)[w]/noperxd[w]
        return xvals,yvals

    def loadmask(self,maskfile):
        ''' Load the mask and incorporate the blemish file
            if necessary.'''
        f = h5py.File(maskfile,"r")
        self.mask = np.array(f['mask'])
        f.close()

    def updatemask(self,winnum=None):
        '''Update the mask with mask creator.'''
        self.mcreator = MaskCreator(data=self.avg_img,inmask=self.mask,winnum=winnum)

    '''Plotting routines. 
    plotsq
    plotlimg
    plotring
    plotcen'''
    def plotsq(self,winnum=0):
        ''' Plot the circular average. Winnum is window (default 0)'''
        plt.figure(winnum)
        plt.gcf().ax.loglog(self.sq_q*self.qperpixel,self.sq)

    def plotlimg(self,winnum=0,plotcen=0):
        ''' Plot the average image log.'''
        limg = np.copy(self.avg_img)*0
        w = np.where(self.avg_img != 0)
        limg[w] = np.log10(self.avg_img[w])
        fig = plt.figure(winnum);
        plt.cla();
        #self.fig.ax.clear()
        fig.ax.autoscale(tight=True)
        im = fig.ax.imshow(limg*self.mask)
        if(plotcen != 0):
            fig.ax.plot(self.xcen,self.ycen,'g+',markersize=18,mew=3)
        im.set_clim(0,3)

    def plotring(self,dq):
        ''' Plot a ring of width dq (in pixels)'''
        th = np.linspace(0., 2*np.pi, 1000)
        plt.gca().plot(dq*np.cos(th)+self.xcen,dq*np.sin(th)+self.ycen)

    def plotcen(self):
        ''' Plot the center of the scattering image.'''
        plt.gca().plot(self.xcen,self.ycen)

    '''End plotting routines.'''
