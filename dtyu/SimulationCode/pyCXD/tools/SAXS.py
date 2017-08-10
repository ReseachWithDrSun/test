from detector.eiger import EigerImages
from tools.mask import openmask, savemask
from tools.average import runningaverage
from tools.circavg import circavg
from tools.optics import calc_q

from fit.SchultzFitObject import SchultzFitObject, Parameters
from fit.fns.formfactors import sqsphereschultz

import matplotlib.pyplot as plt

import numpy as np

class SAXSObj:
    def __init__(self, filename, rdet=None, dpix=None, wv=None, xcen=None, ycen=None, maskfilename=None):
        ''' Initialize the SAXS object. Need some parameters:
                rdet : sample det distance (m)
                dpix : pixel dimensions (m)
                wv : wavelength
                xcen, ycen : x y coordinates of image (as plotted in matplotlib,
                    so the fastest varying index is x, : IMG[y][x])
                only mandatory parameter to begin with is the file name
        ''' 
        self.rdet = rdet
        self.dpix = dpix
        self.xcen = xcen
        self.ycen = ycen
        self.filename = filename
        self.IMGS = EigerImages(filename)
        self.maskfilename = maskfilename
        if(maskfilename is not None):
            self.mask = openmask(maskfilename)
        else:
            self.mask = np.ones(self.IMGS.dims)
        self.average_frames()
        #self.compute_sq()
        #self.qperpixel = calc_q(rdet,dpix,wv)
        self.fitobj = SchultzFitObject()
        self.winnum = 1006
        self.IMGfig = None
        self.sqfig = None
        self.masking = False
        self.IMGmin = None
        self.IMGmax = None
        self.sqfig = None

    def setmask(self, maskfilename):
        self.maskfilename = maskfilename
        self.mask = openmask(maskfilename)
    
    def average_frames(self):
        self.IMG, self.Ivst = runningaverage(self.IMGS)
    
    def compute_sq(self):
        self.sqx,self.sqy = circavg(self.IMG,x0=self.xcen,y0=self.ycen,mask=self.mask)

    def calcq(self):
        '''calculate the q perpixel for the saxs data. '''
        self.qperpixel = calc_q(self.rdet,self.dpix,self.wavelength)

    def plotsq(self):
        self.compute_sq()
        if(self.sqfig is None):
            self.sqfig = plt.figure(self.winnum+1)
            ax = self.sqfig.gca()
            ax.set_xscale('log')
            ax.set_yscale('log')

        self.sqfig.clf()
        self.sqfig.gca().loglog(self.sqx,self.sqy)
        plt.draw()
        

    def plotIMG(self,masking=None):
        if(masking is None):
            masking = self.masking

        if(self.IMGfig is None):
            self.IMGfig = plt.figure(self.winnum)

        self.IMGfig.clf()

        if(masking):
            self.IMGfigimg = self.IMGfig.gca().imshow(self.IMG*self.mask);
        else:
            self.IMGfigimg = self.IMGfig.gca().imshow(self.IMG);

        self.IMGfigimg.set_clim(self.IMGmin,self.IMGmax)
        plt.draw()
        
    
    def setparam(self,xcen=None,ycen=None,dpix=None,rdet=None,wavelength=None):
        if(xcen is not None):
            self.xcen = xcen
        if(ycen is not None):
            self.ycen = ycen
        if(dpix is not None):
            self.dpix = dpix
        if(rdet is not None):
            self.rdet = rdet
        if(wavelength is not None):
            self.wavelength = wavelength
