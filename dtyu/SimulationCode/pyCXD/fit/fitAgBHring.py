#method to fit an AgBH ring (or similar ring) in data
CONST_AgBHpeak = 58.380# in angs
from plot.plcirc import plcirc
from fit.FitObject import FitObject, Parameters
from fit.fitfns import ring2Dfitfunc, ring2Dlorentzfitfunc
import numpy as np
from tools.circavg import circavg
import matplotlib.pyplot as plt

def fitAgBHring(IMG,mask=None,pars=None,plotwin=None,clims=None):
    ''' Fit an AgBH (silver behenate) ring
        plotwin : if set, plot to the window desired.
        pars : a Parameters object of the initial guess Parameters. If not set
            the program will ask you for a best guess.
        mask : a mask
        clims : if specified, these will be the color limits of the image plot windows
            default is to take min and max of *masked* image
    '''
    qAGBH = 2*np.pi/CONST_AgBHpeak
    if(plotwin is not None):
        plt.figure(plotwin)
        if(mask is None):
            mask = np.ones(IMG.shape)
        img = IMG*(mask + .1)
        plt.imshow(img)
        if(clims is None):
            plt.clim(np.min(img*mask),np.max(img*mask))
        plt.draw();plt.pause(.001)
    
    pixellist = np.where(mask.ravel() == 1)
    print("\n\nfitAgBH: An attempt to fit the first silver behenate ring on a 2D area det")
    print("Note if plotting, the masked region is brightened w.r.t. the rest")
    print("Some tips: \n \
           1. Try to set mask to only select region near where the ring is\n\
                for a better fit\n\
            2. Try a fit without the parasitic bg or const scattering first.\n\
        ")
    
    if(pars is None):
        pars = Parameters()
        print("You did not specify parameters, so I will ask you a few questions");
        print("Specify the variables in one of two ways, either:")
        print(": value, min, max")
        print("or")
        print(": value")
        #it is "raw_input" in python 2x


        xcenvals = np.array(input("beam center x position guess: ").split(",")).astype(float)
        pars.add('x0',xcenvals[0],min=xcenvals[0]-40,max=xcenvals[0]+40)
        if(len(xcenvals) == 3):
            pars['x0'].min = xcenvals[1]
            pars['x0'].max = xcenvals[2]

        ycenvals = np.array(input("beam center y position guess: ").split(",")).astype(float)
        pars.add('y0',ycenvals[0],min=ycenvals[0]-40,max=ycenvals[0]+40)
        if(len(ycenvals) == 3):
            pars['y0'].min = ycenvals[1]
            pars['y0'].max = ycenvals[2]

        if(plotwin is not None):
            print("Plotted estimate of S(q) from given xcen, ycen");
            sqx,sqy = circavg(IMG,x0=xcenvals[0],y0=ycenvals[0],mask=mask)
            plt.figure(plotwin+2);plt.cla();
            w = np.where(sqy > 0)
            plt.loglog(sqx[w],sqy[w]);

        ampvals = np.array(input("amplitude of ring: ").split(",")).astype(float)
        pars.add('amp',ampvals[0],min=0,max=1e9)
        if(len(ampvals) == 3):
            pars['amp'].min = ampvals[1]
            pars['amp'].max= ampvals[1]

        ringvals = np.array(input("Ring location (in pixels from center of beam, approx):").split(",")).astype(float)
        pars.add('r0',ringvals[0],min=0,max=4*ringvals[0])
        if(len(ringvals) == 3):
            pars['r0'].min = ringvals[1]
            pars['r0'].max = ringvals[2]

        sigmavals = np.array(input("sigma of ring (FWHM for Lorentzian) :").split(",")).astype(float)
        pars.add('sigma', sigmavals[0],min=.1, max=100)#,vary=False)
        if(len(sigmavals) == 3):
            pars['sigma'].min = sigmavals[1]
            pars['sigma'].max = sigmavals[2]

            
        bgvals = input("1/q^4 background (enter nothing to not vary): ")
        if(len(bgvals) == 0):
            pars.add('bg',0,vary=False)
            pars.add('bgexp', 4.,vary=False)
        else:
            bgvals = np.array(bgvals.split(",")).astype(float)
            pars.add('bg', bgvals[0],vary=False)
            pars.add('bgexp', 4.,vary=True,min=1,max=4)
            if(len(bgvals) == 3):
                pars['bg'].min = bgvals[1]
                pars['bgexp'].max = bgvals[2]
        constvals = input("Constant background (enter nothing to not vary):")
        if(len(constvals) == 0):
            pars.add('const', 0, vary=False)
        else:
            constvals = np.array(constvals.split(",")).astype(float)
            pars.add("const",constvals[0])
            if(len(constvals) == 3):
                pars['const'].min = constvals[1]
                pars['const'].max = constvals[2]
    
    data = IMG.ravel()[pixellist]
    
    fitobj = FitObject(fn=ring2Dlorentzfitfunc,pars=pars)
    
    x = np.arange(IMG.shape[1])
    y = np.arange(IMG.shape[0])
    X,Y = np.meshgrid(x,y)
    xy = np.array([X.ravel()[pixellist],Y.ravel()[pixellist]])
    
    erbs = np.ones(len(pixellist[0]))
    
    datatest = np.zeros(IMG.shape)
    
    fitobj.fit(xy,data,erbs=erbs)
    
    datatest.ravel()[pixellist] = fitobj.fn(xy,fitobj.pars)
    
    XCENFIT = fitobj.pars['x0'].value
    YCENFIT = fitobj.pars['y0'].value
    RFIT = fitobj.pars['r0'].value
    SIGMAFIT = fitobj.pars['sigma'].value
    CONSTFIT = fitobj.pars['const'].value
    BGFIT = fitobj.pars['bg'].value
    BGEXPFIT = fitobj.pars['bgexp'].value
    print("x center fit: {:3.2f}; y center fit: {:3.2f}; radius fit: {:3.2f}".format(XCENFIT,YCENFIT,RFIT))
    print("ring FWHM : {:3.2f}; const background: {:3.2f}; parasitic bg: {:3.2f}/q^{:3.2f}".format(SIGMAFIT,CONSTFIT,BGFIT,BGEXPFIT))
    sqd_x,sqd_y = circavg(IMG,x0=XCENFIT,y0=YCENFIT,mask=mask)
    sqf_x,sqf_y = circavg(datatest,x0=XCENFIT,y0=YCENFIT,mask=mask)
    if(plotwin is not None):
        plt.figure(plotwin);plt.cla();
        plt.imshow(IMG*mask);
        if(clims is None):
            plt.clim(np.min(img*mask),np.max(img*mask))
        plcirc([XCENFIT,YCENFIT],RFIT,'b')
        #plt.clim(np.min(IMG*mask),np.max(IMG*mask))

        plt.figure(plotwin+1);plt.cla();
        plt.imshow(datatest*mask);
        if(clims is None):
            plt.clim(np.min(img*mask),np.max(img*mask))
        #plt.clim(np.min(IMG*mask),np.max(IMG*mask))
        plcirc([XCENFIT,YCENFIT],RFIT,'b')

        plt.figure(plotwin+2);plt.cla();
        plt.loglog(sqd_x,sqd_y)
        plt.loglog(sqf_x,sqf_y,'r')
        plt.gca().autoscale_view('tight')
        plt.draw()
        plt.pause(0.001)
    resp = input("Get det distance?(y for yes, anything else for no): ")
    if(resp == 'y'):
        wv = float(input("wavelength (in angs): "))
        dpix = float(input("pix size (in m): "))
        L,err = getdistfrompeak(qAGBH,wv,dpix,RFIT);
        print("Your length is approximately {:4.2f} m, and error in calc (from Ewald curvature) approx {:4.2f}%".format(L,err))
    
    print("done")
    


def getdistfrompeak(q,wv,dpix,qpix):
    ''' Get the SAXS distance from a measured peak from:
            q - the value of the known peak (in inv angs)
            wv- wavelength (in angs)
            dpix - pixel size in m
            qpix - number of pixels the ring is located on detector.
        returns L,err:
            L - distance (m)
            err - estimated error from Ewald sphere curvature (should be less than
                1% or else don't trust this)
    '''
    a = dpix*qpix
    L = a/(2*np.arcsin(q*wv/4/np.pi))
    at1 = a/L
    at2 = np.arctan(at1)
    err = at1/at2 - 1
    return L, err
