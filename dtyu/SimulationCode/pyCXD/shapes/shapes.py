''' Make shapes of different geometries. Mainly desired for simulations.  
phis = np.linspace(0., np.pi,nphis)
    All shape generators must follow this convention:
    mkshape(pars,img) where pars are the parameters for the shape and 
    img is the image to place the shape into. Only square images have been tried for now.
'''
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.ndimage.interpolation import shift
from tools.circavg import circavg
from plugins.rotate._rotate import rotate
import matplotlib.pyplot as plt

#from shapes.shapefunctions import * 
#from plot.ZoomFigure import *
#from matplotlib import colors
#from tools.rotate import rotate
#from plugins.rotate._rotate import rotate
#import matplotlib.pyplot as plt

class ShapeObj:
    '''General shape object.
        N - image dimensions
        pars - pars to shape generator
        mkshape - the shape generator (mkshape(pars,img))
        pos - the positions of shapes currently added
        img - the image with shapes
        imgtmp - temporary image for the shapegeneration
        fftupdated - tells whether or not the fft was updated
        resolution - the resolution (in m) per pixel
                (resolution = 1 sets units to be pixels)
        unit is a text value of the units
        '''
    def __init__(self,mkshape,pars,N,resolution=1.,unit='rad*pixel'):
        self.resolution=resolution
        self.dims = (N,N)
        self.mkshapegrid()
        #threshold of region that makes nonzero density
        self.thresh = 1e-6
        #base shape params
        self.bshapeparams = pars
        self.mkshape = mkshape
        self.bshape = np.zeros(self.dims)#the base shape
        self.shapemask = np.zeros(self.dims,dtype='float64')
        self.shapemask2 = np.zeros(self.dims,dtype='float64')
        mkshape(self.bshape)
        self.shapeparams = []
        #make sure all arrays are float64 because rotation only works on this 
        # dtype for now
        self.img = np.zeros(self.dims,dtype='float64')
        #temp image to add in new shapes before rotation etc
        self.imgtmp = np.zeros(self.dims,dtype='float64')
        self.imgFFTre = np.zeros(self.dims,dtype='float64')
        self.imgFFTim = np.zeros(self.dims,dtype='float64')
        self.imgFFT2 = np.zeros(self.dims,dtype='float64')
        self.limgFFT2 = np.zeros(self.dims,dtype='float64')
        self.win = 0 #default plotting window
        #self.prerot = self.sub_rotate(self.dims)
        self.fftupdated = False
        self.unit = unit
        self.winnum = 1001
        self.qperpixel = 2*np.pi/N/resolution
        self.xperpixel = resolution

    def clear(self):
        ''' Clear all shapes in shape object'''
        self.img.fill(0)
        self.imgFFTre.fill(0)
        self.imgFFTim.fill(0)
        self.imgFFT2.fill(0)
        self.imgFFT2.fill(0)
        self.shapeparams = []
        self.fftupdated = False
        self.regen()

    def regen(self):
        '''regenerate the shape with the new parameters.'''
        self.bshape *= 0
        self.mkshape(self.bshape)

    def addshape(self,x,y,phi=None,rho=None,overlap=True):
        '''Add another shape with density rho at x0,y0, with rotated by phi.
            Note: ndimage.interpolation.shift interpolates.'''
        #shift is from ndimage
        self.sub_addshape(x,y,phi=phi,rho=rho)
        if(overlap is False): 
            self.shapemask = (self.img > self.thresh).astype('float64')
            self.shapemask2 = (self.imgtmp > self.thresh).astype('float64')
            w = np.where((self.shapemask+self.shapemask2) > 1.5)
            if(len(w[0]) > 0):
                self.imgtmp *= 0 
                self.shapemask *= 0
                self.shapemask2 *= 0
                return 0
            self.shapemask *= 0
            self.shapemask2 *= 0

        self.img += self.imgtmp
        self.imgtmp *= 0
        self.shapeparams.append([x,y,phi,rho])
        self.fftupdated = False
        return 1

    def remshape(self,nshp=None):
        '''Remove the nshp'th shape. If not supplied, remove last shape.'''
        #shift is from ndimage
        if(nshp is None):
            nshp = len(self.shapeparams)-1
        (x,y,phi,rho) = self.shapeparams[nshp]
        del self.shapeparams[nshp]
        self.sub_addshape(x,y,phi=phi,rho=rho)
        self.img -= self.imgtmp
        self.imgtmp *= 0
        self.fftupdated = False

    def sub_addshape(self,x,y,phi,rho):
        ''' Subroutine for adding the shape. Complicated statements because I'm
        trying to minimize the number of reads and writes to arrays. This
        should be investigated again by someone with a better knowledge of
        numpy (either future me or someone else)'''
        if(phi is not None):
            self.rotate(self.bshape,phi,self.imgtmp)
        else:
            self.imgtmp += self.bshape
        if(rho is None):
            #self.imgtmp = self.roll2(self.imgtmp, int(x+.5),int(y+.5))
            self.imgtmp = shift(self.imgtmp,[x,y])
        else:
            self.imgtmp = shift(self.imgtmp*rho,[x,y])
            #self.imgtmp = self.roll2(self.imgtmp*rho, int(x+.5),int(y+.5))

    def roll2(self,img,nx,ny):
        ''' Perform a numpy roll on both axes for a numpy roll object.'''
        return np.roll(np.roll(img,nx,axis=0),ny,axis=1)

    def shift(self,sh,mode='constant'):
        ''' Shift the image by an amount. By default pixels outside are set to zero.
            Set to mode='wrap' if you want the shift to wrap around. 
            sh - an (x,y) pair to shift by'''
        return scipy.ndimage.interpolation.shift(img,sh,mode=mode)

    def rotate(self,img,th,imgr,predef=None):
        '''Rotate a function using no interpolation. Should be faster.
            predef is a bunch of predefined variables. They are obtained by calling sub_rotate.
        '''
        xcen,ycen = img.shape[1]/2.,img.shape[0]/2.
        rotate(img,imgr,th,xcen,ycen)

#        R = np.array([
#            [np.cos(th), np.sin(th)],
#            [-np.sin(th), np.cos(th)]
#        ])
#    
#        imgr *= 0
#        dims = img.shape
#        #img = img.reshape(dims[0]*dims[1])
#        #imgr = imgr.reshape(dims[0]*dims[1])
#        if(predef is None):
#            r,x,y,cen = self.sub_rotate(dims)
#        else:
#            r,x,y,cen = predef
#    
#        rp = np.dot((r-cen),R) + cen#broadcast cen array
#        rp = rp.reshape((dims[0]*dims[1],2))
#        xbins = np.linspace(x[0]-.5,x[-1]+.5,dims[0]+1)
#        ybins = np.linspace(y[0]-.5,y[-1]+.5,dims[1]+1)    
#        xd = np.searchsorted(xbins,rp[:,0],side="right")-1
#        yd = np.searchsorted(ybins,rp[:,1],side="right")-1
#        w = np.where((xd >= 0) & (xd < dims[0]) & (yd >= 0) & (yd < dims[1]))
#        xd = xd[w]
#        yd = yd[w]
#        #rd = xd + yd*dims[0] 
#        #rr = r[w,0] + r[w,1]*dims[0]
#        imgr[r[w,0],r[w,1]] = img[xd,yd]
    
    def sub_rotate(self,dims):
        '''Subroutine to rotate. If you call rotate multiple times, use the
        parameters returned here as the predef parameter to rotate.  It avoids the
        re-creation of many unnecessary routines.'''
        x = np.arange(dims[0])
        y = np.arange(dims[1])
        X,Y = np.meshgrid(x,y)
        r = np.dstack((X,Y))
        r = r.reshape((dims[0]*dims[1],2))
        cen = np.array([dims[0]/2.,dims[1]/2.]).reshape((1,1,2))
        return r,x,y,cen

    def plotrealimg(self,winnum=None):
        self.plotimg(self.img,winnum=winnum)

    def plotimg(self,img,winnum=None,clear=0,logcmap=False,reciprocal=0,cb=None):
        '''Plot to the zoom figure.'''
        #self.fig = ZoomFigure(winnum=winnum,clear=clear)
        if(winnum is None):
            winnum = self.winnum
        self.setscale([0,0],winnum)
        self.fig = plt.figure(winnum)
        self.fig.clear()
        if(logcmap is False):
            normf = None
        else:
            normf = colors.LogNorm(vmin=.01,vmax=1e5)

        if(reciprocal == 0):
            pimg = self.fig.gca().imshow(img,extent=self.extent,norm=normf)
            self.fig.gca().set_xlabel(self.unit)
            self.fig.gca().set_ylabel(self.unit)
        else:
            pimg = self.fig.gca().imshow(img,extent=np.array(self.extent)/self.resolution**2*2.*np.pi/float(self.dims[0]),norm=normf)
            self.fig.gca().set_xlabel(self.unit + "$^{-1}$")
            self.fig.gca().set_ylabel(self.unit + "$^{-1}$")
        if(cb is not None):
            self.fig.fig.colorbar(pimg)
        #need to implement this later
        #self.fig.fig.colorbar(self.fig.ax)

    def setscale(self, logxy, winnum):
        f = plt.figure(winnum)
        ax = f.gca()
        if(logxy[0] == 0):
            ax.set_xscale("linear")
        else:
            ax.set_xscale("log")
        if(logxy[1] == 0):
            ax.set_yscale("linear")
        else:
            ax.set_yscale("log")

    def plotline(self,x,y,clr=None,winnum=None,clear=0,logxy=[1,1]):
        '''Plot a line element, zoomable.'''
        if(winnum is None):
            winnum = self.winnum
        self.setscale(logxy,winnum)
        self.fig = plt.figure(winnum)
        self.fig.gca().plot(x,y,clr)

    #def plotimg(self,winnum=None,clear=0,logcmap=False):
        #''' Plot image of the total shape.'''
        #self.plotzfimg(self.img,winnum=winnum,clear=clear,logcmap=logcmap,cb=None)

    def plotscat(self,winnum=None,logcmap=False):
        '''Plot the scattering (|FFT|^2) pattern for far field diffraction.'''
        if(self.fftupdated == False):
            self.calcFFT()
            self.fftupdated = True
        self.plotimg(self.imgFFT2,winnum=winnum,reciprocal=1,cb=None)

    def plotsq(self,winnum=None,clr=None,logxy=[1,1]):
        '''Plot the circularly averaged structure factor of the data.'''
        if(clr is None):
            clr = 'k'
        #pixel average
        cavgx,cavgy = circavg(self.imgFFT2)
        #q perpixel
        self.sq_q = cavgx
        self.sq = cavgy
        self.dq = 2*np.pi/float(self.dims[0])/self.resolution
        #self.plotzfline(self.sq_q*self.dq,self.sq,clr=clr,winnum=winnum,logxy=logxy)
        if(winnum is None):
            winnum = self.winnum
        fig = plt.figure(winnum)
        plt.loglog(self.sq_q*self.dq,self.sq,clr)
        fig.gca().set_xlabel(self.unit + "$^{-1}$")
        fig.gca().set_ylabel("$|S(q)|^2$")
        

    def calcFFT(self):
        '''Calc the fft.'''
        self.imgFFT = fftshift(fft2(fftshift(self.img,)))
        self.imgFFT2 = np.absolute(self.imgFFT)**2
        self.limgFFT2 = np.zeros(self.imgFFT2.shape)
        w = np.where(self.imgFFT2 > 0)
        self.limgFFT2[w] = np.log10(self.imgFFT2[w])

    def mkarray2d(self,vecs,subvecs=None,basislims=[-3,3,-3,3],PF=None):
        ''' Make an array of 2d structures 
            shapefn - shape generating function
            shapefnpars - shape generating function pars
            vecs - basis vectors (in pixels)
            subvecs - sub basis vectors (in pixels) (if exist)
            basislims - the max number of basis vectors to iterate over
                [xleft,xright,yleft,yright]
            NOTE: not tested yet
        '''
        res = np.zeros(self.dims)
        subimg = np.zeros(self.dims)
        basislimsx = np.arange(basislims[0],basislims[1]+1)
        basislimsy = np.arange(basislims[2],basislims[3]+1)
        for i in basislimsx:
            for j in basislimsy:
                if(PF is not None):
                    print("iteration ({},{})".format(i,j))
                if(subvecs is not None):
                    for k in np.arange(subvecs.shape[0]):
                        vx = i*vecs[0,0] + j*vecs[1,0] + subvecs[k,0]
                        vy = i*vecs[0,1] + j*vecs[1,1] + subvecs[k,1]
                        self.addshape(vx,vy)
                else:
                    vx = i*vecs[0,0] + j*vecs[1,0] 
                    vy = i*vecs[0,1] + j*vecs[1,1]
                    self.addshape(vx,vy)
        #return res

    def mkshapegrid(self):
        '''Make the X,Y mesh for the shape with the resolution in meters/pixel .'''
        x = np.linspace(-self.dims[0]/2., self.dims[0]/2., self.dims[0])*self.resolution
        y = np.linspace(-self.dims[1]/2., self.dims[1]/2., self.dims[1])*self.resolution
        self.X,self.Y = np.meshgrid(x,y)
        #calculate the extent for the images
        rr = self.resolution
        self.extent = [self.X[0,0]-.5*rr, self.X[0,-1]+.5*rr, self.Y[0,0]-.5*rr, self.Y[-1,0]+.5*rr]

#Shape Objects that inherit the base shape

class NmerShape(ShapeObj):
    ''' Make an nmershape object. Inherits the base shape class.'''
    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mknmershape,pars,N,resolution=resolution,unit=unit)

    def mknmershape(self,img):
        ''' Make an n symmetry object . Elements are spheres
            radius      : (r) diameter of spheres (resolution unit)
            distance    : (a) distance between spheres (resolution unit)
            symmetry    : (n) the symmetry (2 is dimer, 3 trimer etc)
            N : number of sampling points per dimension
            X,Y are the arrays of positions
        '''
        #assume square shape image
        dims = img.shape
        r = self.bshapeparams['radius']
        a = self.bshapeparams['distance']
        n = self.bshapeparams['symmetry']
        
        #hexagon
        if(np.abs(n) > 1):
            for i in np.arange(n):
                ax = a*np.cos(2*np.pi/float(n)*i)
                ay = a*np.sin(2*np.pi/float(n)*i)
                img += ((self.X-ax)**2 + (self.Y-ay)**2 < r**2)
        elif(np.abs(n) == 1):
            ''' Move sphere to make COM at center.'''
            img += ((self.X)**2 + (self.Y)**2 < r**2)
        if(n < 0):
            '''Special case if symmetry negative, make object asymmetric.'''
            w = np.where(self.X > 0)
            img[w] *= -1

class NmerShape3DProj(ShapeObj):
    ''' Make an nmershape 3d projection object. Inherits the base shape class.'''
    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mknmershape3dproj,pars,N,resolution=resolution,unit=unit)

    def mknmershape3dproj(self,img):
        ''' Make an n symmetry object . Elements are spheres
            radius      : (r) diameter of spheres (resolution unit)
            distance    : (a) distance between spheres (resolution unit)
            symmetry    : (n) the symmetry (2 is dimer, 3 trimer etc)
            N : number of sampling points per dimension
            X,Y are the arrays of positions
        '''
        #assume square shape image
        dims = img.shape
        r = self.bshapeparams['radius']
        a = self.bshapeparams['distance']
        n = self.bshapeparams['symmetry']
        
        #hexagon
        if(np.abs(n) > 1):
            for i in np.arange(n):
                ax = a*np.cos(2*np.pi/float(n)*i)
                ay = a*np.sin(2*np.pi/float(n)*i)
                img += np.sqrt(np.maximum(0, 1- ((self.X-ax)**2 + (self.Y-ay)**2)/r**2))
        elif(np.abs(n) == 1):
            ''' Move sphere to make COM at center.'''
            img += np.sqrt(np.maximum(0,(1- ((self.X)**2 + (self.Y)**2)/r**2)))
        if(n < 0):
            '''Special case if symmetry negative, make object asymmetric.'''
            w = np.where(self.X > 0)
            img[w] *= -1


class GratingShape(ShapeObj):
    ''' Make a grating shape object. Inherits the base shape class.'''
    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkgrating,pars,N,resolution=1.,unit='pixel')

    def mkgrating(self,img):
        ''' Make an grating of transverse length L (pixels), number n, width a,
        spacing d, with a tilt of phi in an image of NxN pixels.
        length  - (L) The transverse length of grating
        nelem   - (n) The number of elements in grating
        width   - (a) The width of the grating
        spacing - (d) The spacing of the grating
        '''
        L = self.bshapeparams['length']
        n = self.bshapeparams['nelem']
        a = self.bshapeparams['width']
        d = self.bshapeparams['spacing']

        for i in np.arange(n):
            y0 = d*(i - n/2)
            w = np.where((np.abs(self.X) <= L) * (np.abs(self.Y-y0) < a/2.))
            img[w] += 1

class SGratingShape(ShapeObj):
    ''' Make a sinusoidal grating shape object. Inherits the base shape
    class.'''
    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkgrating,pars,N,resolution=1.,unit='pixel')

    def mkgrating(self,img):
        ''' Make an grating of transverse length L (pixels), number n, width a,
        spacing d, with a tilt of phi in an image of NxN pixels.
        pars[0] - (L) The transverse length of grating
        pars[1] - (n) The number of elements in grating
        pars[2] - (a) The width of the grating
        pars[3] - (d) The spacing of the grating
        '''
        L = self.bshapeparams['length']
        n = self.bshapeparams['nelem']
        a = self.bshapeparams['width']
        d = self.bshapeparams['spacing']

        width = n*d
        w = np.where((np.abs(self.X) <= L)* (np.abs(self.Y) < width/2.))
        k = 2*np.pi/width*n
        img[w] = (1+np.cos(k*self.Y[w]))

class SAnnuliShape(ShapeObj):
    ''' Make Sin Annuli shape objects. Inherits the base shape class.
           Radius is the start radius.'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mksannuli,pars,N,resolution=resolution,unit=unit)

    def mksannuli(self,img):
        '''mkg(auss)annulus
            Generate an annulus with a Gaussian spread, not sharp cutoff.
            pars['radius'] - central radius
            pars['rwidth'] - radius width
            pars['numannuli']  - standard deviation sigma
            pars['phi']  - standard deviation sigma
            '''
        rc      = self.bshapeparams['radius']
        dr      = self.bshapeparams['rwidth']
        n       = self.bshapeparams['numannuli']
        phi1    = self.bshapeparams['phi']#phase shift, usually just zero
        k       = 2*np.pi/dr*n
        r       = np.sqrt(self.X**2 + self.Y**2)
        w       = np.where((r > rc)*(r < rc + dr))
        img[w] += (1 + np.cos(k*r[w]+phi1))

class GAnnuliShape(ShapeObj):
    ''' Make Gaussian Annuli shape objects. Inherits the base shape class.'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkgannuli,pars,N,resolution=resolution,unit=unit)

    def mkgannulus(self,pars,img):
        '''mkg(auss)annulus
            Generate an annulus with a Gaussian spread, not sharp cutoff.
            pars['radius'] - central radius
            pars['rwidth'] - radius width
            pars['sigma']  - standard deviation sigma
            '''
        rc      = pars['radius']
        dr      = pars['rwidth']
        sg      = pars['sigma']
        r       = np.sqrt(self.X**2 + self.Y**2)
        img    += np.exp(-(r-rc)**2/(2.*sg**2))
    
    def mkgannuli(self,img):
        '''mkg(auss)annuli Make a series of n annuli starting with radius r0 width dr spaced drr
        apart. They will be placed in an NxN pixel image. These parameters are
        specified by the parameters array:
        pars['radius']    - (r0) mean radius in pixels
        pars['anwidth']   - (dr) annulus width in pixels
        pars['ansep']     - (drr) annulus radial separation in pixels
        pars['numannuli'] - (n) the number of annuli to include
        pars['sigma']     - std dev of Gauss annuli
        '''
        r0      = self.bshapeparams['radius']
        dr      = self.bshapeparams['anwidth']
        drr     = self.bshapeparams['ansep']
        n       = self.bshapeparams['numannuli']
        sg      = self.bshapeparams['sigma']
    
        for i in np.arange(n):
            self.mkgannulus({'radius' : r0+drr*i,'rwidth' : dr, 'sigma' : sg},img)


class AnnuliShape(ShapeObj):
    ''' Make Gaussian Annuli shape objects. Inherits the base shape class.'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkannuli,pars,N,resolution=resolution,unit=unit)

    def mkannulus(self,pars,img):
        ''' Generate an annulus of mean radius r0 and width dr,specified by the
            parameters:
            pars['radius'] - (r0) mean radius (in pixels)
            pars['rwidth'] - (dr) radius width (in pixels)
        '''
        rc      = pars['radius']
        dr      = pars['rwidth']
        r2      = np.sqrt(self.X**2 + self.Y**2)
        w       = np.where((r2 > (rc - dr/2.))*(r2 < (rc + dr/2.)))
        img[w] += 1
    
    def mkannuli(self,img):
        '''Make a series of n annuli starting with radius r0 width dr spaced drr
        apart. They will be placed in an NxN pixel image. These parameters are
        specified by the parameters array:
        radius      - (r0) mean radius in pixels
        anwidth     - (dr) annulus width in pixels
        ansep       - (drr) annulus radial separation in pixels
        numannuli   - (n) the number of annuli to include
        '''
        r0      = self.bshapeparams['radius']
        dr      = self.bshapeparams['anwidth']
        drr     = self.bshapeparams['ansep']
        n       = self.bshapeparams['numannuli']
    
        for i in np.arange(n):
            self.mkannulus({'radius' : r0+drr*i,'rwidth' : dr},img)

class AnnuliShape3DProj(ShapeObj):
    ''' Make Gaussian Annuli toroids projected on a 2d space. Inherits the base
        shape class.
        Pars :
            radius      - (r0) mean radius in pixels
            anwidth     - (dr) annulus width in pixels
            ansep       - (drr) annulus radial separation in pixels
            numannuli   - (n) the number of annuli to include
'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkannuli3dproj,pars,N,resolution=resolution,unit=unit)

    def mkannulus3dproj(self,pars,img):
        ''' Generate an annulus of mean radius r0 and width dr,specified by the
            parameters:
            pars['radius'] - (r0) mean radius (in pixels)
            pars['rwidth'] - (dr) radius width (in pixels)
        '''
        rc      = pars['radius']
        dr      = pars['rwidth']
        r2      = np.sqrt(self.X**2 + self.Y**2)
        #w       = (r2 > (rc - dr/2.)**2)*(r2 < (rc + dr/2.)**2)
        img    += np.sqrt(np.maximum(1 - (2.*(r2 - rc)/dr)**2,0))
        #img[w] += 1
    
    def mkannuli3dproj(self,img):
        '''Make a series of n annuli starting with radius r0 width dr spaced drr
        apart. They will be placed in an NxN pixel image. These parameters are
        specified by the parameters array:
        radius      - (r0) mean radius in pixels
        anwidth     - (dr) annulus width in pixels
        ansep       - (drr) annulus radial separation in pixels
        numannuli   - (n) the number of annuli to include
        '''
        r0      = self.bshapeparams['radius']
        dr      = self.bshapeparams['anwidth']
        drr     = self.bshapeparams['ansep']
        n       = self.bshapeparams['numannuli']
    
        for i in np.arange(n):
            self.mkannulus3dproj({'radius' : r0+drr*i,'rwidth' : dr},img)

class RadWedgesShape(ShapeObj):
    '''Make radial wedges.'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkradialwedges,pars,N,resolution=resolution,unit=unit)

    def mkline(self,img,X,Y,L,T):
        '''Make a horizontal line of length L
            and thickness T.'''
        w = np.where((np.abs(X) < L/2.)*(np.abs(Y) < T/2.))
        img[w] += 1

    def mkgrating(self,img,L,T,s,n):
        ''' Make a grating of transverse length L (pixels), number n, width a,
        spacing d, with a tilt of phi in an image of NxN pixels.
        length  - (L) The transverse length of grating
        width   - (T) The thickness of the grating
        spacing - (s) The spacing of the grating
        nelem   - (n) The number of elements in grating
        '''
        for i in np.arange(n):
            self.mkline(img,self.X,self.Y-(i-n/2.)*s-(s-T),L,T)

    def mkradialwedges(self,img):
        '''Make radial wedges.
        length  - (L) The transverse length of grating
        nelem   - (n) The number of elements in grating
        width   - (a) The width of the grating
        spacing - (d) The spacing of the grating
        nphi    - (nphi) number of phis
        '''
        imgtmp = 0*img
        imgtmp2 = 0*img

        L    = self.bshapeparams['length']
        T    = self.bshapeparams['width']
        s    = self.bshapeparams['spacing']
        n    = self.bshapeparams['nelem']
        nphi = self.bshapeparams['nphi']

        theta = 2*np.pi/nphi

        for i in range(nphi):
            imgtmp *= 0
            imgtmp2 *= 0
            th  = theta*i 
            self.mkgrating(imgtmp,L,T,s,n)
            w = n*s-T
            imgtmp = np.roll(np.roll(imgtmp,int(L/2. + ((n*s-(s-T))/2.)/np.tan(theta/2.)),axis=1),0,axis=0)
            rotate(imgtmp,th,imgtmp2)
            img += imgtmp2

class PhiWedgesShape(ShapeObj):
    '''Make phi wedges grating.'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkphiwedge,pars,N,resolution=resolution,unit=unit)

    def mkline(self,img,X,Y,L,T):
        '''Make a horizontal line of length L
            and thickness T.'''
        w = np.where((np.abs(X) < L/2.)*(np.abs(Y) < T/2.))
        img[w] += 1

    def mkphiwedge(self,img):
        '''make phi wedge gratings of angle theta, thickness t spacing s and
            number n.'''
        T = self.bshapeparams['width']
        s = self.bshapeparams['spacing']
        n = self.bshapeparams['nelem']
        theta = 2*np.pi/n
        for i in range(n):
            r = i*s
            self.mkline(img,self.X,self.Y-r/2,2*r*np.tan(theta/2.),T)

    def mkphiwedges(self,img):
        '''Make phi wedges.'''
        for i in range(12):
            imgtmp *= img*0
            mklinewedges(imgtmp,thetaphiwedges/2.,T,s,nspacings,X,Y)
            imgr = imgtmp*0
            rotate(imgtmp,(thetaphiwedges)*i,imgr)
            img += imgr

class RectShape(ShapeObj):
    '''Make  a square shape.'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkrect,pars,N,resolution=resolution,unit=unit)

    def mkrect(self,img):
        ''' Make a square shape:
            a, b : the width and height of shape
            da, db : the percent polydispersity
        '''
        a = self.bshapeparams['a']
        b = self.bshapeparams['b']
        dd = self.bshapeparams['dd']
        drnd = (np.random.random()-.5)*dd
        dda = drnd*a
        ddb = drnd*b
        w = np.where((np.abs(self.X) <  a/2. - dda)*(np.abs(self.Y) < b/2. - ddb))
        img[w] += 1

class RoundedRectShape(ShapeObj):
    '''Make  a square shape.'''

    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        '''The initialization function for the object.'''
        ShapeObj.__init__(self,self.mkroundedrect,pars,N,resolution=resolution,unit=unit)

    def mkroundedrect(self,img):
        ''' Make a square shape:
            a, b : the width and height of shape
            r, radius of the circle that makes the rounded rect
            da, db : the percent polydispersity(ignored fornow)
        '''
        a = self.bshapeparams['a']
        b = self.bshapeparams['b']
        r = self.bshapeparams['r']
        #dd = self.bshapeparams['dd']
        #drnd = (np.random.random()-.5)*dd
        #dda = drnd*a
        #ddb = drnd*b
        #this finds the rect
        logic1 = (np.abs(self.X) <  a/2.)*(np.abs(self.Y) < b/2.)
        #this logic makes sure that it's within limit bounded by circles
        logic2sub1 = (np.sqrt((np.abs(self.X)-(a/2.-r))**2 + (np.abs(self.Y)-(a/2.-r))**2) < r)
        logic2sub2 = np.abs(self.X) < (a/2.-r)
        logic2sub3 = np.abs(self.Y) < (a/2.-r)
        logic2 = (logic2sub1 + logic2sub2 + logic2sub3 > 0)
        w = np.where(logic1*logic2)
        img[w] += 1
