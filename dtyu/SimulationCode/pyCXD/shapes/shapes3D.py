''' Make shapes of different geometries. Mainly desired for simulations.  
    All shape generators must follow this convention:
    mkshape(pars,img) where pars are the parameters for the shape and 
    img is the image to place the shape into. Only square images have been tried for now.
'''
import numpy as np
#from shapes.shapefunctions import * 
from plot.ZoomFigure import *
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.ndimage.interpolation import shift
from matplotlib import colors
from tools.circavg import circavg
from tools.rotate import rotate
import matplotlib.pyplot as plt

class ShapeObj3D:
    '''General 3D shape object.
        N - image dimensions for slices
        pars - pars to shape generator
        projfn - the projection function of the element
        pos - the positions of shapes currently added (3 vectors)
        rotate - the rotation function
        slice - slice the object
        '''
    def __init__(self,projfn,pars,N,resolution=1.,unit='pixel'):
        self.resolution=resolution
        self.dims = (N,N)
        #base shape params
        self.projfn = projfn
        #positions of elements
        self.vecs   = list()
        #the Euler angles for the subelement
        self.alphas = list()
        self.img = np.zeros((N,N))
        x = np.linspace(-N/2,N/2,N)
        y = np.linspace(-N/2,N/2,N)
        self.X,self.Y = np.meshgrid(x,y)

    def addshape(self,r,alphas=None,rho=None):
        '''Add a shape with vector r'''
        self.vecs.append(r)

    def slice(self,alpha, beta, gamma):
        '''Rotate by Euler angles then slice.
            Right now just ignores this and takes base projection.
        '''
        self.img *= 0
        R = self.makerotationmat(alpha,beta,gamma)
        for i,vec in enumerate(vecs):
            vec = np.dot(R,vec)
            self.img += self.profn(vec,self.alphas[i],self.subelempars)
        
    def makerotationmat(self,alpha,beta,gamma):
        ''' Make the rotation matrix from the Euler angles.'''
        D = np.array([
            [np.cos(alpha), np.sin(alpha),0],
            [-np.sin(alpha),np.cos(alpha),0],
            [0, 0 ,1]
        ])
        C = np.array([
            [0, 0 ,0],
            [0,np.cos(beta), np.sin(beta)],
            [0,-np.sin(beta),np.cos(beta)]
        ])
        B = np.array([
            [np.cos(gamma), np.sin(gamma),0],
            [-np.sin(gamma),np.cos(gamma),0],
            [0, 0 ,0]
        ])
        R = np.dot(D,C)
        R = np.dot(R,B)
        return R

#Shape Objects that inherit the base shape
class SphereShape3D(ShapeObj3D):
    '''Put a combination of spheres in 3d space.'''
    def __init__(self,pars,N,resolution=1.,unit='pixel'):
        ''' Init the 3d sphere function.'''
        ShapeObj3D.__init__(self,self.projfnsphere,pars,N,resolution=resolution,unit=unit)

    def projfnsphere(self,r,alphas,pars):
        '''
            alphas: Euler angles alphas[0,1,2] : alpha, beta, gamma
                Not used for sphere
            'radius' : radius
        '''
        return np.maximum(0,1 - ((self.X-r[0])**2 - (self.Y-r[1])**2)/R**2)
