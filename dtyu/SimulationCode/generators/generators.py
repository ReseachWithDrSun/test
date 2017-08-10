#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage


#det_image = np.zeros( (calibration.height, calibration.width) )


def zingers(calibration, amt=1.0):                         
    
    det_image = np.random.poisson(lam=0.0005*amt, size=(calibration.height, calibration.width))
    #det_image *= np.random.normal(loc=2**18, scale=0.3*2**18, size=(calibration.height, calibration.width))
    det_image *= 2**18
    
    #print('zingers>    min={:d} max={:d} avg={:.1f}+/-{:.1f}'.format(np.min(det_image), np.max(det_image), np.average(det_image), np.std(det_image)))    
    
    return det_image


def diffuse_constant(calibration, avg=10.0, sigma=2.0):
    
    return np.random.normal(loc=avg, scale=sigma, size=(calibration.height, calibration.width))

    
def diffuse_poisson(calibration, lam=10.0):
    
    return np.random.poisson(lam=lam, size=(calibration.height, calibration.width))



def diffuse_low_q_power(calibration, power=-3.0):
    
    q = calibration.q_map()
    rescale = np.power(calibration.get_q_per_pixel(), power)
    
    det_image = np.power(q, power)/rescale
    
    return det_image


def diffuse_low_q_power_aniso(calibration, power=-3.0, epsilon=2.0, chi=0.0):
    
    
    angles = np.radians(calibration.angle_map()-chi)
    q = calibration.q_map()*np.sqrt( np.square(np.sin(angles))+ np.square(epsilon)*np.square(np.cos(angles)) )
    rescale = np.power(calibration.get_q_per_pixel(), power)
    
    det_image = np.power(q, power)/rescale
    
    return det_image


def diffuse_low_q_OrnsteinZernike(calibration, xi=1000.0):
    
    q = calibration.q_map()
    det_image = 1./(1.+np.square(q)*np.square(xi))
    
    return det_image

def diffuse_low_q_OrnsteinZernike_aniso(calibration, xi=1000.0, epsilon=2.0, chi=0.0):
    
    angles = np.radians(calibration.angle_map()-chi)
    q = calibration.q_map()*np.sqrt( np.square(np.sin(angles))+ np.square(epsilon)*np.square(np.cos(angles)) )
    det_image = 1./(1.+np.square(q)*np.square(xi))
    
    return det_image
    
def diffuse_low_q_DebyeBueche(calibration, a=1000.0):
    
    q = calibration.q_map()
    det_image = 1./np.square(1.+np.square(q*a))
    
    return det_image

def diffuse_low_q_DebyeBueche_aniso(calibration, a=1000.0, epsilon=2.0, chi=0.0):
    
    angles = np.radians(calibration.angle_map()-chi)
    q = calibration.q_map()*np.sqrt( np.square(np.sin(angles))+ np.square(epsilon)*np.square(np.cos(angles)) )
    det_image = 1./np.square(1.+np.square(q*a))
    
    return det_image
        
def diffuse_low_q_Guinier(calibration, Rg=1000.0):
    
    q = calibration.q_map()
    det_image = np.exp( -np.square(Rg)*np.square(q)/3. )
    
    return det_image     

def diffuse_low_q_Guinier_aniso(calibration, Rg=1000.0, epsilon=2.0, chi=0.0):
    
    angles = np.radians(calibration.angle_map()-chi)
    q = calibration.q_map()*np.sqrt( np.square(np.sin(angles))+ np.square(epsilon)*np.square(np.cos(angles)) )
    det_image = np.exp( -np.square(Rg)*np.square(q)/3. )
    
    return det_image 


def diffuse_high_q(calibration, sigma=10):
    
    q = calibration.q_map()
    
    det_image = ( 1 - np.exp( -1*np.square(sigma)*np.square(q) ) )
    det_image /= np.max(det_image)
    
    return det_image

def diffuse_high_q_aniso(calibration, sigma=10, epsilon=2.0, chi=0.0):
    
    angles = np.radians(calibration.angle_map()-chi)
    q = calibration.q_map()*np.sqrt( np.square(np.sin(angles))+ np.square(epsilon)*np.square(np.cos(angles)) )
    
    det_image = ( 1 - np.exp( -1*np.square(sigma)*np.square(q) ) )
    det_image /= np.max(det_image)
    
    return det_image

def direct_beam(calibration, width=20.0, height=None, peak=1e12):

    if height is None:
        height = width
        
    x = np.arange(calibration.width) - calibration.x0
    y = np.arange(calibration.height) - calibration.y0
    X, Y = np.meshgrid(x, y)
    #R = np.sqrt(X**2 + Y**2)
        
    fwhm_to_sigma = 2.*np.sqrt(2.*np.log(2))
    sigma_x = width/fwhm_to_sigma
    sigma_y = height/fwhm_to_sigma
        
    beam = peak*np.exp( -np.square(X)/(2.*np.square(sigma_x)) )*np.exp( -np.square(Y)/(2.*np.square(sigma_y)) )
    
    return beam




def S_DW(calibration, sigma=10):
    
    q = calibration.q_map()
    det_image = 1. - np.exp(-np.square(sigma)*np.square(q))
    
    return det_image
    
   


def symmetry_ring(calibration, q0, dq, eta=0.5, chi=0.0, symmetry=2.0):
    
    q = calibration.q_map()
    angles = np.radians(calibration.angle_map()-chi)
    
    det_image = (1-np.square(eta))/( np.square(1+eta) - 4*eta*np.square(np.cos(angles*symmetry/2)) )
    det_image *= np.exp( -np.square(q-q0)/(2.*np.square(dq)) )
    
    return det_image


def symmetry_rings(calibration, q0, dq, eta=0.5, chi=0.0, symmetry=2.0, sigma_DW=10):
    
    num_rings = 1
    
    q = calibration.q_map()
    angles = np.radians(calibration.angle_map()-chi)
    
    det_image = (1-np.square(eta))/( np.square(1+eta) - 4*eta*np.square(np.cos(angles*symmetry/2)) )
    
    q_dep = np.zeros( (calibration.height, calibration.width) )
    for i in range(1, 25, 1):
        c = np.exp( - np.square(sigma_DW)*np.square(q0)*(np.square(i)-1) )
        q_dep_i = c*np.exp( -np.square(q-i*q0)/(2.*np.square(dq)) )
        if np.max(q_dep_i)>0.01:
            num_rings = i
        q_dep += q_dep_i
    
    det_image *= q_dep
    
    det_image /= np.max(det_image)
    
    return det_image, num_rings



def adhoc_rings(calibration, q0s, dq):

    det_image = np.zeros( (calibration.height, calibration.width) )    
    q = calibration.q_map()
    for (q0, intensity) in q0s:
        det_image += intensity*np.exp( -np.square(q-q0)/(2.*np.square(dq)) )
        
    return det_image
        
        
def adhoc_rings_speckled(calibration, q0s, dq, symmetry=4, eta=0.95, num_spots=20):

    det_image = np.zeros( (calibration.height, calibration.width) )    
    q = calibration.q_map()
    angles = np.radians(calibration.angle_map())
    
    for (q0, intensity) in q0s:
        angular = np.zeros( (calibration.height, calibration.width) )    
        for i in range(num_spots):
            chi = np.radians(np.random.uniform(-180, +180))
            c = np.random.uniform(0.001, 1.0)
            angular += c*(1-np.square(eta))/( np.square(1+eta) - 4*eta*np.square(np.cos((angles-chi)*symmetry/2)) )
        
        det_image += intensity*np.exp( -np.square(q-q0)/(2.*np.square(dq)) )*angular
        
    return det_image
        
        
def adhoc_rings_speckled_individual(calibration, q0s, dq, symmetry=4, num_spots=20):

    det_image = np.zeros( (calibration.height, calibration.width) )    
    q = calibration.q_map()
    qx = calibration.qx_map()
    qz = calibration.qz_map()
    #angles = np.radians(calibration.angle_map())
    
    for (q0, intensity) in q0s:
        angular = np.zeros( (calibration.height, calibration.width) )    
        for i in range(num_spots):
            q0c = np.random.normal(q0, dq*2)
            chi = np.radians(np.random.uniform(-180, +180))
            c = np.random.uniform(0.001, 1.0)
            for s in range(symmetry):
                qx0 = q0c*np.sin(chi + s*np.pi/symmetry)
                qz0 = q0c*np.cos(chi + s*np.pi/symmetry)
                angular += c*np.exp( -np.square(qx-qx0)/(2.*np.square(dq)) )*np.exp( -np.square(qz-qz0)/(2.*np.square(dq)) )
        
        det_image += intensity*angular
        
    return det_image



def adhoc_rings_symmetry(calibration, q0s, dq, eta, chi, symmetry):

    det_image = np.zeros( (calibration.height, calibration.width) )    
    q = calibration.q_map()
    angles = np.radians(calibration.angle_map()-chi)
    
    for (q0, intensity) in q0s:
        det_image += intensity*np.exp( -np.square(q-q0)/(2.*np.square(dq)) )
        
        
        
    det_image *= (1-np.square(eta))/( np.square(1+eta) - 4*eta*np.square(np.cos(angles*symmetry/2)) )
    
    return det_image




def coherent_halo(calibration, q0, dq, eta=0.5, chi=0.0, symmetry=2.0, rescale=4.0):
    
    q = calibration.q_map()
    angles = np.radians(calibration.angle_map()-chi)
    
    
    coherent_speckle = diffuse_structured(calibration, avg=10.0, sigma=7.0, rescale=rescale)
        
    det_image = (1-np.square(eta))/( np.square(1+eta) - 4*eta*np.square(np.cos(angles*symmetry/2)) )
    det_image *= np.exp( -np.square(q-q0)/(2.*np.square(dq)) )
    det_image *= coherent_speckle
    
    return det_image


def form_factor_sphere(calibration, radius=10):
    
    qR = calibration.q_map()*radius
    
    det_image = np.square( np.sin(qR)-qR*np.cos(qR) )/( np.power(qR, 6) )
    
    return det_image



def NP_lattice(calibration, peaks, radius=10, sigma_DWr=1e2, q0=2e-2, dq=2e-3):

    q = calibration.q_map()
    qR = calibration.q_map()*radius
    
    form_factor = np.square( np.sin(qR)-qR*np.cos(qR) )/( np.power(qR, 6) )
    DW = np.exp( - np.square(sigma_DWr)*np.square(2.*np.pi*q/q0) )
    ff_part = form_factor*DW/np.square( np.maximum(q, 1e-4) )
    
    num_rings = 1
    I_q0 = 0
    det_image = np.zeros( (calibration.height, calibration.width) )
    for i, (qp, intensity) in enumerate(peaks):
        qc = qp*q0
        current = ff_part*intensity*np.exp( -np.square(q-qc)/(2.*np.square(dq)) )
        current = np.nan_to_num(current)
        if i==0:
            I_q0 = np.max(current)
        elif np.max(current)>0.002*I_q0:
            num_rings = i+1
        
        det_image += current
    
    det_image += form_factor*(1-DW)
    det_image /= np.max(det_image)
    
    
    return det_image, num_rings



def NP_lattice_distorted(calibration, peaks, radius=10, sigma_DWr=1e2, q0=2e-2, dq=2e-3, epsilon=2.0, chi=0.0, mode=0):

    angles = np.radians(calibration.angle_map()-chi)
    q = calibration.q_map()*np.sqrt( np.square(np.sin(angles))+ np.square(epsilon)*np.square(np.cos(angles)) )
    
    if mode<0.5:
        qR = q*radius
    else:
        qR = calibration.q_map()*radius
    
    form_factor = np.square( np.sin(qR)-qR*np.cos(qR) )/( np.power(qR, 6) )
    DW = np.exp( - np.square(sigma_DWr)*np.square(2.*np.pi*q/q0) )
    ff_part = form_factor*DW/np.square( np.maximum(q, 1e-4) )

    num_rings = 1
    I_q0 = 0
    det_image = np.zeros( (calibration.height, calibration.width) )
    for i, (qp, intensity) in enumerate(peaks):
        qc = qp*q0
        current = ff_part*intensity*np.exp( -np.square(q-qc)/(2.*np.square(dq)) )
        current = np.nan_to_num(current)
        if i==0:
            I_q0 = np.max(current)
        elif np.max(current)>0.002*I_q0:
            num_rings = i+1
        
        det_image += current
    
    det_image /= np.max(det_image)
    
    
    return det_image, num_rings





def linear_beamstop(calibration, width=20.0, height=10.0, blur=2.0):
    
    x = np.arange(calibration.width) - calibration.x0
    y = np.arange(calibration.height) - calibration.y0
    X, Y = np.meshgrid(x, y)
    
    beamstop = np.where( np.abs(X)<width, 1, 0 )
    beamstop *= np.where( Y+height>0, 1, 0 )
    
    beamstop = beamstop.astype('float')
    beamstop = ndimage.filters.gaussian_filter( beamstop, blur )
    beamstop = 1-beamstop
    
    return beamstop


def circular_beamstop(calibration, radius=20.0, chi=10.0, blur=2.0):
    
    x = np.arange(calibration.width) - calibration.x0
    y = np.arange(calibration.height) - calibration.y0
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    angles = np.radians(calibration.angle_map()-chi)
    
    beamstop = np.where( np.abs(R)<radius, 1, 0 )
    beamstop2 = np.where( np.abs(angles)<np.radians(2.0), 1, 0 )
    beamstop = np.maximum(beamstop, beamstop2)
    
    beamstop = beamstop.astype('float')
    beamstop = ndimage.filters.gaussian_filter( beamstop, blur )
    beamstop = 1-beamstop
    
    return beamstop


def wedge_beamstop(calibration, chi=10.0, wedge=10, displacement=5.0, blur=2.0):
    
    x = np.arange(calibration.width) - calibration.x0 - displacement*np.sin(np.radians(chi))
    y = np.arange(calibration.height) - calibration.y0 - displacement*np.cos(np.radians(chi))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    angles = np.degrees(np.arctan2(X, Y))
    angles = np.radians(angles-chi)
    
    beamstop = np.where( np.abs(angles)<np.radians(wedge), 1, 0 )
    
    beamstop = beamstop.astype('float')
    beamstop = ndimage.filters.gaussian_filter( beamstop, blur )
    beamstop = 1-beamstop
    
    return beamstop

def window_obstruction_circle(calibration, q0, dq):
    
    #blocking = np.ones( (calibration.height, calibration.width) )
    q = calibration.q_map()
    blocking = 1 - 1./(1. + np.exp(-(q-q0)/dq) )
    
    return blocking


def background_square_window(calibration, x0, y0, size=50, blur=4.0):
    
    
    x = np.arange(calibration.width) - calibration.x0
    y = np.arange(calibration.height) - calibration.y0
    X, Y = np.meshgrid(x, y)
    #R = np.sqrt(X**2 + Y**2)
    
    det_image = np.where(abs(X-x0)<size,1,0)*np.where(abs(Y-y0)<size,1,0)
    
    det_image = det_image.astype('float')
    det_image = ndimage.filters.gaussian_filter( det_image, blur )
    
    return det_image

def background_circle_window(calibration, x0, y0, size=50, blur=4.0):
    
    
    x = np.arange(calibration.width) - calibration.x0 - x0
    y = np.arange(calibration.height) - calibration.y0 - y0
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    det_image = np.where(R<size,1,0)
    det_image = det_image.astype('float')
    det_image = ndimage.filters.gaussian_filter( det_image, blur )
    
    return det_image

def diffuse_structured(calibration, avg=10.0, sigma=2.0, rescale=4.0):
    
    from scipy.interpolate import griddata
    
    # Low-resolution noise
    det_image = np.zeros( (int(calibration.height/rescale), int(calibration.width/rescale)) )
    h, w = det_image.shape
    det_image += np.random.normal(loc=avg, scale=sigma, size=(h, w))
    
    # Determine limits
    dq = calibration.get_q_per_pixel()
    qx_min = np.min(calibration.qx_map())
    qx_max = np.max(calibration.qx_map())
    qz_min = np.min(calibration.qz_map())
    qz_max = np.max(calibration.qz_map())
    
    
    qx = np.linspace(qx_min-dq, qx_max+dq, num=w)
    qz = np.linspace(qz_min-dq, qz_max+dq, num=h)
    QX, QZ = np.meshgrid(qx, qz)    
    
    
    # Upscale to full image size
    points = np.column_stack((QX.ravel(), QZ.ravel()))
    values = det_image.ravel()
    
    remesh_data = griddata(points, values, (calibration.qx_map(), calibration.qz_map()), method='cubic')
    remesh_data = ndimage.filters.gaussian_filter( remesh_data, 2.0 )
    
    return np.maximum( remesh_data, 0.0 )


def slit_streak_H(calibration, length, aspect):
    
    det_image = np.zeros( (calibration.height, calibration.width) )
    
    q = calibration.q_map()
    qx = calibration.qx_map()
    qz = calibration.qz_map()
    
    sigma_l = length
    sigma_w = sigma_l*aspect
    
    # Streak along horizontal
    #det_image += np.exp( -np.square(qz)/sigma )*np.power(q,-2.0)
    det_image += np.exp( -np.square(qz)/sigma_w )*np.exp( -np.square(qx)/sigma_l )
    
    return det_image

def slit_streak_V(calibration, length, aspect):
    
    det_image = np.zeros( (calibration.height, calibration.width) )
    
    q = calibration.q_map()
    qx = calibration.qx_map()
    qz = calibration.qz_map()
    
    sigma_l = length
    sigma_w = sigma_l*aspect
    
    # Streak along horizontal
    det_image += np.exp( -np.square(qx)/sigma_w )*np.exp( -np.square(qz)/sigma_l )
    
    return det_image







def mesostructure(calibration, zoom=0.5):
    
    det_image = np.zeros( (calibration.height, calibration.width) )
    

    import sys
    pyCXD_PATH='../pyCXD/'
    pyCXD_PATH in sys.path or sys.path.append(pyCXD_PATH)
    
    import matplotlib.pyplot as plt
    
    from tools.transforms import rotation_matrix
    from shapes.shapes3DProj import Shape3, sphereprojfn
    
    
    class Shape3special(Shape3):
        def project(self):
            ''' Project the units onto a 2D image. The convention
                is to project onto the x-y plane.'''
            if self.img is None:
                self.img = np.zeros((self.dims[0],self.dims[1]))
            else:
                self.clearimg(self.img)
            curtype = -1
            for vec, typeno in zip(self.vecs, self.types):
                if typeno != curtype:
                    # first clear old shape
                    if(self.typeimg is None):
                        self.typeimg = np.zeros((self.dims[0],self.dims[1]))
                    if curtype >= 0:
                        self.clearimg(self.typeimg, bboxdims=self.typebboxes[curtype])
                    curtype = typeno
                    # make a new type
                    self.gentype(typeno)
                # project vector onto z, round, project current type stored
                self.projecttype((vec[:2]+.5).astype(int))
            

            if False:
                # Create density-fluctuations with defined scale
                from scipy.interpolate import griddata
                rescale = 6.0
                noise = np.zeros( (self.img.shape[0]//rescale, self.img.shape[1]//rescale) )
                noise += np.clip(np.random.normal(loc=0.5, scale=0.25, size=noise.shape), 0, 4.0)
                
                # Upscale to full image size
                x = np.linspace(0, 1, num=noise.shape[1])
                y = np.linspace(0, 1, num=noise.shape[0])
                X, Y = np.meshgrid(x, y)
                points = np.column_stack((Y.ravel(), X.ravel()))

                x = np.linspace(0, 1, num=self.img.shape[1])
                y = np.linspace(0, 1, num=self.img.shape[0])
                X, Y = np.meshgrid(x, y)
                remesh_data = griddata(points, noise.ravel(), (X, Y), method='cubic')
                remesh_data = ndimage.filters.gaussian_filter( remesh_data, rescale/2 )
                remesh_data = ndimage.filters.gaussian_filter( remesh_data, rescale/2 )
                self.img += remesh_data
                
            else:
                # Random density noise
                noise = np.random.uniform( 0, 0.5, self.img.shape ) 
                noise = ndimage.filters.gaussian_filter( noise, 2.0 )
                self.img += noise
            
            self.fimg2 = np.fft.fftshift(np.abs(np.fft.fft2(self.img)))**2        
    
    shp = Shape3special()
    # sphere parameters: radius, density, alpha, beta, gamma (Euler angles)
    # note for sphere euler angles are not necessary, but this prototype still 
    # requires them
    
    radius = 10
    shp.addtype(sphereprojfn,[radius,2,1,1,1],bboxdims=[100,100]);
    
    a = radius*2 # min
    a = radius*2.5
    b = a
    c = a
    span = 1
    for ix in range(-span, span+1):
        for iy in range(-span, span+1):
            for iz in range(-span, span+1):
                shp.addunits([ix*a,iy*b,iz*c],typeno = 0)
                
    
    shp.vecs[:,0] += shp.dims[0]//2 # translate x+500
    shp.vecs[:,1] += shp.dims[1]//2 # translate y+500
    shp.vecs[:,2] += shp.dims[2]//2 # translate z+500
    
    
    #for i in range(100):
    if True:
    
        shp.vecs[:,0] -= shp.dims[0]//2 # translate x+500
        shp.vecs[:,1] -= shp.dims[1]//2 # translate y+500
        shp.vecs[:,2] -= shp.dims[2]//2 # translate z+500
    
        eta = np.random.uniform(0, 2.*np.pi)
        mat = rotation_matrix([0,0,1],eta) # Rotation about z
        shp.transform3D(mat)
        
        v = np.random.uniform(0,1)
        theta = np.arccos(2*v-1)
        mat = rotation_matrix([1,0,0],theta) # Tilt away from z (rot about x)
        shp.transform3D(mat)
        
        phi = np.random.uniform(0, 2.*np.pi)
        mat = rotation_matrix([0,0,1],phi) # Rotation about z
        shp.transform3D(mat)
        
        shp.vecs[:,0] += shp.dims[0]//2 # translate x+500
        shp.vecs[:,1] += shp.dims[1]//2 # translate y+500
        shp.vecs[:,2] += shp.dims[2]//2 # translate z+500
        
        shp.project()
        
        if False:
            #print("{}".format(i))
            plt.figure(0);plt.cla();
            plt.imshow(shp.img)
            plt.figure(1);plt.cla();
            plt.imshow(shp.fimg2);
            plt.clim(0,1e5);
            plt.draw();plt.pause(1)
            #plt.draw();plt.pause(.001)
        
        
    fft_data = ndimage.interpolation.zoom(shp.fimg2, zoom=zoom)
    #fft_data += np.max(fft_data)*1e-4*np.ones((fft_data.shape)) # For testing
    
    # Attenuate edges
    x = np.arange(fft_data.shape[1]) - fft_data.shape[1]/2
    y = np.arange(fft_data.shape[0]) - fft_data.shape[0]/2
    X, Y = np.meshgrid(x, y)
    gauss = np.exp( -np.square(X)/(2.*np.square(fft_data.shape[1]/8)) )*np.exp( -np.square(Y)/(2.*np.square(fft_data.shape[0]/8)) )
    fft_data *= gauss
    
    # Throw-away center
    R = np.sqrt(np.square(X)+np.square(Y))
    fft_data *= 1-np.exp(-R/( fft_data.shape[0]/16 ) )
    
    
    if True:
        # Copy b2 into b1
        b1 = det_image
        b2 = fft_data
        
        # offset
        pos_v = int( calibration.y0 - b2.shape[0]/2 )
        pos_h = int( calibration.x0 - b2.shape[1]/2 )
        v_range1 = slice( max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0) )
        h_range1 = slice( max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0) )

        v_range2 = slice( max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]) )
        h_range2 = slice( max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]) )
        
        b1[v_range1, h_range1] += b2[v_range2, h_range2]
        
    
    
    det_image /= np.max(det_image)

    
    return det_image





def mesostructure_rotation(calibration, zoom=0.5, num_orientations=10):
    
    det_image = np.zeros( (calibration.height, calibration.width) )
    

    import sys
    pyCXD_PATH='../pyCXD/'
    pyCXD_PATH in sys.path or sys.path.append(pyCXD_PATH)
    
    import matplotlib.pyplot as plt
    
    from tools.transforms import rotation_matrix
    from shapes.shapes3DProj import Shape3, sphereprojfn
    
    
    class Shape3special(Shape3):
        def project(self):
            ''' Project the units onto a 2D image. The convention
                is to project onto the x-y plane.'''
            if self.img is None:
                self.img = np.zeros((self.dims[0],self.dims[1]))
            else:
                self.clearimg(self.img)
            curtype = -1
            for vec, typeno in zip(self.vecs, self.types):
                if typeno != curtype:
                    # first clear old shape
                    if(self.typeimg is None):
                        self.typeimg = np.zeros((self.dims[0],self.dims[1]))
                    if curtype >= 0:
                        self.clearimg(self.typeimg, bboxdims=self.typebboxes[curtype])
                    curtype = typeno
                    # make a new type
                    self.gentype(typeno)
                # project vector onto z, round, project current type stored
                self.projecttype((vec[:2]+.5).astype(int))
            

            if False:
                # Create density-fluctuations with defined scale
                from scipy.interpolate import griddata
                rescale = 6.0
                noise = np.zeros( (self.img.shape[0]//rescale, self.img.shape[1]//rescale) )
                noise += np.clip(np.random.normal(loc=0.5, scale=0.25, size=noise.shape), 0, 4.0)
                
                # Upscale to full image size
                x = np.linspace(0, 1, num=noise.shape[1])
                y = np.linspace(0, 1, num=noise.shape[0])
                X, Y = np.meshgrid(x, y)
                points = np.column_stack((Y.ravel(), X.ravel()))

                x = np.linspace(0, 1, num=self.img.shape[1])
                y = np.linspace(0, 1, num=self.img.shape[0])
                X, Y = np.meshgrid(x, y)
                remesh_data = griddata(points, noise.ravel(), (X, Y), method='cubic')
                remesh_data = ndimage.filters.gaussian_filter( remesh_data, rescale/2 )
                remesh_data = ndimage.filters.gaussian_filter( remesh_data, rescale/2 )
                self.img += remesh_data
                
            else:
                # Random density noise
                noise = np.random.uniform( 0, 0.5, self.img.shape ) 
                noise = ndimage.filters.gaussian_filter( noise, 2.0 )
                self.img += noise
            
            self.fimg2 = np.fft.fftshift(np.abs(np.fft.fft2(self.img)))**2        
    
    shp = Shape3special()
    # sphere parameters: radius, density, alpha, beta, gamma (Euler angles)
    # note for sphere euler angles are not necessary, but this prototype still 
    # requires them
    
    radius = 10
    shp.addtype(sphereprojfn,[radius,2,1,1,1],bboxdims=[100,100]);
    
    a = radius*2 # min
    a = radius*2.5
    b = a
    c = a
    span = 1
    for ix in range(-span, span+1):
        for iy in range(-span, span+1):
            for iz in range(-span, span+1):
                shp.addunits([ix*a,iy*b,iz*c],typeno = 0)
                
    
    shp.vecs[:,0] += shp.dims[0]//2 # translate x+500
    shp.vecs[:,1] += shp.dims[1]//2 # translate y+500
    shp.vecs[:,2] += shp.dims[2]//2 # translate z+500
    
    
    fft_data = None
    for i in range(num_orientations):
        print('        orientation {:d}'.format(i))
    
        shp.vecs[:,0] -= shp.dims[0]//2 # translate x+500
        shp.vecs[:,1] -= shp.dims[1]//2 # translate y+500
        shp.vecs[:,2] -= shp.dims[2]//2 # translate z+500
    
        eta = np.random.uniform(0, 2.*np.pi)
        mat = rotation_matrix([0,0,1],eta) # Rotation about z
        shp.transform3D(mat)
        
        v = np.random.uniform(0,1)
        theta = np.arccos(2*v-1)
        mat = rotation_matrix([1,0,0],theta) # Tilt away from z (rot about x)
        shp.transform3D(mat)
        
        phi = np.random.uniform(0, 2.*np.pi)
        mat = rotation_matrix([0,0,1],phi) # Rotation about z
        shp.transform3D(mat)
        
        shp.vecs[:,0] += shp.dims[0]//2 # translate x+500
        shp.vecs[:,1] += shp.dims[1]//2 # translate y+500
        shp.vecs[:,2] += shp.dims[2]//2 # translate z+500
        
        shp.project()
        
        if False:
            #print("{}".format(i))
            plt.figure(0);plt.cla();
            plt.imshow(shp.img)
            plt.figure(1);plt.cla();
            plt.imshow(shp.fimg2);
            plt.clim(0,1e5);
            plt.draw();plt.pause(1)
            #plt.draw();plt.pause(.001)
            
        if fft_data is None:
            fft_data = shp.fimg2
        else:
            fft_data += shp.fimg2
        
        
    fft_data = ndimage.interpolation.zoom(fft_data, zoom=zoom)
    #fft_data += np.max(fft_data)*1e-4*np.ones((fft_data.shape)) # For testing
    
    # Attenuate edges
    x = np.arange(fft_data.shape[1]) - fft_data.shape[1]/2
    y = np.arange(fft_data.shape[0]) - fft_data.shape[0]/2
    X, Y = np.meshgrid(x, y)
    gauss = np.exp( -np.square(X)/(2.*np.square(fft_data.shape[1]/8)) )*np.exp( -np.square(Y)/(2.*np.square(fft_data.shape[0]/8)) )
    fft_data *= gauss
    
    # Throw-away center
    R = np.sqrt(np.square(X)+np.square(Y))
    fft_data *= 1-np.exp(-R/( fft_data.shape[0]/16 ) )
    
    
    if True:
        # Copy b2 into b1
        b1 = det_image
        b2 = fft_data
        
        # offset
        pos_v = int( calibration.y0 - b2.shape[0]/2 )
        pos_h = int( calibration.x0 - b2.shape[1]/2 )
        v_range1 = slice( max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0) )
        h_range1 = slice( max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0) )

        v_range2 = slice( max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]) )
        h_range2 = slice( max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]) )

        b1[v_range1, h_range1] += b2[v_range2, h_range2]
        
    
    
    det_image /= np.max(det_image)

    
    return det_image



