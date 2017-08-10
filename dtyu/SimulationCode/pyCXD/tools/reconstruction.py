'''Reconstruction tools. Eventually it will be an object and will most likely
be changed so for now it is not completely trustable.'''
from numpy.fft import fft2, ifft2, fftshift
import numpy as np

def reconstruct(fimg2, fref,mode='real',phi=1):
    ''' Reconstruct an image with a known reference fref.
        phi is the filtering value for the reference division.
        It is assumed that the sample and reference term were subtracted.
Subtract it manually if your intention is to do so.
        mode = 'real' means normalize by the real part of the refere
        mode = 'imag is the opposite.
'''
    if(mode == 'real'):
        frefpart = fref.real
    elif(mode == 'imag'):
        frefpart = fref.imag
        
    w = np.where(np.absolute(frefpart) > phi)
    shp2freal       =  np.zeros(frefpart.shape)
    shp2freal[w]    = fimg2[w]/2./frefpart[w]

    #real/imag part yield symmetric/antisymmetric image pair
    #imag part only retrievable in the case of non negligible absorption
    # so it's ignored
    shp2reconre = fftshift(ifft2(shp2freal)).real
    #shp2reconim = fftshift(ifft2(shp2freal)).imag
    shp2recon   = shp2reconre #+ shp2reconim
    return shp2recon
