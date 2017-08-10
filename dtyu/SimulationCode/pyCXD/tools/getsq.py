from tools.circavg import circavg
from numpy.fft import fft2, fftshift
import numpy as np

def getsq(img):
    '''get cir avg structure factor from an image.'''
    fimg = np.absolute(fftshift(fft2(img)))**2
    return circavg(fimg)
