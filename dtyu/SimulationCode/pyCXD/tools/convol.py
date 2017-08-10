import numpy as np

def convol1d(a,b=None,axis=-1):
    ''' convolve a with b. If b not specified, perform
        a self convolution.'''
    from numpy.fft import fft, ifft
    if(b is None):
        b = a
    return ifft(fft(a,axis=axis)*np.conj(fft(b,axis=axis)),axis=axis).real

def convol2d(a,b=None,axes=(-2,-1)):
    ''' convolve in 2 dimensions.'''
    from numpy.fft import fft2, ifft2
    if(b is None):
        b = a
    return ifft2(fft2(a,axes=axes)*np.conj(fft2(b,axes=axes)),axes=axes).real
