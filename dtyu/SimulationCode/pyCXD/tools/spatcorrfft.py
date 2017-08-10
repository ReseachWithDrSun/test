''' The spatial correlation fft routine.
    The idea is to use the FFT to perform an autocorrelation of two images
    (convolution). This requires more memory but the calculation is faster
    than the brute force method.
    Requires the numpy fft libraries
    '''
from numpy.fft import fft2, ifft2, fftshift
import numpy as np

def spatcorrfft(pxlst,pxind,IMGS):
    '''Compute the spatial correlation of images.'''
    '''Algorithm:
        1. over time
            1. 

    '''
    #dummy for now
    dimx,dimy = 100,100
    #find the maximum extensiono of region
    imwidth = IMGS.dims[0]
    subpxlst,mask = mkbox(pxlst,dims)
    imgs = IMGS.ldimgs()

    #convolute the mask

    #now iterating over frames



    tsc = np.zeros(len(IMGS),dimx,dimy )

def mkbox(pxlst,dims):
    ''' Make the image box with mask from the pixels selected and the known imgwidth'''

    imgwidth = dims[1]
    #x and y are flipped???
    #matrix notation!!!
    pixely = pxlst%imgwidth
    pixelx = pxlst//imgwidth

    minpixelx = np.min(pixelx)
    minpixely = np.min(pixely)
    maxpixelx = np.max(pixelx)
    maxpixely = np.max(pixely)

    widthx = maxpixelx - minpixelx + 1
    widthy = maxpixely - minpixely + 1

    oldimg = np.zeros(dims)
    oldimg.ravel()[pxlst] = 1
    mask = np.copy(oldimg[minpixelx:maxpixelx+1,minpixely:maxpixely+1])
    subpxlst = np.where(mask.ravel() == 1)[0]
    print("min/maxpixelx: {},{};min/maxpixely: {},{}".format(minpixelx,maxpixelx,minpixely,maxpixely))

    return subpxlst,mask
    
def autocorr(img1,img2,mask=None):
    '''Compute the autocorrelation of two images.
        Right now does not take mask into account.
        todo: take mask into account (requires extra calculations)
    '''
    if(mask is not None):
        img1 *= mask
        img2 *= mask
    imgc = fftshift(ifft2(fft2(img1)*np.conj(fft2(img2))).real)
    if(mask is not None):
        maskc = autocorr(mask,mask)
        w = np.where(maskc > 0)[0]
        if(len(w) > 0):
            imgc[w] /= maskc[w] 
        w = np.where(maskc == 0)
        if(len(w)):
            imgc[w] *= 0
    return imgc
