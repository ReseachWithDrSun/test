import numpy as np
from plugins.rotate._rotate import rotate
from tools.circavg import circavg
from scipy.ndimage.filters import gaussian_filter

def gaussfill(img, mask,sigma=30):
    ''' Fill in masked regions by a Gaussian averaged version.
        img - the image
        mask - the mask
        sigma (default: 30) - the std dev for the Gaussian kernel (twice this
            should be about the longest void in mask)
    '''
    imgf = np.copy(img)
    imgg = gaussian_filter(img,sigma)
    imgmsk = gaussian_filter(mask*1.,sigma)
    imgmskrat = gaussian_filter(mask*0.+1,sigma)
    w = np.where((mask < .1)*(imgmsk > 0))
    imgf[w] = imgg[w]/imgmsk[w]*imgmskrat[w]
    return imgf

def circavgfill(img, mask, x0=None,y0=None,ps=True,poisson=False):
    ''' Fill using circular average.
        img - the image
        mask - the mask
        x0, y0 - the center of the image
        ps - also use point symmetry
    '''
    if(x0 is None):
        x0 = img.shape[1]/2
    if(y0 is None):
        y0 = img.shape[0]/2
    SIMG = img*0
    sqx,sqy = circavg(img,mask=mask,x0=x0,y0=y0,SIMG=SIMG)
    w = np.where((mask < .01))
    imgcavg = np.copy(img)
    if(poisson is False):
        imgcavg[w] = SIMG[w]
    else:
        imgcavg[w] = np.random.poisson(SIMG[w])

    if(ps):
        mask = mask.astype(float)
        img = img.astype(float)
        imgr = imgcavg*0
        maskr = mask*0
        rotate(imgcavg,imgr, np.pi, x0, y0)
        rotate(mask,maskr, np.pi, x0, y0)
        w = np.where((mask < .1)*(maskr > .1))
        imgcavg[w] = imgr[w]

    return imgcavg
