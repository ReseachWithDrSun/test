import numpy as np
'''Make a scattering pattern of intensities digital by adding poisson noise
    and a threshold.'''

def mkdigital(img, mxcounts):
    '''Turn an image into a digital image.
        Treat each value as mean for a poisson distribution.
        Then threshold for the mxcounts value.
        mxcounts is the dynamic range.'''

    w = np.where(img < 0)
    if(len(w[0]) > 0):
        print("Warning, this image has negative values. This is not realistic.\
            Ignoring these values...")
    img[w]  = 0

    pimg = np.random.poisson(img)

    pimg = pimg*(pimg < mxcounts)

    return pimg
