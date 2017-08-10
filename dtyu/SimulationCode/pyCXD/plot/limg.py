import numpy as np

def limg(IMG):
    ''' Take the safe logarithm of an image'''
    w = np.where(IMG != 0)
    limg = np.zeros(IMG.shape)
    if(len(w) > 0):
        limg[w] = np.log10(IMG[w])
    return limg
