import numpy as np
from detector.eiger import EigerImages

def sumimgs(IMGS,imgthresh=None,PF=False,badpixels=None):
    ''' Sum an iterable IMGS together.
        NOTE: Assumes same exposure time for each.
        badpixels must be same dims as IMG (returns pixel mask for masked pixels)
'''
    pxmask = np.zeros(IMGS.dims)
    imgtot = np.zeros(IMGS.dims)
    Ivst = np.zeros((len(IMGS)))
    totpxno = IMGS.dims[0]*IMGS.dims[1]
    # initialize it to zero
    if(badpixels is not None):
        badpixels *= 0

    for i in range(len(IMGS)):
        img = IMGS[i]
        if(imgthresh is not None):
            w = np.where(img.ravel() < imgthresh)
            if(len(w[0]) != totpxno):
                nobad = totpxno-len(w[0])
                if(PF):
                    print("There are {} bad pixels ({:5.2f}% of detector)".format(nobad,nobad/totpxno*100))
                if(badpixels is not None):
                    w2 = np.where(img.ravel() > imgthresh)
                    badpixels.ravel()[w2] = 1
            imgtot.ravel()[w] += img.ravel()[w]
            pxmask.ravel()[w] += 1
            Ivst[i] += np.average(img.ravel()[w])
        else:
            imgtot += img
            Ivst[i] += np.average(img)
    if(imgthresh is None):
        pxmask = np.zeros(IMGS.dims) + len(IMGS)

    return imgtot,Ivst,pxmask

def avgimgs(IMGS,imgthresh=None,badpixels=None):
    ''' Sum an iterable IMGS together.
        NOTE: assumes same exposure time for each.
        badpixels must be same dims as IMG (returns pixel mask for masked pixels)
    '''
    imgtot, ivst, pxmask = sumimgs(IMGS,imgthresh=imgthresh,badpixels=badpixels)
    if(imgthresh is not None):
        w = np.where(pxmask.ravel() != 0)
        imgtot.ravel()[w] /= (pxmask.ravel()[w]).astype(float)
    else:
        imgtot /= float(len(IMGS))

    return imgtot,ivst

def sumlist(filelist,imgthresh=None,DDIR='.',normexp=True,PF=False):
    ''' Sum the images from a list.
        normexp: if True multiply exposure (so you get result in 
            total cts/total secs instead of tot counts/tot frames)
        
        NOTE: If you set normexp to False, it assumes the same exposure time
            for each image.
        If you want to automatically multiply the exposure time
            (convert to cts/sec rather than cts/frame), then you 
            need to set "normexp" to True
        If non uniform exposure times, you MUST set "normexp" to True.
    '''
    IMGS = EigerImages(DDIR + "/" + filelist[0])
    imgtot = np.zeros(IMGS.dims)
    pxmasktot = np.zeros(IMGS.dims)
    ivframe = np.zeros(len(filelist))
    for i, filename in enumerate(filelist):
        IMGS = EigerImages(DDIR + "/" + filename)
        img, ivst, pxmask = sumimgs(IMGS,imgthresh=imgthresh,PF=PF)
        if(normexp):
            imgtot += img*IMGS.exposuretime
        else:
            imgtot += img
        pxmasktot += pxmask
        ivframe[i] += np.sum(img)/float(np.count_nonzero(pxmask))
    return imgtot, ivframe, pxmasktot

def avglist(filelist,imgthresh=None,DDIR=".",normexp=True,PF=False):
    ''' Sum the images from a list.
        normexp: if True multiply exposure (so you get result in 
            cts/sec instead of counts/frame)
        
        NOTE: If you set normexp to False, it assumes the same exposure time
            for each image.
        If you want to automatically multiply the exposure time
            (convert to cts/sec rather than cts/frame), then you 
            need to set "normexp" to True
        If non uniform exposure times, you MUST set "normexp" to True.
    '''
    imgtot, ivframe, pxmask = sumlist(filelist,imgthresh=imgthresh,DDIR=DDIR,PF=PF)
    w = np.where(pxmask.ravel() != 0)
    imgtot.ravel()[w] /= (pxmask.ravel()[w]).astype(float)
    return imgtot, ivframe
