import numpy as np

def rdIpix(pxlst,pxind,imgs,IAVG=None,Ivst=None):
    '''Read the ipix array.'''

    print("Reading Ipix")

    if(Ivst is None):
        Ivst = np.ones(len(imgs))
    Itot = np.average(Ivst)

    Ipix = np.zeros((len(imgs),len(pxlst)))

    if(IAVG is None):
        IAVG = np.ones(len(pxlst))
    else:
        IAVG = IAVG.ravel()[pxlst]
        #check that IAVG is okay
        w = np.where(IAVG == 0)[0]
        if(len(w)):
            print("Warning there are zero entries in IAVG. Setting to 1 for now\
                    but this result will probably not make sense unless you know\
                    what you're doing!")
            IAVG[w] = 1.

    for i in range(len(imgs)):
        Ipix[i] = imgs[i].ravel()[pxlst]/IAVG #*Itot/Ivst[i]
    return Ipix

def compute_h2t(pxlst, pxind, imgs, IAVG=None,Ipix=None,Ivst=None):
    ''' Compute the two time correlation function for a sequenc of images.
            pxlst - the list of pixels to analyze
            pxind - the index of pixels (set them all 1 if you're looking at one
                    partition, or set them to integers if you want to analyze multiple
                    g2's in parallel.
            imgs - an image sequence. It should have a len(img) = no images
                as well as img[i] returns ith image, and a dims member which
                are the image dimensions.
            IAVG - the average image. It should be the full image. Since it's just 
                passed by reference this shouldn't really matter.
    ''' 
    if(IAVG is None):
        #Not good, you should always have an average
        #but if you don't have one....
        IAVG = np.ones(imgs.dims)

    if(Ivst is None):
        #Normalized I versus time
        Ivst = np.ones(len(imgs))

    #just read in the pixels needed into a new image
    if(Ipix is None):
        Ipix = rdIpix(pxlst,pxind,imgs,Ivst=Ivst,IAVG=IAVG)
    elif(Ipix.shape != (len(imgs),len(pxlst))):
        print("Error. Wrong dimensions for Ipix. Aborting...")

    noperbin = np.bincount(pxind)
    w = np.where(noperbin == 0)[0]
    if(len(w) > 0):
        print("Warning, you have some bin id's with nothing in them, ignoring")
        noperbin[w] = 1
        print(noperbin)
    nobins  = len(noperbin) + 1
    notimes = len(imgs)
    # basic algorithm of h2t
    # you look at time differences, which will show up first in a matrix
    # the two time matrix
    ttmat = np.zeros((nobins,notimes,notimes))
    for i in range(notimes):
        #print("")
        for j in range(i,notimes):
            #print("Iteration {},{}...".format(i,j))
            ttelem = imgs[i].ravel()[pxlst]*imgs[j].ravel()[pxlst]
            ttmat[:,i,j] = (np.bincount(pxind,weights=ttelem))/noperbin
            ttmat[:,j,i] = ttmat[:,i,j]
    
    return ttmat

def ttmat2g2(ttmat):
    ''' Average a two time matrix into a g2 correlation function. Average regions
        of constant t2-t1.
        Right now only works on one element, need to have it work on multiple
        elements later.
        '''
    #watch out for dimensions, np.meshgrid flips x and y, difference 
    # between image notation and matrix notation
    t1 = np.arange(ttmat.shape[2])
    t2 = np.arange(ttmat.shape[1])
    T1,T2 = np.meshgrid(t1,t2)
    TAU = (T2-T1).ravel()
    w = np.where(TAU > -1)
    nopertime = np.bincount(TAU[w])
    notimes = len(nopertime)
    #g2 = np.zeros(notimes)
    g2 = np.bincount(TAU.ravel()[w],weights=ttmat[0].ravel()[w])
    w = np.where(nopertime == 0)[0]
    if(len(w) > 0):
        print("Warning, there are zero time entries, this error should not happen.")
        nopertime[w] = 1
    g2 /= nopertime
    return g2
