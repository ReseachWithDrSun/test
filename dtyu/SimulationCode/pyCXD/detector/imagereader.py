from collections import Sequence
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py

'''Python has a nice way to allow users to define their own sliceable objects.
    By slicable, I mean that basically I can take some object C and C[100]
    will return something else.
    Here is the general way to do this.
    For example, this one would work like:

    import matplotlib.pyplot as plt
    imgs = PyFitsReader(filename)
    myimg = imgs[1]
    plt.imshow(myimg)
    #len function knows the size
    print("You have {} images in this sequence".format(len(imgs)))
'''
 
class PyFitsReader(Sequence):
    def __init__(self,filename,bgfile=None):
        hdulist = pyfits.open(filename)
        self.IMGS = np.copy(hdulist[0].data)
        hdulist.close()
        self.dims = self.IMGS.shape[1],self.IMGS.shape[2]
        self.clim0, self.clim1 = None, None
        self.fig = plt.figure(0)
        if(bgfile is not None):
            f = h5py.File(bgfile,"r")
            bgimg = np.copy(f['BGIMG'])
            f.close()
            self.bgimg = bgimg
 
    def __len__(self):
        return self.IMGS.shape[0]
 
    #def append(self, item):
        #self.data.append(item)
 
    #def remove(self, item):
        #self.data.remove(item)
 
    def __repr__(self):
        mystr = "Image sequence, {} frames total".format(self.IMGS.shape[0])
        return mystr
 
    def __getitem__(self, sliced):
        img = self.IMGS[sliced]
        if(hasattr(self,"bgimg")):
            img -= self.bgimg
        return img

    def ldimgs(self,rng=None):
        '''Load images from a range in time.
        rng : array of dims 2  [start,finish]'''
        if(rng is None):
            rng = [0,len(self)-1]
        imgs = np.zeros((len(self),self.dims[0],self.dims[1]))
        for i,ind in enumerate(range(rng[0],rng[1])):
            imgs[i] = np.copy(self[ind])
        return imgs
