''' A simulated detector.
    Use for simulations.
'''
import numpy as np
import h5py
from pims import FramesSequence, Frame

class SimulImages(FramesSequence):
    def __init__(self,imgs,qperpixel):
        '''Initialize the simulated images object.
            Assumes a T by y by x format, where T is time.
            Assumes it's a numpy array but will still make an attempt to cast
            it.
        sdet.dimx
        sdet.dimy
        '''
        self.imgs = np.array(imgs)
        self.dims = self.imgs.shape

        self.wavelength = sim.wavelength
        self.det_distance = sim.L
        self.pxdimx = sim.pxdimx
        self.pxdimy = sim.pxdimy
        self.xcen = sim.xcen
        self.ycen = sim.ycen
        self.dimx = sim.dimx
        self.dimy = sim.dimy

    def get_frame(self,i):
        '''The get_frame object.'''
        return Frame(self.imgs[i%self.dims[0],:,:],frame_no=i)

    def __len__(self):
        '''len method is mandatory.'''
        return self.dims[0]

    @property
    def frame_shape(self):
        return self[0].shape

    @property
    def pixel_type(self):
        return self[0].dtype
