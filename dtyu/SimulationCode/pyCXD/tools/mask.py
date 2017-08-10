import h5py
import numpy as np

def openmask(maskfilename):
    ''' Open a mask with file name. Just a quick shortcut to using hdf5 etc'''
    f = h5py.File(maskfilename,"r")
    mask = np.copy(f['mask'])
    f.close()
    return mask

def savemask(maskfilename,mask):
    ''' Save a mask.'''
    f = h5py.File(maskfilename, "w")
    f['mask'] = mask
    f.close()
