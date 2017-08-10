#list h5dffile contents
import h5py

def listhdf5(filename):
    ''' List the contents of an hdf5 file from a filename.'''
    f = h5py.File(filename,"r")
    _listhdf5(f)
    f.close()

def _listhdf5(f):
    ''' List the contents of an hdf5 file.'''
    for key in f.keys():
        print(key)
        h = f['/' + key]#h.name.split[-1]]
        if(hasattr(h,"visit")):
            listhdf5(h)
        else:
            print("/{}".format(h.name))
            
