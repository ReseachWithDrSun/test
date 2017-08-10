import numpy as np
def digitize(a,b):
    ''' a wrapper for digitize to speed it up'''
    return np.searchsorted(b,a,"right")
