'''Extra interpolation techniques.'''
import numpy as np

def fillin1d(xdata,data,mask,period=None):
    ''' Fill in a 1d array by interpolating missing values.
        axis: if array is more than one dimension,
            run the correlation on the proper axis
        For numpy version 1.10 and higher, you can wrap around 
            for angular values by specifying period =1 (or any nonzero number)
    '''
    data2 = np.copy(data)
    w = np.where(mask < 1e-6)[0]
    w2 = np.where(mask >= 1e-6)[0]
    if(len(w) > 0 and len(w2) > 0):
        if(period is None):
            # this is meant to be backwards compatible with old numpy versions
            data2[w] = np.interp(xdata[w],xdata[w2],data[w2])
        else:
            data2[w] = np.interp(xdata[w],xdata[w2],data[w2])#,period=period)
    return data2
