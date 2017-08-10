#rotation 
import numpy as np

def rotate(img,th,imgr,predef=None):
    '''Rotate a function using no interpolation. Should be faster.
        predef is a bunch of predefined variables. They are obtained by calling sub_rotate.
        Center of rotation is center of array.
    '''
    R = np.array([
        [np.cos(th), np.sin(th)],
        [-np.sin(th), np.cos(th)]
    ])

    imgr *= 0
    dims = img.shape
    #img = img.reshape(dims[0]*dims[1])
    #imgr = imgr.reshape(dims[0]*dims[1])
    if(predef is None):
        r,x,y,cen = sub_rotate(dims)
    else:
        r,x,y,cen = predef

    rp = np.dot((r-cen),R) + cen#broadcast cen array
    rp = rp.reshape((dims[0]*dims[1],2))
    xbins = np.linspace(x[0]-.5,x[-1]+.5,dims[0]+1)
    ybins = np.linspace(y[0]-.5,y[-1]+.5,dims[1]+1)    
    xd = np.searchsorted(xbins,rp[:,0],side="right")-1
    yd = np.searchsorted(ybins,rp[:,1],side="right")-1
    w = np.where((xd >= 0) & (xd < dims[0]) & (yd >= 0) & (yd < dims[1]))
    xd = xd[w]
    yd = yd[w]
    #rd = xd + yd*dims[0] 
    #rr = r[w,0] + r[w,1]*dims[0]
    imgr[r[w,0],r[w,1]] = img[xd,yd]

def sub_rotate(dims):
    '''Subroutine to rotate. If you call rotate multiple times, use the
    parameters returned here as the predef parameter to rotate.  It avoids the
    re-creation of many unnecessary routines.'''
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    X,Y = np.meshgrid(x,y)
    r = np.dstack((X,Y))
    r = r.reshape((dims[0]*dims[1],2))
    cen = np.array([dims[0]/2.,dims[1]/2.]).reshape((1,1,2))
    return r,x,y,cen
