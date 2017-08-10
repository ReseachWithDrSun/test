# a tile shape wrapper for 2d tile shapes.
# maybe this could be made into an object eventually
from plot.plcirc import plcirc
from shapes.shapes import NmerShape3DProj

def mktileshp(r, d, tilebin, N=1000):
    ''' make a tile shape
        r - radius
        d - distance
        tilebin - the tilebin (needs to be n x n array of 0 or 1)
        N - number of pixels per dimension
'''
    pars = {'radius' : r, 'distance' : 10, 'symmetry' : 1}
    shp = NmerShape3DProj(pars, N)
    Nx = tilebin.shape[1]
    Ny = tilebin.shape[0]
    for i in range(Ny):
        for j in range(Nx):
            if(tilebin[i][j] == 1):
                shp.addshape((j-Nx/2)*d, (i-Ny/2)*d)
    return shp
