from shapes.shapes import ShapeObj
from shapes.shapes import NmerShape
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

'''
Make nmer shapes. In this case pentagon (n=5).
    pars[0] - diameter of spheres
    pars[1] - distance between spheres
    pars[2] - number spheres (n=2 dimer, n=3 trimer etc)
'''

N = 1000

#pars = [20,35,5]
pars = {'radius' : 50,#in nm, also change resolution to nm
        'distance' : 1,
        'symmetry' : 1}

nmershp = NmerShape(pars,N,resolution=10.,unit="$\AA$")
nmershp.addshape(-50,-100)
nmershp.addshape(50,100,phi=np.pi/7.)
nmershp.plotimg(winnum=0)
nmershp.calcFFT()
nmershp.plotscat(winnum=1)
