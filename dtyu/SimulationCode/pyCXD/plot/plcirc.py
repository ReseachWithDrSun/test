import numpy as np
from matplotlib.pyplot import plot

def plcirc(rc, R,color=None):
    ''' Plot a circle.'''
    pphi = np.linspace(0., np.pi*2,1000);
    plot(R*np.cos(pphi)+rc[0],  R*np.sin(pphi) + rc[1],color=color,hold=True)
