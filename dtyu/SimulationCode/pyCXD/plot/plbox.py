import numpy as np
from matplotlib.pyplot import plot

def plbox(bxcrd,color='k'):
    ''' Plot a box.
    bxcrd - box coordinates
    '''
    x0,y0,x1,y1 = bxcrd
    plot([x0,x0],[y0,y1],color=color);
    plot([x0,x1],[y1,y1],color=color);
    plot([x1,x1],[y1,y0],color=color);
    plot([x1,x0],[y1,y1],color=color);
    plot([x1,x1],[y1,y0],color=color);
