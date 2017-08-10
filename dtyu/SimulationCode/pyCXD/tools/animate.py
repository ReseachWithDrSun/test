'''Animation tool, words on a sliceable array.'''
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(IMGS,fignum=0):
    ''' Run the matplotlib animation function.'''
    anid = FuncAnimation(fig,draw_frame)
    n = n%self.IMGS.shape[0]
    plt.cla();
    plt.imshow(self.IMGS[n]);
    plt.clim(self.clim0,self.clim1)
    plt.pause(.0001)
