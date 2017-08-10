import matplotlib.pyplot as plt
import numpy as np

def movie(img,winnum=None,delay=.1,axis=0,clims=None):
    ''' cycle through an image as a movie'''
    if(winnum is not None):
        plt.figure(winnum)
    cnt = 0
    dim = img.shape[axis]
    try:
        while(True):
            plt.cla();
            plt.imshow(np.take(img,cnt,axis=axis));
            if(clims is not None):
                plt.clim(clims[0],clims[1])
            plt.draw();
            plt.pause(delay);
            cnt += 1
            cnt = cnt%dim
    except KeyboardInterrupt:
        return
    
