''' This is an example on how to use pims readers to make any format you wish.
This one is a simulated detector.'''
from detector.simul import SimulImages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.ion()

imgs = np.random.random((4,100,100))
imgs = SimulImages(imgs)

fig = plt.figure(0)
#plt.imshow(imgs[0])
animation.FuncAnimation(fig,imgs.get_frame)
