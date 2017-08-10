import numpy as np
import math


class PolarFunction:
    '''
    Args:
        a: disk radius
        s: image size (length)
        c: center position
    '''
    def __init__(self, a, s, c):
        self.a = 0
        self.s = 0
        self.c = None
        self.r = None
        self.phi = None
        self.set_size(a, s, c)

    def set_size(self, a=None, s=None, c=None):
        if a is not None:
            self.a = a
        if s is not None:
            self.s = s
        if c is not None:
            self.c = c
        self.polar_mesh()

    def polar_mesh(self):
        x = np.arange(0., self.s, 1)
        y = np.arange(0., self.s, 1)
        xx, yy = np.meshgrid(x, y)
        xx -= self.c[0] + 1e-16
        yy -= self.c[1] + 1e-16

        self.phi = np.arctan(yy / xx)
        self.phi = self.phi + (xx < 0) * math.pi + np.logical_and(xx >= 0, yy < 0) * 2 * math.pi
        self.r = np.sqrt(xx ** 2 + yy ** 2)
        # reset image origin
        cx = int(round(self.c[0]))
        cy = int(round(self.c[1]))
        if abs(cx - self.c[0]) < 1e-16 and abs(cy - self.c[1]) < 1e-16:
            self.phi[cy][cx] = 0