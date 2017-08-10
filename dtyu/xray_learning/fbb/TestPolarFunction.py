from fbb.PolarFunction import PolarFunction
import numpy as np
import math


def NarrowPeakFunction(w, x):
    waves = np.sin( 2*math.pi/w * (x-1+w/4) )/2 + 1/2
    return np.multiply(waves, np.logical_and(x > 1-w/2, x<=1))


class TestPolarFunction(PolarFunction):
    def __init__(self, a, s, c):
        super(TestPolarFunction, self).__init__(a, s, c)

    def PhiPeriodic(self, n_sym, width, mag, bias):
        x = self.phi * n_sym / (2 * math.pi)
        x -= np.floor(x)
        x = 2 * abs(x - .5)
        x = np.multiply(x, abs(x) <= 1.)
        return NarrowPeakFunction(width, x) * mag + bias

    def RBand(self, lower, upper):
        return np.logical_and(self.r > lower, self.r < upper)

    def RSquared(self, mul=1):
        r2 = np.multiply(self.r, self.r)
        return mul * r2

    def test_function(self):
        return np.multiply(self.PhiPeriodic(4, 0.2, 0.5, 0.1),
                           self.RBand(100, 102))
