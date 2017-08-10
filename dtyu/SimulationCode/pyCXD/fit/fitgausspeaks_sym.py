from lmfit import Model, Parameters
#from fit.FitObject import FitObject, Parameters
import matplotlib.pyplot as plt
import numpy as np
from fit.fitfns import gausspeaks_sym

class GaussPeaks_Sym(Model):
    ''' Gausspeaks_sym requires all other flags except function name.
        Uses Model object from lmfit. General usage example (or see
        documentation online):
        gmod = GaussPeaks_Sym()
        gmod.fit(y,phis=x,amp=1,phi=0,sym=3)
    '''
    def __init__(self, *args, **kwargs):
        super(GaussPeaks_Sym, self).__init__(gausspeaks_sym, *args, **kwargs)
        #Model.__init__(self, gausspeaks_sym, *args, **kwargs)
        #super().__init__(gausspeaks_sym, *args, **kwargs)
        self.set_param_hint('sym', vary=False, value=6)
        self.set_param_hint('sigma', vary=True, value=.01, min=1e-6,max=1e6)
        self.set_param_hint('amp', vary=True, value=1, min=0)
        # phi cannot be out of +/- pi/2
        self.set_param_hint('phi', vary=True, value=0, min=-np.pi/2., max=np.pi/2.)
