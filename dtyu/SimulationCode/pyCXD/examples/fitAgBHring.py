from fit.fitAgBHring import fitAgBHring
import numpy as np

simulate = False

if simulate:
    # just to make the data, you would replace this with real data
    from lmfit import Parameters
    from fit.fitfns import ring2Dlorentzfitfunc
    x = np.arange(1000)
    y = np.arange(1000)
    X,Y = np.meshgrid(x,y)
    
    pars = Parameters()
    pars.add('amp',10)
    pars.add('r0',276)
    pars.add('sigma', 15)
    pars.add('x0', 433)
    pars.add('y0', 345)
    pars.add('bg', 0)
    pars.add('bgexp', 1)
    pars.add('const', 0)
    IMG = ring2Dlorentzfitfunc([X,Y],pars)
    IMG += np.random.randn(IMG.shape[0],IMG.shape[1])*5
    mask = np.ones(IMG.shape,dtype=int)
else:
    IMG = np.load("examples/data/AgBHringdata.npy")
    mask = np.load("examples/data/AgBHringmask.npy")

fitAgBHring(IMG,mask=mask,plotwin=0)
