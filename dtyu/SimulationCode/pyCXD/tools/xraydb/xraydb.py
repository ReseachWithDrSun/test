#xray database
#test reading the database
# note: This is expected to be run from the main directory of pyCXD
CONST_R0 = 2.8179403267e-5#Thomson radius in angs
CONST_AMU2M = 1.66053904e-24 #AMU to mass in grams
CONST_EV2ANGS = 12398.
from tools.xraydb.elems import elems
import numpy as np

def readfqs(Z, energy):
    '''Read the F(q) for a certain Z at a certain energy'''
    elemchar = elems[Z]
    aurho = np.loadtxt("tools/xraydb/db/"+ elemchar + ".nff",skiprows=1)
    print(elemchar)
    energylst = aurho[:,0]
    fqlst = aurho[:,1]
    fq = np.interp(energy, energylst,fqlst)
    fqqlst = aurho[:,2]
    fqq = np.interp(energy, energylst,fqqlst)
    return fq,fqq
