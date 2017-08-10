# xray database and conversion tools
# note: This is expected to be run from the main directory of pyCXD
CONST_R0 = 2.8179403267e-5#Thomson radius in angs
CONST_AMU2M = 1.66053904e-24 #AMU to mass in grams
CONST_EV2ANGS = 12398.
from tools.xraydb.elems import readelems, ELEMS_FILENAME
import numpy as np

# read in the elements
Zs, symbols, massnos, amus = readelems(ELEMS_FILENAME)

def readfqs(Z, energy):
    '''Read the F(q) for a certain Z at a certain energy (in eV)'''
    elemchar = symbols[np.where(Zs==Z)[0]].lower()
    aurho = np.loadtxt("tools/xraydb/db/"+ elemchar + ".nff",skiprows=1)
    print(elemchar)
    energylst = aurho[:,0]
    fqlst = aurho[:,1]
    fq = np.interp(energy, energylst,fqlst)
    fqqlst = aurho[:,2]
    fqq = np.interp(energy, energylst,fqqlst)
    return fq,fqq

def den2eden(rho,elems,weights):
    ''' Density (g/cm^3) to electron density (e-/nm^3)
        rho - density (g/cm^3)
        elems - list of elements
        weights - the number per element in molecule
    '''
    # convert AMU to g/e-
    # (  g/cm^3 ) ( 1 molecule/amutot g ) ( n electrons/ molecule)
    elems = np.array(elems,ndmin=1)
    eleminds = np.zeros(elems.shape,dtype=int)
    # there are isotopes etc so you need to find the element
    # with matching atomic number. the first is the non-isotope
    for i,elem in enumerate(elems):
        eleminds[i] = np.where(Zs == elems[i])[0][0]
    amutot = np.sum(amus[elems]*weights)
    masstot = amutot*CONST_AMU2M #convert to mass in g
    electrons = np.sum(Zs[elems]*weights)
    eden = rho/masstot*electrons # eden in e-/cm^3
    eden *= 1e-21 # to e-/nm^3
    return eden
