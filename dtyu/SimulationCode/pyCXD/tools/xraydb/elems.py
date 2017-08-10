#read in elems
import numpy as np

ELEMS_FILENAME = "tools/xraydb/elems.csv"

def readelems(filename):
    ''' Read the elements from the csv file of type:
        Z (int), symbol (char), mass number (int), AMU (float)
        return a tuple of arrays of each:
        Z (int), symbol (char), mass number (int), AMU (float)
    '''
    lines = [line.rstrip('\n') for line in open(filename)]
    
    nelems = len(lines)-1
    amus = np.zeros(nelems)
    symbols = ["" for x in range(nelems)]
    massnos = np.zeros(nelems,dtype=int)
    Zs = np.zeros(nelems,dtype=int)
    
    for i in range(len(lines)-1):
        a = lines[i+1].split(',');
        Zs[i] = a[0]
        symbols[i] = a[1].strip()
        massnos[i] = a[2]
        amus[i] = a[3]
    return Zs, symbols, massnos, amus
