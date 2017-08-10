import numpy as np
import sys


# python splitter.py [input] [output1] [output2] [dims1]
if __name__ == '__main__':
    fin = sys.argv[1]
    fout1 = sys.argv[2]
    fout2 = sys.argv[3]
    dims1 = int(sys.argv[4])
    nin = np.load(fin)
    nout1 = nin[:dims1,:]
    nout2 = nin[dims1:,:]
    np.save(fout1, nout1)
    np.save(fout2, nout2)
