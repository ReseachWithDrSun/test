#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Imports
########################################

import sys, os
SciAnalysis_PATH='/home/kyager/current/code/SciAnalysis/main/'
SciAnalysis_PATH in sys.path or sys.path.append(SciAnalysis_PATH)

import glob
from SciAnalysis import tools
from SciAnalysis.Data import *
#from SciAnalysis.XSAnalysis.Data import *
#from SciAnalysis.XSAnalysis import Protocols

if len(sys.argv)>1:
    # Use commandline argument
    idx = int( '0x{}'.format( sys.argv[1] ), base=0 )
    infile = '{:08x}'.format(idx)
    
else:
    exit()


root_dir = './'
source_dir = os.path.join(root_dir, '../data', '001303a3_adhocs')
#output_dir = os.path.join(root_dir, './')


infile = '{}/{}.npy'.format(source_dir, infile)
img = np.load(infile)
data = Data2D()
data.data = img
data.plot(show=True)

