#Note: for this to work, you need to ensure that your DDIR is
# pointing to the right data directory. DDIR is set in config.py
#Reading template, to build upon in developing SAXS code
#General libraries
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
#matplotlib.use('Qt4Agg')
#matplotlib.use('GTKAgg')
#matplotlib.use('GTK')
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'toolbar2'
import os.path
import h5py
import numpy as np
from matplotlib.pyplot import *
from numpy.fft import fft2,ifft2,fftshift
from tools.mask import openmask,savemask

from tools.optics import *

#Custom Libaries
#some config parameters that are data independent (data root directory etc)
# This loads SDIR and DDIR (storage and read only data directories)
from config import *
#tools for the xray data
from tools.Xtools import *
#Customized plotting interaction
#from plot.ZoomFigure import ZoomFigure as zfig
#the detector
#from detector.eiger import EigerImages
#mask creator
from tools.MaskCreator import MaskCreator
#logbooks
import tools.SciLog as SciLog

from plot.limg import limg

from tools.convol import convol1d, convol2d

ion()

#laptop
laptop = 0
homews = 0
workws = 1
if (laptop == 1):
    DDIR = "/media/usbhd-sdb2/NSLSII_data/CHX/2015/07"
    SDIR = "../storage"
    SCRIPTSDIR = "pyscripts"
    LISTSDIR = "pylists"
    PYLISTDIR = "../pylists"
    LOGDIR = "/home/julienl/logbooks/bnl/bnl-mesoimaging"
    LOGPDIR = "/home/julienl/logbooks/bnl"
elif (homews==1):
    DDIR = "/media/usbhd-sdb2/NSLSII_data/CHX/2015/07"
    SDIR = "../storage"
    SCRIPTSDIR = "pyscripts"
    LISTSDIR = "pylists"
    PYLISTDIR = "../pylists"
    LOGDIR = "/home/julienl/logbooks/bnl/bnl-mesoimaging"
    LOGPDIR = "/home/julienl/logbooks/bnl"
else:
    DDIR = "/mnt/d1/NSLSII_Data/CHX/2015_08"
    SDIR = "../storage"
    SCRIPTSDIR = "pyscripts"
    LISTSDIR = "pylists"
    PYLISTDIR = "../pylists"
    LOGDIR = "/home/lhermitt/logbooks/bnl/bnl-mesoimaging"
    LOGPDIR = "/home/lhermitt/logbooks/bnl"


#set up default logbook
lbook = SciLog.SciLog(LOGDIR)

ion()
