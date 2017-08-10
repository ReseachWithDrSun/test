#Note: for this to work, you need to ensure that your DDIR is
# pointing to the right data directory. DDIR is set in config.py
#Reading template, to build upon in developing SAXS code
#General libraries
import os.path
import matplotlib.pyplot as plt

#Custom Libaries
#some config parameters that are data independent (data root directory etc)
# This loads SDIR and DDIR (storage and read only data directories)
from config import *
#tools for the xray data
from tools.Xtools import *
#Customized plotting interaction
from plot.ZoomFigure import ZoomFigure as zfig
#the detector
from detector.eiger import EigerImages 

ion()

#Data Setup (eventually this should be saved in a file describing data
# but for now will be saved here)
#Expt setup settings:
expt = ExpPara()
expt.mask_filename = SDIR + "/" + "mask_Wenyan1.hd5"
expt.dsetname = "30/Wenyan_E13_2160_master.h5"
filename = DDIR + "/" + expt.dsetname
det = EigerImages(filename)
#corrections to detector
det.beamx0 = 601
det.beamy0 = 808
det.det_distance = 1.400#in m
det.wavelength = 1.3758913# in angstroms (9keV)

sd1 = SAXSObj(expt,det,PF=1)
sd1.mkqlist()
sd1.qpartition()
sd1.qbinavg()
sd1.plotsq(winnum=0)

sd1.plotlimg(winnum=1)
sd1.plotring(100)
sd1.plotcen()
#sd1.updatemask(10)
#sd1.mcreator.set_clim(0,10)


'''Should we add these extra parameters here?
    -two theta arm, chi arm (angle orthogonal to two theta?)
    -detector specific rotation angles: detphi, dettheta and detchi
    -sample specific? (need to specify a sample orientation though)
'''
