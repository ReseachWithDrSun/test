#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os
SciAnalysis_PATH= '../SciAnalysis'
SciAnalysis_PATH in sys.path or sys.path.append(SciAnalysis_PATH)

from scipy.io import savemat

import numpy as np
from generators import *
import random
#import glob
from SciAnalysis import tools 
from SciAnalysis.Data import *
from SciAnalysis.XSAnalysis.Data import * 
#from SciAnalysis.XSAnalysis import Protocols
from oned import oneD_intensity ##\scripts

import time

from generators import * 
processor = tools.Processor()
protocol = tools.Protocol()
protocol.name = 'tag_generated'
protocol_cal = tools.Protocol()
protocol_cal.name = 'calibration_generated'



# Standard values (e.g. for Eiger 4M)
#q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX)
qMIN = 4e-3
qTYP = 2e-2
qMAX = 1e-1
qZOOM = 1.0
pZOOM = 1.0


# For smaller detector
#q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX)
qMIN = 4e-3*2
qTYP = 2e-2
qMAX = 1e-1
qZOOM = 1.0
pZOOM = 1.0/4.0



def prob(chance):
    return np.random.uniform(0,1)<chance


def define_experiment(run_id):

    tags_experiment = {}
    tags_experiment['experiment.measurement.TSAXS'] = True

    protocol_cal.start_timestamp = time.time()

    calibration = Calibration()
    ##mask_dir = '/home/kyager/current/code/SciAnalysis/main/SciAnalysis/XSAnalysis/masks/'
    mask_dir = os.path.join(SciAnalysis_PATH, 'SciAnalysis/XSAnalysis/masks')
    mask = Mask()

    if False:
        # Bruker Nanostar
        calibration.set_energy(8.04) # keV
        calibration.set_image_size(2048)
        calibration.set_pixel_size(width_mm=140.0)
        calibration.set_distance(1.15508)

        x0, y0 = 1019.0, 1029.5
        x0 = np.random.uniform(0.4*calibration.width, 0.6*calibration.width)
        y0 = np.random.uniform(0.4*calibration.width, 0.6*calibration.width)
        calibration.set_beam_position(x0, y0)
        if x0<0 or x0>calibration.width or y0<0 or y0>calibration.height:
            tags_experiment['instrumental.beamstop.beam off image'] = True

        calibration.max_counts = 2**16

        mask.load(mask_dir+'Bruker_Nanostar_SAXS-mask.png')

        tags_experiment['instrumental.detector.Bruker Nanostar'] = True

        beam = direct_beam(calibration, width=20.0, height=20.0, peak=1e5)


    elif False:
        # CHX (NSLS-II 11-ID)
        calibration.set_energy(8.8984) # CHX
        calibration.set_image_size(2070, 2167) # Eiger 4M
        calibration.set_pixel_size(pixel_size_um=75.0)
        calibration.set_distance(4.755)

        x0, y0 = 838.5, 1833.4
        x0 = np.random.uniform(-0.1*calibration.width, 0.75*calibration.width)
        y0 = np.random.uniform(0.6*calibration.width, 1.1*calibration.width)
        calibration.set_beam_position(x0, y0)
        if x0<0 or x0>calibration.width or y0<0 or y0>calibration.height:
            tags_experiment['instrumental.beamstop.beam off image'] = True


        calibration.max_counts = 2**24

        mask.load(mask_dir+'Eiger4M_all_gaps-mask.png')
        #mask.load(mask_dir+'CHX_Eiger4M-bad_flatfield_10percent-mask.png')
        #mask.load(mask_dir+'CHX_Eiger4M-bad_flatfield_05percent-mask.png')
        #mask.load(mask_dir+'CHX_pipe-2015Oct-mask.png')
        #mask.load(mask_dir+'CHX_bs_streaks-2015Oct-mask.png')

        tags_experiment['instrumental.detector.Eiger 4M'] = True

        beam = direct_beam(calibration, width=8.0, height=2.0, peak=1e5)

    elif False:
        # Conceptual detector (for testing)
        calibration.set_energy(8.8984) # CHX
        calibration.set_image_size(1000,1000) ##(256,256)
        calibration.set_pixel_size(pixel_size_um=75.0)
        calibration.set_distance(4.755/8)

        x0, y0 = 250, 200
        x0 = np.random.uniform(-0.05*calibration.width, 0.25*calibration.width)
        y0 = np.random.uniform(0.7*calibration.height, 0.95*calibration.height)
        calibration.set_beam_position(x0, y0)
        if x0<0 or x0>calibration.width or y0<0 or y0>calibration.height:
            tags_experiment['instrumental.beamstop.beam off image'] = True

        calibration.max_counts = 2**24

        mask.data = np.ones((calibration.height,calibration.width))
        #mask.load(mask_dir+'Eiger4M_all_gaps-mask.png')
        #mask.load(mask_dir+'CHX_Eiger4M-bad_flatfield_10percent-mask.png')
        #mask.load(mask_dir+'CHX_Eiger4M-bad_flatfield_05percent-mask.png')
        #mask.load(mask_dir+'CHX_pipe-2015Oct-mask.png')
        #mask.load(mask_dir+'CHX_bs_streaks-2015Oct-mask.png')

        tags_experiment['instrumental.detector.Conceptual256'] = True


        beam = direct_beam(calibration, width=np.random.uniform(4.0, 10.0), height=np.random.uniform(2.0, 5.0), peak=1e5)

    elif True:
        # Conceptual detector (for testing)
        # a few mods for CNN testing (Ziqiao Guan)
        calibration.set_energy(8.8984)  # CHX
        calibration.set_image_size(1000,1000)##256, 256)
        calibration.set_pixel_size(pixel_size_um=75.0)
        calibration.set_distance(4.755 / 8)

        x0, y0 = 128, 128
        x0 = np.random.uniform(-0.05 * calibration.width, 1.05 * calibration.width)
        y0 = np.random.uniform(-0.05 * calibration.height, 1.05 * calibration.height)
        calibration.set_beam_position(x0, y0)
        if x0 < 0 or x0 > calibration.width or y0 < 0 or y0 > calibration.height:
            tags_experiment['instrumental.beamstop.beam off image'] = True

        calibration.max_counts = 2 ** 16

        mask.data = np.ones((calibration.height, calibration.width))
        # mask.load(mask_dir+'Eiger4M_all_gaps-mask.png')
        # mask.load(mask_dir+'CHX_Eiger4M-bad_flatfield_10percent-mask.png')
        # mask.load(mask_dir+'CHX_Eiger4M-bad_flatfield_05percent-mask.png')
        # mask.load(mask_dir+'CHX_pipe-2015Oct-mask.png')
        # mask.load(mask_dir+'CHX_bs_streaks-2015Oct-mask.png')

        tags_experiment['instrumental.detector.Conceptual256'] = True

        beam = direct_beam(calibration, width=np.random.uniform(4.0, 10.0), height=np.random.uniform(2.0, 5.0),
                           peak=1e5)

    blocking_experiment = np.ones( (calibration.height, calibration.width) )

    if 'instrumental.beamstop.beam off image' not in tags_experiment or tags_experiment['instrumental.beamstop.beam off image']==False:

        # Beamstop
        weights = np.asarray([0.1, 0.1, 0.1])
        weights /= np.sum(weights)
        weights = np.cumsum(weights)
        idx = np.where(np.random.uniform(0,1)<=weights)[0][0]

        if idx==0:
            #blocking_experiment *= linear_beamstop(calibration, width=15.0, height=20.0, blur=3)
            width = pZOOM*np.random.uniform(8, 30)
            height = pZOOM*np.random.uniform(8, 100)
            blur = np.random.uniform(2.0, 10.0)
            blocking_experiment *= linear_beamstop(calibration, width=width, height=height, blur=blur)
            tags_experiment['instrumental.beamstop.linear beamstop'] = True

        elif idx==1:
            radius = pZOOM*np.random.uniform(8, 50)
            blur = np.random.uniform(1.0, 5.0)
            chi = np.random.uniform(-90, +90)
            blocking_experiment *= circular_beamstop(calibration, radius=radius, chi=chi, blur=blur)
            tags_experiment['instrumental.beamstop.circular beamstop'] = True

        elif idx==2:
            blur = np.random.uniform(1.0, 5.0)
            chi = np.random.uniform(-90, +90)
            wedge = np.random.uniform(10, 20)
            displacement = np.random.uniform(-10,-50)
            blocking_experiment *= wedge_beamstop(calibration, chi=chi, wedge=wedge, displacement=displacement, blur=blur)
            tags_experiment['instrumental.beamstop.wedge beamstop'] = True

    if prob(0.05):
        # q0 0.02 .. 0.04, dq 1e-4 .. 1e-2
        q0 = np.random.uniform(0.04, 0.06)
        dq = np.random.uniform(1e-4, 1e-2)
        blocking_experiment *= window_obstruction_circle(calibration, q0, dq)
        tags_experiment['image.problem.windowed obstruction'] = True


    return tags_experiment, calibration, mask, beam, blocking_experiment

def apply_detector(det_image):

    #det_image += diffuse_constant(calibration, avg=10.0, sigma=2.0) # background
    det_image += diffuse_poisson(calibration, lam=0.002) # spurious
    #det_image += zingers(calibration)

    det_image = np.clip(det_image, 0, calibration.max_counts)
    det_image = np.nan_to_num(det_image)

    det_image = np.random.poisson(lam=det_image) # Poisson counting statistics
    det_image = det_image.astype('float64')

    det_image *= mask.data
    det_image = np.rint(det_image).astype('uint32')

    return det_image

def pstats(image, name=''):

    print( '    {}:'.format(name) )
    print( '        min={:.1f} max={:.1f} (span={:.1f})'.format(np.min(image), np.max(image), np.max(image)-np.min(image)) )
    print( '        sum={:.1f} avg={:.1f}+/-{:.1f}'.format(np.sum(image), np.average(image), np.std(image)) )


class Data2D_current(Data2D):

    def _plot_extra(self, **plot_args):

        els = []
        for key in sorted(self.tags.keys()):
            if self.tags[key]:
                els.append(key)

        s = '\n'.join(els)

        xi, xf, yi, yf = self.ax.axis()
        self.ax.text(xi, yf, s, color='white', verticalalignment='top', horizontalalignment='left')


def generate_diffuse(calibration):

    tags = {}
    diffuse = np.zeros( (calibration.height, calibration.width) )


    if prob(0.05): # Constant background
        avg = np.random.uniform(0,2000)
        sigma = np.random.uniform(0.05,1.0)*avg
        diffuse += diffuse_constant(calibration, avg=avg, sigma=sigma)


    if prob(0.3): # diffuse low-q: isotropic
        tag = 'features.main.diffuse low-q: isotropic'
        num = 0
        current = np.zeros( (calibration.height, calibration.width) )

        if prob(0.5):
            c = 10**(np.random.uniform(4,6))
            current += c*diffuse_low_q_power(calibration, power=np.random.uniform(-4.5,-1.8))
            num += 1

        if prob(0.5):
            xi = 10**(np.random.uniform(1,4))
            c = 10**(np.random.uniform(2,5))
            current += c*diffuse_low_q_OrnsteinZernike(calibration, xi=xi)
            num += 1

        if prob(0.5):
            a = 10**(np.random.uniform(1,4))
            c = 10**(np.random.uniform(2,5))
            current += c*diffuse_low_q_DebyeBueche(calibration, a=a)
            num += 1

        if prob(0.5):
            Rg = 10**(np.random.uniform(2,3))
            c = 10**(np.random.uniform(2,5))
            current += c*diffuse_low_q_Guinier(calibration, Rg=Rg)
            num += 1

        #pstats(current, tag)
        diffuse += current
        if num>0 and np.sum(current>10)>400:
            tags[tag] = True
            #print('        ADDED {}'.format(tag))


    if prob(0.15): # diffuse low-q: anisotropic
        tag = 'features.main.diffuse low-q: anisotropic'
        chi = np.random.uniform(-180,+180)
        num = 0
        current = np.zeros( (calibration.height, calibration.width) )

        if prob(0.4):
            c = 10**(np.random.uniform(4,6))
            epsilon = 2.0*np.random.lognormal(mean=0.0, sigma=0.5)
            #epsilon = np.clip(epsilon, 1.1, 4.0)
            current += c*diffuse_low_q_power_aniso(calibration, power=np.random.uniform(-3.8,-1.8), epsilon=epsilon, chi=chi)
            num += 1

        if prob(0.4):
            xi = 10**(np.random.uniform(1,4))
            c = 10**(np.random.uniform(2,5))
            epsilon = 2.0*np.random.lognormal(mean=0.0, sigma=0.5)
            current += c*diffuse_low_q_OrnsteinZernike_aniso(calibration, xi=xi, epsilon=epsilon, chi=chi)
            num += 1

        if prob(0.4):
            a = 10**(np.random.uniform(1,4))
            c = 10**(np.random.uniform(2,5))
            epsilon = 2.0*np.random.lognormal(mean=0.0, sigma=0.5)
            current += c*diffuse_low_q_DebyeBueche_aniso(calibration, a=a, epsilon=epsilon, chi=chi)
            num += 1

        if prob(0.4):
            Rg = 10**(np.random.uniform(2,3))
            c = 10**(np.random.uniform(2,5))
            epsilon = 2.0*np.random.lognormal(mean=0.0, sigma=0.5)
            current += c*diffuse_low_q_Guinier_aniso(calibration, Rg=Rg, epsilon=epsilon, chi=chi)
            num += 1

        #pstats(current, tag)
        diffuse += current
        if num>0 and np.sum(current>20)>500:
            tags[tag] = True
            #print('        ADDED {}'.format(tag))


    if prob(0.2): # diffuse high-q: isotropic
        tag = 'features.main.diffuse high-q: isotropic'

        c = 10**(np.random.uniform(1,3))
        #c = np.random.uniform(10,1000)
        #c = 100*np.random.lognormal(mean=0.0, sigma=0.5)
        sigma = 10**(np.random.uniform(-2, 2.2))
        current = c*diffuse_high_q(calibration, sigma=sigma)

        diffuse += current
        if np.sum(current>10)>400:
            tags[tag] = True


    if prob(0.01): # diffuse high-q: anisotropic
        tag = 'features.main.diffuse high-q: anisotropic'

        chi = np.random.uniform(-180,+180)
        epsilon = 1.0 + 0.5*np.random.lognormal(mean=0.0, sigma=0.25)
        c = 10**(np.random.uniform(1,3))
        sigma = 10**(np.random.uniform(-2, 2.2))
        current = c*diffuse_high_q_aniso(calibration, sigma=sigma, epsilon=epsilon, chi=chi)

        diffuse += current
        if np.sum(current>10)>400:
            tags[tag] = True


    return diffuse, tags




lattice_peaks_BCC = [
    [1, 1],
    [1.41421356237303, 0.5],
    [1.73205080756878, 2],
    [1.99999999999989, 1],
    [2.2360679774997, 2],
    [2.44948974278315, 0.666666666666667],
    [2.64575131106451, 4],
    [2.82842712474607, 0.5],
    [2.99999999999989, 3],
    [3.16227766016829, 2],
    [3.31662479035528, 2],
    [3.46410161513768, 2],
    [3.60555127546386, 6],
    [3.87298334620732, 4],
    [3.99999999999989, 1],
    [4.12310562561754, 4],
    [4.24264068711921, 2.5],
    [4.35889894354057, 6],
    [4.4721359549994, 2],
    [4.5825756949557, 4],
    [4.69041575982326, 2],
    [4.79583152331255, 4],
    [4.89897948556618, 0.666666666666667],
]

lattice_peaks_FCC = [
    [1, 1],
    [1.15470053837927, 0.75],
    [1.63299316185546, 1.5],
    [1.91485421551276, 3],
    [2.00000000000009, 1],
    [2.30940107675854, 0.75],
    [2.51661147842362, 3],
    [2.58198889747169, 3],
    [2.82842712474629, 3],
    [3.00000000000009, 4],
    [3.265986323711, 1.5],
    [3.41565025531999, 6],
    [3.4641016151379, 3.75],
    [3.65148371670117, 3],
    [3.78593889720031, 3],
    [3.82970843102544, 3],
    [4.00000000000009, 1],
    [4.12310562561777, 3],
    [4.16333199893239, 3],
    [4.3204937989387, 6],
    [4.43471156521682, 3],
    [4.76095228569539, 3],
    [4.89897948556646, 1.5],
]

lattice_peaks_hex = [
    [1, 1],
    [1.15470053837881, 3],
    [1.52752523164647, 6],
    [1.99999999999212, 4],
    [2.23606797749517, 6],
    [2.30940107674975, 9],
    [2.516611478418, 6],
    [2.82842712474261, 6],
    [2.99999999999212, 1],
    [3.05505046329294, 12],
]

lattice_peaks_lam = [
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 1],
    [7, 1],
    [8, 1],
    [9, 1],
    [10, 1],
]

peak_names = ['BCC', 'FCC', 'hexagonal', 'lamellar']
peak_types = [lattice_peaks_BCC, lattice_peaks_FCC, lattice_peaks_hex, lattice_peaks_lam]

def generate_sample(calibration):

    tags = {}
    sample_scattering = np.zeros( (calibration.height, calibration.width) )

    #form_factor = np.ones( (calibration.height, calibration.width) )
    #structure_factor = np.ones( (calibration.height, calibration.width) )
    #sample_scattering = form_factor*structure_factor

    num_contributions = 1 + np.random.poisson(0.3)

    weights = np.asarray([0.1, 0.1, 0.3, 0.2, 0.2, 0.05, 0.2, 0.5, 0.2, 0.2])
    weights = np.asarray([0.05, # single symmetry ring
                          0.05, # single symmetry halo
                          0.1, # Symmetry rings
                          0.2, # Sphere form factor
                          0.2, # NP lattice
                          0.02, # NP lattice distorted
                          0.2, # adhoc rings
                          0.2, # speckled rings
                          0.2, # coherent halo
                          0.1, # meso
                          ])
    #weights = np.asarray([0, 0, 0, 0, 0, 1])
    weights /= np.sum(weights)
    weights = np.cumsum(weights)

    for i in range(num_contributions):
        idx = np.where( np.random.uniform(0,1)<=weights )[0][0]

        if idx==0:
            # Single symmetry ring

            chi = np.random.uniform(-180,+180)
            #symmetry = 2*np.random.random_integers(1,6)
            symmetry = np.clip( 2*np.rint( np.random.lognormal(mean=0.0, sigma=0.75) ), 2, 20 )
            #q0 = np.random.uniform(1e-4, 1e-1)
            q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX)
            dq = q0*np.random.uniform(0.01, 0.2) # Peak
            #dq = q0*np.random.uniform(0.3, 0.6) # Halo
            #eta = np.random.uniform(0.2, 0.95)
            eta = np.clip(np.random.normal(loc=0.3, scale=1.0), 0.2, 0.95)

            c = 10**(np.random.uniform(-1,4))
            current = c*symmetry_ring(calibration, q0=q0, dq=dq, eta=eta, chi=chi, symmetry=symmetry)
            sample_scattering += current

            if np.sum(current>=0.05)>100:
                tags['features.main.ring: anisotropic'] = True
                tags['features.variations.symmetry ring: {:d}'.format(int(symmetry))] = True
                if symmetry==2:
                    if abs(chi-0)<15 or abs(chi-180)<15 or abs(chi+180)<15:
                        tags['features.main.ring: oriented OOP'] = True
                    elif abs(chi-90)<15 or abs(chi+90)<15:
                        tags['features.main.ring: oriented IP'] = True
                    else:
                        tags['features.main.ring: oriented other'] = True

                if eta>0.75:
                    tags['features.variations.ring: orientation distribution: sharp'] = True
                elif eta<0.3:
                    tags['features.variations.ring: orientation distribution: broad'] = True
                else:
                    tags['features.variations.ring: orientation distribution: intermediate'] = True

                if dq/q0<0.025:
                    tags['features.variations.ring: width: sharp'] = True
                elif dq/q0>0.1:
                    tags['features.variations.ring: width: broad'] = True
                else:
                    tags['features.variations.ring: width: intermediate'] = True



        elif idx==1:
            # Single symmetry halo

            chi = np.random.uniform(-180,+180)
            #symmetry = 2*np.random.random_integers(1,6)
            symmetry = np.clip( 2*np.rint( np.random.lognormal(mean=0.0, sigma=0.75) ), 2, 20 )
            #q0 = np.random.uniform(1e-4, 1e-1)
            q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX)
            #dq = q0*np.random.uniform(0.01, 0.2) # Peak
            dq = q0*np.random.uniform(0.3, 0.6) # Halo
            eta = np.clip(np.random.normal(loc=0.1, scale=1.0), 0.02, 0.4)


            c = 10**(np.random.uniform(-1,4))
            current = c*symmetry_ring(calibration, q0=q0, dq=dq, eta=eta, chi=chi, symmetry=symmetry)
            current *= S_DW(calibration, sigma=10**(np.random.uniform(1,3)))
            sample_scattering += current

            if np.sum(current>=0.05)>100:
                tags['features.main.halo: anisotropic'] = True
                tags['features.variations.symmetry halo: {:d}'.format(int(symmetry))] = True
                if symmetry==2:
                    if abs(chi-0)<15 or abs(chi-180)<15 or abs(chi+180)<15:
                        tags['features.main.halo: oriented OOP'] = True
                    elif abs(chi-90)<15 or abs(chi+90)<15:
                        tags['features.main.halo: oriented IP'] = True
                    else:
                        tags['features.main.halo: oriented other'] = True


        elif idx==2:
            # Symmetry rings

            chi = np.random.uniform(-180,+180)
            #symmetry = 2*np.random.random_integers(1,6)
            symmetry = np.clip( 2*np.rint( np.random.lognormal(mean=0.0, sigma=0.75) ), 2, 20 )
            #q0 = np.random.uniform(1e-4, 1e-1)
            q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX)
            dq = q0*np.random.uniform(0.01, 0.2) # Peak
            #dq = q0*np.random.uniform(0.3, 0.6) # Halo
            #eta = np.random.uniform(0.2, 0.95)
            eta = np.clip(np.random.normal(loc=0.3, scale=1.0), 0.2, 0.95)

            sigma_DW = 10**(np.random.uniform(1.5,3.0))
            c = 10**(np.random.uniform(-1,4))
            current, num_rings = symmetry_rings(calibration, q0=q0, dq=dq, eta=eta, chi=chi, symmetry=symmetry, sigma_DW=sigma_DW)
            current *= c
            sample_scattering += current

            if np.sum(current>=0.05)>100:
                tags['features.main.ring: anisotropic'] = True
                tags['features.variations.symmetry ring: {:d}'.format(int(symmetry))] = True
                tags['features.variations.higher orders: {:d}'.format(int(num_rings))] = True
                if num_rings>4:
                    tags['features.variations.many rings'] = True
                if symmetry==2:
                    if abs(chi-0)<15 or abs(chi-180)<15 or abs(chi+180)<15:
                        tags['features.main.ring: oriented OOP'] = True
                    elif abs(chi-90)<15 or abs(chi+90)<15:
                        tags['features.main.ring: oriented IP'] = True
                    else:
                        tags['features.main.ring: oriented other'] = True

                    if num_rings>2:
                        tags['sample.lattice.lamellar'] = True


                if eta>0.75:
                    tags['features.variations.ring: orientation distribution: sharp'] = True
                elif eta<0.3:
                    tags['features.variations.ring: orientation distribution: broad'] = True
                else:
                    tags['features.variations.ring: orientation distribution: intermediate'] = True

                if dq/q0<0.025:
                    tags['features.variations.ring: width: sharp'] = True
                elif dq/q0>0.1:
                    tags['features.variations.ring: width: broad'] = True
                else:
                    tags['features.variations.ring: width: intermediate'] = True


        elif idx==3:
            # Sphere form factor

            radius = 10**(np.random.uniform(1.5, 3.5))
            c = 10**(np.random.uniform(0,4))
            current = form_factor_sphere(calibration, radius=radius)
            sample_scattering += c*np.nan_to_num(current)

            if np.sum(current>=0.1)>100:
                tags['image.general.form factor'] = True
                tags['image.general.form factor: sphere'] = True


        elif idx==4:
            # NP lattice

            sigma_DWr = np.clip( np.random.normal(loc=0.1, scale=0.2), 0.01, 0.5)
            q0 = np.clip( np.random.normal(loc=qTYP*3, scale=4e-2), qMIN, qMAX*0.5)
            d0 = 2.*np.pi/q0
            dq = q0*np.random.uniform(0.01, 0.2) # Peak
            #radius = np.clip( 10**(np.random.uniform(1.5, 3.5)), 0, np.pi/q0 ) # Particle shouldn't be bigger than unit-cell
            radius = d0*np.random.uniform(0.01,2.0)

            c = 10**(np.random.uniform(0,4))

            peak_idx = np.random.randint(0, len(peak_types))
            peaks = peak_types[peak_idx]

            current, num_rings = NP_lattice(calibration, peaks=peaks, radius=radius, sigma_DWr=sigma_DWr, q0=q0, dq=dq)
            current *= c
            sample_scattering += np.nan_to_num(current)

            if np.sum(current>=0.1)>100 and np.max(current)>1:
                tags['image.general.structure factor'] = True
                tags['sample.lattice.{}'.format(peak_names[peak_idx])] = True

                tags['features.main.ring: isotropic'] = True
                tags['features.variations.higher orders: {:d}'.format(int(num_rings))] = True
                if num_rings>4:
                    tags['features.variations.many rings'] = True

                if dq/q0<0.025:
                    tags['features.variations.ring: width: sharp'] = True
                elif dq/q0>0.1:
                    tags['features.variations.ring: width: broad'] = True
                else:
                    tags['features.variations.ring: width: intermediate'] = True


        elif idx==5:
            # NP lattice distorted

            sigma_DWr = np.clip( np.random.normal(loc=0.1, scale=0.2), 0.01, 0.5)
            q0 = np.clip( np.random.normal(loc=qTYP*3, scale=4e-2), qMIN, qMAX*0.5)
            d0 = 2.*np.pi/q0
            dq = q0*np.random.uniform(0.01, 0.2) # Peak
            #radius = np.clip( 10**(np.random.uniform(1.5, 3.5)), 0, np.pi/q0 ) # Particle shouldn't be bigger than unit-cell
            radius = d0*np.random.uniform(0.01,2.0)

            c = 10**(np.random.uniform(0,4))

            peak_idx = np.random.randint(0, len(peak_types))
            peaks = peak_types[peak_idx]

            epsilon = 2.0*np.random.lognormal(mean=0.0, sigma=0.5)
            chi = np.random.uniform(-180,180)
            mode = np.random.uniform(0,1)

            current, num_rings = NP_lattice_distorted(calibration, peaks=peaks, radius=radius, sigma_DWr=sigma_DWr, q0=q0, dq=dq, epsilon=epsilon, chi=chi, mode=mode)
            current *= c
            sample_scattering += np.nan_to_num(current)

            if np.sum(current>=0.1)>100 and np.max(current)>1:
                tags['image.general.structure factor'] = True
                tags['sample.lattice.{}'.format(peak_names[peak_idx])] = True

                tags['features.variations.oval ring'] = True
                tags['features.variations.higher orders: {:d}'.format(int(num_rings))] = True
                if num_rings>4:
                    tags['features.variations.many rings'] = True

                if dq/q0<0.025:
                    tags['features.variations.ring: width: sharp'] = True
                elif dq/q0>0.1:
                    tags['features.variations.ring: width: broad'] = True
                else:
                    tags['features.variations.ring: width: intermediate'] = True


        elif idx==6:
            # Adhoc rings

            q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX*0.5)
            dq = q0*np.random.uniform(0.01, 0.04)
            Delq = q0*np.random.uniform(0.1, 1.0)
            num_rings = np.random.randint(3,10)
            q0s = []
            for i in range(num_rings):
                q0s.append( [q0 + i*Delq*np.random.uniform(0.2, 1.5), np.random.uniform(0.1, 1.0)] )


            c = 10**(np.random.uniform(1,4))
            current = c*adhoc_rings(calibration, q0s, dq)
            sample_scattering += current


            if np.sum(current>=0.1)>100 and np.max(current)>1:
                tags['image.general.structure factor'] = True
                tags['features.main.ring: isotropic'] = True
                tags['features.variations.higher orders: {:d}'.format(int(num_rings))] = True
                if num_rings>4:
                    tags['features.variations.many rings'] = True

                if dq/q0<0.025:
                    tags['features.variations.ring: width: sharp'] = True
                elif dq/q0>0.1:
                    tags['features.variations.ring: width: broad'] = True
                else:
                    tags['features.variations.ring: width: intermediate'] = True


        elif idx==7:
            # Speckled rings

            q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX*0.5)
            dq = q0*np.random.uniform(0.005, 0.03)
            Delq = q0*np.random.uniform(0.1, 1.0)
            num_rings = np.random.randint(3,10)
            q0s = []
            for i in range(num_rings):
                q0s.append( [q0 + i*Delq*np.random.uniform(0.2, 1.5), np.random.uniform(0.1, 1.0)] )

            symmetry = 2*np.random.randint(1,4)
            c = 10**(np.random.uniform(2,4))

            if prob(0.99):
                eta = np.random.uniform(0.5, 0.95)
                num_spots = np.random.randint(2,15)
                current = c*adhoc_rings_speckled(calibration, q0s, dq, symmetry=symmetry, eta=eta, num_spots=num_spots)
            else:
                num_spots = np.random.randint(2,15)
                num_spots = 20
                current = c*adhoc_rings_speckled_individual(calibration, q0s, dq, symmetry=symmetry, num_spots=num_spots)
                eta = 0.95


            sample_scattering += current


            if np.sum(current>=0.1)>100 and np.max(current)>1:
                tags['image.general.structure factor'] = True
                tags['features.variations.higher orders: {:d}'.format(int(num_rings))] = True
                if num_rings>4:
                    tags['features.variations.many rings'] = True

                if dq/q0<0.025:
                    tags['features.variations.ring: width: sharp'] = True
                elif dq/q0>0.1:
                    tags['features.variations.ring: width: broad'] = True
                else:
                    tags['features.variations.ring: width: intermediate'] = True

                tags['sample.lattice.symmetry: {:d}'.format(symmetry)] = True
                if eta>0.75:
                    tags['features.main.ring: spotted'] = True
                    tags['sample.type.large grains'] = True
                elif eta>=0.5:
                    tags['features.main.ring: textured'] = True
                    tags['sample.type.polycrystalline'] = True

        elif idx==8:
            # 'Coherent' halo

            chi = np.random.uniform(-180,+180)
            #symmetry = 2*np.random.random_integers(1,6)
            symmetry = np.clip( 2*np.rint( np.random.lognormal(mean=0.0, sigma=0.75) ), 2, 20 )
            #q0 = np.random.uniform(1e-4, 1e-1)
            q0 = np.clip( np.random.normal(loc=qTYP, scale=4e-2), qMIN, qMAX*0.5)

            #dq = q0*np.random.uniform(0.01, 0.2) # Peak
            dq = q0*np.random.uniform(0.1, 0.6) # Halo
            eta = np.clip(np.random.normal(loc=0.1, scale=1.0), 0.0, 0.4)
            rescale = np.random.randint(3.0, 8.0)

            c = 10**(np.random.uniform(1,4))
            current = c*coherent_halo(calibration, q0, dq, eta=eta, chi=chi, symmetry=symmetry, rescale=rescale)
            current *= S_DW(calibration, sigma=10**(np.random.uniform(1,3)))
            sample_scattering += current

            if np.sum(current>=0.05)>100:
                tags['features.other.coherent speckle'] = True

                if eta>0.05:
                    tags['features.main.halo: anisotropic'] = True
                    tags['features.variations.symmetry halo: {:d}'.format(int(symmetry))] = True

                    if symmetry==2:
                        if abs(chi-0)<15 or abs(chi-180)<15 or abs(chi+180)<15:
                            tags['features.main.halo: oriented OOP'] = True
                        elif abs(chi-90)<15 or abs(chi+90)<15:
                            tags['features.main.halo: oriented IP'] = True
                        else:
                            tags['features.main.halo: oriented other'] = True

                else:
                    tags['features.main.halo: isotropic'] = True

        elif idx==9:
            # Mesostructure single

            c = 10**(np.random.uniform(3,5))
            zoom = np.random.uniform(0.2,1.0)

            if prob(0.8):
                current = c*mesostructure(calibration, zoom=zoom)

                if np.sum(current>=10)>50:
                    tags['sample.type.mesostructure'] = True
                    tags['sample.type.single crystal'] = True
                    tags['sample.special.meso lattice SC'] = True

            else:
                num_orientations = np.random.randint(10,100)
                current = c*mesostructure_rotation(calibration, zoom=zoom, num_orientations=num_orientations)

                if np.sum(current>=10)>50:
                    tags['sample.type.mesostructure'] = True
                    tags['sample.type.polycrystalline'] = True
                    tags['sample.special.meso lattice SC'] = True

            sample_scattering += current

    return sample_scattering, tags

def generate_background(calibration):

    tags = {}
    background = np.zeros( (calibration.height, calibration.width) )

    if prob(0.5): # Constant background
        avg = np.random.uniform(0,2000)
        sigma = np.random.uniform(0.05,1.0)*avg
        background += diffuse_constant(calibration, avg=avg, sigma=sigma)

    if prob(0.05):
        avg = np.random.uniform(0,100)
        sigma = np.random.uniform(0.05,0.2)*avg
        rescale = np.random.uniform(2.0, 20.0)
        background += diffuse_structured(calibration, avg=avg, sigma=sigma, rescale=rescale)

    if prob(0.1):
        size = np.random.uniform(20, 300)
        x0 = np.random.uniform(-1,1)*size
        y0 = np.random.uniform(-1,1)*size
        blur = np.random.uniform(3.0, 50.0)
        c = 10**(np.random.uniform(-1,3))
        background += c*background_square_window(calibration, x0, y0, size, blur=blur)
        tags['image.problem.windowed scattering'] = True
        tags['image.problem.windowed scattering: square'] = True

    if prob(0.1):
        size = np.random.uniform(20, 300)
        x0 = np.random.uniform(-1,1)*size
        y0 = np.random.uniform(-1,1)*size
        blur = np.random.uniform(3.0, 50.0)
        c = 10**(np.random.uniform(-1,3))
        background += c*background_circle_window(calibration, x0, y0, size, blur=blur)
        tags['image.problem.windowed scattering'] = True
        tags['image.problem.windowed scattering: circle'] = True

    background = np.clip(background, 0, 2**32)

    if np.max(background)>50:
        tags['features.other.background'] = True
    if np.max(background)>10000 or np.average(background)>1000:
        tags['features.other.high background'] = True

    return background, tags

def generate_defects(calibration):

    tags = {}
    det_image = np.zeros( (calibration.height, calibration.width) )

    if prob(1):

        length = 10**(np.random.uniform(-6,-3))
        aspect = 10**(np.random.uniform(-4,-2))
        c = 10**(np.random.uniform(2,4))

        if prob(0.8):
            det_image += c*slit_streak_H(calibration, length=length, aspect=aspect)
            tags['image.problem.parasitic slit scattering'] = True
            tags['image.problem.parasitic slit scattering: horizontal'] = True

        if prob(0.8):
            c *= np.random.uniform(0.5, 2.0)
            length *= np.random.uniform(0.5, 2.0)
            det_image += c*slit_streak_V(calibration, length=length, aspect=aspect)
            tags['image.problem.parasitic slit scattering'] = True
            tags['image.problem.parasitic slit scattering: vertical'] = True

    return det_image, tags

def simulate_sample(image_idx, run_id, calibration, beam, blocking_experiment):

    tags = {}
    protocol.start_timestamp = time.time()

    np.random.seed( (12980 + run_id + 981039*image_idx) % (2**32 - 1) )

    tags['experiment.measurement.TSAXS'] = True
    tags['image.measured.sample'] = True

    # Add scattering features
    diffuse = np.zeros( (calibration.height, calibration.width) )
    tags_diffuse = {}
    if prob(0.4):
        diffuse, tags_diffuse = generate_diffuse(calibration)

    sample_scattering = np.zeros( (calibration.height, calibration.width) )
    tags_sample = {}
    if True:#prob(0.7):   # zg (ALL SAMPLES)
        sample_scattering, tags_sample = generate_sample(calibration)

    # Add experimental defects
    background = np.zeros( (calibration.height, calibration.width) )
    tags_background = {}
    if prob(0.4):
        background, tags_background = generate_background(calibration)

    defects = np.zeros( (calibration.height, calibration.width) )
    tags_defects = {}
    if prob(0.1):
        defects, tags_defects = generate_defects(calibration)

    # Sum contributions
    det_image = beam + diffuse + sample_scattering + background + defects

    blocking = np.ones( (calibration.height, calibration.width) )
    blocking *= blocking_experiment
    det_image *= blocking

    tags.update(tags_background)

    if np.max(diffuse)>np.average(background)*0.01:
        tags.update(tags_diffuse)
    if np.max(sample_scattering)>np.average(background)*0.01:

        if np.max(sample_scattering)>1000 or np.average(sample_scattering)>5:
            tags_sample['image.general.strong scattering'] = True
        elif np.max(sample_scattering)<10 or np.max(sample_scattering)<np.average(background):
            tags_sample['image.general.weak scattering'] = True
        else:
            tags_sample['image.general.intermediate scattering'] = True

        tags.update(tags_sample)

    if np.max(defects)>np.average(background)*0.01:
        tags.update(tags_defects)

    #pstats(diffuse, 'diffuse')
    #pstats(background, 'background')
    #pstats(sample_scattering, 'sample')
    #pstats(defects, 'defects')
    #pstats(det_image, 'pre-final')

    return det_image, tags

def simulate_dark(image_idx, run_id, calibration, beam, blocking_experiment):

    protocol.start_timestamp = time.time()

    tags = {}
    tags['image.measured.dark'] = True
    tags['image.problem.blocked'] = True

    det_image = np.zeros( (calibration.height, calibration.width) )

    return det_image, tags

def simulate_standardAgBH(image_idx, run_id, calibration, beam, blocking_experiment):

    protocol.start_timestamp = time.time()

    tags = {}
    tags['experiment.measurement.TSAXS'] = True
    tags['image.measured.standard'] = True
    tags['substance.instrumental.AgBH'] = True

    det_image = np.zeros( (calibration.height, calibration.width) )
    det_image += beam

    q0 = 0.1076
    dq = q0*0.05
    q0s = []

    intensities = [1.0000, 0.5965, 0.3822, 0.1798, 0.1123, 0.0825, 0.0627, 0.0462, 0.0391, 0.0332, 0.0352]
    for i in range(11):
        q0s.append( [q0*(i+1), intensities[i]] )
    q0s.append( [ q0*12.7230, 0.0771] )
    q0s.append( [ q0*12.8903, 0.0678] )

    c = 10**(np.random.uniform(2,4))
    det_image += c*adhoc_rings(calibration, q0s, dq)

    det_image += 0.01*c*diffuse_constant(calibration, avg=1.0, sigma=0.4)

    det_image += 10*c*diffuse_low_q_Guinier(calibration, Rg=800.0)

    blocking = np.ones( (calibration.height, calibration.width) )
    blocking *= blocking_experiment
    det_image *= blocking

    return det_image, tags

def simulate_particular(image_idx, run_id, calibration, beam, blocking_experiment):

    protocol.start_timestamp = time.time()

    tags = {}
    tags['experiment.measurement.TSAXS'] = True
    tags['image.measured.sample'] = True
    tags['substance.test.particular'] = True

    det_image = np.zeros( (calibration.height, calibration.width) )
    det_image += beam

    q0 = 0.025*0.1076
    dq = q0*0.05
    q0s = []

    intensities = [1.0000, 0.5965, 0.3822, 0.1798, 0.1123, 0.0825, 0.0627, 0.0462, 0.0391, 0.0332, 0.0352]
    for i in range(11):
        q0s.append( [q0*(i+1), intensities[i]] )
    q0s.append( [ q0*12.7230, 0.0771] )
    q0s.append( [ q0*12.8903, 0.0678] )

    c = 10**(np.random.uniform(2,4))
    det_image += c*adhoc_rings(calibration, q0s, dq)

    det_image += 0.01*c*diffuse_constant(calibration, avg=1.0, sigma=0.4)

    det_image += 10*c*diffuse_low_q_Guinier(calibration, Rg=800.0)

    blocking = np.ones( (calibration.height, calibration.width) )
    blocking *= blocking_experiment
    det_image *= blocking

    return det_image, tags

def simulate_direct(image_idx, run_id, calibration, beam, blocking_experiment):

    protocol.start_timestamp = time.time()

    tags = {}
    tags['experiment.measurement.TSAXS'] = True
    tags['image.measured.direct'] = True

    det_image = np.zeros( (calibration.height, calibration.width) )
    det_image += beam

    return det_image, tags

def simulate_detector(det_image):

    tags = {}

    # Apply detector artifacts
    if prob(0.05):
        det_image += zingers(calibration)
        tags['image.problem.zingers'] = True

    if np.max(det_image)>calibration.max_counts:
        tags['image.problem.saturation'] = True

    det_image = apply_detector(det_image)

    #pstats(det_image, 'final')

    return det_image, tags

def save_data(det_image, run_dir, image_idx_str, tags, intensity):

    # Save raw
    if False:
        outfile = '{}/{}.npy'.format(exp_dir, image_idx_str)
        np.save(outfile, det_image)
    else:
        outfile = '{}/{}.mat'.format(exp_dir, image_idx_str)
        data = {}
        data['detector_image'] = det_image
        savemat(outfile, data)

    # Save tags
    analysis_dir = os.path.join(root_dir, run_dir, 'analysis')
    protocol.end_timestamp = time.time()
    processor.store_results(tags, analysis_dir, image_idx_str, protocol)

    # Save calibration
    protocol_cal.end_timestamp = time.time()
    tags_cal = {}
    tags_cal['x0'] = calibration.x0
    tags_cal['y0'] = calibration.y0
    tags_cal['image_width'] = calibration.width
    tags_cal['image_height'] = calibration.height
    tags_cal['pixel_um'] = calibration.pixel_size_um
    tags_cal['distance_m'] = calibration.distance_m
    tags_cal['dq'] = calibration.get_q_per_pixel()
    tags_cal['wavelength_A'] = calibration.get_wavelength()
    tags_cal['energy_keV'] = calibration.get_energy()
    tags_cal['k_Ainv'] = calibration.get_k()
    processor.store_results(tags_cal, analysis_dir, image_idx_str, protocol_cal)

    # Save thumbnail
    analysis_dir = os.path.join(root_dir, run_dir, 'analysis', 'thumbnails')
    data = Data2D()
    data.data = det_image
    data.set_z_display( [None, None, 'gamma', 0.3] )
    outfile = '{}/{}.jpg'.format(analysis_dir, image_idx_str)
    #data.plot(show=True, cmap=cmap_vge, ztrim=[0.001, 0.001])
    data.plot_image(save=outfile, cmap=cmap_vge, ztrim=[0.001, 0.001])

    # save 1d (zg)
    analysis_dir = os.path.join(root_dir, run_dir, 'analysis', 'oned')
    outfile = '{}/{}.npy'.format(analysis_dir, image_idx_str)
    np.save(outfile, intensity)

    if False:
        # Save tag-thumbnail
        analysis_dir = os.path.join(root_dir, run_dir, 'analysis', 'tagimgs')
        data = Data2D_current()
        data.tags = tags
        data.data = det_image
        data.set_z_display( [None, None, 'gamma', 0.3] )
        outfile = '{}/{}.png'.format(analysis_dir, image_idx_str)
        data.plot(save=outfile, cmap=cmap_vge, ztrim=[0.01, 0.001])


for ir in range(500): ##Number of experiments
#if True:
    # Define the 'experimental run'
    run_id = random.randint(0, 16**8)
    #run_id = 100
    run_str = '{:08x}'.format(run_id)
    #run_comment = 'adhocs'
    #run_comment = 'spots'
    run_comment = 'varied_sm'
    run_dir = '_'.join([run_str, run_comment])

    print('Experiment {:d} ({})'.format(run_id, run_dir))

    root_dir = 'synthetic_data/'
    exp_dir = os.path.join(root_dir, run_dir)
    tools.make_dir(exp_dir)

    analysis_dir = os.path.join(root_dir, run_dir, 'analysis', 'thumbnails')
    tools.make_dir(analysis_dir)
    analysis_dir = os.path.join(root_dir, run_dir, 'analysis', 'results')
    tools.make_dir(analysis_dir)
    analysis_dir = os.path.join(root_dir, run_dir, 'analysis', 'tagimgs')
    tools.make_dir(analysis_dir)
    # 1d curve (zg)
    analysis_dir = os.path.join(root_dir, run_dir, 'analysis', 'oned')
    tools.make_dir(analysis_dir)

    tags_experiment, calibration, mask, beam, blocking_experiment = define_experiment(run_id)


    # Generate a bunch of images
    for image_idx in range(100): ##Number of images per experiment
        tags = {}
        tags.update(tags_experiment)

        image_idx_str = '{:08x}'.format(image_idx)
        print('  Image {:d} ({})'.format(image_idx, image_idx_str))

        if image_idx==0 or prob(0.01):
            det_image, tags_current = simulate_standardAgBH(image_idx, run_id, calibration, beam, blocking_experiment)
        elif True:#prob(0.95):      #zg (ALL SAMPLES)
            det_image, tags_current = simulate_sample(image_idx, run_id, calibration, beam, blocking_experiment)
        elif prob(0.4):
            det_image, tags_current = simulate_particular(image_idx, run_id, calibration, beam, blocking_experiment)
        elif prob(0.5):
            det_image, tags_current = simulate_dark(image_idx, run_id, calibration, beam, blocking_experiment)
        else:
            det_image, tags_current = simulate_direct(image_idx, run_id, calibration, beam, blocking_experiment)

        tags.update(tags_current)


        det_image, tags_current = simulate_detector(det_image)
        tags.update(tags_current)

        # 1d (zg)
        fff = oneD_intensity(im=det_image,xcenter=calibration.x0,ycenter=calibration.y0)
        intensity = fff.cir_ave()[:,0]

        save_data(det_image, run_dir, image_idx_str, tags, intensity)
