import os
import scipy.io
import xml.etree.ElementTree as ET
from oned import *

exps = os.listdir('.')
for e in exps:
    if os.path.isdir(e):
        for f in os.listdir(e):
            fullpath = os.path.join(e, f)
            if os.path.isfile(fullpath):
                name, ext = os.path.splitext(f)
                x = 0
                y = 0
                if ext == '.mat':
                    image = scipy.io.loadmat(fullpath)['detector_image']

                    tag_path = os.path.join(e, 'analysis/results/' + name + '.xml')
                    tag = ET.parse(tag_path)
                    for m in tag.findall('protocol'):
                        if m.attrib['name'] == 'calibration_generated':
                            for r in m.iter('result'):
                                if r.attrib['name'] == 'x0':
                                    x = float(r.attrib['value'])
                                elif r.attrib['name'] == 'y0':
                                    y = float(r.attrib['value'])
                            break
                    oned = oneD_intensity(im=image, xcenter=x, ycenter=y)
                    intensity = oned.cir_ave()[:,0]
                    oned_path = os.path.join(e, 'analysis/oned/' + name + '.npy')

                    if not os.path.isdir(os.path.join(e, 'analysis/oned')):
                        os.mkdir(os.path.join(e, 'analysis/oned'))
                    np.save(oned_path, intensity)
