""" Extract and record all interested tags of a set of real data samples given
a filelist
"""
import numpy as np
import os
import pickle
#from tagio.DatasetHelper import RealDatasetHelper as dh
from tagio.tag import *


def extract_tags(filelist, taglist, filename):
    lt = []
    for f in filelist:
        image_dir, image_name = os.path.split(f)
        name, _ = os.path.splitext(image_name)
        lt.append(os.path.join(image_dir, '../mini_image/' + name + '.xml'))
    ltags = [tagdata(ft, tag_type=tagtype.Synthetic) for ft in lt]
    output = np.zeros([len(lt), len(taglist)], dtype=np.uint8)

    for i, tag in enumerate(ltags):
        if i % 100 == 0:
            print('%d / %d' % (i, len(ltags)))
        for j, tagname in enumerate(taglist):
            output[i, j] = tagname in tag.SimulatedFeatures
    sums = np.sum(output, axis=0)
    print(sums)
    np.save(filename, output)


if __name__ == '__main__':
    with open('/home/shared/mixed_dataset/preprocessed_data/imagelist', 'r') as f:
        l = f.read().splitlines()
    l1 = l[:2300]
    taglist = ['features.main.diffuse high-q',
               'features.main.diffuse low-q',
               'features.main.diffuse low-q: isotropic',
               'features.main.halo',
               'features.main.halo: isotropic',
               'features.main.ring',
               'features.main.ring: anisotropic',
               'features.main.ring: isotropic',
               'features.main.ring: oriented OOP',
               'features.main.ring: spotted',
               'features.variations.higher orders',
               'image.general.strong scattering',
               'image.general.weak scattering',
               'features.variations.symmetry ring',
               'features.variations.symmetry ring: 2',
               'features.variations.symmetry ring: 4',
               #'features.variations.symmetry ring: 6',
               'features.variations.symmetry halo',
               'features.variations.symmetry halo: 2',
               #'features.variations.symmetry halo: 4',
               'features.variations.symmetry halo: 6',
               'sample.type.polycrystalline']
    extract_tags(l1, taglist, '/home/zquan/backup/features_REAL/Y_train.npy')
    l2 = l[2300:]
    extract_tags(l2, taglist, '/home/zquan/backup/features_REAL/Y_test.npy')
