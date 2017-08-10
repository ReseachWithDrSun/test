'''
Analyze positive/negative samples of specific labels (w their FBB coefs)
'''

import numpy as np
from tagio.tag import *
from tagio.DatasetHelper import SyntheticDatasetHelper as SDH
import tagio.xray_data_processor as Processor
import os
import matplotlib.pyplot as plt
import scipy.io


data_dir = '/home/zquan/xray_data/nodefect_50k/'
coef_dir = '/home/zquan/xray_data/fbb_output/nodefect_50k/'
output_dir = '/home/zquan/xray_data/fbb_analysis/'
label_len = 10


def sample(label, n_positive, n_negative):
    idx_positive = np.where(label == 1)[0]
    idx_negative = np.where(label == 0)[0]
    if idx_positive.size > n_positive:
        idx_positive = np.random.permutation(idx_positive)[:n_positive]
    if idx_negative.size > n_negative:
        idx_negative = np.random.permutation(idx_negative)[:n_negative]
    return idx_positive, idx_negative


def visualize(sdh, files, output_template):
    for i, f in enumerate(files):
        print('%d / %d' % (i, len(files)))
        image = scipy.io.loadmat(f)['detector_image']
        image = np.log(1 + image) / np.log(65536)
        expr, name = sdh.split_expr_name(f)
        coef_path = os.path.join(coef_dir, expr, name + '.npy')
        coef = np.reshape(np.load(coef_path), [2, 40, 11])
        amp = np.sqrt(coef[0, :, :] ** 2 + coef[1, :, :] ** 2)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        im = axes[0].imshow(image)
        ima = axes[1].imshow(amp)
        axes[2].plot(np.sum(amp, axis=0))
        fig.colorbar(im, ax=[axes[0]])
        fig.colorbar(ima, ax=[axes[1]])
        fig.suptitle(f)
        fig.savefig(output_template.format(i))
        plt.close(fig)


if __name__ == '__main__':
    sdh = SDH(data_dir)
    l = sdh.get_image_list()
    l = np.random.permutation(l)[:5000]     # sample the entire dataset
    tags = [tagdata(sdh.get_tag_from_image(x), tag_type=tagtype.Synthetic)
            for x in l]
    keywords = ['symmetry halo', 'symmetry ring', 'lattice.symmetry']

    for w in keywords:
        output_dir_w = output_dir + w + '/'
        if not os.path.isdir(output_dir_w):
            os.mkdir(output_dir_w)
        for i, t in enumerate(tags):
            if any(x.find(w) > -1 for x in t.SimulatedFeatures):
                sdh.copy_to(l[i], output_dir_w)

'''
if __name__ == '__main__':
    sdh = SDH(data_dir)
    l = sdh.get_image_list()
    l = np.random.permutation(l)[:5000]     # sample the entire dataset
    tags = [tagdata(sdh.get_tag_from_image(x), tag_type=tagtype.Synthetic)
            for x in l]
    labels = np.zeros([len(l), label_len], dtype=int)
    for i, t in enumerate(tags):
        labels[i, :] = Processor.SimulatedFeatureSelectorSymmetry(t)[0, :label_len]

    indices = [7, 8, 9]
    for idx in indices:
        idx_positive, idx_negative = sample(labels[:, idx], 50, 50)
        if not os.path.isdir(output_dir + str(idx)):
            os.mkdir(output_dir + str(idx))
        visualize(sdh, l[idx_positive], output_dir + str(idx) + '/p-{}.png')
        visualize(sdh, l[idx_negative], output_dir + str(idx) + '/n-{}.png')
'''
