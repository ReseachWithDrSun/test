from fbb.FourierBesselBasis import *
from fbb.TestPolarFunction import TestPolarFunction
from fbb.FBBSparseSolver import FBBSparseSolver as FBBSS
from tagio.DatasetHelper import RealDatasetHelper as dh
from tagio.tag import *
import numpy as np
import scipy.io
import skimage.measure
import matplotlib.pyplot as plt
import os
import time
import multiprocessing
from functools import partial


image_size = 256
basis_size = 600

'''
# M
output_dir = '/home/zquan/xray_data/fbb_output/2016_03_13/'
dict_path = '/home/zquan/xray_data/fbb_output/dictreal.npy'
mask_path = '/home/zquan/xray_data/fbb_output/overmask.npy'
xr, yr = slice(256), slice(256)
xx, yy = 149, 164
rdh = dh('/home/shared/TEST_DATA/real/2016_03_13r/mini_image')
'''

# A
output_dir = '/home/zquan/xray_data/fbb_output/real_A/'
dict_path = '/home/zquan/xray_data/fbb_output/dictreal_A_mask.npy'
mask_path = '/home/zquan/xray_data/fbb_output/A_mask.npy'
xr, yr = slice(256), slice(256)
xx, yy = 129, 194
rdh = dh('/home/shared/CMS_for_ML/2011_04Apr_30-Fang_shapedNPs/mini_image')

'''
# N
output_dir = '/home/zquan/xray_data/fbb_output/real_N/'
dict_path = '/home/zquan/xray_data/fbb_output/dictreal_N_mask.npy'
mask_path = '/home/zquan/xray_data/fbb_output/N_mask.npy'
xr, yr = slice(256), slice(34, 290)
xx, yy = 136, 143
rdh = dh('/home/shared/CMS_for_ML/2016_11Nov_17-First_samples/mini_image')
'''


'''
Load raw image and tag data
'''
def load_data(imgpath):
    t = rdh.get_tag_from_image(imgpath)
    tag = tagdata(t, tag_type=tagtype.Synthetic)
    image = scipy.io.loadmat(imgpath)['detector_image']
    return image, tag


def compute_enet(d, params):
    fid, imgpath = params
    image, tag = load_data(imgpath)
    image = image[yr, xr]
    name = rdh.get_name(imgpath)
    flag_plot = True
    alpha = 1e-4
    l1_ratio = 0.8
    max_iter = 20

    try:
        solver = FBBSS(dictionary=d, basis_size=basis_size, image_size=image_size)
        pre_image = solver.load_image(image, yy, xx)

        input_image = solver.toreal_transform(np.reshape(pre_image, [-1, 1]))
        t1 = time.time()
        coef, n_iter = solver.solveEnet(input_image, alpha=alpha, l1_ratio=l1_ratio,
                                        max_iter=max_iter)
        print('%d: solved in %fs, %d iterations.' % (fid, time.time() - t1, n_iter))
        np.save(os.path.join(output_dir, name + '.npy'), coef)

        if flag_plot:
            # display result
            n_paramsets = 1
            fig, axes = plt.subplots(nrows=2, ncols=n_paramsets + 1,
                                     figsize=(23, 11))
            im = axes[0][0].imshow(pre_image)
            vmin, vmax = np.amin(pre_image), np.amax(pre_image)
            for j in range(n_paramsets):
                x = coef
                y_recon = np.dot(solver.d, x)
                y_recon = solver.tocomplex_transform(y_recon)
                y_recon = np.real(np.reshape(y_recon, [image_size, image_size]))
                axes[0][j + 1].imshow(y_recon, vmin=vmin, vmax=vmax)
                psnr = skimage.measure.compare_psnr(pre_image, y_recon,
                                                    dynamic_range=vmax - vmin)
                axes[0][j + 1].set_title('PSNR = %f' % psnr)
                amp = np.absolute(solver.tocomplex_transform(x))
                amp = np.reshape(amp, [40, 11])
                amp = np.insert(amp, 0, 0, axis=0)
                imamp = axes[1][j + 1].imshow(amp)
                fig.colorbar(imamp, ax=[axes[1][j + 1]])
            fig.colorbar(im, ax=axes[0, :].ravel().tolist())
            fig.savefig(os.path.join(output_dir, name + '.png'))
            plt.close(fig)
    except:
        with open('enet_real.log', 'a') as f:
            f.write('Error: %s' + imgpath)


if __name__ == '__main__':
    l = rdh.get_image_list()
    l1 = []
    D = np.load(dict_path)
    mask = np.load(mask_path)
    # cond = {'D': D, 'mask': mask}

    # verify the solution
    for f in l:
        name = rdh.get_name(f)
        fout = os.path.join(output_dir, name + '.npy')
        fimg = os.path.join(output_dir, name + '.png')
        if not (os.path.isfile(fout) or os.path.isfile(fimg)):
            l1.append(f)
    print('%d / %d' % (len(l1), len(l)))

    # compute_enet(D, (0, l[0]))

    # generate new pool for computing an entire expr
    pool = multiprocessing.Pool()
    func = partial(compute_enet, D)
    pool.map(func, enumerate(l1))
    pool.close()
    pool.join()
