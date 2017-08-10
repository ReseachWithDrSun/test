from fbb.FourierBesselBasis import *
from fbb.TestPolarFunction import TestPolarFunction
from fbb.FBBSparseSolver import FBBSparseSolver as FBBSS
from tagio.DatasetHelper import SyntheticDatasetHelper as SDH
from tagio.tag import *
import numpy as np
import scipy.io
import skimage.measure
import matplotlib.pyplot as plt
import os
import time
import multiprocessing
from functools import partial


output_dir = '../xray_data/fbb_output/sed/' ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
basis_path = os.path.join(output_dir, '../fbb.npy')
# dict_path = os.path.join(output_dir, './dict_toreal.npy')    # './dict.npy')
image_size = 1000 ##256~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
basis_size = 2500 ##600~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sdh = SDH('../SimulationCode/generators/synthetic_data/') ##~~~~~~~~~~~~~~~~~~
B = np.load(basis_path)
image_per_expr = 100 ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Load raw image and tag data
'''
def load_data(imgpath):
    t = sdh.get_tag_from_image(imgpath)
    tag = tagdata(t, tag_type=tagtype.Synthetic)
    image = scipy.io.loadmat(imgpath)['detector_image']
    return image, tag


def compute_enet(d, params):
    fid, imgpath = params
    image, tag = load_data(imgpath)
    flag_plot = False
    alpha = 1e-4
    l1_ratio = 0.8
    max_iter = 20
    expr, name = sdh.split_expr_name(imgpath)
    output_dir_f = output_dir + expr
    if not os.path.isdir(output_dir_f):
        os.mkdir(output_dir_f)

    solver = FBBSS(dictionary=d, basis_size=basis_size, image_size=image_size)
    pre_image = solver.load_image(image, tag.y0, tag.x0)

    input_image = solver.toreal_transform(np.reshape(pre_image, [-1, 1]))
    t1 = time.time()
    coef, n_iter = solver.solveEnet(input_image, alpha=alpha, l1_ratio=l1_ratio,
                                    max_iter=max_iter)
    print('%d: solved in %fs, %d iterations.' % (fid, time.time() - t1, n_iter))
    np.save(os.path.join(output_dir_f, name + '.npy'), coef)

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
        fig.savefig(os.path.join(output_dir_f, name + '.png'))
        plt.close(fig)


if __name__ == '__main__':
    l = sdh.get_image_list()
    assert len(l) % image_per_expr == 0, '#images not divided by #exprs'
    l.sort()    # batch job by experiment

    # pool = multiprocessing.Pool()
    # m = multiprocessing.Manager()
    # lock = m.Lock()
    # func = partial(compute_enet, lock)
    D = None

    for i in range(int(len(l) / image_per_expr)):
        l_expr = l[i * image_per_expr:(i+1) * image_per_expr]
        expr, _ = sdh.split_expr_name(l_expr[0])

        # construct dictionary for the expr
        image, tag = load_data(l_expr[0])
        solver = FBBSS(bases=B, basis_size=basis_size, image_size=image_size)
        pre_image = solver.load_image(image, tag.y0, tag.x0)
        t1 = time.time()
        solver.crop_bases()
        solver.build_dictionary_toreal()
        D = np.asarray(solver.d)
        print('%s: Dictionary generated in %fs.' % (expr, time.time() - t1))

        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # generate new pool for computing an entire expr
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## Use for loop for large data (1000x1000)
        for k in range(image_per_expr):
            compute_enet(D, (k, l_expr[k]))
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## Use multiprocessing pooling for smaller data (256x256)
        #pool = multiprocessing.Pool()
        #func = partial(compute_enet, D)
        #pool.map(func, enumerate(l_expr))
        #pool.close()
        #pool.join()
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # # clean up dictionary afterwards
        # os.remove(os.path.join(output_dir, expr, 'dict.npy'))
