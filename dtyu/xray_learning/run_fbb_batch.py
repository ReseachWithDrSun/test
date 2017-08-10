'''
batch compute the sparse coding of a group of synthetic images with fixed dictionary
ElasticNetCV testing
'''

# from fbb.FourierBesselBasis import FourierBesselBasis as FBB
from fbb.FourierBesselBasis import *
from fbb.TestPolarFunction import TestPolarFunction
from fbb.FBBSparseSolver import FBBSparseSolver as FBBSS
from tagio.DatasetHelper import SyntheticDatasetHelper as SDH
from tagio.tag import *
import numpy as np
import scipy.io
import skimage.measure
import matplotlib.pyplot as plt
import sys
import os
import time


output_dir = '/home/zquan/xray_data/fbb_output/'
basis_path = os.path.join(output_dir, './fbb.npy')
dict_path = os.path.join(output_dir, './dict_toreal.npy')    # './dict.npy')
image_size = 256
basis_size = 600
nnzs = [20, 50, 100]
alphas = [100, 1, 1E-2, 1E-4]


# '''
# Load image and shift to grid point
# '''
# def load_data(imgpath, normalize='log'):
#     t = sdh.get_tag_from_image(imgpath)
#     tag = tagdata(t, tag_type=tagtype.Synthetic)
#     image = scipy.io.loadmat(imgpath)['detector_image']
#     # shift beam center to grid point
#     shifted_image, x, y = shift_image(image, tag.x0, tag.y0)
#     # log normalize
#     if normalize is None:   # (DEPRECATED) log centered
#         shifted_image = 2 * np.log(shifted_image + 1) / np.log(65536) - 1
#     elif normalize == 'log':
#         shifted_image = np.log(shifted_image + 1) / np.log(65536)
#     elif normalize == 'logr2':
#         shifted_image = np.log(shifted_image + 1) / np.log(65536)
#         tpf = TestPolarFunction(image_size * np.sqrt(2), basis_size,
#                                 [int(basis_size / 2), int(basis_size / 2)])
#         r2 = crop_bases(tpf.RSquared())
#         shifted_image = np.multiply(shifted_image, r2)
#     # bounded disk projection
#     xx, yy = np.meshgrid(np.arange(0, image_size, 1),
#                          np.arange(0, image_size, 1))
#     xx -= x
#     yy -= y
#     rr = np.sqrt(xx ** 2 + yy ** 2)
#     # maximum radius w/o discont. 4 corner values
#     rmax = max([x, y, image_size - x, image_size - y])
#     projected_image = np.multiply(shifted_image, rr <= rmax)
#
#     # normalize to unit norm
#     projected_image /= np.sqrt(np.sum(projected_image ** 2))
#     return shifted_image, tag, x, y, rmax


'''
Load raw image and tag data
'''
def load_data(sdh, imgpath):
    t = sdh.get_tag_from_image(imgpath)
    tag = tagdata(t, tag_type=tagtype.Synthetic)
    image = scipy.io.loadmat(imgpath)['detector_image']
    return image, tag


if __name__ == '__main__':
    assert (len(sys.argv) > 2 and sys.argv[1] == '--action'), 'Usage: python run_fbb.py --action [action id]'
    action = int(sys.argv[2])

    sdh = SDH('/home/zquan/selected_dataset')
    # sdh = SDH('/home/zquan/SimulationCode/generators/synthetic_data')
    l = sdh.get_image_list()
    if action == 0:
        # compute sparse representation
        D = np.load(dict_path)
        psnr = np.zeros([len(nnzs), len(l)])
        for i, f in enumerate(l):
            print('%d / %d' % (i, len(l)))
            image, tag, x, y = load_data(f)
            # compute sparse coding
            solver = FBBSS(dictionary=D, basis_size=basis_size, image_size=image_size)
            # shifted_image = solver.load_image(image, tag.y0, tag.x0)
            solver.x, solver.y = x, y
            coef = np.zeros([len(nnzs), D.shape[0]])
            for j, nnz in enumerate(nnzs):
                coef[j, :] = solver.solveOMP(image, nnz)

            expr, name = sdh.split_expr_name(f)
            output_dir_f = output_dir + expr
            if not os.path.isdir(output_dir_f):
                os.mkdir(output_dir_f)
            np.save(os.path.join(output_dir_f, name + '.npy'), coef)

            # # load coefficients
            # coef = np.load(os.path.join(output_dir_f, name + '.npy'))

            # display result
            fig, axes = plt.subplots(nrows=2, ncols=len(nnzs)+1, figsize=(18, 9))
            im = axes[0][0].imshow(image)
            vmin, vmax = np.amin(image), np.amax(image)
            for j, nnz in enumerate(nnzs):
                x = coef[j, :]
                y_recon = np.reshape(np.dot(x, D), [image_size, image_size])
                axes[0][j + 1].imshow(y_recon, vmin=vmin, vmax=vmax)
                axes[1][j + 1].plot(x)
                dr = np.amax(image) - np.amin(image)
                psnr[j][i] = skimage.measure.compare_psnr(image, y_recon, dynamic_range=dr)
                axes[0][j + 1].set_title('NNZ = %d, PSNR = %f' % (nnz, psnr[j][i]))
            fig.colorbar(im, ax=axes[0, :].ravel().tolist())
            fig.savefig(os.path.join(output_dir_f, name + '.png'))
            plt.close(fig)
        np.save(os.path.join(output_dir_f, 'psnr.npy'), psnr)
    elif action == 1:
        # rearrange frequencies
        D = np.load(dict_path)
        n_freq_radial = 30
        n_freq_angular = 20
        len_real = n_freq_radial * (n_freq_angular + 1)
        nnz_ind = 2    # 50
        nnz = nnzs[nnz_ind]
        for i, f in enumerate(l):
            print('%d / %d' % (i, len(l)))
            image, tag, x, y = load_data(f)
            expr, name = sdh.split_expr_name(f)
            # TODO decouple image shifting from solver
            solver = FBBSS(basis_size=basis_size, image_size=image_size)
            # shifted_image = solver.load_image(image, tag.y0, tag.x0)
            solver.x, solver.y = x, y

            output_dir_f = output_dir + expr
            output_dir_freq = output_dir + expr + '/freq'
            if not os.path.isdir(output_dir_freq):
                os.mkdir(output_dir_freq)
            coef = np.load(os.path.join(output_dir_f, name + '.npy'))[nnz_ind, :]
            coef_freq = np.zeros([2, n_freq_radial, n_freq_angular + 1])
            coef_freq[0, :, :] = np.reshape(coef[:len_real], [n_freq_radial, n_freq_angular + 1])
            coef_freq[1, :, 1:] = np.reshape(coef[len_real:], [n_freq_radial, n_freq_angular])
            np.save(os.path.join(output_dir_freq, name + '.npy'), coef_freq)
            # print illustration of |freq|=sqrt(r^2 + i^2)
            mod_freq = np.sum(coef_freq, axis=0)
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 9))
            im = axes[0][0].imshow(image)
            vmin, vmax = np.amin(image), np.amax(image)
            y_recon = np.reshape(np.dot(coef, D), [image_size, image_size])
            axes[0][1].imshow(y_recon, vmin=vmin, vmax=vmax)
            freqs = axes[1][1].imshow(mod_freq)
            dr = np.amax(image) - np.amin(image)
            psnr = skimage.measure.compare_psnr(image, y_recon, dynamic_range=dr)
            axes[0][1].set_title('NNZ = %d, PSNR = %f' % (nnz, psnr))
            fig.colorbar(im, ax=axes[0, :].ravel().tolist())
            fig.colorbar(freqs, ax=axes[1][1])
            fig.savefig(os.path.join(output_dir_freq, name + '.png'))
            plt.close(fig)
    elif action == 2:
        # full fbt
        # zero padding for area outside "rmax"

        # precompute for all centered images
        normalize = 'logr2'
        B = np.load(basis_path)
        B = crop_bases(B)
        xx, yy = np.meshgrid(np.arange(0, image_size, 1),
                             np.arange(0, image_size, 1))
        xx -= 128
        yy -= 128
        rr = np.sqrt(xx ** 2 + yy ** 2)
        for i, f in enumerate(l):
            print('%d / %d' % (i, len(l)))
            image, tag, x, y = load_data(f, normalize=normalize)
            expr, name = sdh.split_expr_name(f)
            output_dir_f = output_dir + expr
            output_dir_fbt = output_dir + expr + '/fbt'
            if normalize is not None:
                output_dir_fbt = output_dir_fbt + '_' + normalize
            if not os.path.isdir(output_dir_fbt):
                os.mkdir(output_dir_fbt)

            coef = np.zeros([31, 21], dtype=np.complex)
            # shifted_image, x, y = shift_image(image, tag.x0, tag.y0)

            rmax = max([x, y, image_size - x, image_size - y])
            # shifted_image = np.multiply(shifted_image, rr <= rmax)
            image = np.multiply(image, rr <= rmax)

            # B = np.load(basis_path)
            # B = crop_bases(B, bsize=basis_size, imgsize=image_size, x=x, y=y)
            for n in range(1, 31):
                for m in range(21):
                    coef[n, m] = np.sum(
                        np.multiply(image, np.conj(B[n, m, :, :])))
            np.save(os.path.join(output_dir_fbt, name + '.npy'), coef)
            recon_image = np.zeros([image_size, image_size], dtype=np.complex)
            for n in range(1, 31):
                for m in range(21):
                    recon_image = recon_image + coef[n, m] * B[n, m, :, :]

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(23, 10), squeeze=False)
            vmin, vmax = np.amin(image), np.amax(image)
            im = axes[0][0].imshow(image)
            im_recon = axes[0][1].imshow(np.real(recon_image),
                                         vmin=vmin, vmax=vmax)
            fbb_mag = np.absolute(coef)
            im_mag = axes[1][0].imshow(fbb_mag)
            im_ang = axes[1][1].imshow(np.angle(coef))
            axes[0][2].plot(np.sum(fbb_mag, axis=0))
            axes[0][2].set_title('Angular sum')
            axes[1][2].plot(np.sum(fbb_mag, axis=1))
            axes[1][2].set_title('Radial sum')
            fig.colorbar(im, ax=[axes[0][0], axes[0][1]])
            fig.colorbar(im_mag, ax=[axes[1][0]])
            fig.colorbar(im_ang, ax=[axes[1][1]])
            fig.savefig(os.path.join(output_dir_fbt, name + '.png'))
            plt.close(fig)
    elif action == 3:
        # complex sparse reconstruction
        method = 'enet'
        fixed_center = False    # toy concentric dataset!
        n_paramsets = 1
        alphas = [1E-4]
        B, D = None, None

        if fixed_center:
            if not os.path.isfile(dict_path) or '--force' in sys.argv[1:]:
                B = np.load(basis_path)
                B = crop_bases(B)
                image, tag = load_data(sdh, l[0])
                solver = FBBSS(bases=B, basis_size=basis_size, image_size=image_size)
                solver.load_image(image, tag.y0, tag.x0)
                solver.build_dictionary_toreal()
                np.save(dict_path, solver.d)
            D = np.load(dict_path)
        else:
            B = np.load(basis_path)

        for i, f in enumerate(l):
            print('%d / %d' % (i, len(l)))
            image, tag = load_data(sdh, f)
            solver, pre_image = None, None
            if fixed_center:
                solver = FBBSS(dictionary=D, basis_size=basis_size, image_size=image_size)
                pre_image = solver.load_image(image, tag.y0, tag.x0)
            else:
                solver = FBBSS(bases=B, basis_size=basis_size, image_size=image_size)
                pre_image = solver.load_image(image, tag.y0, tag.x0)
                t1 = time.time()
                solver.crop_bases()
                solver.build_dictionary_toreal()
                print('Dictionary generated in %fs.' % (time.time() - t1))
            coef = np.zeros([solver.d.shape[1], n_paramsets])
            input_image = np.reshape(pre_image, [-1, 1])
            input_image = solver.toreal_transform(input_image)
            for j, alpha in enumerate(alphas):
                coef[:, j] = solver.solveEnet(input_image, alpha=alphas[j], l1_ratio=0.8, max_iter=20)

            expr, name = sdh.split_expr_name(f)
            output_dir_f = output_dir + expr
            output_dir_method = output_dir_f + '/' + method
            if not os.path.isdir(output_dir_f):
                os.mkdir(output_dir_f)
            if not os.path.isdir(output_dir_method):
                os.mkdir(output_dir_method)
            np.save(os.path.join(output_dir_method, name + '.npy'), coef)

            if not '--noplot' in sys.argv[1:]:
                # display result
                fig, axes = plt.subplots(nrows=2, ncols=n_paramsets+1, figsize=(23, 11))
                im = axes[0][0].imshow(pre_image)
                vmin, vmax = np.amin(pre_image), np.amax(pre_image)
                for j in range(n_paramsets):
                    sx = int(coef.shape[0] / 2)
                    sy = image_size * image_size
                    x = coef[:, j]
                    y_recon = np.dot(solver.d, x)
                    y_recon = solver.tocomplex_transform(y_recon)
                    y_recon = np.real(np.reshape(y_recon, [image_size, image_size]))
                    axes[0][j + 1].imshow(y_recon, vmin=vmin, vmax=vmax)
                    psnr = skimage.measure.compare_psnr(pre_image, y_recon, dynamic_range=vmax-vmin)
                    if method == 'enet':
                        axes[0][j + 1].set_title('PSNR = %f' % psnr)
                    elif False:
                        axes[0][j + 1].set_title('NNZ = %d, PSNR = %f' % (nnzs[j], psnr))
                    amp = np.absolute(solver.tocomplex_transform(x))
                    amp = np.reshape(amp, [40, 11])
                    amp = np.insert(amp, 0, 0, axis=0)
                    imamp = axes[1][j + 1].imshow(amp)
                    fig.colorbar(imamp, ax=[axes[1][j + 1]])
                fig.colorbar(im, ax=axes[0, :].ravel().tolist())
                fig.savefig(os.path.join(output_dir_method, name + '.png'))
                plt.close(fig)

'''
    elif action == 3:
        # compute residual
        normalize = 'log'
        B = np.load(basis_path)
        B = crop_bases(B)
        xx, yy = np.meshgrid(np.arange(0, image_size, 1),
                             np.arange(0, image_size, 1))
        xx -= 128
        yy -= 128
        rr = np.sqrt(xx ** 2 + yy ** 2)

        for i, f in enumerate(l):
            print('%d / %d' % (i, len(l)))
            image, tag, x, y = load_data(f, normalize=normalize)
            expr, name = sdh.split_expr_name(f)
            output_dir_f = output_dir + expr
            output_dir_fbt = output_dir + expr + '/fbt_' + normalize
            output_dir_res = output_dir + expr + '/res_' + normalize
            if not os.path.isdir(output_dir_res):
                os.mkdir(output_dir_res)
            rmax = max([x, y, image_size - x, image_size - y])
            image = np.multiply(image, rr <= rmax)

            coef = np.load(os.path.join(output_dir_fbt, name + '.npy'))
            recon_image = np.zeros([image_size, image_size], dtype=np.complex)
            for n in range(1, 31):
                for m in range(21):
                    recon_image = recon_image + coef[n, m] * B[n, m, :, :]
            res_image = image - np.real(recon_image)
            np.save(os.path.join(output_dir_res, name + '.npy'), res_image)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(res_image)
            fig.colorbar(im)
            fig.savefig(os.path.join(output_dir_res, name + '.png'))
            plt.close(fig)
'''


'''
    elif action == 1:
        # run ElasticNetCV
        n_samples = 1
        np.random.shuffle(l)
        l = l[:n_samples]     # sample a few for CV
        solver = FBBSS(dictionary=D, basis_size=basis_size, image_size=image_size)
        images = np.zeros([image_size * image_size, n_samples])
        for i, f in enumerate(l):
            image, _ = load_data(imgpath=f)
            images[:, i] = np.reshape(image, [image_size * image_size])
        model = solver.runEnetCV(images.ravel(), l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
        print('EnetCV: alpha = %f, l1_ratio = %f' % (model.alpha_, model.l1_ratio_))
        plt.plot(model.alphas_, model.mse_path_.mean(axis=-1))
        y_recon = np.dot(np.transpose(D), model.coef_) + model.intercept_
        y_recon = np.reshape(y_recon, [image_size, image_size])
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
        axes[0][0].imshow(image)
        axes[0][1].imshow(y_recon)
        plt.show()
'''
