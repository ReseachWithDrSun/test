from fbb.FourierBesselBasis import *
from fbb.FourierBesselBasis import FourierBesselBasis as FBB
from fbb.FBBSparseSolver import FBBSparseSolver as FBBSS
from tagio.DatasetHelper import SyntheticDatasetHelper as SDH
from tagio.tag import *
from sklearn.decomposition import SparseCoder as SC
import numpy as np
import scipy.io
#import scipy.ndimage
import skimage.measure
import matplotlib.pyplot as plt
import sys

basis_path = '../xray_data/fbb_output/fbb.npy'
dict_path = '../xray_data/fbb_output/dict_toreal.npy' 
image_size = 1000 ##256
basis_size = 2500 ##600
# nnzs = [10, 20, 50]
# alphas = [0.01, 0.1, 1]
sparse_params = [1E-5, 1E-6, 1E-7]
algorithm = 'lars'


def load_data(imgid):
    # sdh = SDH('/home/zquan/synthetic_data_local')
    sdh = SDH('../SimulationCode/generators/synthetic_data/')
    l = sdh.get_image_list()
    t = sdh.get_tag_from_image(l[imgid])
    tag = tagdata(t, tag_type=tagtype.Synthetic)
    image = scipy.io.loadmat(l[imgid])['detector_image']
    image = 2 * np.log(image + 1) / np.log(65536) - 1
    image /= np.sum(image ** 2)
    return image, tag


if __name__ == '__main__':
    assert (len(sys.argv) > 2 and sys.argv[1] == '--action'), 'Usage: python run_fbb.py --action [action id]'
    action = int(sys.argv[2])
    if action == 0:
        # precompute bases
        fbb = FBB(image_size, basis_size, [int(basis_size / 2), int(basis_size / 2)])
        B = fbb.compute_bases(40, 20)
        np.save(basis_path, B)
    elif action == 1:
        # compute sparse coding
        B = np.load(basis_path)
        solver = FBBSS(bases=B, basis_size=basis_size, image_size=image_size)
        image, tag = load_data(imgid=280)
        shifted_image = solver.load_image(image, tag.y0, tag.x0)
        solver.crop_bases()
        solver.build_dictionary()
        np.save(dict_path, solver.d)
        for i, p in enumerate(sparse_params):
            print('Computing sparse coding %d / %d (%s)' % (i, len(sparse_params), algorithm) )
            if algorithm == 'omp':
                x = solver.solveOMP(image, p)
            elif algorithm == 'lars':
                x = solver.solveLARS(image, p)
            np.save('%s-coef-%d.npy' % (algorithm, i), x)
    elif action == 2:
        # display result
        D = np.load(dict_path)
        image, tag = load_data(imgid=280)
        solver = FBBSS(dictionary=D, basis_size=basis_size, image_size=image_size)
        shifted_image = solver.load_image(image, tag.y0, tag.x0)
        fig, axes = plt.subplots(nrows=2, ncols=len(sparse_params)+1)
        im = axes[0][0].imshow(shifted_image)
        vmin, vmax = np.amin(shifted_image), np.amax(shifted_image)
        for i, p in enumerate(sparse_params):
            x = np.load('%s-coef-%d.npy' % (algorithm, i))
            y_recon = np.reshape(np.dot(x, D), [image_size, image_size])
            axes[0][i + 1].imshow(y_recon, vmin=vmin, vmax=vmax)
            axes[1][i + 1].plot(x[0, :])
            # mse = np.sum((shifted_image - y_recon) ** 2)
            dr = np.amax(shifted_image) - np.amin(shifted_image)
            psnr = skimage.measure.compare_psnr(shifted_image, y_recon, dynamic_range=dr)
            axes[0][i + 1].set_title('Param = %f, PSNR = %f' % (float(p), psnr))
        fig.colorbar(im, ax=axes[0, :].ravel().tolist())
        plt.show()
    elif action == 3:
        # load and display reconstruction
        B = np.load(basis_path)
        solver = FBBSS(bases=B, basis_size=basis_size, image_size=image_size)
        solver.build_dictionary()
        x = np.load('coef-0.npy')
        y_recon = np.reshape(np.dot(x, solver.d), [basis_size, basis_size])
        plt.imshow(y_recon)
        plt.colorbar()
        plt.show()
    elif action == 4:
        # the full transform
        B = np.load(basis_path)
        B = crop_bases(B)
        coef = np.zeros([31, 21], dtype=np.complex)
        image, tag = load_data(imgid=233)
        shifted_image, _, _ = shift_image(image, tag.x0, tag.y0)
        for n in range(1, 31):
            print('%d / 30' % n)
            for m in range(21):
                coef[n, m] = np.sum(np.multiply(shifted_image, np.conj(B[n, m, :, :])))
        np.save('fbtcoef.npy', coef)

        ##recon_image = np.zeros([256, 256], dtype=np.complex) ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        recon_image = np.zeros([1000, 1000], dtype=np.complex)
        for n in range(1, 31):
            for m in range(21):
                recon_image = recon_image + coef[n, m] * B[n, m, :, :]
        fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
        im = axes[0][0].imshow(shifted_image)
        vmin, vmax = np.amin(shifted_image), np.amax(shifted_image)
        axes[0][1].imshow(np.real(recon_image), vmin=vmin, vmax=vmax)
        axes[0][2].imshow(np.imag(recon_image), vmin=vmin, vmax=vmax)
        # psnr = skimage.measure.compare_psnr(shifted_image, recon_image, dynamic_range=vmax - vmin)
        # axes[0][1].set_title('PSNR = %f' % psnr)
        fig.colorbar(im)
        plt.show()

'''
    elif action == 5:
        # test: understanding elastic net reconstruction
        from fbb.TestPolarFunction import TestPolarFunction
        B = np.load(basis_path)
        B = crop_bases(B)
        D = np.load(dict_path)
        coef = np.zeros([31, 21], dtype=np.complex)

        tpf = TestPolarFunction(256, 256, [128, 128])
        p = tpf.PhiPeriodic(6, 0.5, 1, 0)
        solver = FBBSS(dictionary=D, basis_size=basis_size, image_size=image_size)
        pre_image = solver.load_image(p, 128, 128)

        # fbt:
        for n in range(1, 31):
            for m in range(21):
                coef[n, m] = np.sum(np.multiply(pre_image, np.conj(B[n, m, :, :])))

        y_fbt = np.zeros([256, 256], dtype=np.complex)
        for n in range(1, 31):
            for m in range(21):
                y_fbt = y_fbt + coef[n, m] * B[n, m, :, :]
        y_fbt = np.reshape(np.real(y_fbt), [image_size, image_size])
        # linear regression:
        from sklearn.linear_model import LinearRegression
        y = np.reshape(pre_image, [1, -1])
        sy = image_size * image_size
        y = solver.toreal_transform(y).T
        lr = LinearRegression()
        lr.fit(D.T, y)
        # print('Dt shape')
        # print(D.T.shape)
        # print('y shape')
        # print(y.shape)
        # print('coef shape')
        # print(lr.coef_.shape)
        # print('intercept shape')
        # print(lr.intercept_.shape)
        y_lr = np.dot(D.T, lr.coef_.T) + lr.intercept_
        # print('y recon shape')
        # print(y_lr.shape)
        y_lr = np.real(y_lr[:sy] + y_lr[sy:] * 1j)
        # print('y toreal shape')
        # print(y_lr.shape)
        y_lr = np.reshape(y_lr, [image_size, image_size])
        sw = int(lr.coef_.size / 2)
        w_lr = lr.coef_[0, :sw] + lr.coef_[0, sw:] * 1j
        w_lr = np.reshape(w_lr, [30, 21])
        w_lr = np.insert(w_lr, 0, 0, axis=0)
        # # OMP:
        # from sklearn.decomposition import SparseCoder
        # sc = SparseCoder(dictionary=D, transform_n_nonzero_coefs=20)
        # w_omp = sc.transform(y.T)
        # y_omp = np.dot(w_omp, D)
        # y_omp = np.real(y_omp[])
        # elastic net:
        from sklearn.linear_model import ElasticNet
        enet = ElasticNet(alpha=0.1, l1_ratio=0.8)
        enet.fit(D.T, y)
        print('enet coef shape')
        print(enet.coef_.shape)
        y_enet = np.dot(D.T, enet.coef_.T) + enet.intercept_
        y_enet = np.real(y_enet[:sy] + y_enet[sy:] * 1j)
        y_enet = np.reshape(y_enet, [image_size, image_size])
        w_enet = enet.coef_[:sw] + enet.coef_[sw:] * 1j
        w_enet = np.reshape(w_enet, [30, 21])
        w_enet = np.insert(w_enet, 0, 0, axis=0)

        fig, ax = plt.subplots(nrows=2, ncols=4, squeeze=False)
        vmin, vmax = np.amin(pre_image), np.amax(pre_image)
        im = ax[0][0].imshow(pre_image, vmin=vmin, vmax=vmax)
        ax[0][1].imshow(y_fbt, vmin=vmin, vmax=vmax)
        ax[0][2].imshow(y_lr, vmin=vmin, vmax=vmax)
        ax[0][3].imshow(y_enet, vmin=vmin, vmax=vmax)
        imt1 = ax[1][1].imshow(np.absolute(coef))
        imt2 = ax[1][2].imshow(np.absolute(w_lr))
        imt3 = ax[1][3].imshow(np.absolute(w_enet))
        fig.colorbar(im, ax=ax[0, :].ravel().tolist())
        fig.colorbar(imt1, ax=ax[1][1])
        fig.colorbar(imt2, ax=ax[1][2])
        fig.colorbar(imt3, ax=ax[1][3])
        plt.show()
'''


'''
    elif action == 5:
        # test: understanding simple symmetry full fbt
        def generate_tpf(t, p, nsym, width, mag, bias, lower, upper):
            p['detector_image'] = np.multiply(t.PhiPeriodic(nsym, width, mag, bias),
                                              t.RBand(lower, upper))
            return p

        # synthesize new symmetric figures
        from fbb.TestPolarFunction import TestPolarFunction
        # import scipy.io
        tpf = TestPolarFunction(256, 256, [128, 128])
        # p = {}
        # p = generate_tpf(tpf, p, 6, 0.2, 0.5, 0.1, 100, 102)
        p = np.multiply(tpf.PhiPeriodic(4, 0.2, 0.5, 0.1),
                        tpf.RBand(100, 102))
        p /= np.sum(p ** 2)
        # compute
        B = np.load(basis_path)
        B = crop_bases(B)
        coef = np.zeros([31, 21], dtype=np.complex)
        for n in range(1, 31):
            print('%d / 30' % n)
            for m in range(21):
                coef[n, m] = np.sum(np.multiply(p, np.conj(B[n, m, :, :])))

        recon_image = np.zeros([256, 256], dtype=np.complex)
        for n in range(1, 31):
            for m in range(21):
                recon_image = recon_image + coef[n, m] * B[n, m, :, :]
        fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False)
        vmin, vmax = np.amin(p), np.amax(p)
        im = axes[0][0].imshow(p)
        im_recon = axes[0][1].imshow(np.real(recon_image),
                                     vmin=vmin, vmax=vmax)
        im_mag = axes[1][0].imshow(np.absolute(coef))
        im_ang = axes[1][1].imshow(np.angle(coef))
        fig.colorbar(im, ax=[axes[0][0], axes[0][1]])
        fig.colorbar(im_mag, ax=[axes[1][0]])
        fig.colorbar(im_ang, ax=[axes[1][1]])
        plt.show()
'''


'''
    elif action == 4:
        D = np.load('/home/zquan/xray_data/fbb_output/dict.npy')
        image, tag = load_data(imgid=42)
        solver = FBBSS(dictionary=D, basis_size=basis_size, image_size=image_size)
        shifted_image = solver.load_image(image, tag.y0, tag.x0)
        coef, intercept = solver.solveEnet(shifted_image, alpha=0.01, l1_ratio=0.7)
        y_recon = np.dot(np.transpose(D), coef) + intercept
        y_recon = np.reshape(y_recon, [image_size, image_size])
        np.save('recon.npy', y_recon)
        np.save('coef.npy', coef)
        np.save('intercept.npy', intercept)
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
        axes[0][0].imshow(shifted_image)
        axes[0][1].imshow(y_recon)
        plt.show()
'''

'''
    elif action == 3:
        B = np.load(basis_path)
        solver = FBBSS(bases=B, basis_size=basis_size, image_size=image_size)
        image, tag = load_data(imgid=280)
        shifted_image = solver.load_image(image, tag.y0, tag.x0)
        solver.crop_bases()
        solver.view_basis(shifted_image, 3, 0)
'''
