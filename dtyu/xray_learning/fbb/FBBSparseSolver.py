import scipy.ndimage
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from fbb.TestPolarFunction import TestPolarFunction


'''
Handles the sparse solver of FBB approximation and preprocessing of precomputed
bases, i.e. shifting, cropping and dictionary initialization
'''
class FBBSparseSolver:
    '''
    Members:
        b: precomputed FBB
        bsize: size of precomputed FBB
        imgsize: size of the images to reconstruct, and which bases to crop into
        d: dictionary
        x: (shifted) image center
        y: (shifted) image center
    '''
    def __init__(self, bases=None, dictionary=None, basis_size=0, image_size=0):
        self.b = bases
        self.bsize = basis_size
        self.imgsize = image_size
        self.d = dictionary
        self.x = 0
        self.y = 0
        self.rr = None
        self.rmax = 0

    '''
    Shift the beam center of the image to an integer pixel position so that FBB
    can be directly cropped out
    '''
    def load_image(self, image, y0, x0, remap='log', mask=None, medfilt=False):
        # shift image to grid point
        self.x = int(round(x0))
        self.y = int(round(y0))
        output_image = scipy.ndimage.shift(image, [self.x - x0, self.y - y0])
        # range remap
        if remap == 'log':
            output_image = np.log(output_image + 1) / np.log(65536)
        else:
            raise Exception('Unrecognized remapping method')
        # medium filter
        if medfilt:
            output_image = scipy.signal.medfilt(output_image)
        if mask is not None:
            output_image = np.multiply(mask, output_image)
        # bounded disk projection
        xx, yy = np.meshgrid(np.arange(0, self.imgsize, 1),
                             np.arange(0, self.imgsize, 1))
        xx -= self.x
        yy -= self.y
        self.rr = np.sqrt(xx ** 2 + yy ** 2)     # save rmap for basis normalization
        # compute max radius w/o broken rings
        self.rmax = max([self.x, self.y, self.imgsize - self.x, self.imgsize - self.y])
        output_image = np.multiply(output_image, self.rr <= self.rmax)
        # normalize to unit norm
        output_image /= np.sqrt(np.sum(output_image ** 2))
        return output_image

    '''
    Crop the bases to match the image to reconstruct as a rectangular window of
    the full size bases
        Requires: x, y determined by shifting image
        Args: mask, array, a mask of valid pixels
    '''
    def crop_bases(self, mask=None):
        # self.b = self.b[1:, :, :, :]
        self.b = self.b[:, :, int(self.bsize / 2 - self.y):int(self.bsize / 2 - self.y + self.imgsize), int(self.bsize / 2 - self.x):int(self.bsize / 2 - self.x + self.imgsize)]
        if mask is not None:
            for i in range(self.b.shape[0]):
                for j in range(self.b.shape[1]):
                    self.b[i, j, :, :] = np.multiply(mask, self.b[i, j, :, :])

    '''
    Build the dictionary
        Requires: Cropped bases
    '''
    def build_dictionary(self):
        # split real and imaginary parts
        real = np.real(self.b[1:, :, :, :])
        imag = np.imag(self.b[1:, 1:, :, :])
        # flatten all
        real = np.reshape(real, [real.shape[0] * real.shape[1], -1])
        imag = np.reshape(imag, [imag.shape[0] * imag.shape[1], -1])
        # # remove all zero images from pure real/imaginary bases
        # real = real[np.where(np.sum(abs(real), axis=1) > 1E-16)[0], :]
        # imag = imag[np.where(np.sum(abs(imag), axis=1) > 1E-16)[0], :]
        # normalization
        self.d = np.concatenate([real, imag], axis=0)
        self.d /= np.sqrt(np.sum(self.d ** 2, axis=1))[:, np.newaxis]

    '''
    Build the complex dictionary; no real/imaginary split
        Requires: Cropped basis
        # Args: r: int, range of support
    '''
    def build_dictionary_complex(self):
        self.d = np.asarray(self.b[1:, :, :, :])
        # self.d = np.transpose(self.d, axes=(2, 3, 0, 1))
        # self.d = np.reshape(self.d, [self.d.shape[0], self.d.shape[1], -1])
        # self.d = np.transpose(self.d, axes=(2, 0, 1))
        self.d = np.reshape(self.d, [self.d.shape[0] * self.d.shape[1], -1]).T
        self.d /= np.sqrt(np.sum(self.d ** 2, axis=1))[:, np.newaxis]

    '''
    Convert the complex valued dictionary/vector to real for sparse reconstruction
    Note: sklearn SparseCoder uses transposed version of everything
        Args: a, input vector or matrix
              t, bool, transpose (SparseCoder format) or regular (ElasticNet and
              standard sparse reconstruction theory format)
    '''
    def toreal_transform(self, a, t=False):
        assert a.ndim == 2, 'a must be a 2D matrix or (1, .) row vector'
        s = a.shape
        if not t and s[1] == 1:
            return np.append(np.real(a), np.imag(a), axis=0)
        elif t and s[0] == 1:
            return np.reshape(np.append(np.real(a), np.imag(a)), [1, -1])
        else:
            aa = np.zeros([2 * s[0], 2 * s[1]])
            aa[:s[0], :s[1]] = np.real(a)
            if not t:
                aa[:s[0], s[1]:] = -np.imag(a)
                aa[s[0]:, :s[1]] = np.imag(a)
            else:
                aa[:s[0], s[1]:] = np.imag(a)
                aa[s[0]:, :s[1]] = -np.imag(a)
            aa[s[0]:, s[1]:] = np.real(a)
            return aa

    def tocomplex_transform(self, aa, t=False):
        if aa.ndim == 1:
            s = int(aa.size / 2)
            return aa[:s] + aa[s:] * 1j
        elif aa.ndim == 2:
            if not t and aa.shape[1] == 1:
                s = int(aa.shape[0])
                return aa[:s, 0] + aa[s:, 0] * 1j
            elif t and aa.shape[0] == 1:
                s = int(aa.shape[1])
                return aa[0, :s] + aa[0, s:] * 1j
            else:
                raise ValueError()
        else:
            raise ValueError()

    '''
    Build the dictionary that converts complex problem to real problem
        Requires: Cropped basis, calculated rmap and rmax
    '''
    def build_dictionary_toreal(self, t=False):
        # drop m=0 and odd angular symmetry; fortran style storage
        dict_complex = np.asarray(self.b[1:, ::2, :, :], order='F')
        # bounded disk projection
        for m in range(dict_complex.shape[0]):
            for n in range(dict_complex.shape[1]):
                dict_complex[m, n, :, :] = np.multiply(dict_complex[m, n, :, :],
                                                       self.rr <= self.rmax)
        # flatten r/a frequencies & image pixels
        dict_complex = np.reshape(dict_complex, [dict_complex.shape[0] * dict_complex.shape[1], -1])
        if not t:
            # t flag on: "transposed" version -> what we get from the reshape, vice versa
            # TODO: favor "untransposed" from the begining of computing bases to speedup calculation
            dict_complex = dict_complex.T
        self.d = self.toreal_transform(dict_complex, t=t)
        # if mask is not None:
        #     # trim the empty pixels
        #     pass
        if not t:
            self.d /= np.sqrt(np.sum(self.d ** 2, axis=0))
        else:
            self.d /= np.sqrt(np.sum(self.d ** 2, axis=1))[:, np.newaxis]

    '''
    Solve the sparse coding problem with OMP algorithm.
        Requires: computed dictionary
    '''
    def solveOMP(self, image, nnz):
        from sklearn.decomposition import SparseCoder
        y = np.reshape(image, [1, -1])
        coder = SparseCoder(dictionary=self.d, transform_n_nonzero_coefs=nnz)
        return coder.transform(y)

    def solveCD(self, image, alpha):
        from sklearn.decomposition import SparseCoder
        y = np.reshape(image, [1, -1])
        coder = SparseCoder(dictionary=self.d, transform_algorithm='lasso_cd', transform_alpha=alpha)
        return coder.transform(y)

    def view_basis(self, image, n, m):
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # axes[0].imshow(image)
        # axes[1].imshow(np.real(self.b[n, m, :, :]))
        # plt.show()
        plt.imshow(image - 0.01 * np.real(self.b[n, m, :, :]))
        plt.show()

    # '''
    # Perform cross validation to determine elastic net reg. parameters
    # '''
    # def runEnetCV(self, images, l1_ratio=0.5):
    #     from sklearn.linear_model import ElasticNetCV
    #     enetcv = ElasticNetCV(l1_ratio=l1_ratio, cv=5)
    #     enetcv.fit(np.transpose(self.d), images)
    #     return enetcv.alpha_, enetcv.l1_ratio_

    '''
    Solve the sparse problem with Elastic net objective function (L1+L2 error)
        Note: this is not transposed
    '''
    def solveEnet(self, image, alpha, l1_ratio, max_iter=100, tol=1E-4, stride=2):
        from sklearn.linear_model import ElasticNet

        # if mask is not None:
        #     # if mask exists, image and dict are trimmed to reduce zero terms
        #     mask = np.reshape(mask, [-1, 1])
        #     mask = np.concatenate([mask, mask], axis=0)
        #     image = image[mask > 0, :]
        #     # save dict w/o zero pixels to reduce ops
        #     # d = d[mask > 0, :]

        # cancel out the 1 / n_samples term for regression
        scale = image.shape[0]
        enet = ElasticNet(alpha=alpha/scale, l1_ratio=l1_ratio,
                          fit_intercept=False, max_iter=max_iter, tol=tol)
        enet.fit(self.d[::stride, :], image[::stride, :])
        return enet.coef_, enet.n_iter_

    '''
    Solve the complex valued sparse problem with SGP algorithm
    '''
    def solveSPG(self, image, tau=0, sigma=0, x0=[], opts={}):
        from spgl1 import spgl1
        y = np.reshape(image, [image.size])
        return spgl1(self.d, y, tau, sigma, x0, opts)

    '''
    Solve the complex valued -> real valued sparse problem with OMP
    '''
    def solveTorealOMP(self, image, nnz):
        from sklearn.decomposition import SparseCoder
        s = image.size
        y = self.toreal_transform(np.reshape(image, [1, -1]))
        coder = SparseCoder(dictionary=self.d, transform_n_nonzero_coefs=nnz)
        return coder.transform(y)
