import scipy.special as ss
import numpy as np
import math
from fbb.PolarFunction import PolarFunction

import scipy.ndimage


class FourierBesselBasis(PolarFunction):
    def __init__(self, a, s, c):
        super(FourierBesselBasis, self).__init__(a, s, c)

    def Phi(self, m):
        return 1 / np.sqrt(2 * math.pi) * np.exp(1j * m * self.phi)

    def R(self, n, m):
        # find the n-th positive root of J_m(.)
        xmn = ss.jn_zeros(m, n)[-1]
        # compute square root of N_n^{(m)}
        sNnm = self.a * ss.jn(m + 1, xmn) / np.sqrt(2)
        # find the energy level/allowed values k_{mn}
        kmn = xmn / self.a
        return 1 / sNnm * ss.jn(m, kmn * self.r)

    def Psi(self, n, m):
        return np.multiply(self.Phi(m), self.R(n, m))
        # p = np.multiply(self.Phi(m), self.R(n, m))
        # if np.isnan(p).sum() > 0:
        #     print('FBB Psi: NaN at (%d, %d)' % (n, m))
        # return p

    def compute_bases(self, maxn, maxm):
        B = np.zeros([maxn + 1, maxm + 1, self.s, self.s]).astype(np.complex)
        # B[0, :] are left blank so that n indices start from 1
        for n in range(1, maxn + 1):
            print('%d / %d' % (n, maxn))
            for m in range(maxm + 1):
                B[n, m, :, :] = self.Psi(n, m)
        return B


# TODO: image-adaptive-basis helper class
def crop_bases(b, bsize=600, imgsize=256, x=128, y=128):
    if b.ndim == 4:
        b = b[:, :, int(bsize / 2 - y):int(bsize / 2 - y + imgsize), int(bsize / 2 - x):int(bsize / 2 - x + imgsize)]
    elif b.ndim == 2:
        b = b[int(bsize / 2 - y):int(bsize / 2 - y + imgsize), int(bsize / 2 - x):int(bsize / 2 - x + imgsize)]
    return b


def shift_image(image, x0, y0):
    x = int(round(x0))
    y = int(round(y0))
    shifted_image = scipy.ndimage.shift(image, [x - x0, y - y0])
    return shifted_image, x, y
