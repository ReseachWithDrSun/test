""" compute features for comparison.
"""
import numpy as np
import os
import scipy.io
import scipy.signal
import skimage.feature
import pickle
from phog_features.phog import PHogFeatures


def phog(image):
    pHog = PHogFeatures()
    feature = pHog.get_features(image, pyramid_levels=4)
    return np.reshape(feature, -1)


def lbp(image):
    lbp_digits = skimage.feature.local_binary_pattern(image, 24, 3, method='uniform')
    lbp_histograms = []
    for i in range(8):
        for j in range(8):
            patch = lbp_digits[32*i:32*(i+1), 32*j:32*(j+1)]
            h = np.histogram(patch, bins=26, range=(0, 26))
            lbp_histograms.append(h[0])
    feature = np.concatenate(lbp_histograms)
    return np.reshape(feature, -1)


def compute_features(filelist, feature_function, filename):
    features = []
    for f in filelist:
        image = scipy.io.loadmat(f)['detector_image'].astype(np.float64)
        #image = image[:256, :]
        image = scipy.signal.medfilt2d(image)
        features.append(feature_function(image))
    output = np.stack(features, axis=0)
    np.save(filename, output)


feature_map = {
    'phog': phog,
    'lbp': lbp
}


if __name__ == '__main__':
    with open('/home/shared/mixed_dataset/preprocessed_data/imagelist', 'r') as f:
        l = f.read().splitlines()
    l1 = l[:2300]
    for key, func in feature_map.items():
        print(key)
        compute_features(l1, func, 'X_train_' + key + '.npy')
    l2 = l[2300:]
    for key, func in feature_map.items():
        print(key)
        compute_features(l2, func, 'X_test_' + key + '.npy')
