import numpy as np
import os
import yaml


'''
convert numpy format data to libsvm format.

Args:
    X, numpy array, label, integer type
    Y, numpy array, feature vectors
    output, string, output file path
'''
def convert_to_libsvm(X, Y, output):
    assert X.shape[0] == Y.shape[0], 'Label and feature sizes don\'t match'
    with open(output, 'w') as f:
        for i in range(Y.shape[0]):
            l = ['-1', '+1'][Y[i]]
            for j in range(X.shape[1]):
                if abs(X[i][j]) > 1e-16:
                    l = l + ' %d:%f' % (j + 1, X[i][j])
            l = l + '\n'
            f.write(l)


'''
Rescale the feature vectors (matrices) to [0, 1] for SVM classification.

Args:
    mat, numpy array, the matrix to rescale
    mode, int, 0 - rescale each dimension separately;
               1 - rescale the entire matrix with the same scale and shift.
    mins, numpy array or numerical, known minimums from training data
    maxs, numpy array or numerical, known maximums
'''
def rescale_features(mat, mode=0, mins=None, maxs=None):
    rescaled = mat
    if mode == 0:
        if mins is None:
            mins = np.amin(mat, axis=0)
            maxs = np.amax(mat, axis=0)
        for i in range(rescaled.shape[1]):
            if mins is None:
                rescaled[:,i] = (rescaled[:,i] - np.amin(rescaled[:,i])) / (np.amax(rescaled[:,i]) - np.amin(rescaled[:,i]))
            else:
                rescaled[:,i] = (rescaled[:,i] - mins[i]) / (maxs[i] - mins[i])
    elif mode == 1:
        if mins is None:
            mins = np.amin(mat)
            maxs = np.amax(mat)
            rescaled = (rescaled - np.amin(rescaled)) / (np.amax(rescaled) - np.amin(rescaled))
        else:
            rescaled = (rescaled - mins) / (maxs - mins)
    else:
        raise ValueError('Unsupported rescale mode')
    return rescaled, mins, maxs


if __name__ == '__main__':
    DATASET = 'val'
    FC_LAYER = 1
    c_logmax = 66000

    oned_binary_dir = '../../xray_data/synthetic_output_features/'
    mat_ground_truth = oned_binary_dir + 'gt_%s.npy' % DATASET
    mat_fc = oned_binary_dir + 'fc%d_%s.npy' % (FC_LAYER, DATASET)
    fc = np.load(mat_fc)

    mat_ground_truth_r = oned_binary_dir + 'gt_%s_r.npy' % DATASET
    mat_fc_r = oned_binary_dir + 'fc%d_%s_r.npy' % (FC_LAYER, DATASET)
    yml_rescale = oned_binary_dir + 'rescale.yml'
    print('Rescaling features...')
    if DATASET == 'train':
        fc, fc_mins, fc_maxs = rescale_features(fc)

        config = {}
        if os.path.isfile(yml_rescale):
            with open(yml_rescale, 'r') as f:
                config = yaml.load(f)
        config['fc%d_mins' % FC_LAYER] = fc_mins
        config['fc%d_maxs' % FC_LAYER] = fc_maxs
        np.save(mat_fc_r, fc)
        with open(yml_rescale, 'w') as f:
            yaml.dump(config, f)
    elif DATASET == 'val':
        config = {}
        try:
            with open(yml_rescale, 'r') as f:
                config = yaml.load(f)
        except:
            print('Failed to load precomputed scale and shift')
        fc_mins = config['fc%d_mins' % FC_LAYER]
        fc_maxs = config['fc%d_maxs' % FC_LAYER]
        fc, _, _ = rescale_features(fc, mins=fc_mins, maxs=fc_maxs)
        np.save(mat_fc_r, fc)
    else:
        raise ValueError
    ############################################################################

    #X = np.concatenate([fc, oneds], axis=1)
    X = fc
    Y = np.load(mat_ground_truth).astype(np.int)

    if DATASET == 'train':
        # can't use all synthetic data, sample some data for training
        sample_idx = np.arange(0, 95000)
        np.random.shuffle(sample_idx)
        sample_idx = sample_idx[:5000]
        X = X[sample_idx, :]
        Y = Y[sample_idx, :]
    for i in range(Y.shape[1]):
        print('Processing label %d' % i)
        output_path = oned_binary_dir + 'data_fc%d_label%d_%s' % (FC_LAYER, i, DATASET)
        convert_to_libsvm(X, Y[:, i], output_path)
