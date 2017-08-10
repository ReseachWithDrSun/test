from tagio.tag import *
from tagio.DatasetHelper import SyntheticDatasetHelper as SDH
import tagio.xray_data_processor as Processor
import numpy as np
import sys
import os
from fbb.FBBSparseSolver import FBBSparseSolver as FBBSS
import scipy.io
import pickle


output_dir = '../xray_data/fbb_output/' ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
input_dir = '../xray_data/fbb_output/' ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
feature_len = 880
label_len = 10
train_test_split = 45000
mask = np.array([0, 1, 2, 3, 4, 9, 10])

sdh = SDH('../xray_data/fbb_output/synthetic_data') ##~~~~~~~~~~~~~~~~~~~~~

flag_list_source = ''    # indicates where to read the file list to process

if flag_list_source == '':    # generate new and randomize
    l = sdh.get_image_list()
    l = np.random.permutation(l)
    print (l)
    # with open(os.path.join(output_dir, 'imagelist'), 'wb') as f:
    #     pickle.dump(l, f)
    with open(os.path.join(output_dir, 'imagelist'), 'w') as f:
        for s_file in l:
            f.write(s_file + '\n')
elif flag_list_source == 'list':   # read from generated to maintain the split
    # with open(os.path.join(output_dir, 'imagelist'), 'rb') as f:
    #     l = pickle.load(f)
    with open(os.path.join(output_dir, 'imagelist'), 'r') as f:
        l = f.read().splitlines()
else:
    pass    # happens one time when I deleted the imagelist by accident
# solver = FBBSS()

flag_datatype = 'cnn'
# gather enet training set

if flag_datatype == 'cnn':
    batch_size = 5000
    ind = 0
    for i in range(int(len(l) / batch_size)):
        print('%d / %d' % (i, int(len(l) / batch_size)))
        with open(os.path.join(output_dir, 'batch-%d.bin' % i), 'wb') as f:
            for j in range(batch_size):
                t = sdh.get_tag_from_image(l[ind])
                tag = tagdata(t, tag_type=tagtype.Synthetic)
                label = Processor.SimulatedFeatureSelectorSymmetry(tag)
                flag_rearrange = False
                if flag_rearrange:
                    label_w = np.zeros([1, 32], dtype=np.uint8)
                    # remove structure and image detect tags
                    label_w[:, :len(mask)] = label[:, mask]
                    # rearrange the labels for masked evaluation
                    f.write(label_w.tostring())
                else:
                    f.write(label.tostring())

                flag_pos = False
                # save beam center
                if flag_pos:
                    arr_pos = np.array([tag.x0, tag.y0]).astype(np.float32)
                    f.write(arr_pos.tostring())

                flag_fbt = True
                if flag_fbt:
                    # fetch calculated sparse coefs
                    expr, name = sdh.split_expr_name(l[ind])
                    # fbt coefs
                    coef_path = os.path.join(input_dir, '/analysis/oned/', expr, name + '.npy')
                    coef = np.load(coef_path).astype(np.float32)
                    # coef = np.reshape(coef, [2, 40, 11])    # it's raw data in binary anyway...
                    f.write(coef.tostring())

                # legacy 1D curve placeholder
                flag_oned = False
                if flag_oned:
                    f.write(bytes([0] * 1024))

                flag_image = True
                if flag_image:
                    # write log-image
                    image = scipy.io.loadmat(l[ind])['detector_image']
                    image = 255 * np.log(1 + image) / np.log(65536)
                    image = image.astype(np.uint8)
                    f.write(image.tostring())

                ind += 1
elif flag_datatype == 'svm':
    def normalize(v, flag=''):
        v_normalized = None
        if flag == '':
            v_normalized = v / np.linalg.norm(v) * 100
        elif flag == 'log':
            v_normalized = v / np.amax(np.abs(v)) * 255
            v_normalized = np.multiply(np.sign(v_normalized),
                                       np.log(1 + np.abs(v_normalized)) / np.log(256))
        return v_normalized
    flag = ''

    if flag_list_source == 'list':
        X = np.zeros([len(l), feature_len])
        Y = np.zeros([len(l), label_len])
        for i, image in enumerate(l):
            if i % 100 == 0:
                print('%d / %d' % (i, len(l)))
            t = sdh.get_tag_from_image(image)
            tag = tagdata(t, tag_type=tagtype.Synthetic)
            label = Processor.SimulatedFeatureSelectorSymmetry(tag)

            # fetch calculated sparse coefs
            expr, name = sdh.split_expr_name(image)
            # fbt coefs
            coef_path = os.path.join(input_dir, '/analysis/oned/', expr, name + '.npy')
            coefs = normalize(np.load(coef_path), flag='')
            X[i, :] = coefs[0, :]
            Y[i, :] = label[0, :]
    elif flag_list_source == 'bin':
        # this is not intended for regular use and there are hardcoded stuff
        import struct
        X = np.zeros([50000, feature_len])
        Y = np.zeros([50000, label_len])
        label_length = 32
        fbt_length = 880
        image_length = 65536
        record_bytes = label_length + fbt_length * 4 + image_length
        for i in range(10):
            print('%d / 10' % i)
            with open(os.path.join(output_dir, 'batch-%d.bin' % i), 'rb') as f:
                for j in range(5000):
                    f.seek(j * record_bytes)
                    label = struct.unpack('10B', f.read(10))
                    f.seek(j * record_bytes + label_length)
                    coefs = struct.unpack('880f', f.read(fbt_length * 4))
                    X[i * 5000 + j, :] = normalize(np.asarray(coefs), flag=flag)   #generate log-enhanced samples
                    Y[i * 5000 + j, :] = np.asarray(label)

    np.save(os.path.join(output_dir, 'X_train_' + flag + '.npy'), X[:train_test_split, :])
    np.save(os.path.join(output_dir, 'Y_train.npy'), Y[:train_test_split, :])
    np.save(os.path.join(output_dir, 'X_test_' + flag + '.npy'), X[train_test_split:, :])
    np.save(os.path.join(output_dir, 'Y_test.npy'), Y[train_test_split:, :])
