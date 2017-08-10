# from tagio.tag import *
from tagio.DatasetHelper import RealDatasetHelper as DH
# import tagio.xray_data_processor as Processor
import numpy as np
import sys
import os
# from fbb.FBBSparseSolver import FBBSparseSolver as FBBSS
import scipy.io
import pickle


output_dir = '/home/shared/mixed_dataset/preprocessed_data/'
input_dir = '/home/shared/mixed_dataset/cropped/'
coef_dir = '/home/shared/mixed_dataset/fb/'
#train_test_split = 2000
np.seterr(all='raise')

rdh = DH(input_dir)

flag_list_source = ''    # indicates where to read the file list to process

if flag_list_source == '':    # generate new and randomize
    l = rdh.get_image_list()
    l1 = []
    for f in l:
        # verify coef.
        name = rdh.get_name(f)
        if os.path.isfile(coef_dir + name + '.npy'):
            l1.append(f)
    print('%d / %d' % (len(l1), len(l)))
    l1 = np.random.permutation(l1)
    with open(os.path.join(output_dir, 'imagelist'), 'w') as f:
        for s_file in l1:
            f.write(s_file + '\n')
elif flag_list_source == 'list':   # read from generated to maintain the split
    with open(os.path.join(output_dir, 'imagelist'), 'r') as f:
        l1 = f.read().splitlines()
else:
    pass    # happens one time when I deleted the imagelist by accident
# solver = FBBSS()

flag_datatype = 'cnn'
# gather enet training set


def write_binary(filelist, filename):
    with open(filename, 'wb') as f:
        for i in range(len(filelist)):
            if i % 100 == 0:
                print('%d / %d' % (i, len(filelist)))

            # 10 tags are not used in real dataset
            label_w = np.zeros([1, 32], dtype=np.uint8)
            f.write(label_w.tostring())

            # fetch calculated sparse coefs
            name = rdh.get_name(filelist[i])
            # fbt coefs
            coef_path = os.path.join(coef_dir, name + '.npy')
            coef = np.load(coef_path).astype(np.float32)
            # coef = np.reshape(coef, [2, 40, 11])    # it's raw data in binary anyway...
            f.write(coef.tostring())

            # write log-image
            image = scipy.io.loadmat(filelist[i])['detector_image']
            #image = image[:256, :]
            try:
                image = 255 * np.log(1 + image) / np.log(65536)
            except Exception:
                print('Error raised: ' + filelist[i])
            image = image.astype(np.uint8)
            f.write(image.tostring())


if flag_datatype == 'cnn':
    write_binary(l1[:2300], os.path.join(output_dir, 'batch-0.bin'))
    write_binary(l1[2300:], os.path.join(output_dir, 'batch-1.bin'))

'''
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
            coef_path = os.path.join(input_dir, expr, name + '.npy')
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
'''
