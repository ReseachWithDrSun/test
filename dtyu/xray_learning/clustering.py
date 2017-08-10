from tagio.DatasetHelper import SyntheticDatasetHelper as SDH
import tagio.xray_data_processor as Processor
import numpy as np
import sys
import os
import scipy.io
from sklearn.cluster import KMeans
import pickle


output_dir = '/home/zquan/xray_data/fbb_output/'
input_dir = '/home/zquan/xray_data/fbb_output/nodefect_50k/'
feature_len = 880
nclusters = 10
sdh = SDH('/home/zquan/xray_data/nodefect_50k')

with open(os.path.join(output_dir, 'imagelist'), 'rb') as f:
    l = pickle.load(f)
l = np.array(l)
all_coefs = np.zeros([len(l), feature_len], dtype=np.float32)
for i in range(len(l)):
    # fetch calculated sparse coefs
    expr, name = sdh.split_expr_name(l[i])
    # fbt coefs
    coef_path = os.path.join(input_dir, expr, name + '.npy')
    coef = np.load(coef_path).astype(np.float32)
    all_coefs[i, :] = coef

print('Begin k-means clustering...')
km = KMeans(n_clusters=nclusters)
km.fit(all_coefs)

for i in range(nclusters):
    print(i)
    with open(os.path.join(output_dir, 'imagelist-cluster-%d' % i), 'w') as f:
        li = l[km.labels_ == i]
        for ff in li:
            f.write(ff + '\n')
