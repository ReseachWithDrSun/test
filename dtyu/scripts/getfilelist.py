import hashlib
import os


path_filelist_0 = '/home/zquan/xray_data/fbb_output/temp/imagelist'
path_filelist_1 = '/home/zquan/xray_data/fbb_output/temp/imagelist_out'
# filelist_0 = []
filelist_1 = []
path_batch_0 = '/home/zquan/xray_data/fbb_output/temp/batch-0.bin'
path_batch_1 = ['/home/zquan/xray_data/fbb_output/batch-%d.bin' % i for i in range(10)]
len_record = 32 + 880 * 4 + 65536
hashes_0 = []
hashes_1 = []
length_0 = os.path.getsize(path_batch_0)
assert length_0 % len_record == 0, 'Corrupted binary ' + path_batch_0

with open(path_batch_0, 'rb') as f:
    n_record_0 = length_0 // len_record
    for i in range(n_record_0):
        if i % 10 == 0:
            print('Hashing 0: %d / %d' % (i, n_record_0))
        f.seek(i * len_record)
        record = f.read(len_record)
        hashes_0.append(hashlib.md5(record).hexdigest())
for pb in path_batch_1:
    length_1 = os.path.getsize(pb)
    assert length_1 % len_record == 0, 'Corrupted binary ' + pb
    with open(pb, 'rb') as f:
        n_record_1 = length_1 // len_record
        for i in range(n_record_1):
            if i % 10 == 0:
                print('Hashing %s: %d / %d' % (pb, i, n_record_1))
            f.seek(i * len_record)
            record = f.read(len_record)
            hashes_1.append(hashlib.md5(record).hexdigest())
with open(path_filelist_0) as f:
    filelist_0 = f.read().splitlines()
for i, h1 in enumerate(hashes_1):
    print('Matching %d / %d' % (i, n_record_0))
    filelist_1.append(filelist_0[hashes_0.index(h1)])
with open(path_filelist_1, 'w') as f:
    for ff in filelist_1:
        f.write(ff + '\n')
