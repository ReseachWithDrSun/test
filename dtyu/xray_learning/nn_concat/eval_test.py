from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nn import eval


if __name__ == '__main__':
    ids = [ ['45f18c5c_varied_sm', '{:08x}'.format(i)]
            for i in range(200)]
    pred_all, gt_all = eval.evaluate('../xray_data/experiment', ids)
    print(pred_all)
    print(gt_all)

