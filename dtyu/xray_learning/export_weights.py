from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
import pickle


def export_weights(ckpt, out):
	reader = tf.train.NewCheckpointReader(ckpt)
	weights = {}
	tensor_list = reader.debug_string().decode('utf-8').split('\n')[:-1]
	tensor_list = [tensor_str.split(' ')[0] for tensor_str in tensor_list]
	for tensor in tensor_list:
		print(tensor)
		try:
			weights[tensor] = reader.get_tensor(tensor)
		except:
			print('Not found: ' + tensor)
	with open(out, 'wb') as f:
		pickle.dump(weights, f)


if __name__ == '__main__':
	if not all(args in sys.argv for args in ['--ckpt', '--out']):
		print('Usage: export_weights.py --ckpt [Checkpoint file] --out [Output file]')
		exit(1)
	ckpt = sys.argv[sys.argv.index('--ckpt') + 1]
	out = sys.argv[sys.argv.index('--out') + 1]
	export_weights(ckpt, out)