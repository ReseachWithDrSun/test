''' Command interface for running the xray data processor.
'''

import sys
import os
import math
import pickle
import yaml
import numpy as np
from six.moves import xrange
import tagio.xray_data_processor as Processor
from tagio.tag import tagtype

# command:
#   python run_processor.py --action-id [0-3]

tag_type_map = {'real': tagtype.Real, 'synthetic': tagtype.Synthetic}


if __name__ == '__main__':
	index = -1

	config = {}
	if os.path.isfile('run_processor.yml'):
		config = yaml.safe_load(open('run_processor.yml'))
		config['tag_type'] = tag_type_map[config['_tag_type'].lower()]
	else:
		print('Loading default')
		# load default
		config['tag_type'] = tagtype.Synthetic
		config['tag_dir'] = '../xray_data/experiment'
		config['image_dir'] = '../xray_data/experiment'
		config['prefix'] = 'synthetic_data/'
		config['tag_path'] = '../xray_data/experiment_binary/stags.bin'
		config['record_path_pl'] = '../xray_data/experiment_binary/stagged-batch-{}.bin'
		config['image_size'] = 256
		config['batch_size'] = 5000

	if '--action-id' in sys.argv:
		index = int(sys.argv[sys.argv.index('--action-id') + 1])

	if index == 0:
		# load all tags (some may not have coresponding imgs)
		tags = Processor.process_xray_tags(config['tag_dir'],
			image_dir=config['image_dir'],
			prefix=config['prefix'],
			#prefix='/home/kyager/BNL/MachineLearning/synthetic/data/',
			tag_type=config['tag_type'])
		np.random.shuffle(tags)
		print('%d tags loaded.' % len(tags))
		# if tag_type == tagtype.Synthetic:
		# 	for key, value in Processor.SimulatedFeatures.items():
		# 		print(key + ': ' + str(value))
		with open(config['tag_path'], 'wb') as f:
			pickle.dump(tags, f)
	elif index == 1:
		# preprocess images and create a new save for valid tags
		with open(config['tag_path'], 'rb') as f:
			tags = pickle.load(f)
		# print(str(len(tags)) + ' tags loaded.')
		if config['tag_type'] == tagtype.Real:
			Processor.process_xray_images(
				tags, config['image_dir'], config['record_path'],
				size=(config['image_size'], config['image_size']),
				tag_type=tagtype.Real)
		else:
			# hardcoded for parsing synthetic data:
			# no resize, no precision conversion, no valid tag check, batch output filename
			num_batches = math.ceil(len(tags) / config['batch_size'])
			for i in xrange(num_batches):
				batch_end = min((i+1)*config['batch_size'], len(tags))
				Processor.process_xray_images(
					tags[i*config['batch_size']:batch_end],
					config['image_dir'],
					config['record_path_pl'].format(i),
					tag_type=config['tag_type'],
					offset=1056)
				print('Batch %d completed.' % i)
	elif index == 2:
		# add in the labels
		# 288 = label_len + oned_len
		with open(config['tag_path'], 'rb') as f:
			tags = pickle.load(f)
		if config['tag_type'] == tagtype.Real:
			labels = Processor.generate_xray_labels(tags, Processor.MainImageFeatureSelector, label_len=32)
			tag_stats = labels.sum(axis=0)
			print(tag_stats)
			Processor.update_labels(config['record_path'], labels,
				1056 + config['image_size'] * config['image_size'])
		else:
			labels = Processor.generate_xray_labels(tags, Processor.SimulatedFeatureSelector, label_len=32)
			tag_stats = labels.sum(axis=0)
			print(tag_stats)
			num_batches = math.ceil(len(tags) / config['batch_size'])
			for i in xrange(num_batches):
				batch_end = min((i+1)*config['batch_size'], len(tags))
				Processor.update_labels(config['record_path_pl'].format(i),
					labels[i*config['batch_size']:batch_end],
					1056 + config['image_size'] * config['image_size'])
				print ('{} / {} batches updated.'.format(i + 1, num_batches))
	elif index == 3:
		# update 1d curves
		with open(config['tag_path'], 'rb') as f:
			tags = pickle.load(f)
		if config['tag_type'] == tagtype.Real:
			pass	# TODO
		else:
			num_batches = math.ceil(len(tags) / config['batch_size'])
			for i in xrange(num_batches):
				batch_end = min((i+1)*config['batch_size'], len(tags))
				Processor.update_oned(tags[i*config['batch_size']:batch_end],
					config['record_path_pl'].format(i),
					1056 + config['image_size'] * config['image_size'],
					32)
				print ('{} / {} batches updated.'.format(i + 1, num_batches))
	elif index == 8:
		# label stats
		with open(config['tag_path'], 'rb') as f:
			tags = pickle.load(f)
			if config['tag_type'] == tagtype.Real:
				pass	# not used
			else:
				labels = Processor.generate_xray_labels(tags, Processor.SimulatedFeatureSelector, label_len=32)
				tag_stats = labels.sum(axis=0)
				print(tag_stats)
	elif index == 9:
		# list tags
		with open(config['tag_path'], 'rb') as f:
			tags = pickle.load(f)
			if config['tag_type'] == tagtype.Real:
				pass
			else:
				all_tagnames = set()
				for tag in tags:
					all_tagnames |= set(tag.SimulatedFeatures)
				for tagname in all_tagnames:
					print(tagname)
	else:
		print('Command: python run_processor.py --action-id [0-2]')
