# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Output the feature vectors with labels to use in SVM.

Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
import yaml

from nn_fbbenet import model
from nn_fbbenet import nn_input

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('architecture', '5layer',
#                            """Network acrhitecture.""")
# tf.app.flags.DEFINE_string('eval_dir', '../xray_data/fbb_output/',
#                            """Directory where to write event logs.""")
# tf.app.flags.DEFINE_string('eval_data', 'val',
#                            """Either 'test' or 'train_eval'.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', '../xray_data/fbb_output/',
#                            """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
# tf.app.flags.DEFINE_integer('num_examples',
#                             nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
#                             """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
eval_dir = '' # FLAGS.eval_dir + FLAGS.architecture + '_eval'
checkpoint_dir = '' # FLAGS.checkpoint_dir + FLAGS.architecture + '_train'
# if FLAGS.eval_data == 'test':
#   num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
# else:
#   num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL
batch_size = FLAGS.batch_size
config_path = './run_config.yml'
run_config = yaml.safe_load(open(config_path))


def eval_once(saver, keys_op, probs_op, fc_op, labels, architecture='joint'):
  """Run Eval once.
  """
  # if dataset == 'train':
  #   num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  # elif dataset == 'test':
  #   num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(run_config['num_examples'] / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * batch_size    # note this is rounded up a little bit
      result_size = [total_sample_count, model.NUM_CLASSES]
      fc_all = np.zeros([total_sample_count, fc_op.get_shape()[1].value])
      probs_all = np.zeros(result_size)
      gt_all = np.zeros(result_size)
      keys_all = []
      step = 0
      while step < num_iter and not coord.should_stop():
        keys, probs, fc, gt = sess.run([keys_op, probs_op, fc_op, labels])
        fc_all[step*batch_size:(step+1)*batch_size,:] = fc
        probs_all[step*batch_size:(step+1)*batch_size,:] = probs
        gt_all[step*batch_size:(step+1)*batch_size,:] = gt
        s_keys = [key.decode() for key in keys]
        keys_all.extend(s_keys)
        if step % 10 == 0:
          print('%d / %d' % (step + 1, num_iter))
        step += 1

      # Compute precision @ 1.
    #   if mask is not None:
    #     pred_all = pred_all[:, mask]
    #     gt_all = gt_all[:, mask]
    #   gt_all = gt_all.astype(np.float32)
      np.save('X_%s.npy' % architecture, fc_all)
      # np.save('Y_%s.npy' % dataset, gt_all)
      np.save('probs_%s.npy' % architecture, probs_all)
      with open('keys_%s' % architecture, 'w') as f_keys:
        for s in keys_all:
          f_keys.write(s + '\n')
    #   pred_all_th = (pred_all > 0.5).astype(np.int)
    #   true_count = (pred_all_th == gt_all).astype(np.int).sum()
    #   precision = true_count / gt_all.size
    #   ap = average_precision_score(gt_all, pred_all_th, average=None)
    #   map = np.mean(ap)
    #   print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    #   print('AP = ')
    #   print(ap)
    #   print('mAP = %f' % map)
      #np.save('pred.npy', pred_all)
      #np.save('gt.npy', gt_all)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def output_features(architecture='joint'):
  """Write the labels+feature vectors for SVM."""
  mask = None  # used for inconsistent tags (real data)
  with tf.Graph().as_default() as g:
    # fetch *training* data for output values
    # if dataset == 'train':
    #   keys, coefs, images, labels = model.distorted_inputs(shuffle=False, num_threads=1, nodistort=True)
    # elif dataset == 'test':
    #   keys, coefs, images, labels = model.inputs(eval_data=False, num_threads=1)
    # else:
    #   raise Exception
    keys, coefs, images, labels = model.inputs(num_threads=1)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _, _, fc = model.inference(coefs, images, architecture=architecture)
    probs = tf.sigmoid(logits)
    #
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    #
    # # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()
    #
    # summary_writer = tf.train.SummaryWriter(eval_dir, g)

    while True:
    #   pred_all, gt_all = eval_once(saver, summary_writer, probs, labels, summary_op, mask)
      eval_once(saver, keys, probs, fc, labels, architecture=architecture)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  try:
    architecture = argv[argv.index('-a') + 1]
  except Exception:
    architecture = 'joint'
  output = '-o' in argv[1:]
  print('Architecture: ' + architecture)
  global eval_dir, checkpoint_dir
  eval_dir = run_config['train_dir'] + architecture + '_eval'
  checkpoint_dir = run_config['train_dir'] + architecture + '_train'

  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)

  output_features(architecture=architecture)


if __name__ == '__main__':
  tf.app.run()
