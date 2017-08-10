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

from nn import model
from nn import nn_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('architecture', '5layer',
                           """Network acrhitecture.""")
tf.app.flags.DEFINE_string('eval_dir', '../xray_data/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'val',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../xray_data/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples',
                            nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
eval_dir = FLAGS.eval_dir + FLAGS.architecture + '_eval'
checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.architecture + '_train'
if FLAGS.eval_data == 'train':
  num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
elif FLAGS.eval_data == 'val':
  num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_VAL
#num_examples = nn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
batch_size = FLAGS.batch_size


def eval_once(saver, keys_op, fc1_op, fc2_op, logits, oneds_op, labels):
  """Run Eval once.
  """
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

      num_iter = int(math.ceil(num_examples / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * batch_size    # note this is rounded up a little bit
      result_size = [total_sample_count, model.NUM_CLASSES]
      fc1_all = np.zeros([total_sample_count, fc1_op.get_shape()[1].value])
      fc2_all = np.zeros([total_sample_count, fc2_op.get_shape()[1].value])
      oneds_all = np.zeros([total_sample_count, oneds_op.get_shape()[1].value])
      gt_all = np.zeros(result_size)
      keys_all = np.atleast_1d([])
      step = 0
      while step < num_iter and not coord.should_stop():
        keys, fc1, fc2, oneds, gt = sess.run([keys_op, fc1_op, fc2_op, oneds_op, labels])
        fc1_all[step*batch_size:(step+1)*batch_size,:] = fc1
        fc2_all[step*batch_size:(step+1)*batch_size,:] = fc2
        oneds_all[step*batch_size:(step+1)*batch_size,:] = oneds
        gt_all[step*batch_size:(step+1)*batch_size,:] = gt
        keys_all = np.append(keys_all, keys)
        if step % 10 == 0:
          print('%d / %d' % (step + 1, num_iter))
        step += 1

      # Compute precision @ 1.
    #   if mask is not None:
    #     pred_all = pred_all[:, mask]
    #     gt_all = gt_all[:, mask]
      gt_all = gt_all.astype(np.float32)
      oned_binary_path = '../xray_data/synthetic_oned_binary/'
      np.save(oned_binary_path + 'fc1_%s.npy' % FLAGS.eval_data, fc1_all)
      np.save(oned_binary_path + 'fc2_%s.npy' % FLAGS.eval_data, fc2_all)
      np.save(oned_binary_path + 'gt_%s.npy' % FLAGS.eval_data, gt_all)
      np.save(oned_binary_path + 'keys_%s.npy' % FLAGS.eval_data, keys_all)
      np.save(oned_binary_path + 'oneds_%s.npy' % FLAGS.eval_data, oneds_all)
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


def output_features():
  """Write the labels+feature vectors for SVM."""
  mask = None  # used for inconsistent tags (real data)
  with tf.Graph().as_default() as g:
    # fetch *training* data for output values
    if FLAGS.eval_data == 'train':
      keys_op, images, oneds_op, labels = model.distorted_inputs(shuffle=False)
    elif FLAGS.eval_data == 'val':
      keys_op, images, oneds_op, labels = model.inputs(False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    fc1_op, fc2_op, logits = model.inference(images)
    # probs = tf.sigmoid(logits)
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
      eval_once(saver, keys_op, fc1_op, fc2_op, logits, oneds_op, labels)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #alexnet.maybe_download_and_extract()
  # if tf.gfile.Exists(eval_dir):
  #   tf.gfile.DeleteRecursively(eval_dir)
  # tf.gfile.MakeDirs(eval_dir)
  # if '--test' in argv[1:]:
  #   ids = [ ['45f18c5c_varied_sm', '{:08x}'.format(i)]
  #           for i in range(200)]
  #   evaluate('../xray_data/experiment', ids)
  # else:
  #   evaluate()
  output_features()


if __name__ == '__main__':
  tf.app.run()
