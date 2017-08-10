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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from nn_concat import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('architecture', '5layer_concat_dropout',
                           """Network acrhitecture.""")
tf.app.flags.DEFINE_string('train_dir', '../xray_data/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
train_dir = FLAGS.train_dir + FLAGS.architecture + '_train'
loader_dir = FLAGS.train_dir + '5layer_dropout' + '_train'


def train(from_checkpoint=False):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Get images and labels for CIFAR-10.
    _, images, oneds, labels = model.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    _, _, logits, weights_all = model.inference(images, oneds, dropout=FLAGS.architecture=='5layer_concat_dropout')

    # Calculate loss.
    loss = model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
    # Saver for loading fixed conv weights.
    loader = tf.train.Saver(weights_all)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    if from_checkpoint:
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step_cnt = global_step.eval(session=sess)
            print('Resuming from checkpoint: step %d' % global_step_cnt)
        else:
            print('Checkpoint not found')
            from_checkpoint = False
    if not from_checkpoint:
        print('Training (fine-tuning) from loader: vanilla 5 layer model')
        if tf.gfile.Exists(train_dir):
            tf.gfile.DeleteRecursively(train_dir)
        tf.gfile.MakeDirs(train_dir)
        ckpt = tf.train.get_checkpoint_state(loader_dir)
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(init)
            loader.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('Loader not found')
        global_step_cnt = 0

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    for step in xrange(global_step_cnt, FLAGS.max_steps):
      start_time = time.time()

      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  #alexnet.maybe_download_and_extract()
  train(from_checkpoint=not '--reset' in argv)


if __name__ == '__main__':
  tf.app.run()
