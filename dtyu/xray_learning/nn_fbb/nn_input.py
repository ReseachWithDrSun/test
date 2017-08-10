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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 227

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 7 # this is not mutually exclusive! it's actually real_label_bytes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3000
NUM_EXAMPLES_PER_EPOCH_FOR_VAL = 500
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2385

# import math
# UINT32_MAX = 4294967295
# c_log_image = UINT32_MAX / math.log(1 + UINT32_MAX)


def read_data(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class DataRecord(object):
    pass
  result = DataRecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 32  # 2 for CIFAR-100
  coef_length = 1230   # sparse coefficients, float32
  record_bytes = label_bytes + coef_length * 4

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  label_tensor = tf.decode_raw(value, tf.uint8)
  result.label = tf.cast(
    tf.slice(label_tensor, [0], [NUM_CLASSES]), tf.int32
  )

  coef_tensor = tf.decode_raw(value, tf.float32)
  result.coef = tf.slice(coef_tensor, [int(label_bytes / 4)], [630])

  return result


def _generate_image_and_label_batch(coef, label, min_queue_examples,
                                    batch_size, shuffle, num_threads=16):
  """
  1D curve added.

  Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  #num_preprocess_threads = 16
  if shuffle:
    coefs, labels = tf.train.shuffle_batch(
        [coef, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    coefs, labels = tf.train.batch(
        [coef, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # # Display the training images in the visualizer.
  # tf.image_summary('images', images, max_images=20)

  return coefs, labels


def distorted_inputs(data_dir, batch_size, shuffle=True):
  """
  1D curve added.

  Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'fbb_train_batch.bin')]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue)
  coef = read_input.coef
  # reshaped_coef = tf.reshape(coef, [coef.get_shape()[0].value, 1, 1])
  # whitened_coef = tf.image.per_image_whitening(reshaped_coef)
  # restored_coef = tf.reshape(whitened_coef, [coef.get_shape()[0].value])
  normalized_coef = coef / tf.reduce_max(tf.abs(coef))

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(normalized_coef, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=shuffle)


def inputs(eval_data, data_dir, batch_size, num_threads=16):
  """
  1D curve added.

  Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    # validation
    filenames = [os.path.join(data_dir, 'fbb_val_batch.bin')]
                 #'data_batch_%d.bin' % i)
                 #for i in xrange(1, 6)]
    #num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_VAL
  else:
    # test
    raise Exception()
    filenames = [os.path.join(data_dir, 'tagged.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue)
  coef = read_input.coef
  # reshaped_coef = tf.reshape(coef, [coef.get_shape()[0].value, 1, 1])
  # whitened_coef = tf.image.per_image_whitening(reshaped_coef)
  # restored_coef = tf.reshape(whitened_coef, [coef.get_shape()[0].value])
  normalized_coef = coef / tf.reduce_max(tf.abs(coef))

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(normalized_coef, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False, num_threads=num_threads)
