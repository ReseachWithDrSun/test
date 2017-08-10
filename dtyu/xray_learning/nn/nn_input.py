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
NUM_CLASSES = 15 # this is not mutually exclusive! it's actually real_label_bytes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 95000
NUM_EXAMPLES_PER_EPOCH_FOR_VAL = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2385

import math
UINT32_MAX = 4294967295
c_log_image = UINT32_MAX / math.log(1 + UINT32_MAX)


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
  oned_length = 256 # 1D curve, int32
  result.height = 256
  result.width = 256
  result.depth = 1
  image_bytes = result.height * result.width * result.depth

  record_bytes = label_bytes + oned_length * 4 + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  record_tensor = tf.decode_raw(value, tf.uint8)
  oned_tensor = tf.decode_raw(value, tf.float32)
  result.label = tf.cast(
    tf.slice(record_tensor, [0], [NUM_CLASSES]), tf.int32
  )
  result.oned = tf.slice(oned_tensor, [int(label_bytes / 4)], [oned_length])
  image = tf.reshape(
    tf.slice(record_tensor, [label_bytes + oned_length * 4], [image_bytes]),
    [result.depth, result.height, result.width]
  )
  result.image = tf.transpose(image, [1, 2, 0])

  # image_bytes = image_length * 2
  # # Every record consists of a label followed by the image, with a
  # # fixed number of bytes for each.
  # record_bytes = label_bytes + image_bytes
  #
  # # Read a record, getting filenames from the filename_queue.  No
  # # header or footer in the CIFAR-10 format, so we leave header_bytes
  # # and footer_bytes at their default of 0.
  # reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  # result.key, value = reader.read(filename_queue)
  #
  # # perform 2 decoding to fetch data of different types
  # record_bytes = tf.decode_raw(value, tf.uint8)
  # image_tensor = tf.decode_raw(value, tf.int16)
  # result.label = tf.cast(
  #   tf.slice(record_bytes, [0], [NUM_CLASSES]), tf.int32
  # )
  # label_length = int(label_bytes / 2)  # note: label is not correctly decoded with int16 conv
  # image = tf.reshape(tf.slice(image_tensor, [label_length], [image_length]),
  #   [result.depth, result.height, result.width])
  # result.image = tf.transpose(image, [1, 2, 0])

  return result


def _generate_image_and_label_batch(key, image, oned, label, min_queue_examples,
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
    keys, images, oneds, label_batch = tf.train.shuffle_batch(
        [key, image, oned, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    keys, images, oneds, label_batch = tf.train.batch(
        [key, image, oned, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images, max_images=20)

  return keys, images, oneds, label_batch    #tf.reshape(label_batch, [batch_size, NUM_CLASSES])


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
  filenames = [os.path.join(data_dir, 'stagged-batch-%d.bin' % i)
               for i in xrange(19)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input.image, tf.float32)
  read_oned = read_input.oned

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  #distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
  #  width, height)
  distorted_image = tf.image.resize_images(reshaped_image, height, width)
  # # Image processing for training the network. Note the many random
  # # distortions applied to the image.
  #
  # # Randomly crop a [height, width] section of the image.
  # distorted_image = tf.random_crop(reshaped_image, [height, width, 1])
  #
  # # Randomly flip the image horizontally.
  # distorted_image = tf.image.random_flip_left_right(distorted_image)
  #
  # # Because these operations are not commutative, consider randomizing
  # # the order their operation.
  # distorted_image = tf.image.random_brightness(distorted_image,
  #                                              max_delta=63)
  # distorted_image = tf.image.random_contrast(distorted_image,
  #                                            lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(read_input.key, float_image, read_oned, read_input.label,
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
    filenames = [os.path.join(data_dir, 'stagged-batch-19.bin')]
                 #'data_batch_%d.bin' % i)
                 #for i in xrange(1, 6)]
    #num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_VAL
  else:
    # test
    filenames = [os.path.join(data_dir, 'tagged.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue)
  #reshaped_image = tf.cast(read_input.image, tf.float32)
  reshaped_image = read_input.image
  read_oned = read_input.oned

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  ## Crop the central [height, width] of the image.
  #resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
  #                                                       width, height)
  resized_image = tf.image.resize_images(reshaped_image, height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(read_input.key, float_image, read_oned, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False, num_threads=num_threads)


def get_synthetic_record(data_dir, exp_no, image_no):
  """grab one image and its label with specific experiment and image number

  Args:
    data_dir: string, path of synthetic dataset
    exp_no: string, experiment number
    image_no: string, image number

  Returns:
    image: single preprocessed image
    label: single label
  """
  from tagio.tag import tagdata, tagtype
  import tagio.xray_data_processor as Processor
  import scipy.io
  import os
  import numpy as np

  image_path = exp_no + '/' + image_no + '.mat'
  image_path = os.path.join(data_dir, image_path)
  tag_path = exp_no + '/analysis/results/' + image_no + '.xml'
  tag_path = os.path.join(data_dir, tag_path)

  image = scipy.io.loadmat(image_path)['detector_image']
  image = Processor._filter_image(image)
  image = np.expand_dims(image, axis=2)
  # atm image processing is done at real time instead of pushing in graphs
  # when data is specified by list of filenames
  image = image[14:241, 14:241]
  image = (image - np.mean(image)) / np.std(image)

  tag = tagdata(tag_path, tag_type=tagtype.Synthetic)
  label = Processor.SimulatedFeatureSelector(tag)

  return image, label


def synthetic_input_by_ids(data_dir, ids):
  """generate an input tensor from a list of specified images with labels

  Args:
    data_dir: string, path of synthetic dataset
    ids: array of [exp_no, image_no] records

  Returns:
    image_tensor
    label_tensor
  """
  images = []
  labels = []
  #width = IMAGE_SIZE
  #height = IMAGE_SIZE
  for record_id in ids:
    image, label = get_synthetic_record(data_dir, record_id[0], record_id[1])
    images.append(image)
    labels.append(label)
  images = np.stack(images, axis=0)
  labels = np.concatenate(labels, axis=0)

  image_tensor = tf.cast(tf.convert_to_tensor(images), tf.float32)
  label_tensor = tf.cast(tf.slice(tf.convert_to_tensor(labels),
                            [0,0], [-1, NUM_CLASSES]), tf.int32)

  return image_tensor, label_tensor
