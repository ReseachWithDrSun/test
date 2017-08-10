# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

""" DataSet loader modified from tf mnist example
cf. tensorflow.contrib.learn.python.learn.datasets.mnist
"""

"""Functions for downloading and reading MNIST data."""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from enum import Enum
import sys

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
#from tensorflow.python.platform import gfile

flag_test = False

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class LabelType(Enum):
  AllFeatures = 0
  MainFeatures = 1


class DataSet(object):

  def __init__(self,
               images,
               labels,
               flatten=True,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      #assert images.shape[3] == 1
      if flatten:
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._raw_labels = None
    self._name = ''
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(images, tags, which,
                   label_type=LabelType.MainFeatures,
                   flatten=True,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32):
#def read_data_sets(train_dir,
#                   fake_data=False,
#                   one_hot=False,
#                   dtype=dtypes.float32):
  if fake_data:

    def fake():
      return DataSet([], [], flatten=flatten, fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  VALIDATION_SIZE = 300

  num_images = len(images)
  # just pick one arbitrarily to determine size info
  one_image = next(iter(images.values()))
  size_image = one_image.size
  train_images = numpy.empty([num_images, size_image[0], size_image[1]])
  test_images = []

  train_labels = numpy.empty([num_images])
  test_labels = []
  train_raw_labels = []
  name_images = []
  validation_raw_labels = []

  next_idx = 0

  positive_count = 0
  for name, img in images.items():
    train_images[next_idx,:,:] = img
    name_images.append(name)

    if label_type == LabelType.MainFeatures:
      train_labels[next_idx] = int(which in tags[name].MainImageFeatures)
      train_raw_labels.append(tags[name].MainImageFeatures)
      if which in tags[name].MainImageFeatures:
        positive_count = positive_count + 1
    else:
      train_labels[next_idx] = int(which in tags[name].ImageFeatures)
      train_raw_labels.append(tags[name].ImageFeatures)
      if which in tags[name].ImageFeatures:
        positive_count = positive_count + 1
    next_idx = next_idx + 1
  if flag_test:
    print('Load completed with ' + which + ' positive count = ' + str(positive_count))
  if one_hot:
    train_labels = dense_to_one_hot(train_labels, 2)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  validation_raw_labels = train_raw_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  train_raw_labels = train_raw_labels[VALIDATION_SIZE:]

  train = DataSet(train_images, train_labels, flatten=flatten, dtype=dtype)
  validation = DataSet(validation_images, validation_labels, flatten=flatten, dtype=dtype)
  #test = DataSet(test_images, test_labels, fake_data=True, one_hot=one_hot, dtype=dtype)
  test = fake()
  train._raw_labels = train_raw_labels
  train._names = name_images
  validation._raw_labels = validation_raw_labels

  return base.Datasets(train=train, validation=validation, test=test)


if __name__ == '__main__':
  if '--test-tag' in sys.argv:
    flag_test = True
    from tag_processing import tagdata
    i = sys.argv.index('--test-tag')
    tags = numpy.atleast_1d(numpy.load('./pre/tags.npy'))[0]
    data = numpy.atleast_1d(numpy.load('./pre/data.npy'))[0]
    dataset = read_data_sets(data, tags, sys.argv[i + 1])