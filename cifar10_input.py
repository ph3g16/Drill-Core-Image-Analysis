# -*- coding: utf-8 -*-

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

import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.


# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue, im_size):
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

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = im_size
  result.width = im_size
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

# validate the filename inputs using the following line:
#  value = tf.Print(value, [result.key])

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, phased_image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
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
  num_preprocess_threads = 16
  if shuffle:
    images, phases, label_batch = tf.train.shuffle_batch(
        [image, phased_image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, phases, label_batch = tf.train.batch(
        [image, phased_image, label],
        batch_size=batch_size,
        num_threads=16,
        capacity=min_queue_examples + 3 * batch_size,
#        enqueue_many = False
        min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, phases, tf.reshape(label_batch, [batch_size])


def distorted_inputs(im_size, data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filelist = os.listdir(data_dir)
  filenames = []
  for f in filelist:
      if f.startswith("train_batch"):
          filenames.append(os.path.join(data_dir, f))
          
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue, im_size)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  
  # Randomly crop a [height, width] section of the image.
  height = im_size
  width = im_size
  reshaped_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally - take this out if your data has some kind of horizontal bias, e.g. due to noise
  reshaped_image = tf.image.random_flip_left_right(reshaped_image)

  # At this point we split the image into two streams.
  # "phased_image" is pre-processed to highlight structural features within the image
  # "float_image" is just a zero to one floating point representation of the raw RGB pixel values
  # phased image sensitizes the ntwork to colour gradiants whereas the float image sensitizes the network to actual colours

  # distort image prior to standaradization (for some reason this massively aids training of phased images)
  phased_image = tf.image.random_brightness(reshaped_image,
                                               max_delta=63)
  phased_image = tf.image.random_contrast(phased_image,
                                             lower=0.2, upper=1.8)
  # Subtract off the mean and divide by the variance of the pixels.
  phased_image = tf.image.per_image_standardization(reshaped_image)
  
  # Set float_image (can be thought of as a vanilla representation of the image)
  float_image = reshaped_image / 256

  # Set the shapes of tensors.
  phased_image.set_shape([height, width, 3])
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 1
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, phased_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(im_size, mapping, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    mapping: bool, indicating if one should use the raw or pre-classified eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filelist = os.listdir(data_dir)
  filenames = []
  
  if mapping:
#      from Raw_Image_Processor import file_name
      for f in filelist:
            if f.startswith("raw_batch"):
                filenames.append(os.path.join(data_dir, f))
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
      for f in filelist:
            if f.startswith("eval_batch"):
                filenames.append(os.path.join(data_dir, f))
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue, im_size)
  
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  height = im_size
  width = im_size
  
  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  phased_image = tf.image.per_image_standardization(reshaped_image)
  
  float_image = reshaped_image / 256
  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  if mapping:
      return _generate_image_batch(float_image, phased_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
  else:
      return _generate_image_and_label_batch(float_image, phased_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
      
      
def _generate_image_batch(image, phased_image, label, min_queue_examples,
                                    batch_size, shuffle):

    images, phases, label_batch = tf.train.batch(
        [image, phased_image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=1,
        enqueue_many = False)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, phases, 0