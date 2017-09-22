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

"""Evaluation for CIFAR-10.
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
from shutil import rmtree
import time
import math
import csv

import numpy as np
import tensorflow as tf

import cifar10

save_directory = "Results/Validation/"
eval_dir = "eval_log"
checkpoint_dir = "train_log"
num_classes = cifar10.num_classes
im_size = cifar10.im_size

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 100000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

def eval_once(saver, summary_writer, evaluation_argument, summary_op, checkpoint_dir, mapping=False):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
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

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      output = []
      confusion_matrix = np.zeros((num_classes, num_classes), dtype=int) # initialises confusion matrix
      if mapping:   # if in mapping mode generate a map, if in default mode (variable set to False by default) then tally predictions instead.
          while step < num_iter and not coord.should_stop():
            step += 1
            hold = sess.run(evaluation_argument)
            for i in range (len(hold)):        
                output.append(hold[i])
#            hold = sess.run(output_classification).indices
#            for i in range (len(hold)):        
#                output.append(hold[i][0])
          accuracy = None  # set accuracy stat (gets added to the tensorboard summary later)
                
      else: # else compare training labels against predictions
          
          # run the graph to produce prediction and training label arrays
          while step < num_iter and not coord.should_stop():        
              predictions, truths = sess.run(evaluation_argument)
              for ref in range(FLAGS.batch_size):
                  confusion_matrix[predictions[ref],truths[ref]] += 1   # adds one to the appropriate entry in the confusion matrix
                  # notice that the vertical component (row num) is the machine prediction and the horizontal component (col number) is the label
              step += 1      

          # Compute stats (notice that we are still in the "else" section - accuracy figures not produced)          
          accuracy = sum (confusion_matrix[x][x] for x in range(num_classes)) / total_sample_count
          # calculate precision and recall stats for each category, compile as a list ready for CSV printing
          precision = ["Precision"]
          recall = ["Recall"]
          for x in range(num_classes):
              if confusion_matrix[x, x] != 0:
                  precision.append(confusion_matrix[x, x] / sum(confusion_matrix[x]))
                  recall.append(confusion_matrix[x, x] / sum(confusion_matrix[:,x]))
              else:
                  precision.append(0)
                  recall.append(0)
          
          # convert confusion matrix to percentage figures (this makes it easier to read)
#          confusion_matrix = np.around((100 * confusion_matrix) / total_sample_count, decimals=3)
          confusion_matrix = 100*np.true_divide(confusion_matrix, confusion_matrix.sum(axis=0, keepdims=True))
        
          print('%s: accuracy = %.3f' % (datetime.now(), accuracy))
          
          # save results in CSV format
          save_name = "confusion_matrix" + time.strftime('_%a_%I_%M_%p_%S')
          with open(save_directory + save_name + ".csv", 'w', newline='') as csvfile:
              datawriter = csv.writer(csvfile, delimiter=',',
                                      quotechar='|', quoting=csv.QUOTE_MINIMAL)
              # write headings
              datawriter.writerow(["-"] + ["Cat {}".format(i) for i in range(num_classes)])
              # add outcome matrix row by row
              confusion_matrix = confusion_matrix.tolist()
              for row in confusion_matrix:
                  row_num = confusion_matrix.index(row)
                  row.insert(0, "Cat {}".format(row_num))
                  datawriter.writerow(row)
              datawriter.writerow(precision)
              datawriter.writerow(recall)
              datawriter.writerow(["Accuracy", accuracy])
          print("Confusion matrix saved as {}.csv in {}".format(save_name, save_directory))
          
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Accuracy @ 1', simple_value=accuracy)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return(output)


def evaluate(mapping=False):
  """Eval CIFAR-10 for a number of steps."""
        
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, phases, labels = cifar10.inputs(mapping)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images, phases)

    # Calculate and evaluate predictions (used with pre-classified data).
    if mapping:
        # evaluation_argument = tf.nn.top_k(logits, 1)
        evaluation_argument = tf.argmax(logits, 1)
    else:
        #evaluation_argument = tf.nn.in_top_k(logits, labels, 1)
        evaluation_argument = tf.argmax(logits, 1), labels    

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    return(eval_once(saver, summary_writer, evaluation_argument, summary_op, checkpoint_dir, mapping=mapping))

def map_interface(pixel_count): # function created to allow prediction mapping and labelling of raw unlabelled data
    
    FLAGS.num_examples = pixel_count    # ensures that the program generates at least as many predictions as there are pixels in the raw data
                                        # it will overshoot and generate this many preidctions plus some to reach the next multiple of batch_size
                                        # this is fine but we will need to delete the excess predictions later   
    
    return(evaluate(mapping=True))    

def main(argv=None):  # pylint: disable=unused-argument

  if tf.gfile.Exists(eval_dir):
      rmtree(eval_dir, ignore_errors=True)
#    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()