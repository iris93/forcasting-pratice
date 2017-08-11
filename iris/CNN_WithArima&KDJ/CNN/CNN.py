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
adjust by iris ,to 4 layers CNN;
you can make pool=pool2,pool = pool3 or pool=pool4 to decide how many layers that you want to use ;
you can change the kernel of every convolutional layer in the model function;
you can change the chanel of every layer as long as you make the input chanel changes at the same time .
when you run ,you will find the demesion dosen't match.go to change the fc1_shape0 to adjust the first demension.

Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# # import CNN_config.ini 2 this CNN model
# from CNN_config_2model import *

model_path = "/home/iris/workspace/stock_forecasting/iris/model_save/CNN/model_7.ckpt"  
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

fig = []
def drawpic(realTargets,predictions,fig_index,show_index):
    fig.append(plt.figure(fig_index))
    ax = fig[fig_index].add_subplot(111)
    ax.grid(True)
    p1 = ax.plot(realTargets)
    p2 = ax.plot(predictions)
    ax.legend((p1[0], p2[0]), ('real', 'pred'), loc='best', fancybox=True, framealpha=0.5)
    if fig_index==show_index: plt.show(block=True)

def main(argv=None):  # pylint: disable=unused-argument
  # # Get the data.
  # '''you can dajust the parameters in CNN_config.ini or here below
  # '''
  IMAGE_SIZE = 10
  NUM_CHANNELS = 1
  PIXEL_DEPTH = 2
  NUM_LABELS = 2
  VALIDATION_start=200
  VALIDATION_end = 250
  SEED = 66478
  BATCH_SIZE = 50
  NUM_EPOCHS = 200
  EVAL_BATCH_SIZE =50
  EVAL_FREQUENCY = 100 
  iteration = 50
  fc1_shape0 = 128
  fc1_shape1 =128
  learning_rate_ini = 0.0005
  decay_rate = 0.95
  test_length=50
  run_train=False  #True if running the train_data to train the  model.
  run_train=True
  
  print ("          IMAGE_SIZE=",IMAGE_SIZE,";        NUM_CHANNELS=",NUM_CHANNELS,";           PIXEL_DEPTH=",PIXEL_DEPTH,";               NUM_LABELS=",NUM_LABELS)
  print ("VALIDATION_start=",VALIDATION_start,";     SEED=",SEED,";                         NUM_EPOCHS=",NUM_EPOCHS,";        BATCH_SIZE=",BATCH_SIZE)
  print ("VALIDATION_start=",VALIDATION_end,  ";    EVAL_BATCH_SIZE=",EVAL_BATCH_SIZE,";     EVAL_FREQUENCY=",EVAL_FREQUENCY,";     iteration=",iteration)
  print ("            fc1_shape0=",fc1_shape0,";   fc1_shape1=",fc1_shape1,";                  learning_rate_ini=",learning_rate_ini,";     decay_rate=",decay_rate)
  # '''

  data_dir = '../CNN_inputdata/'
  # run cnn with stock data ====================start
  train_data = numpy.load(data_dir+'trX.npy')
  train_labels = numpy.load(data_dir+'trY.npy')
  train_length = len(train_labels)
  print ("total train_length:",train_length)
  print ("test_length:",test_length)
  train_data = numpy.reshape(train_data,[train_length,IMAGE_SIZE,IMAGE_SIZE,1])
  train_labels = numpy.reshape(train_labels,[train_length,])

  # run cnn with stock data =======================end
  
  # Generate a validation set.
  validation_data = train_data[VALIDATION_start:VALIDATION_end, ...]
  validation_labels = train_labels[VALIDATION_start:VALIDATION_end]
  test_data = train_data[(train_length-test_length):train_length,...]
  test_labels = train_labels[train_length-test_length:train_length]
  train_data = train_data[VALIDATION_end:train_length-test_length,...]
  train_labels = train_labels[VALIDATION_end: train_length-test_length]

  print (len(validation_data))
  print (len(train_data))
  print (len(test_data))
  num_epochs = NUM_EPOCHS
  # '''

  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  #intialize weights
  conv1_weights = tf.Variable(
      tf.truncated_normal([2, 2, NUM_CHANNELS, 8],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([8], dtype=data_type()))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [2, 2, 8, 16], stddev=0.1,
      seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[16], dtype=data_type()))
  conv3_weights = tf.Variable(tf.truncated_normal(
      [2, 2, 16, 32], stddev=0.1,
      seed=SEED, dtype=data_type()))
  conv3_biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=data_type()))
  conv4_weights = tf.Variable(tf.truncated_normal(
      [2, 2, 32, 32], stddev=0.1,
      seed=SEED, dtype=data_type()))
  conv4_biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=data_type()))
  #fully connected weights
  fc1_weights = tf.Variable(  # fully connected, depth 512./'
      tf.truncated_normal([fc1_shape0, fc1_shape1],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc1_shape1], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([fc1_shape1, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv1 = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool1 = tf.nn.max_pool(relu1,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    conv2 = tf.nn.conv2d(pool1,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    
    conv3 = tf.nn.conv2d(pool2,
                        conv3_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    pool3 = tf.nn.max_pool(relu3,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    
    conv4 = tf.nn.conv2d(pool3,
                        conv4_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv3_biases))
    pool4 = tf.nn.max_pool(relu4,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')   
    
    pool = pool4
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      learning_rate_ini,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      decay_rate,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.5).minimize(loss)
                                                    

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    # print ("size",size)
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  saver = tf.train.Saver() 
  if run_train: 
    # Create a local session to run the training.
    print ('run trainning session')
    start_time = time.time()
    with tf.Session() as sess:
      # Run all the initializers to prepare the trainable parameters.
      tf.initialize_all_variables().run()
      print('Initialized!')
      # Loop through training steps.
      # print (train_size)
      # print (BATCH_SIZE)
      for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        i = 0
        for i in range(iteration):
          _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction],feed_dict=feed_dict)
        if step % EVAL_FREQUENCY == 0:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                 1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
          print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
          print('Validation error: %.1f%%' % error_rate(
              eval_in_batches(validation_data, sess), validation_labels))
          sys.stdout.flush()
      test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
      print('Test error: %.1f%%' % test_error)
      # Save model weights to disk  
      save_path = saver.save(sess, model_path)  
      print ("Model saved in file: %s" % save_path)
  else:
    print ('run loaded session')
    with tf.Session() as sess:
      # Initialize variables  
      tf.initialize_all_variables().run()
      load_path = saver.restore(sess, model_path)  
      print ("Model restored from file: %s" % model_path)
      # Finally print the result!
      test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
      print('Test error: %.1f%%' % test_error)


if __name__ == '__main__':
  tf.app.run()
