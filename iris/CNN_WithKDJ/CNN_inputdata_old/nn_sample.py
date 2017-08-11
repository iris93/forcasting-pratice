# coding: utf-8
'''
Classic Neural Network Backpropagation
Consist of:
- 5 Input (Open, High, Low, Close, Volume)      OR
  8 Input (Open, High, Low, Close, Volume, K, D, J)
- 2 Hidden layers
- 10 Neurons in each Hidden layers
- 1 Output, selected option that is: Open, High, Low or Close
'''
from itertools import izip                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # Fauzi Fanny
import numpy as np
import tensorflow as tf
tf.InteractiveSession()

from preparing_input import InitPreprocessing
init_instance = InitPreprocessing()

# https://discuss.tradewave.net/t/code-for-kdj-indicator/508/2
# https://www.tradingview.com/script/h51ZIRlX-KDJ-Indicator-iamaltcoin/

# Configuration
output='close' # stock prices option: open, high, low, close
kdj='yes' # or kdj='no'
kdj_style='google' # or kdj_style='', kdj_short_period=3, kdj_long_period=5
kdj_short_period=3 # D (Divergen)
kdj_long_period=5  # K

# Get data from Source (Yahoo)
# X_train, y_train, X_test, y_test = init_instance.get_stock(output=output, kdj=kdj, kdj_style=kdj_style, kdj_short_period=kdj_short_period, kdj_long_period=kdj_long_period)
X_train, y_train, X_test, y_test = init_instance.get_stock_transform(output=output, kdj=kdj, kdj_style=kdj_style, kdj_short_period=kdj_short_period, kdj_long_period=kdj_long_period)

# Max X_train
X_train_max_open, X_train_max_high, X_train_max_low, X_train_max_close, X_train_max_volume = None, None, None, None, None
X_train_max_K, X_train_max_D, X_train_max_J = None, None, None
# Max y_train
y_train_max_open, y_train_max_high, y_train_max_low, y_train_max_close = None, None, None, None
# Max X_test
X_test_max_open, X_test_max_high, X_test_max_low, X_test_max_close, X_test_max_volume = None, None, None, None, None
X_test_max_K, X_test_max_D, X_test_max_J = None, None, None
# Max y_test
y_test_max_open, y_test_max_high, y_test_max_low, y_test_max_close = None, None, None, None

if kdj == 'yes':
	X_train_max_open, X_train_max_high, X_train_max_low, X_train_max_close, X_train_max_volume, X_train_max_K, X_train_max_D, X_train_max_J = init_instance.get_X_train_max(kdj=kdj)
	y_train_max_open, y_train_max_high, y_train_max_low, y_train_max_close = init_instance.get_y_train_max()
	X_test_max_open, X_test_max_high, X_test_max_low, X_test_max_close, X_test_max_volume, X_test_max_K, X_test_max_D, X_test_max_J = init_instance.get_X_test_max(kdj=kdj)
	y_test_max_open, y_test_max_high, y_test_max_low, y_test_max_close = init_instance.get_y_test_max()
else:
	X_train_max_open, X_train_max_high, X_train_max_low, X_train_max_close, X_train_max_volume = init_instance.get_X_train_max()
	y_train_max_open, y_train_max_high, y_train_max_low, y_train_max_close = init_instance.get_y_train_max()
	X_test_max_open, X_test_max_high, X_test_max_low, X_test_max_close, X_test_max_volume = init_instance.get_X_test_max()
	y_test_max_open, y_test_max_high, y_test_max_low, y_test_max_close = init_instance.get_y_test_max()


# Weight Initialization
def init_weight(shape, name):
	init = tf.random_uniform(shape, -1.0, 1.0)
	# init = tf.truncated_normal(shape, stddev=1.0/np.sqrt(float(shape[0])))
	return tf.Variable(init, name=name)

# Bias Initialization
def init_bias(shape, name):
	init = tf.zeros(shape)
	return tf.Variable(init, name=name)

# Model definition
def model(X, W, B):
	m = tf.matmul(X, W) + B
	# Relu for instead sigmoid, Sigmoid only for final layer
	L = tf.nn.relu(m)
	return L


# ---- INPUT ----
X = None
if kdj == 'no':
	# Tensor [None x 5]
	# X = tf.placeholder('float32', [42,5], name='input')
	X = tf.placeholder('float32', [None,5], name='input')
else:
	# Tensor [None x 8]
	# X = tf.placeholder('float32', [42,8], name='input')
	X = tf.placeholder('float32', [None,8], name='input')
# ---------------

# --- OUTPUT ----
# IMPORTANT: OUTPUT ROWS must be SAME as INPUT ROWS  <-----

# IF INPUT 42x5, THEN OUTPUT 42x1 (IN OUR CASE), Because W3 column = 1
# Y = tf.placeholder('float32', [42,1], name='output')
Y = tf.placeholder('float32', [None,1], name='output')
# ---------------

# 1st Weight connected to INPUT
W1 = None
if kdj == 'no':
	W1 = init_weight([5, 10], 'W1') # W1, 5x10
else:
	W1 = init_weight([8, 10], 'W1') # W1, 8x10
B1 = init_bias([10], 'B1') # B1 col should be same to W1 col

W2 = init_weight([10, 10], 'W2') # W2, 10x10
B2 = init_bias([10], 'B2') # B2 col should be same to W2 col

# Last Weight connected to OUTPUT
W3 = init_weight([10, 1], 'W3') # W3, 10x1
B3 = init_bias([1], 'B3') # B3 col should be same to W3 col
# -------------------------------------

# Compose a model ------
L2 = model(X,  W1, B1)
L3 = model(L2, W2, B2)

# -------------------------------------
# Hypothesis/ target/ Y_model/ logits
# 1st, tf.sigmoid
hypothesis = tf.sigmoid(tf.matmul(L3, W3) + B3)
# 2nd, tf.nn.sigmoid
# hypothesis = tf.nn.sigmoid(tf.matmul(L3, W3) + B3) # more decreasing than tf.sigmoid
# 3rd, keep on Relu
# hypothesis = model(L3, W3, B3)
# -------------------------------------

# loss function ---------------------
with tf.name_scope('loss') as scope:
	# 1st, cross_entropy, on backpropagation
	loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
	# 2nd, MSE
	# loss = tf.reduce_mean(tf.pow(tf.sub(hypothesis, Y), 2.0))
	# 3rd, softmax_cross_entropy
	# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
# -----------------------------------

# -----------------------------------
# Only todo batch training with static input
with tf.name_scope('train') as scope:
	lr = tf.Variable(1.0) # Learning rate 
	optimizer = tf.train.GradientDescentOptimizer(lr)
	batch_train = optimizer.minimize(loss)
# -----------------------------------

# -----------------------------------
# Launch the graph
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	# saver = tf.train.Saver()

	for epoch in range(20000):
		# IMPORTANT: Training must be performed before other processes

		# sess.run(batch_train, febatch_ed_dict={ X: X_train, Y: y_train })
		_, loss_value = sess.run([batch_train, loss], feed_dict={ X: X_train, Y: y_train }) 
		# print sess.run(loss, feed_dict={ X: X_train, Y: y_train })

		# print sess.run(W1) # The value of W1 is change
		# print sess.run(W2) # The value of W2 is change

		if epoch % 100 == 0:
			print ( 'Epoch: {0} Loss: {1}'.format(epoch, loss_value) )

	# correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
	# # print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction], feed_dict={ X: X_train, Y: y_train })
	# # print 'hypothesis: %s' % sess.run(hypothesis, feed_dict={ X: X_train, Y: y_train })
	# # print 'hypothesis+0.5: %s' % sess.run(tf.floor(hypothesis+0.5), feed_dict={ X: X_train, Y: y_train })
	# print 'correct_prediction: %s' % sess.run(correct_prediction, feed_dict={ X: X_train, Y: y_train })
	# print 'accuracy: %s ' % sess.run(accuracy, feed_dict={ X: X_train, Y: y_train })


	# Prediction ------------------------------------------------------------
	# predict = sess.run(hypothesis, feed_dict={ X: X_test })
	predict = sess.run(hypothesis, feed_dict={ X: X_train })
	# predict = sess.run(hypothesis, feed_dict={ X: X_train[0:10] })

	# yt = y_train
	# pr = prediction
	y, p, yt, pr = None, None, [], []

	for i, j in izip(y_train, predict):

		if output == 'open':
			y = i[0] * y_train_max_open
			p = j[0] * y_train_max_open

		elif output == 'high':
			y = i[0] * y_train_max_high
			p = j[0] * y_train_max_high

		elif output == 'low':
			y = i[0] * y_train_max_low
			p = j[0] * y_train_max_low

		else:
			y = i[0] * y_train_max_close
			p = j[0] * y_train_max_close

		yt.append(y)
		pr.append(p)

	print ( '-'*40 )
	print ( 'y_train: {}'.format(yt) )
	print ( '-'*40 )
	print ( 'predict: {}'.format(pr) )
