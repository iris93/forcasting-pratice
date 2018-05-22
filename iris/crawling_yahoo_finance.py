from __future__ import print_function
# from pprint import pprint
import tensorflow as tf
from yahoo_finance import Share
yahoo = Share('YHOO')

# print yahoo.get_open()
# print yahoo.get_price()
# print yahoo.get_trade_datetime()

print ('Wait...')
# pprint(yahoo.get_historical('2014-04-25', '2014-04-29'))
stock_train = yahoo.get_historical('2012-06-09', '2016-06-01')
stock_test  = yahoo.get_historical('2016-04-01', '2016-07-01')

X_train, y_train, X_test, y_test= [], [], [], []

# X_train, y_train
# Assign into normal python array
for stock in stock_train:
	X_train.append([ float(stock['Open']), float(stock['High']), float(stock['Low']), float(stock['Volume']) ])
	y_train.append([ float(stock['Close']) ])

# Assign into tensor variable
X_train = tf.Variable([ X_train ])
y_train = tf.Variable([ y_train ])


# X_test
# Assign into normal python array
for stock in stock_test:
	X_test.append([float(stock['Open']), float(stock['High']), float(stock['Low']), float(stock['Volume']) ])
	y_test.append([float(stock['Close'])])


# Assign into tensor variable
X_test = tf.Variable([ X_test ])



# Create session initialization
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Convert matrix into numpy.ndarray
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test  = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test  = tf.convert_to_tensor(y_test, dtype=tf.float32)

# X_train_shape = X_train.get_shape()
# train data
t = sess.run(X_train)
train_data = t[0]

# target data
t = sess.run(y_train)
target_data = t[0]

# test data
t = sess.run(X_test)
test_data = t[0]

#test target
t = sess.run(y_test)
test_target = t[0]

# Sample how to read the training data
for i, r in enumerate(train_data):
	print ('Open:{0} High:{1} Low:{2} Volume:{3}'.format(r[0], r[1], r[2], r[3]))
	print(i)

sess.close()
