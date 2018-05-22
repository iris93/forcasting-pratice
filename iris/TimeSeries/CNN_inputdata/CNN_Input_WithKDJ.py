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
from itertools import izip
 # Fauzi Fanny
import numpy as np
import tensorflow as tf
tf.InteractiveSession()
import urllib2
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
# print len(X_train)
print (X_train)
# print len(y_train)
print (y_train)
# print len(X_test)
# print (X_test)
# print len(y_test)
# print (y_test)
X_set = []
trX = []
trY = []
for i in range(0,len(X_train)-8):
	for j in range(0,8):
		X_set.append(X_train[i+j])
	trX.append(X_set)
	X_set = []
	open1, high1, low1, close1, volume1, K1, D1, J1 = X_train[i+7]
	open2, high2, low2, close2, volume2, K2, D2, J2 = X_train[i+8]
	if close2-close1>0:
		trY.append([1])
	else:
		trY.append([0])
print np.shape(trX),np.shape(trY)

np.save('trX',np.asarray(trX).astype(np.float32))
np.save('trY',np.asarray(trY).astype(np.float32))

# create testing set,teX,teY
teX = []
teY = []
X_teset = []
X_teset1 = []
for i in range(0,len(X_test)-8):
	for j in range(0,8):
		X_teset1.append(X_test[i+j])
	# print X_teset1
	# print "i:",i
	teX.append(X_teset1)
	X_teset = []
	X_teset1 =[]
	open1, high1, low1, close1, volume1, K1, D1, J1 = X_train[i+7]
	open2, high2, low2, close2, volume2, K2, D2, J2 = X_train[i+8]
	# open, high, low, close, volume, K,  D, J= X_test[i+8]
	if close2-close1>0:
		teY.append([1])
	else:
		teY.append([0])
# print "teX",teX[0:2]
print np.shape(teX),np.shape(teY)
np.save('teX',np.asarray(teX).astype(np.float32))
np.save('teY',np.asarray(teY).astype(np.float32))