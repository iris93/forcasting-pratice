from __future__ import print_function
# from pprint import pprint
import tensorflow as tf
import numpy as np
from yahoo_finance import Share
yahoo = Share('YHOO')

# print yahoo.get_open()
# print yahoo.get_price()
# print yahoo.get_trade_datetime()

print ('Wait...')
# pprint(yahoo.get_historical('2014-04-25', '2014-04-29'))
stock_train = yahoo.get_historical('2016-01-01', '2016-06-01')
# print (stock_train)
stock_test  = yahoo.get_historical('2016-06-01', '2016-07-01')

X_train, y_train, X_test, y_test= [], [], [], []

# X_train, y_train
# Assign into normal python array
for stock in stock_train:
	X_train.append([ float(stock['Open']), float(stock['High']), float(stock['Low']), float(stock['Close']), float(stock['Volume']) ])
	y_train.append([ float(stock['Close']) ])

# X_test
# Assign into normal python array
for stock in stock_test:
	X_test.append([float(stock['Open']), float(stock['High']), float(stock['Low']), float(stock['Close']), float(stock['Volume']) ])
	y_test.append([float(stock['Close'])])
y_train= np.reshape(y_train,len(y_train))
y_test= np.reshape(y_test,len(y_test))
np.save('trdata',np.asarray(y_train).astype(np.float32))
np.save('tedata',np.asarray(y_test).astype(np.float32))

print (y_train)