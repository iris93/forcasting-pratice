# coding: utf-8
'''
1. Production : Crawling original data from yahoo stock finance
2. Development: Alternatively load offline excel file
3. Implement div by max and (shift if necessary) to convert
   real data from big floating-point into decimal between 0 to 1
''' 
from __future__ import absolute_import
from __future__ import division

import os
import sys
import ConfigParser
import numpy as np
import pandas as pd
from itertools import izip
from yahoo_finance import Share

from log import logger
from config_parser import basedir, init_preprocessing
config = ConfigParser.ConfigParser()
config_file = os.path.join(basedir, 'config.ini')

# Loading module config_parser to obtain configuration parameters
development, yahoo_excel = None, None
yahoo_historical_train_from, yahoo_historical_train_to = None, None
yahoo_historical_test_from, yahoo_historical_test_to = None, None
try:
	config.readfp(open(config_file))
	development, yahoo_excel, yahoo_historical_train_from, yahoo_historical_train_to, yahoo_historical_test_from, yahoo_historical_test_to = init_preprocessing(config)
except:
	logger.debug('Error init_preprocessing(). Exit!')
	sys.exit()


yahoo = Share('YHOO') if development == '0' else None

class InitPreprocessing:
	'''
	Class for initialization and preprocessing input for neuron input

   How to use:

   # import module
   from preparing_input import InitPreprocessing

   # create a new instance
   init_instance = InitPreprocessing()

   # -----------------------------------
   # get_stock(), get real stock data
   # kdj='no', for input without KDJ otherwise set kdj='yes'
   # kdj only support for production setting, see config.ini

   # kdj_style='google', automatically kdj_short_period=3 and kdj_long_period=5
   #    for example: output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5

   # kdj_style='', need to set kdj_short_period and kdj_long_period
   #    for example: output='close', kdj='yes', kdj_style='', kdj_short_period=3, kdj_long_period=6
   X_train, y_train, X_test, y_test = init_instance.get_stock(output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5)

   # -----------------------------------
   # get_stock_transform(), get transform stock data
   # kdj='no', for input without KDJ otherwise set kdj='yes'
   # kdj only support for production setting, see config.ini

   # kdj_style='google', automatically kdj_short_period=3 and kdj_long_period=5
   #    for example: output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5

   # kdj_style='', need to set kdj_short_period and kdj_long_period
   #    for example: output='close', kdj='yes', kdj_style='', kdj_short_period=3, kdj_long_period=6
   X_train, y_train, X_test, y_test = init_instance.get_stock_transform(output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5)

   # -----------------------------------
   # get_X_train_max(), get maximum value
   # get_X_test_max(), get maximum value
   kdj='yes'   # for production with kdj
   X_train_max_open, X_train_max_high, X_train_max_low, X_train_max_close, X_train_max_volume, X_train_max_K, X_train_max_D, X_train_max_J = init_instance.get_X_train_max(kdj=kdj)
   X_test_max_open, X_test_max_high, X_test_max_low, X_test_max_close, X_test_max_volume, X_test_max_K, X_test_max_D, X_test_max_J = init_instance.get_X_test_max(kdj=kdj)
   y_train_max_open, y_train_max_high, y_train_max_low, y_train_max_close = init_instance.get_y_train_max()
   y_test_max_open, y_test_max_high, y_test_max_low, y_test_max_close = init_instance.get_y_test_max()
	'''

	def __init__(self):
		# Parameters X_train for transformation
		self.X_train_max_open   = -100.0
		self.X_train_max_high   = -100.0
		self.X_train_max_low    = -100.0
		self.X_train_max_close  = -100.0
		self.X_train_max_volume = -10000000000.0

		self.X_train_max_K      = -10000000000.0
		self.X_train_max_D      = -10000000000.0
		self.X_train_max_J      = -10000000000.0

		# Parameters y_train for transformation
		self.y_train_max_open   = -100.0
		self.y_train_max_high   = -100.0
		self.y_train_max_low    = -100.0
		self.y_train_max_close  = -100.0
	
		# Parameters X_test for transformation
		self.X_test_max_open    = -100.0
		self.X_test_max_high    = -100.0
		self.X_test_max_low     = -100.0
		self.X_test_max_close   = -100.0
		self.X_test_max_volume  = -10000000000.0

		self.X_test_max_K       = -10000000000.0
		self.X_test_max_D       = -10000000000.0
		self.X_test_max_J       = -10000000000.0

		# Parameters y_test for transformation
		self.y_test_max_open    = -100.0
		self.y_test_max_high    = -100.0
		self.y_test_max_low     = -100.0
		self.y_test_max_close   = -100.0

	# get maximum X_train
	# IF kdj='no', THEN the return value only max 4 stock prices and 1 volume
	# otherwise added by max K, max D and max J
	def get_X_train_max(self, kdj='no'):
		if kdj == 'no':
			return self.X_train_max_open, self.X_train_max_high, self.X_train_max_low, self.X_train_max_close, self.X_train_max_volume
		return self.X_train_max_open, self.X_train_max_high, self.X_train_max_low, self.X_train_max_close, self.X_train_max_volume, self.X_train_max_K, self.X_train_max_D, self.X_train_max_J

	# get maximum y_train
	def get_y_train_max(self):
		return self.y_train_max_open, self.y_train_max_high, self.y_train_max_low, self.y_train_max_close

	# get maximum X_test
	# IF kdj='no', THEN the return value only max 4 stock prices and 1 volume
	# otherwise added by max K, max D and max J
	def get_X_test_max(self, kdj='no'):
		if kdj == 'no':
			return self.X_test_max_open, self.X_test_max_high, self.X_test_max_low, self.X_test_max_close, self.X_test_max_volume
		return self.X_test_max_open, self.X_test_max_high, self.X_test_max_low, self.X_test_max_close, self.X_test_max_volume, self.X_test_max_K, self.X_test_max_D, self.X_test_max_J

	# get maximum y_test
	def get_y_test_max(self):
		return self.y_test_max_open, self.y_test_max_high, self.y_test_max_low, self.y_test_max_close

	# RSV value for KDJ indicator
	# http://pastebin.com/CVrVn9F6
	# https://groups.google.com/forum/#!topic/google-finance/QFwZRV412TU 
	def rsv_indicator(self, close, low_prev, high_prev):
		return 100 * (close-np.min(low_prev))/(np.max(high_prev)-np.min(low_prev))

	# https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py
	def kdj_indicator(self, rsv, kdj_short_period, kdj_long_period):
		rsv = pd.DataFrame(rsv)
		# rsv = pd.DataFrame(rsv[kdj_long_period:])
		# rsv = pd.Series(rsv[kdj_long_period:])

		# get K by EMA, lag=kdj_long_period
		# K = pd.ewma(rsv, span=kdj_long_period) # deprecated
		K = rsv.ewm(ignore_na=False, span=kdj_long_period, min_periods=0, adjust=True).mean()
		# K = pd.ewma(np.log(rsv), span=kdj_long_period)

		# get D (Divergen of K) by MA
		# lag = kdj_short_period
		# D = pd.rolling_mean(K, kdj_short_period) # deprecated
		K.dropna(inplace=True)
		D = K.rolling(window=kdj_short_period, center=False).mean()
		# D = pd.rolling_mean(np.log(K), kdj_short_period) # lag kdj_short_period

		# get J
		# (see bottom code), https://discuss.tradewave.net/t/code-for-kdj-indicator/508/2
		# https://www.tradingview.com/script/h51ZIRlX-KDJ-Indicator-iamaltcoin/
		J = 3*K - 2*D
		return K, D, J

	# Evaluation process whether using development or production data
	# for development does not support KDJ, the input only consist of 4 stock prices and 1 volume
	# for production support with or without KDJ
	# output only single stock price: [open], [high], [low] or [close]
	# We determine the KDJ based on short period for (D) and long period for (K)
	def evaluate(self, output, kdj, kdj_style, kdj_short_period, kdj_long_period):
		# kdj='yes', then the input data added by 3 columns: K, D and J
		# kdj_style='google', then automatically kdj_short_period=3 and kdj_long_period=5
		# if kdj_style is not 'google', then kdj_short_period and kdj_long_period must be typed manually
		X_train, y_train, X_test, y_test = [], [], [], []

		# Development purpose, get oflline excel
		if development == '1':
			xls = pd.ExcelFile(yahoo_excel)
			df1 = xls.parse('Sheet1') # for train
			df2 = xls.parse('Sheet2') # for test

			# X_train and y_train
			for i, row in enumerate(df1.values):
				date, open, high, low, close, volume, adjclose = row

				# Train input
				X_train.append([ open, high, low, close, volume ])

				# Train output
				if output == 'open':
					y_train.append([ open ])
				elif output == 'high':
					y_train.append([ high ])
				elif output == 'low':
					y_train.append([ low ])
				else:
					y_train.append([ close ])

				# Get maximum value for each of stock columns
				self.X_train_max_open = open if open > self.X_train_max_open else self.X_train_max_open
				self.X_train_max_high = high if high > self.X_train_max_high else self.X_train_max_high
				self.X_train_max_low = low if low > self.X_train_max_low else self.X_train_max_low
				self.X_train_max_close = close if close > self.X_train_max_close else self.X_train_max_close
				self.X_train_max_volume = volume if volume > self.X_train_max_volume else self.X_train_max_volume

				self.y_train_max_open  = self.X_train_max_open
				self.y_train_max_high  = self.X_train_max_high
				self.y_train_max_low   = self.X_train_max_low
				self.y_train_max_close = self.X_train_max_close


			# X_test and y_test
			for i, row in enumerate(df2.values):
				date, open, high, low, close, volume, adjclose = row
				X_test.append([ open, high, low, close, volume ])

				if output == 'open':
					y_test.append([ open ])
				elif output == 'high':
					y_test.append([ high ])
				elif output == 'low':
					y_test.append([ low ])
				else:
					y_test.append([ close ])

				# Get maximum value for each of stock columns
				self.X_test_max_open = open if open > self.X_test_max_open else self.X_test_max_open
				self.X_test_max_high = high if high > self.X_test_max_high else self.X_test_max_high
				self.X_test_max_low = low if low > self.X_test_max_low else self.X_test_max_low
				self.X_test_max_close = close if close > self.X_test_max_close else self.X_test_max_close
				self.X_test_max_volume = volume if volume > self.X_test_max_volume else self.X_test_max_volume

				self.y_test_max_open  = self.X_test_max_open
				self.y_test_max_high  = self.X_test_max_high
				self.y_test_max_low   = self.X_test_max_low
				self.y_test_max_close = self.X_test_max_close


		# Production purpose
		else:
			# Crawling historical data from yahoo stock finance
			stock_train = yahoo.get_historical(yahoo_historical_train_from, yahoo_historical_train_to)
			stock_test = yahoo.get_historical(yahoo_historical_test_from, yahoo_historical_test_to)

			rsv_train = []
			# X_train and y_train
			for i, stock in enumerate(stock_train):
				open, high = float(stock['Open']), float(stock['High'])
				low, close = float(stock['Low']), float(stock['Close'])
				volume = float(stock['Volume'])

				# KDJ calculation is started after 'kdj_long_period' if not kdj_style='google'
				# IF kdj_long_period == 5, THEN K can be calculated after iter > 5
				rsv = 0.0
				if kdj == 'yes':
					# low_prev is a list for previous low price
					# high_prev is a list for previous high price
					low_prev, high_prev = [], []
					if kdj_style == 'google':
						if i > 4: # start creating a list after iter > 5
							for k in range(1, 6): # get 5 previous trading sessions
								st = stock_train[i-k]
								_lo, _hi = float(st['Low']), float(st['High'])
								low_prev.append(_lo)
								high_prev.append(_hi)
							rsv = self.rsv_indicator(close, low_prev, high_prev)
					else:
						if i > kdj_long_period-1:
							for k in range(1, kdj_long_period+1): # get [kdj_long_period] previous trading sessions
								st = stock_train[i-k]
								_lo, _hi = float(st['Low']), float(st['High'])
								low_prev.append(_lo)
								high_prev.append(_hi)
							rsv = self.rsv_indicator(close, low_prev, high_prev)

				rsv_train.append(rsv)

				X_train.append([ open, high, low, close, volume ])

				if output == 'open':
					y_train.append([ open ])
				elif output == 'high':
					y_train.append([ high ])
				elif output == 'low':
					y_train.append([ low ])
				else:
					y_train.append([ close ])

				if kdj == 'no':
					# Get maximum value for each of stock columns
					self.X_train_max_open = open if open > self.X_train_max_open else self.X_train_max_open
					self.X_train_max_high = high if high > self.X_train_max_high else self.X_train_max_high
					self.X_train_max_low = low if low > self.X_train_max_low else self.X_train_max_low
					self.X_train_max_close = close if close > self.X_train_max_close else self.X_train_max_close
					self.X_train_max_volume = volume if volume > self.X_train_max_volume else self.X_train_max_volume

					self.y_train_max_open  = self.X_train_max_open
					self.y_train_max_high  = self.X_train_max_high
					self.y_train_max_low   = self.X_train_max_low
					self.y_train_max_close = self.X_train_max_close


			rsv_test = []
			# X_test and y_test
			for stock in stock_test:

				open, high = float(stock['Open']), float(stock['High'])
				low, close = float(stock['Low']), float(stock['Close'])
				volume = float(stock['Volume'])

				# KDJ calculation is started after 'kdj_long_period' if not kdj_style='google'
				# IF kdj_long_period == 5, THEN K can be calculated after iter > 5
				# see, line 144 for rsv_indicator
				rsv = 0.0
				if kdj == 'yes':
					# low_prev is a list for previous low price
					# high_prev is a list for previous high price
					low_prev, high_prev = [], []
					if kdj_style == 'google':
						if i > 4:
							for k in range(1, 6): # get 5 previous trading sessions
								st = stock_train[i-k]
								_lo, _hi = float(st['Low']), float(st['High'])
								low_prev.append(_lo)
								high_prev.append(_hi)
							rsv = self.rsv_indicator(close, low_prev, high_prev)
					else:
						if i > kdj_long_period-1:
							for k in range(1, kdj_long_period+1): # get [kdj_long_period] previous trading sessions
								st = stock_train[i-k]
								_lo, _hi = float(st['Low']), float(st['High'])
								low_prev.append(_lo)
								high_prev.append(_hi)
							rsv = self.rsv_indicator(close, low_prev, high_prev)

				rsv_test.append(rsv)

				X_test.append([ open, high, low, close, volume ])

				if output == 'open':
					y_test.append([ open ])
				elif output == 'high':
					y_test.append([ high ])
				elif output == 'low':
					y_test.append([ low ])
				else:
					y_test.append([ close ])

				if kdj == 'no':
					# Get maximum value for each of stock columns
					self.X_test_max_open = open if open > self.X_test_max_open else self.X_test_max_open
					self.X_test_max_high = high if high > self.X_test_max_high else self.X_test_max_high
					self.X_test_max_low = low if low > self.X_test_max_low else self.X_test_max_low
					self.X_test_max_close = close if close > self.X_test_max_close else self.X_test_max_close
					self.X_test_max_volume = volume if volume > self.X_test_max_volume else self.X_test_max_volume

					self.y_test_max_open  = self.X_test_max_open
					self.y_test_max_high  = self.X_test_max_high
					self.y_test_max_low   = self.X_test_max_low
					self.y_test_max_close = self.X_test_max_close


			# Obtain K, D, J for train and test
			if kdj == 'yes':
				K_train, D_train, J_train = None, None, None
				K_test, D_test, J_test = None, None, None

				if kdj_style == 'google':
					K_train, D_train, J_train = self.kdj_indicator(rsv_train, 3, 5)
					K_test, D_test, J_test = self.kdj_indicator(rsv_test, 3, 5)
				else:
					K_train, D_train, J_train = self.kdj_indicator(rsv_train, kdj_short_period, kdj_long_period)
					K_test, D_test, J_test = self.kdj_indicator(rsv_test, kdj_short_period, kdj_long_period)

				# Pandas DataFrame to List
				K_train = map( float, K_train[0].tolist() )
				D_train = map( float, D_train[0].tolist() )
				J_train = map( float, J_train[0].tolist() )

				# Pandas DataFrame to List
				K_test = map( float, K_test[0].tolist() )
				D_test = map( float, D_test[0].tolist() )
				J_test = map( float, J_test[0].tolist() )


				# X_train and y_train
				# Remove Index value in X_train and y_train before index kdj_long_period to get rid of NaN
				trX, trY = [], []
				for i, X in enumerate(X_train[kdj_long_period:], start=kdj_long_period):
					# X_train New
					open, high, low, close, volume = X
					K, D, J = K_train[i], D_train[i], J_train[i]
					trX.append([ open, high, low, close, volume, K, D, J ])

					# y_train New
					trY.append( [ y_train[i][0] ] )

					# Get maximum value for each of stock columns
					self.X_train_max_open = open if open > self.X_train_max_open else self.X_train_max_open
					self.X_train_max_high = high if high > self.X_train_max_high else self.X_train_max_high
					self.X_train_max_low = low if low > self.X_train_max_low else self.X_train_max_low
					self.X_train_max_close = close if close > self.X_train_max_close else self.X_train_max_close
					self.X_train_max_volume = volume if volume > self.X_train_max_volume else self.X_train_max_volume

					self.X_train_max_K = K if K > self.X_train_max_K else self.X_train_max_K
					self.X_train_max_D = D if D > self.X_train_max_D else self.X_train_max_D
					self.X_train_max_J = J if J > self.X_train_max_J else self.X_train_max_J

					self.y_train_max_open  = self.X_train_max_open
					self.y_train_max_high  = self.X_train_max_high
					self.y_train_max_low   = self.X_train_max_low
					self.y_train_max_close = self.X_train_max_close


				# X_test and y_test
				# Remove Index value in X_test and y_test before index kdj_long_period to get rid of NaN
				teX, teY = [], []
				for i, X in enumerate(X_test[kdj_long_period:], start=kdj_long_period):
					# X_test New
					open, high, low, close, volume = X
					K, D, J = K_test[i], D_test[i], J_test[i]
					teX.append([ open, high, low, close, volume, K, D, J ])

					# y_test New
					teY.append([ y_test[i][0] ])

					# Get maximum value for each of stock columns
					self.X_test_max_open = open if open > self.X_test_max_open else self.X_test_max_open
					self.X_test_max_high = high if high > self.X_test_max_high else self.X_test_max_high
					self.X_test_max_low = low if low > self.X_test_max_low else self.X_test_max_low
					self.X_test_max_close = close if close > self.X_test_max_close else self.X_test_max_close
					self.X_test_max_volume = volume if volume > self.X_test_max_volume else self.X_test_max_volume

					self.X_test_max_K = K if K > self.X_test_max_K else self.X_test_max_K
					self.X_test_max_D = D if D > self.X_test_max_D else self.X_test_max_D
					self.X_test_max_J = J if J > self.X_test_max_J else self.X_test_max_J

					self.y_test_max_open  = self.X_test_max_open
					self.y_test_max_high  = self.X_test_max_high
					self.y_test_max_low   = self.X_test_max_low
					self.y_test_max_close = self.X_test_max_close


				X_train, y_train, X_test, y_test = trX, trY, teX, teY


		return X_train, y_train, X_test, y_test


	# Real stock data
	def get_stock(self, output='close', kdj='no', kdj_style='google', kdj_short_period=3, kdj_long_period=5):
		# X_train, y_train, X_test, y_test
		return self.evaluate(output, kdj, kdj_style, kdj_short_period, kdj_long_period)

	# Transform stock data
	def get_stock_transform(self, output='close', kdj='no', kdj_style='google', kdj_short_period=3, kdj_long_period=5):
		X_train, y_train, X_test, y_test = self.evaluate(output, kdj, kdj_style, kdj_short_period, kdj_long_period)

		# Transformation by logarithm still not enough to convert become 0-1
		# for instead using dividing and (shifting if necessary)
		# X_train and y_train
		trX, trY = [], []
		for X, y in izip(X_train, y_train):
			# X_train
			op, hi, lo, cl, vo, k, d, j = None, None, None, None, None, None, None, None
			if kdj == 'no':
				op, hi, lo, cl, vo = X
			else:
				op, hi, lo, cl, vo, k, d, j = X

			op = float('{0:.2f}'.format(op/self.X_train_max_open))
			hi = float('{0:.2f}'.format(hi/self.X_train_max_high))
			lo = float('{0:.2f}'.format(lo/self.X_train_max_low))
			cl = float('{0:.2f}'.format(cl/self.X_train_max_close))
			vo = float('{0:.2f}'.format(vo/self.X_train_max_volume))

			if kdj == 'no':
				trX.append([ op, hi, lo, cl, vo ])
			else:
				k = float('{0:.2f}'.format(k/self.X_train_max_K))
				d = float('{0:.2f}'.format(d/self.X_train_max_D))
				j = float('{0:.2f}'.format(j/self.X_train_max_J))
				trX.append([ op, hi, lo, cl, vo, k, d, j ])

			# y_train
			if output == 'open':
				open = y[0]
				open = float('{0:.2f}'.format(open/self.y_train_max_open))
				trY.append([ open ])
			elif output == 'high':
				high = y[0]
				high = float('{0:.2f}'.format(high/self.y_train_max_high))
				trY.append([ high ])
			elif output == 'low':
				low = y[0]
				low = float('{0:.2f}'.format(low/self.y_train_max_low))
				trY.append([ low ])
			else:
				close = y[0]
				close = float('{0:.2f}'.format(close/self.y_train_max_close))
				trY.append([ close ])


		# X_test and y_test
		teX, teY = [], []
		for X, y in izip(X_test, y_test):
			# X_test
			op, hi, lo, cl, vo, k, d, j = None, None, None, None, None, None, None, None
			if kdj == 'no':
				op, hi, lo, cl, vo = X
			else:
				op, hi, lo, cl, vo, k, d, j = X

			op = float('{0:.2f}'.format(op/self.X_test_max_open))
			hi = float('{0:.2f}'.format(hi/self.X_test_max_high))
			lo = float('{0:.2f}'.format(lo/self.X_test_max_low))
			cl = float('{0:.2f}'.format(cl/self.X_test_max_close))
			vo = float('{0:.2f}'.format(vo/self.X_test_max_volume))

			if kdj == 'no':
				teX.append([ op, hi, lo, cl, vo ])
			else:
				k = float('{0:.2f}'.format(k/self.X_test_max_K))
				d = float('{0:.2f}'.format(d/self.X_test_max_D))
				j = float('{0:.2f}'.format(j/self.X_test_max_J))
				teX.append([ op, hi, lo, cl, vo, k, d, j ])

			# y_test
			if output == 'open':
				open = y[0]
				open = float('{0:.2f}'.format(open/self.y_test_max_open))
				teY.append([ open ])
			elif output == 'high':
				high = y[0]
				high = float('{0:.2f}'.format(high/self.y_test_max_high))
				teY.append([ high ])
			elif output == 'low':
				low = y[0]
				low = float('{0:.2f}'.format(low/self.y_test_max_low))
				teY.append([ low ])
			else:
				close = y[0]
				close = float('{0:.2f}'.format(close/self.y_test_max_close))
				teY.append([ close ])

		# X_train, y_train, X_test, y_test
		return trX, trY, teX, teY
