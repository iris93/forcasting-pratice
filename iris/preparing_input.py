# coding: utf-8
'''
1. Production : Crawling original data from yahoo stock finance
2. Development: Alternatively load offline excel file
3. Implement logarithm to convert real data from big floating-point
   into decimal between 0 to 1
''' 
from __future__ import absolute_import
from __future__ import division

import os
import sys
import ConfigParser
import pandas as pd
from itertools import izip
from sklearn.cross_validation import train_test_split
# from yahoo_finance import Share

from log import logger
from config_parser import basedir, init_preprocessing
config = ConfigParser.ConfigParser()
config_file = os.path.join(basedir, 'config.ini')

# Loading module config_parser to obtain configuration parameters
development, cross_validation, yahoo_excel = None, None, None
yahoo_historical_train_from, yahoo_historical_train_to = None, None
yahoo_historical_test_from, yahoo_historical_test_to = None, None
try:
	config.readfp(open(config_file))
	development, cross_validation, yahoo_excel, yahoo_historical_train_from, yahoo_historical_train_to, yahoo_historical_test_from, yahoo_historical_test_to = init_preprocessing(config)
except:
	logger.debug('Error init_preprocessing(). Exit!')
	sys.exit()


# yahoo = Share('YHOO') if development == '0' else None

class InitPreprocessing:
	'''
	Class for initialization and preprocessing input for neuron input

   How to use:

   # import module
   from preparing_input import InitPreprocessing

   # create a new instance
   init_instance = InitPreprocessing()

   # get_tensor(), get real stock data in tensor format
   X_train, y_train, X_test, y_test = init_instance.get_tensor()

   # get_tensor_transform(), get transform stock data in tensor format
   X_train, y_train, X_test, y_test = init_instance.get_tensor_transform()

	'''

	def __init__(self):
		# Parameters X_train for transformation
		self.X_train_max_open   = -100.0
		self.X_train_max_high   = -100.0
		self.X_train_max_low    = -100.0
		self.X_train_max_close  = -100.0
		self.X_train_max_volume = -100.0

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
		self.X_test_max_volume  = -100.0

		# Parameters y_test for transformation
		self.y_test_max_open    = -100.0
		self.y_test_max_high    = -100.0
		self.y_test_max_low     = -100.0
		self.y_test_max_close   = -100.0

	# def get maximum X_train
	def get_X_train_max(self):
		return self.X_train_max_open, self.X_train_max_high, self.X_train_max_low, self.X_train_max_close, self.X_train_max_volume

	# get maximum y_train
	def get_y_train_max(self):
		return self.y_train_max_open, self.y_train_max_high, self.y_train_max_low, self.y_train_max_close

	# get maximum X_test
	def get_X_test_max(self):
		return self.X_test_max_open, self.X_test_max_high, self.X_test_max_low, self.X_test_max_close, self.X_test_max_volume

	# get maximum y_test
	def get_y_test_max(self):
		return self.y_test_max_open, self.y_test_max_high, self.y_test_max_low, self.y_test_max_close

	# Evaluation process whether use development or production data
	# output: open, high, low, close
	def evaluate(self, output):
		X_train, y_train, X_test, y_test = [], [], [], []

		# Development purpose, get oflline excel
		if development == '1':
			xls = pd.ExcelFile(yahoo_excel)
			df1 = xls.parse('Sheet3')
			df2 = xls.parse('Sheet4')

			# X_train and y_train
			for i, row in enumerate(df1.values):
				date, open, high, low, close, volume = row
				X_train.append([ open, high, low, close, volume ])
				if output == 'open':
					y_train.append([ open ])
				elif output == 'high':
					y_train.append([ high ])
				elif output == 'low':
					y_train.append([ low ])
				else:
					y_train.append([ close ])

				if cross_validation == '0':
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


			# When X_test and y_test are produced by cross_validation
			if cross_validation == '1':
				# sklearn format X_test in 2nd order
				X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

				for trX, trY, teX, teY in izip(X_train, y_train, X_test, y_test):
					# X_train
					open, high, low, close, volume = trX

					# Get maximum value for each of stock columns
					self.X_train_max_open = open if open > self.X_train_max_open else self.X_train_max_open
					self.X_train_max_high = high if high > self.X_train_max_high else self.X_train_max_high
					self.X_train_max_low = low if low > self.X_train_max_low else self.X_train_max_low
					self.X_train_max_close = close if close > self.X_train_max_close else self.X_train_max_close
					self.X_train_max_volume = volume if volume > self.X_train_max_volume else self.X_train_max_volume

					# y_train
					if output == 'open':
						open = trY
						self.y_train_max_open = open if open > self.y_train_max_open else self.y_train_max_open
					elif output == 'high':
						high = trY
						self.y_train_max_high = high if high > self.y_train_max_high else self.y_train_max_high
					elif output == 'low':
						low = trY
						self.y_train_max_low = low if low > self.y_train_max_low else self.y_train_max_low
					else:
						close = trY
						self.y_train_max_close = close if close > self.y_train_max_close else self.y_train_max_close

					# X_test
					open, high, low, close, volume = teX

					# Get maximum value for each of stock columns
					self.X_test_max_open = open if open > self.X_test_max_open else self.X_test_max_open
					self.X_test_max_high = high if high > self.X_test_max_high else self.X_test_max_high
					self.X_test_max_low = low if low > self.X_test_max_low else self.X_test_max_low
					self.X_test_max_close = close if close > self.X_test_max_close else self.X_test_max_close
					self.X_test_max_volume = volume if volume > self.X_test_max_volume else self.X_test_max_volume

					# y_test
					if output == 'open':
						open = teY
						self.y_train_max_open = open if open > self.y_train_max_open else self.y_train_max_open
					elif output == 'high':
						high = teY
						self.y_train_max_high = high if high > self.y_train_max_high else self.y_train_max_high
					elif output == 'low':
						low = teY
						self.y_train_max_low = low if low > self.y_train_max_low else self.y_train_max_low
					else:
						close = teY
						self.y_test_max_close = close if close > self.y_test_max_close else self.y_test_max_close

			else:
				# X_test
				# Assign into normal python array
				for i, row in enumerate(df2.values):
					date, open, high, low, close, volume = row
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
		# else:
		# 	# Crawling historical data from yahoo stock finance
		# 	stock_train = yahoo.get_historical(yahoo_historical_train_from, yahoo_historical_train_to)
		# 	stock_test = yahoo.get_historical(yahoo_historical_test_from, yahoo_historical_test_to)

		# 	for stock in stock_train:

		# 		open, high = float(stock['Open']), float(stock['High'])
		# 		low, close = float(stock['Low']), float(stock['Close'])
		# 		volume = float(stock['Volume'])

		# 		X_train.append([ open, high, low, close, volume ])
		# 		if output == 'open':
		# 			y_train.append([ open ])
		# 		elif output == 'high':
		# 			y_train.append([ high ])
		# 		elif output == 'low':
		# 			y_train.append([ low ])
		# 		else:
		# 			y_train.append([ close ])

		# 		if cross_validation == '0':
		# 			# Get maximum value for each of stock columns
		# 			self.X_train_max_open = open if open > self.X_train_max_open else self.X_train_max_open
		# 			self.X_train_max_high = high if high > self.X_train_max_high else self.X_train_max_high
		# 			self.X_train_max_low = low if low > self.X_train_max_low else self.X_train_max_low
		# 			self.X_train_max_close = close if close > self.X_train_max_close else self.X_train_max_close
		# 			self.X_train_max_volume = volume if volume > self.X_train_max_volume else self.X_train_max_volume

		# 			self.y_train_max_open  = self.X_train_max_open
		# 			self.y_train_max_high  = self.X_train_max_high
		# 			self.y_train_max_low   = self.X_train_max_low
		# 			self.y_train_max_close = self.X_train_max_close


		# 	# When X_test and y_test are produced by cross_validation
		# 	if cross_validation == '1':
		# 		# sklearn format X_test in 2nd order
		# 		X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

		# 		for trX, trY, teX, teY in izip(X_train, y_train, X_test, y_test):
		# 			# X_train
		# 			open, high, low, close, volume = trX

		# 			# Get maximum value for each of stock columns
		# 			self.X_train_max_open = open if open > self.X_train_max_open else self.X_train_max_open
		# 			self.X_train_max_high = high if high > self.X_train_max_high else self.X_train_max_high
		# 			self.X_train_max_low = low if low > self.X_train_max_low else self.X_train_max_low
		# 			self.X_train_max_close = close if close > self.X_train_max_close else self.X_train_max_close
		# 			self.X_train_max_volume = volume if volume > self.X_train_max_volume else self.X_train_max_volume

		# 			# y_train
		# 			if output == 'open':
		# 				open = trY
		# 				self.y_train_max_open = open if open > self.y_train_max_open else self.y_train_max_open
		# 			elif output == 'high':
		# 				high = trY
		# 				self.y_train_max_high = high if high > self.y_train_max_high else self.y_train_max_high
		# 			elif output == 'low':
		# 				low = trY
		# 				self.y_train_max_low = low if low > self.y_train_max_low else self.y_train_max_low
		# 			else:
		# 				close = trY
		# 				self.y_train_max_close = close if close > self.y_train_max_close else self.y_train_max_close

		# 			# X_test
		# 			open, high, low, close, volume = teX

		# 			# Get maximum value for each of stock columns
		# 			self.X_test_max_open = open if open > self.X_test_max_open else self.X_test_max_open
		# 			self.X_test_max_high = high if high > self.X_test_max_high else self.X_test_max_high
		# 			self.X_test_max_low = low if low > self.X_test_max_low else self.X_test_max_low
		# 			self.X_test_max_close = close if close > self.X_test_max_close else self.X_test_max_close
		# 			self.X_test_max_volume = volume if volume > self.X_test_max_volume else self.X_test_max_volume

		# 			# y_test
		# 			if output == 'open':
		# 				open = teY
		# 				self.y_test_max_open = open if open > self.y_test_max_open else self.y_test_max_open
		# 			elif output == 'high':
		# 				high = teY
		# 				self.y_test_max_high = high if high > self.y_test_max_high else self.y_test_max_high
		# 			elif output == 'low':
		# 				low = teY
		# 				self.y_test_max_low = low if low > self.y_test_max_low else self.y_test_max_low
		# 			else:
		# 				close = teY
		# 				self.y_test_max_close = close if close > self.y_test_max_close else self.y_test_max_close

		# 	else:
		# 		# X_test
		# 		# Assign into normal python array
		# 		for stock in stock_test:

		# 			open, high = float(stock['Open']), float(stock['High'])
		# 			low, close = float(stock['Low']), float(stock['Close'])
		# 			volume = float(stock['Volume'])

		# 			X_test.append([ open, high, low, close, volume ])

		# 			if output == 'open':
		# 				y_test.append([ open ])
		# 			elif output == 'high':
		# 				y_test.append([ high ])
		# 			elif output == 'low':
		# 				y_test.append([ low ])
		# 			else:
		# 				y_test.append([ close ])

		# 			# Get maximum value for each of stock columns
		# 			self.X_test_max_open = open if open > self.X_test_max_open else self.X_test_max_open
		# 			self.X_test_max_high = high if high > self.X_test_max_high else self.X_test_max_high
		# 			self.X_test_max_low = low if low > self.X_test_max_low else self.X_test_max_low
		# 			self.X_test_max_close = close if close > self.X_test_max_close else self.X_test_max_close
		# 			self.X_test_max_volume = volume if volume > self.X_test_max_volume else self.X_test_max_volume

		# 			self.y_test_max_open  = self.X_test_max_open
		# 			self.y_test_max_high  = self.X_test_max_high
		# 			self.y_test_max_low   = self.X_test_max_low
		# 			self.y_test_max_close = self.X_test_max_close


		return X_train, y_train, X_test, y_test


	# Real stock data
	def get_tensor(self, output='close'):
		# X_train, y_train, X_test, y_test
		return self.evaluate(output)

	# Transform stock data
	def get_tensor_transform(self, output='close'):
		X_train, y_train, X_test, y_test = self.evaluate(output)

		# Transformation by logarithm still not enough to convert become 0-1
		# for instead using dividing and shifting
		trX, trY = [], []
		teX, teY = [], []
		for X, y in izip(X_train, y_train):
			# X_train
			op, hi, lo, cl, vo = X
			op = (op/self.X_train_max_open)-0.1111111111111111
			hi = (hi/self.X_train_max_high)-0.1111111111111111
			lo = (lo/self.X_train_max_low)-0.1111111111111111
			cl = (cl/self.X_train_max_close)-0.1111111111111111
			vo = float(vo)
			vo = (vo/self.X_train_max_volume)-0.1111111111111111
			trX.append([ op, hi, lo, cl, vo ])

			# y_train
			if output == 'open':
				open = y[0]
				open = (open/self.y_train_max_open)-0.1111111111111111
				trY.append([ open ])
			elif output == 'high':
				high = y[0]
				high = (high/self.y_train_max_high)-0.1111111111111111
				trY.append([ high ])
			elif output == 'low':
				low = y[0]
				low = (low/self.y_train_max_low)-0.1111111111111111
				trY.append([ low ])
			else:
				close = y[0]
				close = (close/self.y_train_max_close)-0.1111111111111111
				trY.append([ close ])


		for X, y in izip(X_test, y_test):
			# X_test
			op, hi, lo, cl, vo = X
			op = (op/self.X_test_max_open)-0.1111111111111111
			hi = (hi/self.X_test_max_high)-0.1111111111111111
			lo = (lo/self.X_test_max_low)-0.1111111111111111
			cl = (cl/self.X_test_max_close)-0.1111111111111111
			vo = (vo/self.X_test_max_volume)-0.1111111111111111
			teX.append([ op, hi, lo, cl, vo ])

			# y_test
			if output == 'open':
				open = y[0]
				open = (open/self.y_test_max_open)-0.1111111111111111
				teY.append([ open ])
			elif output == 'high':
				high = y[0]
				high = (high/self.y_test_max_high)-0.1111111111111111
				teY.append([ high ])
			elif output == 'low':
				low = y[0]
				low = (low/self.y_test_max_low)-0.1111111111111111
				teY.append([ low ])
			else:
				close = y[0]
				close = (close/self.y_test_max_close)-0.1111111111111111
				teY.append([ close ])

		return trX, trY, teX, teY
