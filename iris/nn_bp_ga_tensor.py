# coding: utf-8
'''
Classic Neural Network Backpropagation
Consist of:
- 5 Input (Open, High, Low, Close, Volume)      OR
  8 Input (Open, High, Low, Close, Volume, K, D, J)
- 2 Hidden layers
- 10 Neurons in each Hidden layers
- 1 Output, selected option that is: Open, High, Low or Close

Genetic Algorithm
purpose: to Update Weight
'''

from itertools import izip
import numpy as np
import tensorflow as tf
tf.InteractiveSession()

from preparing_input import InitPreprocessing
init_instance = InitPreprocessing()

# Configuration
output='close' # stock prices option: open, high, low, close
kdj='no' # or kdj='no'
kdj_style='google' # or kdj_style='', kdj_short_period=3, kdj_long_period=5
kdj_short_period=3 # D (Divergen)
kdj_long_period=5  # K

X_train, y_train, X_test, y_test = init_instance.get_stock_transform(output=output, kdj=kdj, kdj_style=kdj_style, kdj_short_period=kdj_short_period, kdj_long_period=kdj_long_period)

