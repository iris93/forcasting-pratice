#arima_package.py
#-*-coding:utf-8-*-2
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
from pylab import *
#matplotlib online
from matplotlib.pylab import rcParams
import scipy as sp
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
import os
import scipy as sp

def arima(list,predict_date_start,predict_date_end):
	data = pd.Series(data = list[1],index = list[0])
	data.index.name = 'the_index'
	data_log = np.log(data[:predict_date_start])
	data_log.dropna(inplace=True)
	data_log_diff = data_log - data_log.shift()
	data_log_diff.dropna(inplace=True)
	lag_acf = acf(data_log_diff, nlags=20)
	lag_pacf = pacf(data_log_diff, nlags=20, method='ols')
	for m in range(0,20):
		if(lag_acf[m] <= 1.96/np.sqrt(len(data_log_diff)) ):
			p = m
			break	
	for k in range(0,20):
		if(lag_pacf[k] <= 1.96/np.sqrt(len(data_log_diff)) ):
			q = k
			break
	arma_mod20 = sm.tsa.ARMA(data_log/10,(p,q)).fit()
	predict_sunspots = arma_mod20.predict(predict_date_start,predict_date_end,dynamic=True)
	predictions_ARIMA = np.exp(predict_sunspots *10)
	# #recover dif
	# predict_sunspots[0] += data_log[-2]

	# for d in range(1,len(predict_sunspots)):
	# 	predict_sunspots[d] += predict_sunspots[d-1]
	# #recover log
	# log_recover = np.exp(predict_sunspots)
	# log_recover.dropna(inplace=True)
	# predictions_ARIMA = log_recover
	print('predictions_ARIMA=',predictions_ARIMA)
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
data = []
data = pd.read_csv('data.csv', index_col='YMD',date_parser=dateparse)
ts = data['1']
listdata = []
listdata.append(data.index)
listdata.append(data['1'])
print listdata
# print(list[1])
arima(listdata,'20140111','20140117')

listdata = []
