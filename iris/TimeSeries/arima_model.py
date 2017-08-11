#-*-coding:utf-8-*-2
from __future__ import print_function
import os
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA, _arma_predict_out_of_sample
from statsmodels.tsa.arima_model import ARIMA

item = 'Open'

# so we can see the key steps are below:
# step 1:data initialize
dta =  np.load('trdata.npy')
dta=np.array(dta,dtype=np.float)
length=len(dta)
dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700','1803'))
dta_train = dta[0:90]
dta_test = dta[90:-1]
print("dta_test",dta_test)


d= 2
# step 3:ARMA
p=7
q=1
arma_mod = sm.tsa.ARIMA(dta_train,order=(p,d,q)).fit(disp=False)

# step 4:predict
predict_sunspots = arma_mod.predict(dta_train.index[-1],'1803',typ='levels')
# predict_sunspots = arma_mod.predict(typ='levels')
predict_sunspots = arma_mod.predict('1789','1803',typ='levels')
print("predict_sunspots",predict_sunspots)
# visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta_test.ix['1789':].plot(ax=ax, label="Original",color="blue")
dta_train.ix['1700':].plot(ax=ax,color='green')
predict_sunspots.plot(ax=ax,color='red',label="Prediction")
# plt.plot(dta_test, label="Original: '{}'".format(item))
# plt.plot(predict_sunspots, label="Prediction: '{}'".format(item), color='red')
# plt.legend(loc='best')
plt.show()

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
# using yahoo_finance5 data
basedir = os.getcwd()
# filename = os.path.join(basedir, 'stock.xlsx')
filename = os.path.join(basedir, 'yahoo_finance5.xlsx')
xls = pd.ExcelFile(filename)

# This version only support for training, not support for testing
df_train = xls.parse('Sheet1', index_col='Date') # train
df_test  = xls.parse('Sheet2', index_col='Date') # test
# print df_train.index
df_train.index=pd.to_datetime(df_train.index)
df_test.index=pd.to_datetime(df_test.index)
Xy = 'train'
ts_train= df_train[item] if Xy=='train' else df_test[item]
ts_train = ts_train.sort_index()
ts_test = df_test[item]
# ts_test = ts_test.diff(1)
ts_test = ts_test.sort_index()

d= 1
p=7
q=1
arma_mod = sm.tsa.ARIMA(ts_train,order=(p,d,q),freq='D').fit(disp=False)

# step 4:predict
predict_sunspots = arma_mod.predict('2016-06-15','2016-08-15',typ='levels')
print("predict_sunspots",predict_sunspots)
# visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax = ts_train.ix['2011-08-16':].plot(ax=ax, label="Original",color="blue")
ts_test.ix['2016-06-16':].plot(ax=ax,color='green')
predict_sunspots.plot(ax=ax,color='red',label="Prediction")
# plt.plot(dta_test, label="Original: '{}'".format(item))
# plt.plot(predict_sunspots, label="Prediction: '{}'".format(item), color='red')
# plt.legend(loc='best')
plt.show()

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
'''
# build a model
model = ARIMA(ts, order=(p,d,q),freq='D')
# fitting model
fitting = model.fit(disp=False)
print ( "ARIMA: fitting history: '{}'.".format(item) )
# typ: levels, is selected due to 'Yahoo!' sometimes is not provide daily stock price, then the interval is not always daily
# predicted = fitting.predict('1789','1790',typ='levels')
# predicted = fitting.predict()
# forecast, fcasterr, conf_int = fitting.forecast(steps=1, alpha=.05)
# print ("predicted",predicted)
# print ( "ARIMA: predict level history: '{}'.".format(item) )

# plt.title("timeseries history: '{}'".format(item))
# plt.plot(ts, label="Original: '{}'".format(item))
# plt.plot(predicted, label="Prediction: '{}'".format(item), color='red')
# plt.legend(loc='best')
# plt.show()

test_predicted = fitting.predict('2016-06-15','2016-07-15')
plt.title("timeseries history: '{}'".format(item))
plt.plot(ts_test, label="Original: '{}'".format(item))
plt.plot(test_predicted, label="Prediction: '{}'".format(item), color='red')
plt.legend(loc='best')
print(test_predicted)
plt.show()
'''