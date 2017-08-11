
# coding: utf-8

# In[20]:


'''
1. Get data Offline Excel, Yahoo! Stock Finance
2. Calculate by ARMA/ARIMA to obtain prediction 'n next days' each of stock item
3. Calculate KDJ from output ARMA/ ARIMA
4. The No.3's output become input GA

'''

import os
import numpy as np
import pandas as pd
from itertools import izip

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pylab as plt
from matplotlib.finance import candlestick
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from statsmodels.tsa.arima_model import ARMA, _arma_predict_out_of_sample
from statsmodels.tsa.arima_model import ARIMA


# Load Excel file
# basedir = os.path.abspath(os.path.dirname(__file__))
basedir = os.getcwd()
# filename = os.path.join(basedir, 'stock.xlsx')
filename = os.path.join(basedir, 'yahoo_finance5.xlsx')
xls = pd.ExcelFile(filename)

df_train = xls.parse('Sheet1', index_col='Date') # train
print df_train.index
df_train.index=pd.to_datetime(df_train.index)
print df_train.index

class BuildModel:
    def __init__(self,
        p=3,
        d=1,
        q=2,
        n_days=1,
        fc=1): # iteration for n days forecast

        # ARMA/ ARIMA
        self.p = p
        self.d = d
        self.q = q
        
        # how many days a head to predict
        self.n_days = n_days
        
        # fc
        self.fc = fc
    
    def predict_arima_next_days(self, item):
        ts = df_train[item]
        ts = ts.sort_index()
        ts_last_day = ts[self.fc] # later, it will be stored as output
        ts = ts[0:self.fc]
        model = ARIMA(ts, order=(self.p, self.d, self.q))
        fitting = model.fit(disp=False)
        
        forecast, fcasterr, conf_int = fitting.forecast(steps=self.n_days, alpha=.05)
        # ts:          history until 1 day before self.fc
        # ts[self.fc]: last day
        # forecast:    1 day forecast (time equalto ts[self.fc])
        return ts, ts_last_day, forecast
    
    def predict_arma_next_days(self, item):
        ts = df_train[item]
        ts = ts.sort_index()
        ts_last_day = ts[self.fc] # later, it will be stored as output
        ts = ts[0:self.fc]
        
        model = ARMA(ts, order=(self.p, self.q), freq='D')
        fitting = model.fit(disp=False)
        params = fitting.params
        residuals = fitting.resid
        p = fitting.k_ar
        q = fitting.k_ma
        k_exog = fitting.k_exog
        k_trend = fitting.k_trend
        # n_days forecasting
        forecast = _arma_predict_out_of_sample(params, self.n_days, residuals, p, q, k_trend, k_exog, endog=ts, exog=None, start=len(ts))
        # ts:          history until 1 day before self.fc
        # ts[self.fc]: last day
        # forecast:    1 day forecast (time equalto ts[self.fc])
        return ts, ts_last_day, forecast
        





def item_prediction(instance, d, item):
    ts, n_days_real, n_days_forecast = None, None, None
    if d == 0:
        ts, n_days_real, n_days_forecast = instance.predict_arma_next_days(item)
    else:
        ts, n_days_real, n_days_forecast = instance.predict_arima_next_days(item)
    
    # n_days_real : output format is float
    # n_days_forecast: output format is a list
    ts_real = ts.append(pd.DataFrame([n_days_real]))
    ts_forecast = ts.append(pd.DataFrame(n_days_forecast))
    ts_real = map( float, ts_real[0].tolist() )
    ts_forecast = map ( float, ts_forecast[0].tolist() )
    return ts_real, ts_forecast, n_days_real, n_days_forecast[0]


# -----------------------------------------------------------
print 'Started...'
# Configuration ARMA/ARIMA, KDJ and Output target price
# NOTICE:
#    d=1 , automatically use ARIMA
#    d=0 , automatically use ARMA
p, d, q = 2, 1, 0
n_days = 1 # How many 1 days prediction/ DONOT CHANGE


# Start Produce Weight for GA
print 'Produce trX and trY for GA.'

trX, trY = [], []
start = 27 # 7
stop = 208 # 208
for fc in range(start, stop): # fc = forecast n days
    an = BuildModel(p, d, q, n_days, fc)
    ts_real_op, ts_fc_op, n_days_real_op, n_days_fc_op = item_prediction(an, d, 'Open')
    ts_real_hi, ts_fc_hi, n_days_real_hi, n_days_fc_hi = item_prediction(an, d, 'High')
    ts_real_lo, ts_fc_lo, n_days_real_lo, n_days_fc_lo = item_prediction(an, d, 'Low')
    ts_real_cl, ts_fc_cl, n_days_real_cl, n_days_fc_cl = item_prediction(an, d, 'Close')
    ts_real_vo, ts_fc_vo, n_days_real_vo, n_days_fc_vo = item_prediction(an, d, 'Volume')
    trX.append([ n_days_fc_op, n_days_fc_hi, n_days_fc_lo, n_days_fc_cl, n_days_fc_vo ])
    trY.append([ n_days_real_cl ])

print trY[0:4]
yy = df_train['Close']
yy = yy.sort_index()
yy = yy[start:]
print yy[0:4]

# Normalization
op_max, hi_max, lo_max, cl_max, vo_max = -10000000000.0, -10000000000.0, -10000000000.0, -10000000000.0, -10000000000.0
out_max = -10000000000.0 # Target, can be 'Close'
for i, _X in enumerate(trX):
    op, hi, lo, cl, vo = _X
    op_max = op if op > op_max else op_max
    hi_max = hi if hi > hi_max else hi_max
    lo_max = lo if lo > lo_max else lo_max
    cl_max = cl if cl > cl_max else cl_max
    vo_max = vo if vo > vo_max else vo_max
    
    out = trY[i][0] # can be 'Close'
    out_max = out if out > out_max else out_max

X_new, y_new = [], []  # transformation
for _X, _y in izip(trX, trY):
    op, hi, lo, cl, vo = _X
    op /= op_max
    hi /= hi_max
    lo /= lo_max
    cl /= cl_max
    vo /= vo_max
    X_new.append([ op, hi, lo, cl, vo ])

    out = _y[0]/out_max
    y_new.append([ out ])

# print 'X_new: {}'.format(X_new)
# print 'y_new: {}'.format(y_new)


# In[ ]:



