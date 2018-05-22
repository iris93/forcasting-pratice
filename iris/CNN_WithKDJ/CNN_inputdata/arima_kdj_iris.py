
# coding: utf-8

# In[4]:


'''
1. Get data Offline Excel, Yahoo! Stock Finance
2. Calculate by ARMA/ARIMA to obtain prediction 'n next days' each of stock item
3. Calculate KDJ from output ARMA/ ARIMA
4. The No.3's output become input NN
5. Train the input by NN BP
   Classic Neural Network Backpropagation
   Consist of:
   - 5 Input (Open, High, Low, Close, Volume)      OR
     8 Input (Open, High, Low, Close, Volume, K, D, J)
   - 2 Hidden layers
   - N (can be configured) Neurons in each Hidden layers
   - 1 Output, selected option that is: Open, High, Low or Close
6. Final result: Slice the 'n next days' last forecasting result from NN

'''

#------------------------------------------------------------------
# MUST READ
# Uptil 10-08-2016
# This code only available to provide prediction with data source started from 'last month' history data until 'today'. Period only 1 month.
# The result is quite bad for prediction which started from last year history data.
#------------------------------------------------------------------


# matplotlib candlestick
# http://stackoverflow.com/questions/34482714/importerror-in-candlestick-example-from-matplotlib-documentation-cannot-import
# http://stackoverflow.com/questions/36334665/how-to-plot-ohlc-candlestick-with-datetime-in-matplotlib
# http://matplotlib.org/examples/pylab_examples/finance_demo.html


# pip install statsmodels==0.8.0rc1 --trusted-host pypi.v2ex.com
# http://stackoverflow.com/questions/25331286/can-statsmodel-arima-forecast-multiple-steps-ahead-using-exogenous-variable
# prediction = dynamic_arima_results.forecast(steps=8, exog=X_pred)

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
import tensorflow as tf
tf.InteractiveSession()


# http://people.duke.edu/~rnau/411arim.htm
# ARIMA(p,d,q) forecasting equation
# ARIMA(1,0,0) = first-order autoregressive model
# ARIMA(0,1,0) = random walk 
# ARIMA(1,1,0) = differenced first-order autoregressive model 
# ARIMA(0,1,1) without constant = simple exponential smoothing 
# ARIMA(0,1,1) with constant = simple exponential smoothing with growth 
# ARIMA(0,2,1) or (0,2,2) without constant = linear exponential smoothing 
# ARIMA(1,1,2) with constant = damped-trend linear exponential smoothing 
# Spreadsheet implementation

# PUT DATE into DataFrame
# http://stackoverflow.com/questions/27931571/arma-predict-for-out-of-sample-forecast-does-not-work-with-floating-points

# Load Excel file
# basedir = os.path.abspath(os.path.dirname(__file__))
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
# print df_train.index
class BuildModel:
    '''
    Consist of:
    1. ARMA (p, q)/ ARIMA Model (p, d, q)
    2. KDJ Indicator
    3. Neural Network Backpropagation
    '''
    def __init__(self,
        p=3,                # Auto Regression (AR), lag p
        d=1,                # Differencing NaN shifting for ARIMA, lag d
        q=2,                # Moving Average (MA), lag q
        n_days=1,      # How many days a head to be predicted
        fc=1,          
        kdj=False,          # KDJ indicator support
        kdj_short_period=3, # to calculate D (Divergen) K
        kdj_long_period=5,  # to calculate K
        output='Close'):    # What is the price target for final prediction (After NN)
                            # Example: 'Open', 'High', 'Low', 'Close'
        
        # ARMA/ ARIMA
        self.p = p # AR lag p
        self.d = d # differencing, ARIMA only
        self.q = q # MA lag q
        
        # how many days a head to predict
        self.n_days = n_days

        #fc
        self.fc = fc

        # KDJ
        self.kdj = kdj
        self.kdj_short_period = kdj_short_period
        self.kdj_long_period  = kdj_long_period

        # y_train, final target price for NN
        self.output = output

        # The following parameters for transformation input NN
        # that is maximum values each of stock prices, volume and/or K, D, J
        self.X_max_op = -10000000000.0
        self.X_max_hi = -10000000000.0
        self.X_max_lo = -10000000000.0
        self.X_max_cl = -10000000000.0
        self.X_max_vo = -10000000000.0

        self.X_max_K  = -10000000000.0
        self.X_max_D  = -10000000000.0
        self.X_max_J  = -10000000000.0

        self.y_max_op = -10000000000.0
        self.y_max_hi = -10000000000.0
        self.y_max_lo = -10000000000.0
        self.y_max_cl = -10000000000.0
        self.y_max_vo = -10000000000.0


    # -------------------------------------------------------------
    # 1st Section
    # ARMA/ ARIMA Model

    # predict single parameter by arima for X_train or X_test
    # item: 'Open', 'High', 'Low', 'Close', 'Volume'
    # Xy: 'train', 'test'
    # http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/

    # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima_model.py
    def predict_arima_next_days(self, item, Xy='train'):
        ts = df_train[item] if Xy=='train' else df_test[item]
        # sorting index Date
        ts = ts.sort_index()

        ts_last_day = ts[self.fc] # later, it will be stored as output
        ts = ts[0:self.fc]
        # build a model
        model = ARIMA(ts, order=(self.p, self.d, self.q))
        # fitting model
        fitting = model.fit(disp=False)
        # print ( "ARIMA: fitting model: '{}'.".format(item) )

        # compute the standard errors (from git source)
        # fcasterr = self._forecast_error(steps)
        # conf_int = self._forecast_conf_int(forecast, fcasterr, alpha)
        forecast, fcasterr, conf_int = fitting.forecast(steps=self.n_days, alpha=.05)
        # print ( "ARIMA: forecast '{0}' for next {1} days.".format(item, self.n_days))
        return ts, ts_last_day,forecast


    # http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.tsa.arima_model.ARMAResults.plot_predict.html
    # http://stackoverflow.com/questions/18616588/arma-out-of-sample-prediction-with-statsmodels
    # http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.tsa.arima_model.ARMAResults.predict.html

    # predict by arma, without d (differencing), then recent value is not lost
    # predict single parameter by arma for X_train or X_test
    # item: 'Open', 'High', 'Low', 'Close', 'Volume'
    # Xy: 'train', 'test'

    # forecasting for n_days (new prediction n steps a head)
    # http://stackoverflow.com/questions/18616588/arma-out-of-sample-prediction-with-statsmodels
    # http://codego.net/9067598/
    def predict_arma_next_days(self, item, Xy='train'):
        ts = df_train[item] if Xy=='train' else df_test[item]
        # sorting index Date
        ts = ts.sort_index()
        ts_last_day = ts[self.fc] # later, it will be stored as output
        ts = ts[0:self.fc]
        # http://statsmodels.sourceforge.net/devel/examples/generated/ex_dates.html
        # https://groups.google.com/forum/#!msg/pystatsmodels/_ItLBVpePIY/nBiP3fn4kDkJ
        # https://github.com/statsmodels/statsmodels/issues/1857
        # http://stackoverflow.com/questions/27931571/arma-predict-for-out-of-sample-forecast-does-not-work-with-floating-points
        # http://statsmodels.sourceforge.net/devel/examples/generated/ex_dates.html
        # https://bicorner.com/2015/11/16/time-series-analysis-using-ipython/
        # http://stackoverflow.com/questions/35593759/python-arima-model-predicted-values-are-shifted
        # http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html
        # freq : str {'B','D','W','M','A', 'Q'}
        #     'B' - business day, ie., Mon. - Fri.
        #     'D' - daily
        #     'W' - weekly
        #     'M' - monthly
        #     'A' - annual
        #     'Q' - quarterly
        # Example: model = ARMA(ts, (self.p, self.q))

        # build a model
        model = ARMA(ts, order=(self.p, self.q), freq='D')
        # fitting model
        fitting = model.fit(disp=False)
        # print ( "ARMA: fitting model: '{}'.".format(item) )
        # preparing input for forecasting out of sample data
        params = fitting.params
        residuals = fitting.resid
        p = fitting.k_ar
        q = fitting.k_ma
        k_exog = fitting.k_exog
        k_trend = fitting.k_trend
        # n_days forecasting
        forecast = _arma_predict_out_of_sample(params, self.n_days, residuals, p, q, k_trend, k_exog, endog=ts, exog=None, start=len(ts))
        # print ( "ARMA: forecast '{0}' for next {1} days.".format(item, self.n_days))
        # ts:          history until 1 day before self.fc
        # ts[self.fc]: last day
        # forecast:    1 day forecast (time equalto ts[self.fc])
        return ts,ts_last_day, forecast


    # -------------------------------------------------------------
    # 2nd Section
    # KDJ Indicator Model

    # RSV indicator
    # http://pastebin.com/CVrVn9F6
    # https://groups.google.com/forum/#!topic/google-finance/QFwZRV412TU
    # cl: is stock 'Close'
    # lo_prev: is a list low previous trading sessions
    # hi_prev: is a list high previous trading sessions
    def rsv_indicator(self, cl, lo_prev, hi_prev):
        return 100 * (cl-np.min(lo_prev))/(np.max(hi_prev)-np.min(lo_prev))

    # KDJ indicator
    # https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py
    def kdj_indicator(self, rsv):
        rsv = pd.DataFrame(rsv)
        # get K by EMA, lag=self.kdj_long_period
        K = rsv.ewm(ignore_na=False, span=self.kdj_long_period, min_periods=0, adjust=True).mean()
        # get D (Divergen of K) by MA
        K.dropna(inplace=True)
        D = K.rolling(window=self.kdj_short_period, center=False).mean()
        # get J
        # (see bottom code), https://discuss.tradewave.net/t/code-for-kdj-indicator/508/2
        # https://www.tradingview.com/script/h51ZIRlX-KDJ-Indicator-iamaltcoin/
        J = 3*K - 2*D
        print ( 'Calculate KDJ Indicator.' )
        return K, D, J

    # KDJ calculation procedure
    def kdj_calc(self, open, high, low, close, volume):
        
        print ( 'Length data before KDJ in: {}'.format(len(open)) )

        # X is total input list for NN
        # y is output list for NN
        rsv_list, X, y = [], [], []
        for i, op in enumerate(open):
            hi, lo, cl, vo = high[i], low[i], close[i], volume[i]
            # KDJ calculation is started after 'kdj_long_period'
            # IF kdj_long_period == 5, THEN K can be calculated after iter>5
            rsv = 0.0
            if self.kdj:
                # lo_prev is a list for previous low price
                # hi_prev is a list for previous high price
                lo_prev, hi_prev = [], []
                if i > kdj_long_period-1: # start creating a list after iter>kdj_long_period-1
                    for k in range(1, kdj_long_period+1): # get [kdj_long_period] previous trading sessions
                        _lo, _hi = low[i-k], high[i-k]
                        lo_prev.append(_lo)
                        hi_prev.append(_hi)
                    rsv = self.rsv_indicator(cl, lo_prev, hi_prev)
                else: pass
            else: pass

            rsv_list.append(rsv)
            X.append([ op, hi, lo, cl, vo ])

            if self.output=='Open':
                y.append([ op ])
            elif self.output=='High':
                y.append([ hi ])
            elif self.output=='Low':
                y.append([ lo ])
            else:
                y.append([ cl ])

            if self.kdj:
                # do nothing, the process will be continued later outside loop
                pass
            else:
                # Get maximum value for each of stock columns
                # max X inputs
                self.X_max_op = op if op > self.X_max_op else self.X_max_op
                self.X_max_hi = hi if hi > self.X_max_hi else self.X_max_hi
                self.X_max_lo = lo if lo > self.X_max_lo else self.X_max_lo
                self.X_max_cl = cl if cl > self.X_max_cl else self.X_max_cl
                self.X_max_vo = vo if vo > self.X_max_vo else self.X_max_vo
                # max y output
                self.y_max_op = self.X_max_op
                self.y_max_hi = self.X_max_hi
                self.y_max_lo = self.X_max_lo
                self.y_max_cl = self.X_max_cl
                self.y_max_vo = self.X_max_vo


        if self.kdj:
            K, D, J = self.kdj_indicator(rsv_list)
            # Convert from pandas DataFrame to a list
            K = map( float, K[0].tolist() )
            D = map( float, D[0].tolist() )
            J = map( float, J[0].tolist() )

            # plt.title('KDJ Indicator')
            # plt.plot(K, label='K')
            # plt.plot(D, label='D')
            # plt.plot(J, label='J')
            # plt.legend(loc='best')
            # plt.show()

            # Remove Index value in X and y before index self.kdj_long_period to get rid of NaN
            # X_new and Y_new for list input and output new X and y with index started from 'self.kdj_long_period'
            X_new, y_new = [], []
            for i, row in enumerate(X[self.kdj_long_period:], start=self.kdj_long_period):
                op, hi, lo, cl, vo = row
                _K, _D, _J = K[i], D[i], J[i]

                # X new for input list NN
                X_new.append([ op, hi, lo, cl, vo, _K, _D, _J ])

                # y new for output list NN
                y_new.append([ y[i][0] ])

                # Get maximum value for each of stock columns
                # max 4 stock price and 1 volume
                # max X inputs
                self.X_max_op = op if op > self.X_max_op else self.X_max_op
                self.X_max_hi = hi if hi > self.X_max_hi else self.X_max_hi
                self.X_max_lo = lo if lo > self.X_max_lo else self.X_max_lo
                self.X_max_cl = cl if cl > self.X_max_cl else self.X_max_cl
                self.X_max_vo = vo if vo > self.X_max_vo else self.X_max_vo

                # max K, D, J
                self.X_max_K = _K if _K > self.X_max_K else self.X_max_K
                self.X_max_D = _D if _D > self.X_max_D else self.X_max_D
                self.X_max_J = _J if _J > self.X_max_J else self.X_max_J

                # max y output
                self.y_max_op = self.X_max_op
                self.y_max_hi = self.X_max_hi
                self.y_max_lo = self.X_max_lo
                self.y_max_cl = self.X_max_cl
                self.y_max_vo = self.X_max_vo

            X, y = X_new, y_new

        else: pass
        
        print ( 'Length data after KDJ in: {}'.format(len(X)) )

        return X, y


    # -------------------------------------------------------------
    # 3rd Section
    # Get original stock data and transformation

    # get original stock data without transformation
    def get_stock_without_transformation(self, open, high, low, close, volume):
        print ( 'Load stock data without transformation.' )
        # X, y
        return self.kdj_calc(open, high, low, close, volume)

    # get stock data after transformation
    # ready for NN training
    def get_stock_with_transformation(self, open, high, low, close, volume):
        print ( 'Load stock data with transformation.' )
        X, y = self.kdj_calc(open, high, low, close, volume)

        # Transformation by logarithm still not enough to convert become 0-1 or small floating-point values
        # for instead is divided by its max value and (no need for shifting)
        X_new, y_new = [], []
        for _X, _y in izip(X, y):
            # X inputs
            op, hi, lo, cl, vo, k, d, j = None, None, None, None, None, None, None, None
            if self.kdj:
                op, hi, lo, cl, vo, k, d, j = _X
            else:
                op, hi, lo, cl, vo = _X

            op /= self.X_max_op
            hi /= self.X_max_hi
            lo /= self.X_max_lo
            cl /= self.X_max_cl
            vo /= self.X_max_vo

            if self.kdj:
                k /= self.X_max_K
                d /= self.X_max_D
                j /= self.X_max_J
                X_new.append([ op, hi, lo, cl, vo, k, d, j ])
            else:
                X_new.append([ op, hi, lo, cl, vo ])

            # y output
            if self.output=='Open':
                op = _y[0]/self.y_max_op
                y_new.append([ op ])
            elif self.output=='High':
                hi = _y[0]/self.y_max_hi
                y_new.append([ hi ])
            elif self.output=='Low':
                lo = _y[0]/self.y_max_lo
                y_new.append([ lo ])
            else:
                cl = _y[0]/self.y_max_cl
                y_new.append([ cl ])

        return X_new, y_new

    # get maximum X inputs
    # purposed: for re-transformation after NN prediction
    def get_X_max(self):
        if self.kdj:
            return self.X_max_op, self.X_max_hi, self.X_max_lo, self.X_max_cl, self.X_max_vo, self.X_max_K, self.X_max_D, self.X_max_J
        else:
            return self.X_max_op, self.X_max_hi, self.X_max_lo, self.X_max_cl, self.X_max_vo

    # get maximum y output
    # purposed: for re-transformation after NN prediction
    def get_y_max(self):
        return self.y_max_op, self.y_max_hi, self.y_max_lo, self.y_max_cl,self.y_max_vo




# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------
# Get the prediction each of items/price & volume, started from history data until n days forecast
# ARMA/ ARIMA
# instance: is instance of class BuildModel
# item: is stock prices and volume:
# e.g: 'Open', 'High', 'Low', 'Close', 'Volume'
# d: is from p, d, q. d=0 for ARMA, and d=1 for ARIMA
def item_prediction(instance, d, item='Close'):
    ts, n_days_real, n_days_forecast = None, None, None

    if d == 0:
        # full prediction based on length sample timeseries data,
        # from start date history data until last date, it is not include forecast for next n days
        # ts, predicted = instance.predict_arma_single_item(item)
        ts, n_days_real, n_days_forecast = instance.predict_arma_next_days(item)
    else:
        # full prediction based on length sample timeseries data,
        # from start date history data until last date, it is not include forecast for next n days
        # ts, predicted = instance.predict_arima_single_item(item)
        # n_days forecast, here only forecast for next n days (without history data)
        ts, n_days_real, n_days_forecast = instance.predict_arima_next_days(item)
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
kdj = True # True/False
kdj_short_period = 3
kdj_long_period = 5
output = 'Close'

# Start Produce Weight for GA
print 'Produce trX and trY for GA.'

fc_data, real_data = [], []
start = 27 # 7
stop = 1207 # 208
for fc in range(start, stop): # fc = forecast n days
    an = BuildModel(p, d, q, n_days, fc,kdj,kdj_short_period,kdj_long_period,output)
    ts_real_op, ts_fc_op, n_days_real_op, n_days_fc_op = item_prediction(an, d, 'Open')
    ts_real_hi, ts_fc_hi, n_days_real_hi, n_days_fc_hi = item_prediction(an, d, 'High')
    ts_real_lo, ts_fc_lo, n_days_real_lo, n_days_fc_lo = item_prediction(an, d, 'Low')
    ts_real_cl, ts_fc_cl, n_days_real_cl, n_days_fc_cl = item_prediction(an, d, 'Close')
    ts_real_vo, ts_fc_vo, n_days_real_vo, n_days_fc_vo = item_prediction(an, d, 'Volume')
    fc_data.append([ n_days_fc_op, n_days_fc_hi, n_days_fc_lo, n_days_fc_cl, n_days_fc_vo ])
    real_data.append([ n_days_real_op,n_days_real_hi,n_days_real_lo,n_days_real_cl,n_days_real_vo])

#get kdj and real data normalization
real_op, real_hi, real_lo, real_cl, real_vo=[], [], [], [], []
for data in real_data:
    real_op.append(data[0])
    real_hi.append(data[1])
    real_lo.append(data[2])
    real_cl.append(data[3])
    real_vo.append(data[4])
# Real_x_train, Real_y_train = an.get_stock_without_transformation(real_op, real_hi, real_lo, real_cl, real_vo)
Real_x_train, Real_y_train = an.get_stock_with_transformation(real_op, real_hi, real_lo, real_cl, real_vo)
# print Real_x_train, Real_y_train 

max_list = []  #max_op, max_hi, max_lo, max_cl,max_vo,max_K,max_D,max_J,so 8 parameter in Real_x_train and Fc_x_train
# its for re-transform, back to original value of stock price
max_op, max_hi, max_lo, max_cl,max_vo,max_K,max_D,max_J= an.get_X_max()
max_list.append([max_op, max_hi, max_lo, max_cl,max_vo,max_K,max_D,max_J])

# forcast data Normalization and K,D,J
fc_op, fc_hi, fc_lo, fc_cl, fc_vo=[], [], [], [], []
for data in fc_data:
    fc_op.append(data[0]/max_op)
    fc_hi .append(data[1]/max_hi)
    fc_lo .append(data[2]/max_lo)
    fc_cl .append(data[3]/max_cl)
    fc_vo .append(data[4]/max_vo)
Fc_x_train, Fc_y_train = an.get_stock_with_transformation(fc_op, fc_hi, fc_lo, fc_cl, fc_vo)
# print Fc_x_train
np.save('max_list',np.asarray(max_list).astype(np.float32))
np.save('Real_x_train',np.asarray(Real_x_train).astype(np.float32))
np.save('Fc_x_train',np.asarray(Fc_x_train).astype(np.float32))