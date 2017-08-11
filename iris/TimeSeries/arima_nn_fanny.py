
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
filename = os.path.join(basedir, 'stock.xlsx')
xls = pd.ExcelFile(filename)

# This version only support for training, not support for testing
df_train = xls.parse('Sheet1', index_col='Date') # train
df_test  = xls.parse('Sheet2', index_col='Date') # test


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
        n_days=1,           # How many days a head to be predicted
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


    # -------------------------------------------------------------
    # 1st Section
    # ARMA/ ARIMA Model

    # predict single parameter by arima for X_train or X_test
    # item: 'Open', 'High', 'Low', 'Close', 'Volume'
    # Xy: 'train', 'test'
    # http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
    def predict_arima_single_item(self, item, Xy='train'):
        ts = df_train[item] if Xy=='train' else df_test[item]
        ts = ts.sort_index()

        # shifting
        # ts_diff = ts - ts.shift(1)
        # ts_diff.dropna(inplace=True)
        # model = ARIMA(ts_diff, order=(self.p, self.d, self.q))
        
        # build a model
        model = ARIMA(ts, order=(self.p, self.d, self.q))
        # fitting model
        fitting = model.fit(disp=False)
        print ( "ARIMA: fitting history: '{}'.".format(item) )
        # typ: levels, is selected due to 'Yahoo!' sometimes is not provide daily stock price, then the interval is not always daily
        predicted = fitting.predict(typ='levels')
        print ( "ARIMA: predict level history: '{}'.".format(item) )
        # concatenated = pd.concat([ts, predicted], axis=1, keys=['original', 'predicted'])
        # return concatenated.head(10)
        # return concatenated

        plt.title("timeseries history: '{}'".format(item))
        plt.plot(ts, label="Original: '{}'".format(item))
        plt.plot(predicted, label="Prediction: '{}'".format(item), color='red')
        plt.legend(loc='best')
        plt.show()
        
        return ts, predicted
  
    # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima_model.py
    def predict_arima_next_days(self, item, Xy='train'):
        ts = df_train[item] if Xy=='train' else df_test[item]
        # sorting index Date
        ts = ts.sort_index()
        # build a model
        model = ARIMA(ts, order=(self.p, self.d, self.q))
        # fitting model
        fitting = model.fit(disp=False)
        print ( "ARIMA: fitting model: '{}'.".format(item) )

        # compute the standard errors (from git source)
        # fcasterr = self._forecast_error(steps)
        # conf_int = self._forecast_conf_int(forecast, fcasterr, alpha)
        forecast, fcasterr, conf_int = fitting.forecast(steps=self.n_days, alpha=.05)
        print ( "ARIMA: forecast '{0}' for next {1} days.".format(item, self.n_days))
        return ts, forecast


    # http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.tsa.arima_model.ARMAResults.plot_predict.html
    # http://stackoverflow.com/questions/18616588/arma-out-of-sample-prediction-with-statsmodels
    # http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.tsa.arima_model.ARMAResults.predict.html

    # predict by arma, without d (differencing), then recent value is not lost
    # predict single parameter by arma for X_train or X_test
    # item: 'Open', 'High', 'Low', 'Close', 'Volume'
    # Xy: 'train', 'test'
    
    def predict_arma_single_item(self, item, Xy='train'):
        ts = df_train[item] if Xy=='train' else df_test[item]
        # sorting index Date
        ts = ts.sort_index()
        # build a model
        model = ARMA(ts, order=(self.p, self.q))
        # fitting model
        fitting = model.fit(disp=False)
        print ( "ARMA: fitting history: '{}'.".format(item) )
        # predict length is equal to original data
        predicted = fitting.predict()
        print ( "ARMA: predict history: '{}'.".format(item) )
        # concatenated = pd.concat([ts, predicted], axis=1, keys=['original', 'predicted'])
        # return concatenated

        plt.title("timeseries history: '{}'".format(item))
        plt.plot(ts, label="Original: '{}'".format(item))
        plt.plot(predicted, label="Prediction: '{}'".format(item), color='red')
        plt.legend(loc='best')
        plt.show()

        return ts, predicted

    # forecasting for n_days (new prediction n steps a head)
    # http://stackoverflow.com/questions/18616588/arma-out-of-sample-prediction-with-statsmodels
    # http://codego.net/9067598/
    def predict_arma_next_days(self, item, Xy='train'):
        ts = df_train[item] if Xy=='train' else df_test[item]
        # sorting index Date
        ts = ts.sort_index()
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
        print ( "ARMA: fitting model: '{}'.".format(item) )
        # preparing input for forecasting out of sample data
        params = fitting.params
        residuals = fitting.resid
        p = fitting.k_ar
        q = fitting.k_ma
        k_exog = fitting.k_exog
        k_trend = fitting.k_trend
        # n_days forecasting
        forecast = _arma_predict_out_of_sample(params, self.n_days, residuals, p, q, k_trend, k_exog, endog=ts, exog=None, start=len(ts))
        print ( "ARMA: forecast '{0}' for next {1} days.".format(item, self.n_days))
        return ts, forecast
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


        if self.kdj:
            K, D, J = self.kdj_indicator(rsv_list)
            # Convert from pandas DataFrame to a list
            K = map( float, K[0].tolist() )
            D = map( float, D[0].tolist() )
            J = map( float, J[0].tolist() )

            plt.title('KDJ Indicator')
            plt.plot(K, label='K')
            plt.plot(D, label='D')
            plt.plot(J, label='J')
            plt.legend(loc='best')
            plt.show()

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
        return self.y_max_op, self.y_max_hi, self.y_max_lo, self.y_max_cl


    # -------------------------------------------------------------
    # 4th Section
    # Neural Network

    # Weight Initialization
    @staticmethod
    def init_weight(shape, name):
        init = tf.random_uniform(shape, -1.0, 1.0)
        return tf.Variable(init, name=name)

    # Bias Initialization
    @staticmethod
    def init_bias(shape, name):
        init = tf.zeros(shape)
        return tf.Variable(init, name=name)

    # Model definition
    @staticmethod
    def NN_model(X, W, B):
        m = tf.matmul(X, W) + B
        # Relu for instead sigmoid, Sigmoid only for final layer
        L = tf.nn.relu(m)
        return L




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
    ts, predicted, n_days_forecast = None, None, None

    if d == 0:
        # full prediction based on length sample timeseries data,
        # from start date history data until last date, it is not include forecast for next n days
        ts, predicted = instance.predict_arma_single_item(item)
        print ( 'Length data before ARMA n days forecast: {}'.format(len(predicted)) )

        # n_days forecast, here only forecast for next n days (without history data)
        ts, n_days_forecast = instance.predict_arma_next_days(item)

    else:
        # full prediction based on length sample timeseries data,
        # from start date history data until last date, it is not include forecast for next n days
        ts, predicted = instance.predict_arima_single_item(item)
        print ( 'Length data before ARIMA n days forecast: {}'.format(len(predicted)) )

        # n_days forecast, here only forecast for next n days (without history data)
        ts, n_days_forecast = instance.predict_arima_next_days(item)


    # join n_days_forecast into ts
    ts = ts.append(pd.DataFrame(n_days_forecast))

    # join n_days_forecast into full prediction
    predicted = predicted.append(pd.DataFrame(n_days_forecast))

    print ( 'Length data after ARMA predict n days: {}'.format(len(predicted)) )
    print ( "AR/IMA '{0}' data n days forecast are: {1}".format(item, n_days_forecast) )

    ts = map( float, ts[0].tolist() )
    print ( "Get ts: '{}'' full prediction...".format(item) )

    predicted = map ( float, predicted[0].tolist() )
    print ( "Get predicted: '{}' full prediction...".format(item) )

    return ts, predicted


# -----------------------------------------------------------
print 'Started...'
# Configuration ARMA/ARIMA, KDJ and Output target price
# NOTICE:
#    d=1 , automatically use ARIMA
#    d=0 , automatically use ARMA
p, d, q = 1, 0, 0
n_days = 3 # How many n steps/days prediction
kdj = True # True/False
kdj_short_period = 3
kdj_long_period = 5
output = 'Close'

# NN
n_nodes_in_hidden = 10 # number of nodes in hidden layer
learning_rate = 1.0
epochs = 20000 # number of iteration
7-------------------------------------------------------------------------------------99999999999999999999999999999999999999999888888888888888887878777777
# -----------------------------------------------------------
# Create an instance
an = BuildModel(
    p=p, d=d, q=q, n_days=n_days,
    kdj=kdj,
    kdj_short_period=kdj_short_period,
    kdj_long_period=kdj_long_period,
    output=output)

# -----------------------------------------------------------
# ARMA/ ARIMA
# NOTICE/ WARNING:
# Be carefull Excel sample data, Yahoo! in certain line has no data alias '-'
# It must be changed with other value
# numpy cannot cast dtype('float64') to dtype('int64'), then make sure
# all the value is in floating-point (especially stock 'Volume')

# stock 'Open' price
# result is data ts (ts_op) and prediction (pr_op)
ts_op, pr_op = item_prediction(instance=an, d=d, item='Open')

# stock 'High' price
# result is data ts (ts_hi) and prediction (pr_hi)
ts_hi, pr_hi = item_prediction(instance=an, d=d, item='High')

# stock 'Low' price
# result is data ts (ts_lo) and prediction (pr_lo)
ts_lo, pr_lo = item_prediction(instance=an, d=d, item='Low')

# stock 'Close' price
# result is data ts (ts_cl) and prediction (pr_cl)
ts_cl, pr_cl = item_prediction(instance=an, d=d, item='Close')

# stock 'Volume'
# result is data ts (ts_vo) and prediction (pr_vo)
ts_vo, pr_vo = item_prediction(instance=an, d=d, item='Volume')


# -----------------------------------------------------------
# Load X inputs and y output
# without transformation
# X_train, y_train = an.get_stock_without_transformation(ts_op, ts_hi, ts_lo, ts_cl, ts_vo)
# X_train, y_train = an.get_stock_without_transformation(pr_op, pr_hi, pr_lo, pr_cl, pr_vo)

# with transformation
# X_train, y_train = an.get_stock_with_transformation(ts_op, ts_hi, ts_lo, ts_cl, ts_vo)
X_train, y_train = an.get_stock_with_transformation(pr_op, pr_hi, pr_lo, pr_cl, pr_vo)

# Get y max
# its for re-transform, back to original value of stock price
y_max_op, y_max_hi, y_max_lo, y_max_cl = an.get_y_max()



# -----------------------------------------------------------
# Neural Network
print ( 'Start NN...' )
# INPUT Nodes ----
X = None
if kdj:
    # Tensor [None x 8]
    # X = tf.placeholder('float32', [42,8], name='Input')
    X = tf.placeholder('float32', [None,8], name='Input')
else:
    # Tensor [None x 5]
    # X = tf.placeholder('float32', [42,5], name='Input')
    X = tf.placeholder('float32', [None,5], name='Input')
print ( 'Init: X Input nodes.')
# ----------------

# OUTPUT Nodes ---
# IMPORTANT: OUTPUT ROWS must equal to INPUT ROWS  <-----
# IF INPUT 42x5, THEN OUTPUT 42x1 (IN OUR CASE), Because W3 column = 1
# Y = tf.placeholder('float32', [42,1], name='Output')
Y = tf.placeholder('float32', [None,1], name='Output')
print ( 'Init: Y Output node.' )
# ----------------

# 1st Weight connected to INPUT
W1 = None
if kdj:
    # W1 = an.init_weight([8, 6], 'W1') # W1, 8x6
    W1 = an.init_weight([8, n_nodes_in_hidden], 'W1') # W1, 8 x n_nodes_in_hidden
else:
    # W1 = an.init_weight([5, 6], 'W1') # W1, 5x6
    W1 = an.init_weight([5, n_nodes_in_hidden], 'W1') # W1, 5 x n_nodes_in_hidden
B1 = an.init_bias([n_nodes_in_hidden], 'B1') # B1 col should be same to W1 col
print ( 'Init: W1 and B1.' )

# W2 = an.init_weight([6, 6], 'W2') # W2, 6x6
W2 = an.init_weight([n_nodes_in_hidden, n_nodes_in_hidden], 'W2') # W2, n_nodes_in_hidden x n_nodes_in_hidden
# B2 = an.init_bias([6], 'B2') # B2 col should be same to W2 col
B2 = an.init_bias([n_nodes_in_hidden], 'B2') # B2 col should be same to W2 col
print ( 'Init: W2 and B2.' )

# Last Weight connected to OUTPUT
# W3 = an.init_weight([6, 1], 'W3') # W3, 6x1
W3 = an.init_weight([n_nodes_in_hidden, 1], 'W3') # W3, n_nodes_in_hidden x 1
B3 = an.init_bias([1], 'B3') # B3 col should be same to W3 col
print ( 'Init: W3 and B3.' )
# -------------------------------------

# Compose a model ------
L2 = an.NN_model(X,  W1, B1)
print ( 'Build 1st Hidden Layer.' )
L3 = an.NN_model(L2, W2, B2)
print ( 'Build 2nd Hidden Layer.' )

# -------------------------------------
# Hypothesis/ target/ Y_model/ logits
# 1st, tf.sigmoid
hypothesis = tf.sigmoid(tf.matmul(L3, W3) + B3)
print ( 'Calculate hypothesis.' )
# 2nd, tf.nn.sigmoid
# hypothesis = tf.nn.sigmoid(tf.matmul(L3, W3) + B3) # more decreasing than tf.sigmoid
# 3rd, keep on Relu
# hypothesis = model(L3, W3, B3)
# -------------------------------------

# loss function ---------------------
with tf.name_scope('loss') as scope:
    # 1st, cross_entropy, on backpropagation
    loss = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))
    print ( 'Define lost function.' )
    # 2nd, MSE
    # loss = tf.reduce_mean(tf.pow(tf.sub(hypothesis, Y), 2.0))
    # 3rd, softmax_cross_entropy
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
# -----------------------------------

# -----------------------------------
# Only todo batch training with static input
with tf.name_scope('train') as scope:
    lr = tf.Variable(learning_rate) # Learning rate 
    optimizer = tf.train.GradientDescentOptimizer(lr)
    batch_train = optimizer.minimize(loss)
    print ( 'Init Optimizer.' )
# -----------------------------------

# -----------------------------------
# Launch the graph
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    # saver = tf.train.Saver()

    for epoch in range(epochs):
        # IMPORTANT: Training must be performed before other processes

        # sess.run(batch_train, febatch_ed_dict={ X: X_train, Y: y_train })
        _, loss_value = sess.run([batch_train, loss], feed_dict={ X: X_train, Y: y_train }) 
        # sess.run(loss, feed_dict={ X: X_train, Y: y_train })

        # print sess.run(W1) # The value of W1 is change
        # print sess.run(W2) # The value of W2 is change

        if epoch % 100 == 0:
            print ( 'Epoch: {0} Loss: {1}'.format(epoch, loss_value) )

    # correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    # # print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction], feed_dict={ X: X_train, Y: y_train })
    # # print 'hypothesis: %s' % sess.run(hypothesis, feed_dict={ X: X_train, Y: y_train })
    # # print 'hypothesis+0.5: %s' % sess.run(tf.floor(hypothesis+0.5), feed_dict={ X: X_train, Y: y_train })
    # print 'correct_prediction: %s' % sess.run(correct_prediction, feed_dict={ X: X_train, Y: y_train })
    # print 'accuracy: %s ' % sess.run(accuracy, feed_dict={ X: X_train, Y: y_train })


    # Prediction ------------------------------------------------------------
    # predict = sess.run(hypothesis, feed_dict={ X: X_test })
    predict = sess.run(hypothesis, feed_dict={ X: X_train })

    # yt = y_train
    # pr = prediction
    y, p, yt, pr = None, None, [], []

    for i, j in izip(y_train, predict):

        if output=='Open':
            y = i[0] * y_max_op
            p = j[0] * y_max_op

        elif output=='High':
            y = i[0] * y_max_hi
            p = j[0] * y_max_hi

        elif output=='Low':
            y = i[0] * y_max_lo
            p = j[0] * y_max_lo

        else:
            y = i[0] * y_max_cl
            p = j[0] * y_max_cl

        yt.append(y)
        pr.append(p)

    print ( '-'*40 )
    print ( 'y_train: {}'.format(yt) )
    print ( '-'*40 )
    print ( 'predict: {}'.format(pr) )
    
    plt.title('Prediction: {}'.format(output))
    if output=='Open':
        plt.plot(ts_op[kdj_long_period:], label='Original')
    elif output=='High':
        plt.plot(ts_hi[kdj_long_period:], label='Original')
    elif output=='Low':
        plt.plot(ts_lo[kdj_long_period:], label='Original')
    elif output=='Close':
        plt.plot(ts_cl[kdj_long_period:], label='Original')
    else: pass
    plt.plot(yt, label='AR/IMA', color='green')
    plt.plot(pr, label='AR/IMA+NN', color='red')
    plt.legend(loc='best')
    plt.show()
    
    print ( "{} days forecast of '{}' price by NN is: {}".format(n_days, output, pr[-n_days:]) )
    # n days forecast only
    arma_pred_output = None
    if output=='Open':
        # arma_pred_output = ts_op[-n_days:]
        arma_pred_output = pr_op[-n_days:]
    elif output=='High':
        # arma_pred_output = ts_hi[-n_days:]
        arma_pred_output = pr_hi[-n_days:]
    elif output=='Low':
        # arma_pred_output = ts_lo[-n_days:]
        arma_pred_output = pr_lo[-n_days:]
    elif output=='Close':
        # arma_pred_output = ts_cl[-n_days:]
        arma_pred_output = pr_cl[-n_days:]
    else: pass
    
    print ('Day\tAR/IMA\t\tNN')
    for i, nn_pred_output in enumerate(pr[-n_days:]):
        print ( '{}\t{}\t{}'.format(i+1, arma_pred_output[i], nn_pred_output) )



print ( 'FINISH.')

