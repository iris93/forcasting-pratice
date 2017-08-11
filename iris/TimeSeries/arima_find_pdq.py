#-*-coding:utf-8-*-2
# please visit http://blog.csdn.net/u010414589/article/details/49622625 
from __future__ import print_function
import os
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


# dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422, 6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355, 10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767, 12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232, 13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248, 9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722, 11999,9390,13481,14795,15845,15271,14686,11054,10395]

# print (len(dta))
# # initial the data series
# dta=np.array(dta,dtype=np.float)
# dta=pd.Series(dta)
# print (dta)
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2090'))
# print (dta)
# dta.plot(figsize=(12,8))
# plt.title('original data')
# plt.show()

'''
=====================you can initialize your data and  make it similar to example  above  =====
'''

# dta =  np.load('trdata.npy')
# length=len(dta)
# dta=np.array(dta,dtype=np.float)
# dta=pd.Series(dta)
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700','1803'))

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
dta = df_train['Open']
'''
# to decide which difference  ======================================  choose d
'''
# data difference--first
# fig = plt.figure(figsize=(12,8))
# ax1= fig.add_subplot(111)
# diff1 = dta.diff(1)
# diff1.plot(ax=ax1)
# plt.title('diff1 data')
# plt.show()

# data difference--second
# fig = plt.figure(figsize=(12,8))
# ax2= fig.add_subplot(111)
# diff2 = dta.diff(2)
# diff2.plot(ax=ax2)
# plt.title('diff2 data')
# plt.show()
d = 3
dta= dta.diff(d)     #maybe we can choose d according to the acf and pacf plot
dta = dta[d:300]    #need to change dta form to fit the following steps.
'''
#===========================================================  choose  p,q
'''
fig = plt.figure("before ARMA, d:{}".format(d),figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
plt.title('Autocorrelation---------------------find q')
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
plt.title('Partial Autocorrelation-------------find p')
plt.show()

# choose the model with smallest aic,bic and hqic============test p q
arma_mod20 = sm.tsa.ARMA(dta,(7,0)).fit()
arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
arma_mod40 = sm.tsa.ARMA(dta,(7,1)).fit()
arma_mod50 = sm.tsa.ARMA(dta,(8,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)

'''
# according to aic,bic and hqic we choose arma_mode20 as our final model ==========================================================model test
'''
#在指数平滑模型下，观察ARIMA模型的残差是否是平均值为0且方差为常数的正态分布（服从零均值、方差不变的正态分布），同时也要观察连续残差是否（自）相关。
resid = arma_mod20.resid
fig = plt.figure('after ARMA',figsize=(12,8))
ax3 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax3)
ax4 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax4)
plt.show()
# 德宾-沃森（Durbin-Watson）检验。德宾-沃森检验,简称D-W检验，是目前检验自相关性最常用的方法，但它只使用于检验一阶自相关性。因为自相关系数ρ的值介于-1和1之间，所以 0≤DW≤４。并且DW＝O＝＞ρ＝１　　 即存在正自相关性
#检验结果是2.02424743723，说明不存在自相关性。
print(sm.stats.durbin_watson(arma_mod20.resid.values))

'''
# ===================================================观察是否符合正态分布
'''
# 这里使用QQ图，它用于直观验证一组数据是否来自某个分布，或者验证某两组数据是否来自同一（族）分布。在教学和软件中常用的是检验数据是否来自于正态分布。QQ图细节，下次再更。
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.title('qqplot')
plt.show()
'''
# =====================================================Ljung-Box检验
'''
# Ljung-Box test是对randomness的检验,或者说是对时间序列是否存在滞后相关的一种统计检验。对于滞后相关的检验，我们常常采用的方法还包括计算ACF和PCAF并观察其图像，但是无论是ACF还是PACF都仅仅考虑是否存在某一特定滞后阶数的相关。LB检验则是基于一系列滞后阶数，判断序列总体的相关性或者说随机性是否存在
# 时间序列中一个最基本的模型就是高斯白噪声序列。而对于ARIMA模型，其残差被假定为高斯白噪声序列，所以当我们用ARIMA模型去拟合数据时，拟合后我们要对残差的估计序列进行LB检验，判断其是否是高斯白噪声，如果不是，那么就说明ARIMA模型也许并不是一个适合样本的模型。
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
# 检验的结果就是看最后一列前十二行的检验概率（一般观察滞后1~12阶），如果检验概率小于给定的显著性水平，比如0.05,0.10等就拒绝原假设，其原假设是相关系数为零。就结果来看，如果取显著性水平为0.05，那么相关系数与零没有显著差异，即为白噪声序列。

# '''
# ==========================================================模型预测

# predict_sunspots = arma_mod20.predict('2090', '2100', dynamic=True)
# predict_sunspots = arma_mod20.predict('1803', '1813', dynamic=True)
# print(predict_sunspots)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax = dta.ix['2001':].plot(ax=ax)
# ax = dta.ix['1700':].plot(ax=ax)
# predict_sunspots.plot(ax=ax)
# plt.title('final prediction')
# plt.show()


'''
# *********************************************summarize****************************************

# so we can see the key steps are below:

# step 1:data initialize
# line 19
# step 2:diff
d= 1
dta= dta.diff(d)
dta= dta[1:]
# step 3:ARMA
p=7
q=0
arma_mod20 = sm.tsa.ARMA(dta,(p,q)).fit()
# step 4:predict
predict_sunspots = arma_mod20.predict('2090', '2100', dynamic=True)
print(predict_sunspots)
# '''
