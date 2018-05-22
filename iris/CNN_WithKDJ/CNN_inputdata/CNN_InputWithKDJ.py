# coding: utf-8
'''
Classic Neural Network Backpropagation
Consist of:
- 5 Input (Open, High, Low, Close, Volume)      OR
  8 Input (Open, High, Low, Close, Volume, K, D, J)
- 2 Hidden layers
- 10 Neurons in each Hidden layers
- 1 Output, selected option that is: Open, High, Low or Close
'''
from itertools import izip
 # Fauzi Fanny
import numpy as np
import urllib2

# step 1 : load data and split data into train part and test part
Real_x_train = np.load('Real_x_train.npy')
print len(Real_x_train),len(Real_x_train[0])

train_len = 500
test_len =100
test_len = train_len + test_len

X_train_real = Real_x_train[0:train_len]

X_test_real = Real_x_train[train_len:test_len]

# '''
# step 2 : create training set trX, trY
X_trset = []
trX = []
trY = []
for i in range(0,len(X_train_real)-8):
	for j in range(0,8):
		X_trset.append(X_train_real[i+j])
	trX.append(X_trset)
	X_trset = []
	open1, high1, low1, close1, volume1, K1, D1, J1 = X_train_real[i+7]
	open2, high2, low2, close2, volume2, K2, D2, J2 = X_train_real[i+8]
	if close2-close1>0:
		trY.append([1])
	else:
		trY.append([0])
print np.shape(trX),np.shape(trY)

np.save('trX',np.asarray(trX).astype(np.float32))
np.save('trY',np.asarray(trY).astype(np.float32))

# step 3 : create testing set,teX,teY
teX = []
teY = []
X_teset = []
for i in range(0,len(X_test_real)-8):
	for j in range(0,8):
		X_teset.append(X_test_real[i+j])
	teX.append(X_teset)
	X_teset =[]
	open1, high1, low1, close1, volume1, K1, D1, J1 = X_test_real[i+7]
	open2, high2, low2, close2, volume2, K2, D2, J2 = X_test_real[i+8]
	if close2-close1>0:
		teY.append([1])
	else:
		teY.append([0])
# print "teX",teX[0:2]
print np.shape(teX),np.shape(teY)
np.save('teX',np.asarray(teX).astype(np.float32))
np.save('teY',np.asarray(teY).astype(np.float32))
# '''
print teX, teY