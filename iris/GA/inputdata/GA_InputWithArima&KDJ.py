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
import pickle

# step 1 : load data and split data into train part and test part
Real_x_train = np.load('Real_x_train.npy')
Fc_x_train = np.load('Fc_x_train.npy')
max_list = np.load('max_list.npy')
print max_list
print len(Real_x_train),len(Real_x_train[0])
print len(Fc_x_train),len(Fc_x_train[0])
train_start = 0
train_end = 409
# test_len =108
# test_len = train_len + test_len
X_train_real = Real_x_train[train_start:train_end]
X_train_fc = Fc_x_train[train_start:train_end]

# X_test_real = Real_x_train[train_len:test_len]
# X_test_fc = Fc_x_train[train_len:test_len]

# '''
# step 2 : create training set trX, trY
X_trset = []
trX = []
trY = []
data_dem = 9
for i in range(0,len(X_train_real)-data_dem):
	for j in range(1,data_dem):
		# X_train_real[i+j].extend(X_train_real[i+j][3]-X_train_real[i+j-1][3])
		tmp=X_train_real[i+j].tolist()
		if X_train_real[i+j][3]-X_train_real[i+j-1][3]>0: t = [1]
		else:t=[0]
		tmp.extend(t)
		X_trset.extend(tmp)
	tmp =X_train_fc[i+data_dem].tolist()
	if X_train_fc[i+data_dem][5]-X_train_real[i+data_dem-1][6]: t= [1]
	else: t = [0]
	tmp.extend(t)
	X_trset.extend(tmp)
	# print X_trset
	trX.append(X_trset)
	X_trset = []
	open1, high1, low1, close1, volume1, K1, D1, J1 = X_train_real[i+data_dem-1]
	open2, high2, low2, close2, volume2, K2, D2, J2 = X_train_real[i+data_dem]
	# if close2-close1>0:
	# 	trY.append([1])
	# else:
	# 	trY.append([0])
	trY.append(close2)
print np.shape(trX),np.shape(trY)

train_data = {
	"trX":trX,
	"trY":trY}
output = open('train_data.pkl','wb')
pickle.dump(train_data,output)
output.close()
