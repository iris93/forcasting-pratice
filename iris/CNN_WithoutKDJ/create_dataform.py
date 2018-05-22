# create dataform.py
import tensorflow as tf
import numpy as np
import pandas as pd
from preparing_input import InitPreprocessing

sess = tf.Session()
init_instance = InitPreprocessing()
# X_train, y_train, X_test, y_test = init_instance.get_tensor()
X_train, y_train, X_test, y_test = init_instance.get_tensor_transform()
# X_train, y_train, X_test, y_test = init_instance.get_normal_matrix()

# create training set,trX,trY
X_set = []
trX = []
trY = []
for i in range(0,len(X_train)-5):
	for j in range(0,5):
		X_set.append(X_train[i+j])
	trX.append(X_set)
	X_set = []
	open, high, low, close, volume = X_train[i+5]
	if close-open>0:
		trY.append([1])
	else:
		trY.append([0])
print np.shape(trX),np.shape(trY)

np.save('trX',np.asarray(trX).astype(np.float32))
np.save('trY',np.asarray(trY).astype(np.float32))

# create testing set,teX,teY
teX = []
teY = []
X_teset = []
X_teset1 = []
for i in range(0,len(X_test)-5):
	for j in range(0,5):
		X_teset1.append(X_test[i+j])
	# print X_teset1
	# print "i:",i
	teX.append(X_teset1)
	X_teset = []
	X_teset1 =[]
	open, high, low, close, volume = X_test[i+5]
	if close-open>0:
		teY.append([1])
	else:
		teY.append([0])
print "teX",teX[0:2]
print np.shape(teX),np.shape(teY)
np.save('teX',np.asarray(teX).astype(np.float32))
np.save('teY',np.asarray(teY).astype(np.float32))

