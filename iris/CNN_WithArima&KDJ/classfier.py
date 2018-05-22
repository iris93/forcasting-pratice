# -*- coding: utf-8 -*-
# import bayes
import numpy 
from numpy import*
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import pickle
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

def dumpTree(tree, filename):
    with open(filename,'wb') as fp:
        pickle.dump(tree, fp)       

def loadTree(filename):
    with open(filename,'rb') as fp:
        return pickle.load(fp)
def classifytest():
    digits = datasets.load_digits()
    data_dir = 'CNN_inputdata/'
    train_x = numpy.load(data_dir+'xSample.npy')
    train_y = numpy.load(data_dir+'ySample.npy')
    # print train_x,train_y
    train_length = len(train_y)
    train_data    = []
    train_labels  = []
    for i in xrange(0,train_length-1):
        train_data.append(train_x[i])
        if train_y[i+1]<train_y[i]:
            train_labels.append([0])
        else :train_labels.append([1])
    digits.data,digits.target = train_data,train_labels
    train_data,test_data,train_target,test_target = train_test_split(digits.data, digits.target)
    # test_data,test_target = train_data,train_target
    # '''
    # bayes方法实现分类
    # clf = MultinomialNB().fit(train_data,train_target)
    # clf = BernoulliNB().fit(train_data,train_target)
    #  
    # clf = MultinomialNB().fit(digits.data,digits.target)

    # 实现svm分类
    clf = svm.SVC(gamma=0.001, C=100.)    #建立预测器
    clf.fit(train_data,train_target)

    joblib.dump(clf, 'classifier_model/svm_stock.pkl')
    clf_load = joblib.load('classifier_model/svm_stock.pkl')
    # '''

    # '''
    #knn分类
    # clf=KNeighborsClassifier(n_neighbors=3)
    # clf.fit(train_data,train_target)
    # dumpTree(clf, 'knntest_booklist2.pkl')
    # clf_load = loadTree('knntest_booklist2.pkl')
    # '''
    testResult=clf_load.predict(test_data)
    realResult = test_target
    testLen = len(testResult)
    errorCount = 0
    totalError = 0
    for i in range(len(testResult)):
        if testResult[i] != realResult[i]:
            errorCount +=1
    totalError += float(errorCount)/testLen
    totalRight = 1 - totalError
    print "number of testing:",testLen,"errorCount :",errorCount,"total right rate",totalRight

# 测试分类器
classifytest()