#-*-coding:utf-8-*-2
import pandas as pd 
import numpy as np 

# section 1: read data

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%M-%d')
data = []
data = pd.read_csv('5years_day.csv',  index_col='Date')

# print (data.Open)
# print (data.High)
# print (data.Low)
# print (data.Close)
# print (data.Volume)
# print (data.AdjClose)
# print data[0:10]
listdata = [[],[],[],[],[],[]]
length = len(data)
# length=1258
for i in xrange(length):
	temp = length-i-1
	# temp = i
	# listdata.append([data.index[temp],data.Open[temp],data.High[temp],data.Low[temp],data.Close[temp],data.Volume[temp]])
	listdata[0].append(data.index[temp])
	listdata[1].append(data.Open[temp])
	listdata[2].append(data.High[temp])
	listdata[3].append(data.Low[temp])
	listdata[4].append(data.Close[temp])
	listdata[5].append(data.Volume[temp])
print len(listdata),len(listdata[0])
print listdata[1]
# listdata = np.reshape(listdata,[7,1258])
# print listdata[0].transpose()

# section 2 :calcultate K,D,J
