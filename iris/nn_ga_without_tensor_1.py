# coding: utf-8
from itertools import izip
import random 
import numpy as np

from preparing_input import InitPreprocessing
init_instance = InitPreprocessing()

# Configuration
output='close' # stock prices option: open, high, low, close
kdj='no' # or kdj='no'
kdj_style='google' # or kdj_style='', kdj_short_period=3, kdj_long_period=5
kdj_short_period=3 # D (Divergen)
kdj_long_period=5  # K

X_train, y_train, X_test, y_test = init_instance.get_stock_transform(output=output, kdj=kdj, kdj_style=kdj_style, kdj_short_period=kdj_short_period, kdj_long_period=kdj_long_period)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

np.random.seed(1)
syn0 = 2*np.random.random((5,1)) -1

def runForward_propagation(X, theta):
	return sigmoid(np.dot(X, theta))

# target - y
def costFunction(X, y, theta):
	m = float(len(X))
	hThetaX = np.array(runForward_propagation(X, theta))
	return np.sum(np.abs(y - hThetaX))

def sigmoid(x):
	return 1 / (1 + np.exp(-x))




# Start [Init pop]
# pop = 100
# mut rate = 5%
# num generations = 250
# chromosome/solution length = 3
# winners/per gen = 20
params = [100, 0.05, 500, 5, 20]

# initialize current population to random values within
# range for initial solutions
curPop = np.random.choice(np.arange(-15,15,step=0.01), size=(params[0],params[3]), replace=False)

nextPop = np.zeros((curPop.shape[0], curPop.shape[1]))

fitVec = np.zeros((params[0], 2)) # 1st col is indices, 2nd col is cost

# iterate through num generations
for i in range(params[2]):
	# Calculate cost/error of each solution and add it to fitVec
	# Each element of fitVec is an array consisting of the index of the solution in curPop and its cost, e.g.[0, 2.54] means
	# that the 0th element in curPop(first solution) has an error of 2.54

	# fitVec = np.array(
	# 	[
	# 	np.array(
	# 		[
	# 		x, np.sum(costFunction(X_train, y_train, curPop[x].reshape(5,1)))
	# 		]
	# 		) for x in range(params[0])
	# 	]
	# 	)
	# fitVec = np.array([np.array([x, np.sum(costFunction(X_train, y_train, curPop[x].reshape(5,1)))]) for x in range(params[0])])

	fitVec, cF = [], 0
	for x in range(params[0]):
		cF += costFunction(X_train, y_train, curPop[x].reshape(5,1))
		fitVec.append( np.array([ x, cF ]) )
	fitVec = np.array(fitVec)

	print("(Gen: #%s) Total error: %s\n" % (i, np.sum(fitVec[:,1])))

	# hold our winning solutions until we move them to next generation
	winners = np.zeros((params[4], params[3])) # 20x5

	# for n in range(20)
	for n in range(len(winners)):
		# Now we're in a loop to populate the winners matrix
		selected = np.random.choice(range(len(fitVec)), params[4]/2, replace=False) # params[4]=20
		# we will randomly choose a subset of our population, and then choose the best sol from that subset and add to our winners array
		wnr = np.argmin(fitVec[selected,1])

		# So the array 'selected' contains 10 random solutions (actually the indices to 10 solutions) from our population.
		# Now we reference fitVec to find the actual elements, use np.argmin() to pick the one with the smallest error/cost
		# and assign the index of that winning element to the variable,'wnr'

		# we reference the winner in curPop, the array of all solutions of the current generation, and copy it to our 'winners' array.
		winners[n] = curPop[int(fitVec[selected[wnr]][0])]

	nextPop[:len(winners)] = winners #populate new gen with winners
	#nextPop is the array containing all the solutions for the next generation.
	#We populate the first 20 elements of nextPop with our winning solutions from 'winners' array.

	# nextPop[len(winners):] = np.array([np.array(np.random.permutation(np.repeat(winners[:, x], ((params[0] - len(winners))/len(winners)), axis=0))) for x in range(winners.shape[1])]).T
	# nextPop[len(winners):] = np.array(
	# 	[
	# 	np.array(
	# 		np.random.permutation(
	# 			np.repeat(winners[:, x], ( (params[0] - len(winners)) /len(winners) ), axis=0)
	# 		)
	# 	) for x in range(winners.shape[1])
	# 	]
	# 	).T

	nrperm_list = []
	for x in range(winners.shape[1]):
		nrepeat = np.repeat( winners[:, x], ( (params[0] - len(winners)) / len(winners) ), axis=0 )
		nrperm  = np.random.permutation( nrepeat )
		nrperm_list.append( np.array( nrperm ) )
	nextPop[len(winners):] = np.array( nrperm_list ).T


	#Populate the rest of the generation with offspring of mating pairs
	# np.repeat() will duplicate our 20x3 matrix to create a 80x3 matrix.
	# We already populated the first 20 elements of nextPop with the winners from last generation.
	# Now we want to populate the last 80 elements with their offspring. and np.random.perm will do the necessary crossover function.
	
	# curPop = np.multiply(nextPop, np.matrix([np.float(np.random.normal(0, 2, 1)) if random.random() < params[1] else 1 for x in range(nextPop.size)]).reshape(nextPop.shape))
	# curPop = np.multiply(
	# 	nextPop, np.matrix(
	# 		[
	# 		np.float(np.random.normal(0, 2, 1)) if random.random() < params[1] else 1 for x in range(nextPop.size)
	# 		]
	# 	).reshape(nextPop.shape)
	# )
	#randomly mutate part of the population
	nf_list = []
	for x in range(nextPop.size):
		nf = None
		if random.random() < params[1]:
			nf = np.float(np.random.normal(0, 2, 1))
		else:
			nf = 1
		nf_list.append(nf)
	curPop = np.multiply( nextPop, np.matrix(nf_list).reshape(nextPop.shape) )




best_soln = curPop[np.argmin(fitVec[:,1])]

#X = np.array([[0,1,1],[1,1,1],[0,0,1],[1,0,1]])
result = np.round(runForward_propagation(X_train, best_soln.reshape(5,1)))
# print("Best Sol'n:\n%s\nCost:%s" % (best_soln,np.sum(costFunction(X_train, y_train, best_soln.reshape(5,1)))))
# print("When X = \n%s \nhThetaX = \n%s" % (X_train[:,:5], result,))
