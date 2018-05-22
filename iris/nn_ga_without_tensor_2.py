# coding: utf-8
# http://iamtrask.github.io/2015/07/12/basic-python-network/
# http://outlace.com/Simple-Genetic-Algorithm-in-15-lines-of-Python/
import random
import numpy as np

from preparing_input import InitPreprocessing
init_instance = InitPreprocessing()

# Configuration
output='close' # stock prices option: open, high, low, close
kdj='yes' # or kdj='no'
kdj_style='google' # or kdj_style='', kdj_short_period=3, kdj_long_period=5
kdj_short_period=3 # D (Divergen)
kdj_long_period=5  # K

X_train, y_train, X_test, y_test = init_instance.get_stock_transform(output=output, kdj=kdj, kdj_style=kdj_style, kdj_short_period=kdj_short_period, kdj_long_period=kdj_long_period)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test  = np.asarray(X_test)
y_test  = np.asarray(y_test)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def runForward_propagation(X, W):
	return sigmoid(np.dot(X, W))

# target - y
def costFunction(X, y, W):
	m = float(len(X))
	hThetaX = np.array(runForward_propagation(X, W))
	return np.sum(np.abs(y - hThetaX))


population  = 100
mut_rate    = 0.05 # 5%
generations = 500
chromo      = 8 # chromosom/solution length
winner      = 20 # winner/per gen
# params = [100, 0.05, 250, 3, 20]

# Init current population
# Create random -15<r<15 , rows=population=100 , cols=chromo=8
current_population = np.random.choice(np.arange(-15, 15, step=0.01), size=(population, chromo), replace=False)
next_population = np.zeros((current_population.shape[0], current_population.shape[1]))
fitVec = np.zeros((population, 2)) # 1st col is indices, 2nd col is cost

for i in range(generations):

	# fitVec = np.array([np.array([x, np.sum(costFunction(X_train, y_train, current_population[x].reshape(8,1)))]) for x in range(population)])
	fitVec, cF = [], 0
	for i in range(population):
		cF += costFunction(X_train, y_train, current_population[i].reshape(8, 1))
		fitVec.append( np.array([ i, cF ]) )
	fitVec = np.array(fitVec)

	print( '(Gen: {0}) Total error: {1}\n'.format(i, np.sum(fitVec[:,1])) )

	# hold our winning solutions until we move them to next generation
	winners = np.zeros((winner, chromo))

	for n in range(len(winners)):
		selected = np.random.choice(range(len(fitVec)), winner/2, replace=False)
		wnr = np.argmin(fitVec[selected,1])

		# print 'selected: %s' % selected
		# print 'wnr: %s' % wnr
		# print 'fitVec[selected]: %s' % fitVec[selected]
		# print 'fitVec[selected,1]: %s' % fitVec[selected,1]
		# print 'selected[wnr]: %s' % selected[wnr]
		# print 'fitVec[selected[wnr]]: %s' % fitVec[selected[wnr]]
		# print 'fitVec[selected[wnr]][0]: %s' % fitVec[selected[wnr]][0]
		# break
	# break
		winners[n] = current_population[int(fitVec[selected[wnr]][0])]

	# populate new gen with winners
	# next_population is the array containing all the solutions for the next generation.
	next_population[:len(winners)] = winners


	# next_population[len(winners):] = np.array([np.array(np.random.permutation(np.repeat(winners[:, x], ((population - len(winners))/len(winners)), axis=0))) for x in range(winners.shape[1])]).T
	nrperm_list = []
	for x in range(winners.shape[1]):
		nrepeat = np.repeat( winners[:, x], ( (population - len(winners)) / len(winners) ), axis=0 )
		nrperm  = np.random.permutation( nrepeat )
		nrperm_list.append( np.array( nrperm ) )
	next_population[len(winners):] = np.array( nrperm_list ).T


	# randomly mutate part of the population
	# current_population = np.multiply(next_population, np.matrix([np.float(np.random.normal(0, 2, 1)) if random.random() < mut_rate else 1 for x in range(next_population.size)]).reshape(next_population.shape))
	nf_list = []
	for x in range(next_population.size):
		nf = None
		if random.random() < mut_rate:
			nf = np.float(np.random.normal(0, 2, 1))
		else:
			nf = 1
		nf_list.append(nf)
	current_population = np.multiply( next_population, np.matrix(nf_list).reshape(next_population.shape) )


best_soln = current_population[np.argmin(fitVec[:,1])]
result = np.round( runForward_propagation(X_train, best_soln.reshape(8,1)) )

print( "Best Sol'n:\n{0}\nCost:{1}".format(best_soln, np.sum(costFunction(X_train, y_train, best_soln.reshape(8,1)))) )
# print( "When X = \n{0} \nhThetaX = \n{1}" % (X_train[:,:8], result,))

print ('result: {}'.format(result))
