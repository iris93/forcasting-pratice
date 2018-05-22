import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralNet import neuralNetwork

# number of input, hidden and output
input_nodes  = 5
hidden_nodes = 100
output_nodes = 1

# Check learning rate performance:
# test by 0.1, 0.2, ...
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load yahoo stock finance
# http://finance.yahoo.com/q/hp?s=%5Eixic+historical+prices
xls = pd.ExcelFile('yahoo.xlsx')
df1 = xls.parse('Sheet1')
df2 = xls.parse('Sheet2')
# idmax = argmax

max_open = df1.Open.max() * 0.99
max_high = df1.High.max() * 0.99
max_low = df1.Low.max() * 0.99
max_close = df1.Close.max() * 0.99
max_volume = df1.Volume.max() * 0.99

# training the neural network
# number of times the training data set is used for training
epoch = 1
SSE = []
MSE = []
error_sum = 0;
square_error = 1
target_error = 0.01 # max error target must be achieved

for i in range(10):
# while square_error > target_error:
	error_sum = 0

	for i, row in enumerate(df1.values):
		date, _open, _high, _low, _close, _volume, _adjclose = row
		_op = (_open/max_open) - 0.11
		_hi = (_high/max_high) - 0.11
		_lo = (_low/max_low) - 0.11
		_cl = (_close/max_close) - 0.11
		_vo = (_volume/max_volume) - 0.11

		# scales and shift the inputs
		inputs = np.array([ _op, _hi, _lo, _cl, _vo ])

		# create the target output value
		targets = np.zeros(output_nodes) - 0.11

		# all _close is target label
		targets[0] = _cl
		n.fit(inputs, targets)

	output_errors = n.final_error()[0][0]
	square_error = np.power(output_errors, 2)
	print (square_error, 'square')
	error_sum += square_error
	SSE.append(error_sum)
	MSE.append(error_sum/epoch)

	print '\rEpoch: %s' % epoch
	epoch += 1


print (MSE, 'MSE')
fig = plt.figure()
plt.plot(SSE)
plt.show(block=True)

plt.plot(MSE)
plt.show(block=True)


print ' ----- '

# test the neural network
for i, row in enumerate(df2.values):
	date, _open, _high, _low, _close, _volume, _adjclose = row
	_op = (_open/max_open) - 0.11
	_hi = (_high/max_high) - 0.11
	_lo = (_low/max_low) - 0.11
	_cl = (_close/max_close) - 0.11
	_vo = (_volume/max_volume) - 0.11

	# correct answer is _close
	real_value = _close
	print (real_value, "real value")

	# # scales and shift the inputs
	inputs = np.array([ _op, _hi, _lo, _cl, _vo ])

	# query the network
	outputs = n.predict(inputs)
	outputs = (outputs + 0.11) * max_close
	print (outputs, 'prediction value')

	print ' '
