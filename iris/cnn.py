# CNN.py
import tensorflow as tf
import numpy as np
import pandas as pd



trX = np.load('trX.npy')
trY = np.load('trY.npy')
teX = np.load('teX.npy')
teY = np.load('teY.npy')

# batch_size = 128
# test_size = 256

batch_size = 100
test_size = 5

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))





def model(X, w, w2, w3, w_o ,p_keep_conv, p_keep_hidden):


	l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 5, 5, 32)
		strides=[1, 1, 1, 1], padding='SAME'))
	l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 5, 5, 32)
		strides=[1, 2, 2, 1], padding='SAME')
	l1 = tf.nn.dropout(l1, p_keep_conv)



	l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                 # l2a shape=(?, 5, 5, 64)
		strides=[1, 1, 1, 1], padding='SAME'))
	l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 5 ,5, 64)
		strides=[1, 2, 2, 1], padding='SAME')
	l2 = tf.reshape(l2, [-1, w3.get_shape().as_list()[0]])    
	l2 = tf.nn.dropout(l2, p_keep_conv)

	# l3 = tf.nn.relu(tf.add(tf.matmul(l2, w3),bias3))
	l3 = tf.nn.relu(tf.matmul(l2, w3))
	l3 = tf.nn.dropout(l3, p_keep_hidden)

	# l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
	# 	strides=[1, 1, 1, 1], padding='SAME'))
	# l3 = tf.nn.max_pool(l3a, ksize=[1, 1, 1, 1],              # l3 shape=(?, 4, 4, 128)
	# 	strides=[1, 1, 1, 1], padding='SAME')
	# l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
	# l3 = tf.nn.dropout(l3, p_keep_conv)

	# l4 = tf.nn.relu(tf.matmul(l3, w4))
	# l4 = tf.nn.dropout(l4, p_keep_hidden)
	# pyx = tf.matmul(l4, w_o)


	pyx = tf.matmul(l3, w_o)
	# pyx = tf.add(tf.matmul(l3, w_o),out)
	return pyx


trX = trX.reshape(-1,5,5,1)
teX = teX.reshape(-1,5,5,1)

X = tf.placeholder("float", [None, 5, 5, 1])
Y = tf.placeholder("float", [None, 2])

w = init_weights([2, 2, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([2, 2, 32, 64])     # 3x3x32 conv, 64 outputs
# w3 = init_weights([2, 2, 64, 128])    # 3x3x32 conv, 128 outputs
# w4 = init_weights([128 * 4 * 4, 625]) # 
# w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)
w3 = init_weights([64*2*2, 1024])
w_o = init_weights([1024, 2]) 

bias1 = tf.Variable(tf.random_normal([32])),
bias2 = tf.Variable(tf.random_normal([64])),
bias3 = tf.Variable(tf.random_normal([1024])),
out = tf.Variable(tf.random_normal([2]))

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
# py_x = model(X, w, w2, w3, w_o, p_keep_conv, p_keep_hidden)
py_x = model(X, w, w2, w3, w_o, p_keep_conv, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
# predict_op = tf.argmax(py_x, 1)
predict_op = tf.equal(tf.argmax(py_x,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predict_op, tf.float32))


# Launch the graph in a session
with tf.Session() as sess:
	tf.initialize_all_variables().run()
	print type(trX)
	for i in range(100):
		training_batch = zip(range(0, len(trX), batch_size),range(batch_size, len(trX), batch_size))
		for start, end in training_batch:
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5})
			# print sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5})

		# test_indices = np.arange(len(teX)) # Get A Test Batch
		# np.random.shuffle(test_indices)
		# test_indices = test_indices[0:test_size]
		# print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
			# sess.run(predict_op, feed_dict={X: teX[test_indices],
			# 	Y: teY[test_indices],
			# 	p_keep_conv: 1.0,
			# 	p_keep_hidden: 1.0})))
		# print "py_x",sess.run(py_x, feed_dict={X: teX[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5})
			if i % 10 == 0:
				loss, acc = sess.run([cost, accuracy], feed_dict={X: trX[start:end],
					Y: trY[start:end],
					p_keep_conv: 1.0,
					p_keep_hidden: 1.0})
				print  " Minibatch Loss= " + \
	 				"{:.6f}".format(loss) + ", Training Accuracy= " + \
	 				"{:.5f}".format(acc)
	print "Optimization Finished!"

		# Calculate accuracy for 256 mnist test images
	print "Testing Accuracy:", \
		sess.run(accuracy, feed_dict={X: teX,
			Y: teY,
			p_keep_conv: 1.0,
			p_keep_hidden: 1.0})
	print "result", sess.run(py_x,feed_dict={X:teX,p_keep_conv: 1.0,p_keep_hidden: 1.0})
