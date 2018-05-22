import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../luyiying/tf/MNIST_data", one_hot=True)
#
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = - tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x: batch_xs, y_: batch_ys})
# interSess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float"))
# print accuracy.eval(feed_dict={x: mnist.test.images , y_: mnist.test.labels})
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
