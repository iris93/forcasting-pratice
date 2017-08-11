import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

# define the rule to get label of xt
def div(xt):
    label1 = int(abs(xt[0]) < 0.5)
    label2 = int(abs(xt[1]) < 0.5)
    return label1 + label2

# create the train data
def train_data():
    inputs = [[random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(100000)]
    labels = np.asarray([div(x_t) for x_t in inputs])
    labels = (np.arange(3) == labels[:, None]).astype(np.float32)

    print(inputs[0])
    print(div(inputs[0]))
    print(labels[0])
    return inputs, labels

# calculate the accuracy 
def accuracy(predictions, train_labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(train_labels, 1)) / predictions.shape[0]

# define the NN model
# dataset:x_train; train_labels: y_train; 
# batch_size: the size you put into training every time ,otherwise if you put all data at once,it will take very long time
# data_count:demesion of x_train;  label_count:demension of y_train
# num_steps: it depends on the length of dataset and batch_size,len(dataset)>=batch*num_steps
def dig_nn(dataset, train_labels, batch_size, data_count, label_count,num_steps):
    # define the graph
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, data_count))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, label_count))
        # define the hidden layer
        hidden_node_count = [10, 10]
        # define the input weight and input bias
        wi = tf.Variable(tf.truncated_normal([data_count, hidden_node_count[0]]))
        bi = tf.Variable(tf.zeros([hidden_node_count[0]]))

        y1 = tf.matmul(tf_train_dataset, wi) + bi
        h1 = tf.nn.relu(y1)

        # define the hidden0 weight and bias
        w0 = tf.Variable(tf.truncated_normal([hidden_node_count[0], hidden_node_count[1]]))
        b0 = tf.Variable(tf.zeros([hidden_node_count[1]]))

        y2 = tf.matmul(h1, w0) + b0
        h2 = tf.nn.relu(y2)

        # define the output weight and bias
        wo = tf.Variable(tf.truncated_normal([hidden_node_count[1], label_count]))
        bo = tf.Variable(tf.zeros([label_count]))

        # define the train_prediction,loss and optimizer
        logits = tf.matmul(h2, wo) + bo
        train_prediction = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # num_steps = 10
    # define session to run the graph
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        fig_index = 0;
        for step in range(num_steps):
            batch_data = dataset[step * batch_size: (step + 1) * batch_size]
            batch_labels = train_labels[step * batch_size: (step + 1) * batch_size]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            # run the optimizer,loss,train_prediction which defined before
            for iterationg in xrange(10000): 
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            # evaluation
            if step % 1 == 0:
                print('=' * 80)
                cur_first_data = dataset[step * batch_size: (step + 1) * batch_size][0]
                print('current first data [%f, %f]' % (cur_first_data[0], cur_first_data[1]))
                print('current first predict: [%f,%f]' % (predictions[0][0],predictions[0][1]))
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

if __name__ == '__main__':
    inputs, labels = train_data()
    dig_nn(inputs, labels, 100, 2, 3,1000)