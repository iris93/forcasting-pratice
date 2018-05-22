import tensorflow as tf
import numpy as np

# A demo shows how to run session
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[3]])
product = tf.matmul(matrix1,matrix2)

sess = tf.Session()

result = sess.run(product)
print result
sess.close()

'''
# Another way to calculate.
with tf.Session() as sess:
    result = sess.run(product)
    print result
'''
interSess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()
sub = tf.sub(x, a)
print sub.eval()

'''
# Creates a constant tensor.
# interSess = tf.InteractiveSession()
hello = tf.constant('Hello, TensorFlow!',name="hello")
print "hello:",hello
print hello.eval()
'''

# tensor = tf.constant([1,2,3,4,5,6,7])
tensor = tf.constant(-1.0,shape=[2,3])
print tensor.eval()
tensor_part = tensor[0,1]
print "tensor_part.eval:",tensor_part.eval()
'''
'''
# '''
#Creates a tensor with all elements set to zero.
zeros = tf.zeros([3, 4])
print "zeros:",zeros
print zeros.eval()
# '''
# Creates a tensor with all elements set to 1.
ones = tf.ones([4,5])
print "ones:",ones
print ones.eval()

# '''
# Creates a tensor with all elements set to zero.
zeros_like = tf.zeros_like(ones)
print "zeros_like:",zeros_like
print zeros_like.eval()

# '''
# Creates a tensor with all elements set to 1.
ones_like = tf.ones_like(zeros)
print "ones_like:",ones_like
print ones_like.eval()
# '''
# Creates a tensor filled with a scalar value.
fill_value = tf.fill([2,3],8,name="fill_value")
print "fill_value:",fill_value
print "fill_value.eval():"
print fill_value.eval()

# '''
# sequence
# Generates values in an interval= (stop - start) / num - 1
start =10.0
stop = 12.0
num = 3
linspace = tf.linspace(start, stop, num, name="linspace")
print "linspace.eval():"
print linspace.eval()

# '''
# Creates a sequence of integers that begins at start and extends by increments of delta up to but not including limit
start =3
limit = 18
delta = 3
rangetest1 = tf.range(start, limit, delta)
print rangetest1.eval()

rangetest2 = tf.range(0,5)
print rangetest2.eval()

# '''
# Random Tensors
# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([2, 3], mean=-1, stddev=4)
# Each time we run these ops, different results are generated
print "norm.eval():1"
print norm.eval()
print "norm.eval():2"
print norm.eval()

# '''
# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)
print "shuff.eval():"
print shuff.eval()

# '''
#  Set an op-level seed to generate repeatable sequences across sessions.
norm = tf.random_normal([2, 3], seed=1234)
sess = tf.Session()
print "sess1:"
print(sess.run(norm))
print(sess.run(norm))
sess = tf.Session()
print "sess2:"
print(sess.run(norm))
print(sess.run(norm))

# Use random uniform values in [0, 1) as the initializer for a variable of shape
# [2, 3]. The default type is float32.
var = tf.Variable(tf.random_uniform([2, 3]), name="var")
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print "sess.run(var)"
print(sess.run(var))


# Outputs random values from a truncated normal distribution.
truncated_normal = tf.truncated_normal([2, 3], mean = 3.0, stddev = 2.0)
print "truncated_normal:"
print truncated_normal.eval()

# '''
# Outputs random values from a uniform distribution.
random_uniform = tf.random_uniform([2, 3], minval = 0, maxval = 100,dtype=tf.float32)
print "random_uniform:"
print random_uniform.eval()


# Randomly crops a tensor to a given size.
random_crop = tf.random_crop(random_uniform, [2,2],seed=None, name=None)
print "random_crop.eval():"
print random_crop.eval()


# Draws samples from a multinomial distribution.
samples = tf.multinomial(tf.log([[0.5, 0.5]]), 10)
print "samples.eval():"
print samples.eval()

# '''
# To generate different sequences across sessions, set neither graph-level nor op-level seeds:
a = tf.random_uniform([1])
b = tf.random_normal([1])

print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A3'
  print(sess2.run(a))  # generates 'A4'
  print(sess2.run(b))  # generates 'B3'
  print(sess2.run(b))  # generates 'B4'

# To generate the same repeatable sequence for an op across sessions, set the seed for the op:
a = tf.random_uniform([1], seed=1)
b = tf.random_normal([1])

# Repeatedly running this block with the same graph will generate the same
# sequence of values for 'a', but different sequences of values for 'b'.
print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A1'
  print(sess2.run(a))  # generates 'A2'
  print(sess2.run(b))  # generates 'B3'
  print(sess2.run(b))  # generates 'B4'

# To make the random sequences generated by all ops be repeatable across sessions, set a graph-level seed:

tf.set_random_seed(1234)
a = tf.random_uniform([1])
b = tf.random_normal([1])

# Repeatedly running this block with the same graph will generate different
# sequences of 'a' and 'b'.
print("Session 1")
with tf.Session() as sess1:
  print(sess1.run(a))  # generates 'A1'
  print(sess1.run(a))  # generates 'A2'
  print(sess1.run(b))  # generates 'B1'
  print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
  print(sess2.run(a))  # generates 'A1'
  print(sess2.run(a))  # generates 'A2'
  print(sess2.run(b))  # generates 'B1'
  print(sess2.run(b))  # generates 'B2'
  
