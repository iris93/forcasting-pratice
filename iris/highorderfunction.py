import tensorflow as tf
import numpy as np

# This map operator repeatedly applies the callable fn to a sequence of elements from first to last. The elements are made of the tensors unpacked from elems. dtype is the data type of the return value of fn. Users must provide dtype if it is different from the data type of 
# '''
fn = lambda x: x * x*x
elems = [1, 2, 3, 4, 5, 6]
# squares = tf.map_fn( lambda x: x * x,elems)
squares=tf.map_fn(fn, elems, dtype=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)
print "squares:",squares
interSess = tf.InteractiveSession()
print "squares.eval():",squares.eval()
# '''


# This foldl operator repeatedly applies the callable fn to a sequence of elements from first to last. The elements are made of the tensors unpacked from elems on dimension 0. The callable fn takes two tensors as arguments. The first argument is the accumulated value computed from the preceding invocation of fn. If initializer is None, elems must contain at least one element, and its first element is used as the initializer.
elems = [1, 2, 3, 4, 5, 6]
total = tf.foldl(lambda a, x: a + x, elems)

fn = lambda a, x: a + x
total = tf.foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)
print total.eval()
# ((((1*2)*3)*4)*5)*6
# '''
# almost same function as tf.foldr
elems = [1, 2, 3, 4, 5, 6]
total = tf.foldr(lambda a, x: a + x, elems)
print total.eval()



elems = [1, 2, 3, 4, 5, 6]
total = tf.scan(lambda a, x: a + x, elems)
print total.eval()
[1+initializer,1+initializer+2,1+initializer+3,...]

fn = lambda a, x: a * x
total = tf.scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)
print total.eval()
  # sum == [1, 3, 6, 10, 15, 21]
