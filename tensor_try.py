"""
This is me beginning to learn how to utilized tensor flow.

Tutorial is from:
https://www.tensorflow.org/get_started/get_started
"""

# First goal is upload the package

import tensorflow as tf 

# Now we'll make two nodes and pring
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# This produces nodes that when evaluated would output 3.0 and 4.0

# Now we're going to create tensor objects
sess = tf.Session()
print(sess.run([node1, node2]))

# This will produce [3.0, 4.0]

# Now we'll combine the nodes to something more complicated
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

# This produced the following output:
# ('node3: ', <tf.Tensor 'Add:0' shape=() dtype=float32>)
# ('sess.run(node3): ', 7.0)

# Now let's do something more interesting and add "placeholders"
# that we can add a value to later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # Same as tf.add(a, b)

# Now we'll add values to the placeholders
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# We got this output:
# 7.5
# [ 3.  7.]

# Let's add more complexity to these operations
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# This produced the following result
# 22.5

# Now we'll add trainable parameters to a graph
W = tf.Variable([.3], tf.float32) # Parameter 1
b = tf.Variable([-.3], tf.float32) # Parameter 2
x = tf.placeholder(tf.float32) # Here is our input
linear_model = W * x + b # The model

# Now we can initialize the variables
init = tf.global_variables_initializer()
sess.run(init)

# and now we can run the linear model
print(sess.run(linear_model, {x: [1,2,3,4]}))

# And we get:
# [ 0.          0.30000001  0.60000002  0.90000004]

# Now we'll make an error functio to evaluate the model
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

# Producing this loss value:
# 23.66

# We could reassign the values of W and b to get the correct output
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Producing this loss value:
# 0.0

# But now we'll actually train the model to get this loss...