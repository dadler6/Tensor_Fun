"""
This is me beginning to learn how to utilized tensor flow.

Tutorial is from:
https://www.tensorflow.org/get_started/get_started

NOT MY CODE! I'm just using it to learn!
"""

# First goal is upload the package

import tensorflow as tf 
import numpy as np

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

# Simple gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to default
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W, b]))

# Results in
# [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]]


############## HERE'S THE FULL PROGRAM ##############
# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

## HERE'S ANOTHER VERSION ##
# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, 
    num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
estimator.evaluate(input_fn=input_fn)
