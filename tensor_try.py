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