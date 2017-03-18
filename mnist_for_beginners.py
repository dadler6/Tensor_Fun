"""
Trying the MNIST for beginners tutorial.

Tutorial is from:
https://www.tensorflow.org/get_started/get_started

NOT MY CODE! I'm just using it to learn!
"""

# Import tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
def main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

    x = tf.placeholder(tf.float32, [None, 784])

    # Making a model y = softmax(Wx + b)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Now we implement the model
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # matmul is matrix multiplication
    # softmax is the softmax function

    # Now we'll initiate the cost function
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        )

    # Here's the training step
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
        cross_entropy
        )

    # Now we'll initiate a session
    sess = tf.InteractiveSession()

    # Initialize the variables
    tf.global_variables_initializer().run()

    # And train 1000 times
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Here we use "stochastic gradient descent" because we are taking
        # our data in batches of 100 and training

    # Now let's see how well we did
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


main()

