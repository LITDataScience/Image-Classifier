import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y = mnist.train.next_batch(10000)
test_x, test_y = mnist.test.next_batch(500)

K_neighbours = 8
x_train = tf.placeholder(tf.float32, [None, 784])
x_test = tf.placeholder(tf.float32, [784])

distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_train, x_test)), reduction_indices = 1)))

indices = tf.nn.top_k(distance, k=K_neighbours, sorted=False)

init = tf.global_variables_initializer()
correct_class = 0
num_steps = 2000

with tf.Session() as session:
    session.run(init)

    for i in range(len(test_x)):
        values_indices = session.run(indices, feed_dict={x_train: train_x, x_test: test_x[i, :]})

        # predicting label for test data
        counter = np.zeros(10)
        for j in range(K_neighbours):
            counter[np.argmax(train_y[values_indices.indices[j]])] += 1

        prediction = np.argmax(counter)

        if prediction == np.argmax(test_y[i]):
            correct_class += 1.0

    print('Accuracy:', (correct_class / len(test_x)) * 100, '%')