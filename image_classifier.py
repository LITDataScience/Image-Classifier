import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
type(mnist)

image_shape = mnist.train.images[1].shape
# plt.imshow(mnist.train.images[1].reshape(28,28))
# plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')
# mnist.train.images[1].max()
# plt.imshow(mnist.train.images[1].reshape(784,1))
# plt.imshow(mnist.train.images[1].reshape(784,1),cmap='gist_gray',aspect=0.02)
x = tf.placeholder("float",shape=[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b
y_true = tf.placeholder("float",[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        ## PROVIDING BATCHES IN THE IN-BUILT METHOD PROVIDED BY TENSORFLOW
        batch_x , batch_y = mnist.train.next_batch(150)
        sess.run(train,feed_dict={
            x:batch_x,
            y_true:batch_y
        })
        
    ## TESTING THE MODEL
    matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    
    accuracy = tf.reduce_mean(tf.cast(matches,"float"))
    
    print(sess.run(accuracy,feed_dict={
        x:mnist.test.images,
        y_true:mnist.test.labels
    }))
