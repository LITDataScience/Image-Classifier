import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


### Initializing weights
def init_weights(shape):
    random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(random_dist)

### Initializing Bias
def init_bias(shape):
    bias = tf.constant(0.1, shape=shape) # The bias takes the shape of the tensor
    return tf.Variable(bias)


def conv2d(x, W):
    """
        Purpose: 2D Convolution using tensorflow's inbuilt api
        :param x: This is our input tensor. Shape = [batch, height of the image, width of the img, channel]
        Channel is nothing but Grayscale or Colored
        :param W: The Filter/Kernel. Shape = [filter height, filter width, no. of channels coming in, no. of channels coming out]
        :param padding ['SAME']: To pad with zeros.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

### Pooling layer
def max_pool_2x2(x):
    # x: This is our input tensor. Shape = [batch, height of the image, width of the img, channel]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    """
        Purpose: Creates a convolutional layer.
    """
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    """
        Purpose: Normal fully connected layer.
    """
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

### Adding Placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])


### Adding layers
x_image = tf.reshape(x,[-1,28,28,1])

# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can change the 32 output, that essentially represents the amount of filters used
# You need to pass in 32 to the next input though, the 1 comes from the original input of
# a single image.
convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
convo_1_pooling = max_pool_2x2(convo_1)


# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can actually change the 64 output if you want, you can think of that as a representation
# of the amount of 6by6 filters used.
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2_pooling = max_pool_2x2(convo_2)


# Why 7 by 7 image? Because we did 2 pooling layers, so (28/2)/2 = 7
# 64 then just comes from the output of the previous Convolution
convo_2_flat = tf.reshape(convo_2_pooling, [-1,7*7*64])
## Using rectified linear unit for activation
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024)) # 1024 = no. of neurons


# NOTE THE PLACEHOLDER HERE!
hold_prob = tf.placeholder(tf.float32) # created holding probabilities
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)


y_pred = normal_full_layer(full_one_dropout,10)

### LOSS
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

### OPTIMIZE
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

### Creating a Training Session
steps = 5000

with tf.Session() as sess:
    score = None
    sess.run(init)
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 100 == 0:
            print('Currently on step {}'.format(i))
            print('Training Accuracy is: ', score)
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            # we have booleans, but we're casting here to floats to take the average accuracy score.
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            ### During prediction/test, we don't want any neuron to be dropped. Hence, dropout = 1
            score = sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels, hold_prob: 1.0})