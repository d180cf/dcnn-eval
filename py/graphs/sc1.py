import tensorflow as tf
from . import nnu

# applies 3x3 convolutions, then a dense layer, then readout
def make_dcnn(images, labels, learning_rate, is_training, n_layers = 4, n_filters = 64):    
    def conv(x, k, n):
        _, _, _, f = x.get_shape().as_list() # [-1, 9, 9, 13]
        b = nnu.bias([n])
        w = nnu.weights([k, k, f, n])
        return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')

    def dense(x, n):
        _, _, n, f = x.get_shape().as_list() # [-1, 9, 9, 64]
        m = n*n*f
        x = tf.reshape(x, [-1, m])
        b = nnu.bias([n])
        w = nnu.weights([m, n])
        return tf.matmul(x, w) + b

    def readout(x):
        _, n = x.get_shape().as_list() # [-1, 64]
        b = nnu.bias([1])
        w = nnu.weights([n, 1])
        return tf.matmul(x, w) + b

    x = images
    print(1, x.shape)

    for i in range(n_layers):
        x = conv(x, 3, n_filters)
        x = tf.nn.relu(x)
        print(2, x.shape)

    x = dense(x, n_filters)
    x = tf.nn.relu(x)
    print(3, x.shape)

    x = readout(x)
    x = tf.sigmoid(x)
    print(4, x.shape)

    y = tf.reshape(x, [-1])
    e = tf.losses.mean_squared_error(labels, y)

    optimizer = tf.train.AdamOptimizer()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        minimize = optimizer.minimize(e)
        return (y, e, minimize)