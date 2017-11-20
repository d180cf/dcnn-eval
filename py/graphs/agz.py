import tensorflow as tf
from . import nnu

# an AlphaGo Zero style value network
# the number of multiply-add ops per residual block:
#   2*N*N*9*F*F = 6 M, if N=9, F=64
def make_dcnn(images, labels, learning_rate, is_training, n_resblocks = 3, n_filters = 64):
    def bnorm(x):
        return tf.layers.batch_normalization(x, axis=3, training=is_training)

    # applies a [k, k, n] convolution with 0 padding
    def conv(x, k, n):
        f = int(x.shape[3]) # [-1, 9, 9, 13]
        w = nnu.weights([k, k, f, n])
        return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')

    # a residual block with 2 convolutions [k, k, n]
    def resb(x, k, n):
        y = tf.identity(x)
        x = conv(x, 3, n_filters)
        x = bnorm(x)
        x = tf.nn.relu(x)
        x = conv(x, 3, n_filters)
        x = bnorm(x)
        return tf.nn.relu(x + y)

    # a fully connected layer with n outputs
    def conn(x, n):
        _, _, n, f = x.shape # [-1, 9, 9, 64]
        m = int(n*n*f)
        x = tf.reshape(x, [-1, m])
        w = nnu.weights([m, int(n)])
        b = nnu.bias([n])
        return tf.matmul(x, w) + b

    # a fully connected layer with 1 output
    def readout(x):
        _, n = x.shape # [-1, 64]
        w = nnu.weights([int(n), 1])
        b = nnu.bias([1])
        return tf.matmul(x, w) + b

    x = images
    print(0, x.shape)

    # the 1st conv layer to align the input
    x = conv(x, 3, n_filters)
    x = bnorm(x)
    x = tf.nn.relu(x)
    print(1, x.shape)

    # a few residual blocks
    for i in range(n_resblocks):
        x = resb(x, 3, n_filters)
        print(2, x.shape)

    # the final 1x1:1 convolution
    x = conv(x, 1, 1)
    x = bnorm(x)
    x = tf.nn.relu(x)
    print(3, x.shape)

    # the fully connected layer
    x = conn(x, n_filters)
    x = tf.nn.relu(x)
    print(4, x.shape)

    # readout with 0..1 output
    x = readout(x)
    x = tf.sigmoid(x)
    print(5, x.shape)

    y = tf.reshape(x, [-1])
    e = tf.losses.mean_squared_error(labels, y)

    optimizer = tf.train.AdamOptimizer()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        minimize = optimizer.minimize(e)
        return (y, e, minimize)
