import tensorflow as tf
from . import nnu

# a few residual blocks followed by a fully connected layer
def make_dcnn(images, labels, learning_rate, is_training, d=3, n=64):
    def bnorm(x):
        return tf.layers.batch_normalization(x, name='bnorm', training=is_training)

    def dense(x, n, bias):
        return nnu.fconn(x, n, name='dense', use_bias=bias)

    print(0, images.shape)
    _, _, N, F = images.shape

    x = tf.reshape(images, [-1, N*N*F])
    print(1, x.shape)

    with tf.variable_scope('align'):
        x = dense(x, n, True)
        # x = bnorm(x)
        x = tf.nn.relu(x)
        print(2, x.shape)

    for i in range(d):
        with tf.variable_scope('resb%d' % (i + 1)):
            y = tf.identity(x)

            with tf.variable_scope('1'):
                x = dense(x, n, True)
                # x = bnorm(x)
                x = tf.nn.relu(x)
                print(3, x.shape)

            with tf.variable_scope('2'):
                x = dense(x, n, True)
                # x = bnorm(x)
                x = x + y
                x = tf.nn.relu(x)
                print(3, x.shape)

    with tf.variable_scope('readout'):
        x = dense(x, 1, True)
        x = tf.tanh(x)
        print(4, x.shape)

    y = tf.reshape(x, [-1])
    e = tf.losses.mean_squared_error(labels, y)

    optimizer = tf.train.AdamOptimizer()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        minimize = optimizer.minimize(e)
        return (y, e, minimize)
