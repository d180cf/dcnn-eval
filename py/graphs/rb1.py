import tensorflow as tf
from . import nnu

# a few residual blocks followed by a fully connected layer
def make_dcnn(images, labels, learning_rate, is_training, d = 3, n = 64):
    print(0, images.shape)
    _, _, N, F = images.shape

    x = tf.reshape(images, [-1, N*N*F])
    print(1, x.shape)

    x = nnu.fconn(x, n, name='internal')
    x = tf.nn.relu(x)
    print(2, x.shape)

    for i in range(d):
        with tf.name_scope('residual-' + str(i)):
            y = tf.identity(x)

            x = nnu.fconn(x, n, name='dense-' + str(i))
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)

            x = nnu.fconn(x, n, name='dense-' + str(i))
            x = tf.layers.batch_normalization(x, training=is_training)
            x = x + y
            x = tf.nn.relu(x)

            print(3, x.shape)

    x = nnu.fconn(x, 1, name='readout')
    x = tf.sigmoid(x)
    print(4, x.shape)

    y = tf.reshape(x, [-1])
    e = tf.losses.mean_squared_error(labels, y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        minimize = optimizer.minimize(e)
        return (y, e, minimize)
