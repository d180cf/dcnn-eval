import tensorflow as tf
from . import nnu

# a few residual blocks followed by a fully connected layer
def make_dcnn(images, labels, learning_rate, d = 3, n = 64):
    print(0, images.shape)
    (_, _, N, F) = images.shape

    x = tf.reshape(images, [-1, N*N*F])
    print(1, x.shape)

    x = nnu.fconn(x, n, name='internal')
    x = tf.nn.relu(x)
    print(2, x.shape)

    for i in range(d):
        y = tf.identity(x)
        x = fconn(x, n, name='conn-0')
        x = tf.nn.relu(x)
        x = fconn(x, n, name='conn-1')
        x = tf.nn.relu(x + y)
        print(3, x.shape)

    x = nnu.fconn(x, 1, name='readout')
    x = tf.sigmoid(x)
    print(4, x.shape)

    y = tf.reshape(x, [-1])
    e = tf.losses.mean_squared_error(labels, y)
    return (y, e, tf.train.GradientDescentOptimizer(learning_rate).minimize(e))