import tensorflow as tf
from . import nnu

# perhaps the simplest NN possible: a weighthed sum of all features
# maximum observed accuracy:
#   0.70 when d=0 n=16
#   0.80 when d=1 n=16
#   0.84 when d=1 n=64
#   0.85 when d=1 n=128
#   0.80 when d=2 n=16
#   0.86 when d=2 n=64
def make_dcnn(images, labels, learning_rate, is_training, d = 2, n = 64):
    print(0, images.shape)
    _, _, N, F = images.shape

    x = tf.reshape(images, [-1, N*N*F])
    print(1, x.shape)

    for i in range(d):
        x = nnu.fconn(x, n, name='internal')
        x = tf.nn.relu(x)
        print(2, x.shape)

    x = nnu.fconn(x, 1, name='readout')
    x = tf.tanh(x)
    print(3, x.shape)

    y = tf.reshape(x, [-1])
    e = tf.losses.mean_squared_error(labels, y)
    return (y, e, tf.train.GradientDescentOptimizer(learning_rate).minimize(e))