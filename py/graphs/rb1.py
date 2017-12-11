import tensorflow as tf
from . import nnu

# a few residual blocks followed by a fully connected layer
def make_dcnn(images, labels, bmoves, learning_rate, is_training, d=3, n=64):
    def bnorm(x):
        return x # tf.layers.batch_normalization(x, name='bnorm', training=is_training)

    def dense(x, n, bias):
        return nnu.fconn(x, n, name='dense', use_bias=bias)

    print(0, images.shape)
    _, _, N, F = images.shape

    x = tf.reshape(images, [-1, N*N*F])
    print(1, x.shape)

    with tf.variable_scope('align'):
        x = dense(x, n, True)
        x = bnorm(x)
        x = tf.nn.relu(x)
        print(2, x.shape)

    for i in range(d):
        with tf.variable_scope('resb%d' % (i + 1)):
            y = tf.identity(x) # TODO: is this needed?

            with tf.variable_scope('1'):
                x = dense(x, n, True)
                x = bnorm(x)
                x = tf.nn.relu(x)
                print(3, x.shape)

            with tf.variable_scope('2'):
                x = dense(x, n, True)
                x = bnorm(x)
                x = x + y
                x = tf.nn.relu(x)
                print(3, x.shape)

    v = x # value, output = 0..1
    p = x # policy, output = [N, N]

    v_err = 0 # mean squared error for value
    p_err = 0 # cross entropy for policy

    # the "value head" of the NN predicts the outcome
    with tf.variable_scope('eval'):
        v = dense(v, 1, True)
        v = tf.sigmoid(v)
        v = tf.reshape(v, [-1])
        v_err = tf.losses.mean_squared_error(labels, v)
        print(4, v.shape, v_err.shape)

    # the "policy head" selects the best moves
    with tf.variable_scope('move'):
        p = dense(p, int(N)**2, True)
        p_err = tf.losses.softmax_cross_entropy(tf.reshape(bmoves, [-1, int(N)**2]), p)
        print(5, p.shape, p_err.shape)
        p = tf.nn.softmax(p)
        p = tf.reshape(p, [-1, N, N])

    e = v_err + p_err # cross-entropy and MSE losses are weighted equally
    optimizer = tf.train.AdamOptimizer()

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        return (v, p, e, optimizer.minimize(e))
