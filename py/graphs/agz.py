import tensorflow as tf
from . import nnu

# An AlphaGoZero-style value network applied to a NxN:F input:
#
# 1. conv 3x3:W, batch norm, relu.
# 2. A few residual blocks with the following structure:
#   2.1. conv 3x3:W, batch norm, relu.
#   2.2. conv 3x3:W, batch norm.
#   2.3. the skip connection
#   2.4. relu
# 3. conv 1x1:1, batch norm, relu.
# 4. A fully connected layer with M outputs, followed by relu.
# 5. A fully connected layer with 1 output, followed by tanh.
#
# Highest observed accuracy: 0.?? with N = ?, W = ?, M = ?
def make_dcnn(images, labels, learning_rate, n_resblocks = 1, n_filters = 64, n_output = 64):
    def bnorm(x):
        n_out = int(x.shape[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]))
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]))
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
        ema = tf.train.ExponentialMovingAverage(decay=0.999)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return batch_mean, batch_var

        mean, var = tf.cond(is_training,
            mean_var_with_update,
            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.001)

    # applies a [k, k, n] convolution with 0 padding
    def conv(x, k, n):
        f = int(x.shape[3]) # [-1, 9, 9, 5]
        w = nnu.weights([k, k, f, n])
        return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')

    # a residual block with 2 convolutions [k, k, n]
    def resb(x, k, n):
        y = conv(x, 3, n_filters)
        # y = bnorm(y)
        y = tf.nn.relu(y)
        y = conv(x, 3, n_filters)
        # y = bnorm(y)        
        y = tf.nn.relu(y + x)
        return y

    # a fully connected layer with n outputs
    def conn(x, n):
        s = x.shape # [-1, 9, 9, 32]
        m = int(s[1]*s[2]*s[3])
        x = tf.reshape(x, [-1, m])
        w = nnu.weights([m, n])
        return tf.matmul(x, w)

    # a fully connected layer with 1 output
    def readout(x):
        n = int(x.shape[1]) # [-1, 128]
        w = nnu.weights([n, 1])
        return tf.matmul(x, w)

    x = images
    print(1, x.shape)

    # the first conv layer that shirnks the input
    x = conv(x, 3, n_filters)
    # x = bnorm(x)
    x = tf.nn.relu(x)
    print(2, x.shape)

    # a few residual blocks
    for i in range(n_resblocks):
        x = resb(x, 3, n_filters)
        print(3, x.shape)

    # the final 1x1:1 convolution
    x = conv(x, 1, 1)
    # x = bnorm(x)
    x = tf.nn.relu(x)
    print(4, x.shape)

    # the fully connected layer with a few outputs
    x = conn(x, n_output)
    x = tf.nn.relu(x)
    print(5, x.shape)

    # the fully connected layer with 1 output
    x = readout(x)
    x = tf.tanh(x)
    print(6, x.shape)

    # y is in -1..1 range, labels are in 0..1 range
    y = tf.reshape(x, [-1])
    y = (y + 1)/2
    e = tf.losses.mean_squared_error(y, labels)
    return (y, e, tf.train.GradientDescentOptimizer(learning_rate).minimize(e))