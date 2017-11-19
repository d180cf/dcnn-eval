import tensorflow as tf
from . import nnu

# www.cs.cityu.edu.hk/~hwchun/research/PDF/Julian%20WONG%20-%20CCCT%202004%20a.pdf
# highest observed accuracy: 0.82
def make_dcnn(images, labels, learning_rate, is_training):
    print(0, images.shape)
    (_, _, N, F) = images.shape
    N = int(N)
    F = int(F)
    
    def conv(n):
        _krnl = nnu.weights([n, n, F, 1])
        _bias = nnu.bias([1])
        _conv = tf.nn.relu(tf.nn.conv2d(images, _krnl, [1, 1, 1, 1], 'VALID') + _bias)
        return tf.reshape(_conv, [-1, (N - n + 1)**2])

    def conn(x, n):
        m = int(x.shape[1])
        _krnl = nnu.weights([m, n])
        _bias = nnu.bias([n])
        return tf.matmul(x, _krnl) + _bias

    # shape = [(N - 0)**2 + (N - 1)**2 + (N - 2)**2 + ...]
    hlayer = tf.concat([conv(n + 1) for n in range(N)], 1)

    # a few fully connected layers
    for n in [50, 30]:
        hlayer = tf.nn.relu(conn(hlayer, n))

    output = tf.sigmoid(conn(hlayer, 1))
    output = tf.reshape(output, [-1])
    e = tf.losses.mean_squared_error(labels, output)
    return (output, e, tf.train.AdamOptimizer(1e-4).minimize(e))