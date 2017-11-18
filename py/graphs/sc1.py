import tensorflow as tf
from . import nnu

# applies 3x3 convolutions, then a dense layer, then readout
# highest observed accuracy: 0.80
def make_dcnn(images, labels, learning_rate, n_conv = 3, n_filters = 64, n_output = 64):    
    def conv(x, k, n):
        b = nnu.bias([n])
        f = int(x.shape[3]) # [-1, 9, 9, 5]
        w = nnu.weights([k, k, f, n])
        return tf.nn.relu(tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME'))

    def dense(x, n):
        s = x.shape # [-1, 9, 9, 32]
        m = int(s[1]*s[2]*s[3])
        x = tf.reshape(x, [-1, m])
        b = nnu.bias([n])
        w = nnu.weights([m, n])
        return tf.nn.relu(tf.matmul(x, w) + b)

    def readout(x):
        n = int(x.shape[1]) # [-1, 128]
        b = nnu.bias([1])
        w = nnu.weights([n, 1])
        return tf.sigmoid(tf.matmul(x, w) + b)

    x = images
    print(1, x.shape)

    # 3x3 convolutions with 16 filters
    for i in range(n_conv):
        x = conv(x, 3, n_filters)
        print(2, x.shape)

    # dense layer
    x = dense(x, n_output)
    print(3, x.shape)

    # readout
    y = readout(x)
    print(4, y.shape)

    y = tf.reshape(y, [-1])    
    e = tf.losses.mean_squared_error(y, labels)
    return (y, e, tf.train.GradientDescentOptimizer(learning_rate).minimize(e))