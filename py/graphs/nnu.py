import tensorflow as tf

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

# a fully connected layer with n outputs
def fconn(x, n, name=None):
    with tf.name_scope(name):
        m = int(x.shape[1])
        w = weights([m, n])
        b = bias([n])
        return tf.matmul(x, w) + b
