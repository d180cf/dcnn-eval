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

# a residual block: two fully connected layers + a skip connection
def resb(x, n, name=None):
    with tf.name_scope(name):        
        y = tf.identity(x)
        x = fconn(x, n, name='conn-0')
        x = tf.nn.relu(x)
        x = fconn(x, n, name='conn-1')
        return tf.nn.relu(x + y)