import os
import sys
import json
import time
import datetime
import numpy as np
import random
import tensorflow as tf

print('Python %s' % (sys.version))
print('TensorFlow %s' % (tf.__version__))

ds_main = sys.argv[1] # .tfrecords file with the main dataset
ds_test = sys.argv[2] # .tfrecords file with the test dataset
N = int(sys.argv[3]) # board frame size, e.g. 11 x 11
F = int(sys.argv[4]) # the number of features features, e.g. 5
vars_file = sys.argv[5] # the checkpoint file for weights
err_log = sys.argv[6] # results on the validation set
logs_path = sys.argv[7] # tensorboard logs

SHUFFLE_WINDOW = 8192
BATCH_SIZE = 256

print('Target frame: %dx%d' % (N, N))
print('Features: %d' % (F))

T = time.time()
print('T = ' + datetime.datetime.now().isoformat())

def tprint(text):
    print('[T+%.1fs] %s' % (time.time() - T, text))

def parse(example):
    features = tf.parse_single_example(example, {
        "size": tf.FixedLenFeature((), tf.int64),
        "label": tf.FixedLenFeature((2), tf.int64),
        "shape": tf.FixedLenFeature((3), tf.int64),
        "planes": tf.VarLenFeature(tf.int64),
        "target": tf.VarLenFeature(tf.int64) })

    size = features["size"] # area size
    label = features["label"] # [0, 1] or [1, 0]
    shape = features["shape"] # [N + 2, N + 2, F] where F is the number of features and N is the size of the board
    planes = features["planes"] # the features tensor with the shape above
    target = features["target"] # list of [target.x, target.y] pointers

    shape = tf.cast(shape, tf.int32) # otherwise TF crashes with weird CPU/GPU related error

    planes = tf.sparse_tensor_to_dense(planes) # when TF was writing the file, it apparently compressed it
    planes = tf.reshape(planes, shape)

    target = tf.sparse_tensor_to_dense(target)
    target = tf.reshape(target, [-1, 2])

    count = tf.shape(target)[0]
    index = tf.random_uniform([1], 0, count, tf.int32)[0]

    t = target[index]

    tx = t[0]
    ty = t[1]

    # `image` = 11 x 11 slice around [tx, ty] from `planes`, padded with 0s
    image = tf.pad(planes, [[N//2, N//2], [N//2, N//2], [0, 0]])
    image = image[tx : tx + N, ty : ty + N, :]
    
    # tranpose randomly
    transpose = tf.random_uniform([1], 0, 2, tf.int32)[0]
    image = tf.cond(
        transpose > 0,
        lambda: tf.transpose(image, [1, 0, 2]),
        lambda: image)

    # rotate up to 3 times
    rotate = tf.random_uniform([1], 0, 4, tf.int32)[0]
    image = tf.image.rot90(image, rotate)

    return (label[1], image)

def make_dataset(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(SHUFFLE_WINDOW)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

tprint('Initializing the main dataset...')    
dataset_main = make_dataset(ds_main)

tprint('Initializing the test dataset...')    
dataset_test = make_dataset(ds_test)

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W,
        strides=[1, 1, 1, 1],
        padding='VALID')

images = tf.placeholder(tf.float32, shape=[None, N, N, F])
labels = tf.placeholder(tf.float32, shape=[None])
phase_train = tf.placeholder(tf.bool)

# perhaps the simplest NN possible: a weighthed sum of all features;
# highest observed accuracy: 0.60
def make_dcnn_fc1():
    x = tf.reshape(images, [-1, N*N*F])
    b = bias([1])
    w = weights([N*N*F, 1])
    y = tf.tanh(tf.matmul(x, w) + b)
    y = tf.reshape(y, [-1])
    e = tf.losses.mean_squared_error(2 * labels - 1, y)
    return ((y + 1)/2.0, e, tf.train.GradientDescentOptimizer(0.5).minimize(e))

# the next simplest network: weighted sum + a hidden layer with 2 values
# highest observed accuracy: 0.65
def make_dcnn_fc2():
    x = tf.reshape(images, [-1, N*N*F])
    b = bias([2])
    w = weights([N*N*F, 2])
    y = tf.matmul(x, w) + b
    y = tf.tanh(y)

    b = bias([1])
    w = weights([2, 1])
    y = tf.matmul(y, w) + b
    y = tf.tanh(y)

    y = tf.reshape(y, [-1])
    y = (y + 1)/2
    e = tf.losses.mean_squared_error(labels, y)
    return (y, e, tf.train.GradientDescentOptimizer(learning_rate).minimize(e))

# applies 3x3 convolutions, then a dense layer, then readout
# highest observed accuracy: 0.80
def make_dcnn_sc1(n_conv = 3, n_filters = 64, n_output = 64):
    def conv(x, k, n):
        b = bias([n])
        f = int(x.shape[3]) # [-1, 9, 9, 5]
        w = weights([k, k, f, n])
        return tf.nn.relu(tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME'))

    def dense(x, n):
        s = x.shape # [-1, 9, 9, 32]
        m = int(s[1]*s[2]*s[3])
        x = tf.reshape(x, [-1, m])
        b = bias([n])
        w = weights([m, n])
        return tf.nn.relu(tf.matmul(x, w) + b)

    def readout(x):
        n = int(x.shape[1]) # [-1, 128]
        b = bias([1])
        w = weights([n, 1])
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
def make_dcnn_agz(n_resblocks = 1, n_filters = 64, n_output = 64):
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

        mean, var = tf.cond(phase_train,
            mean_var_with_update,
            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.001)

    # applies a [k, k, n] convolution with 0 padding
    def conv(x, k, n):
        f = int(x.shape[3]) # [-1, 9, 9, 5]
        w = weights([k, k, f, n])
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
        w = weights([m, n])
        return tf.matmul(x, w)

    # a fully connected layer with 1 output
    def readout(x):
        n = int(x.shape[1]) # [-1, 128]
        w = weights([n, 1])
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

# www.cs.cityu.edu.hk/~hwchun/research/PDF/Julian%20WONG%20-%20CCCT%202004%20a.pdf
# highest observed accuracy: 0.82
def make_dcnn_1():
    def conv(n):
        _krnl = weights([n, n, F, 1])
        _bias = bias([1])
        _conv = tf.nn.relu(conv2d(images, _krnl) + _bias)
        return tf.reshape(_conv, [-1, (N - n + 1)**2])

    def conn(x, n):
        m = int(x.shape[1])
        _krnl = weights([m, n])
        _bias = bias([n])
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

def save_vars():
    data = []
    for x in tf.trainable_variables():
        y = x.eval().reshape([-1])
        data.append([x.name, x.shape.as_list(), y.tolist()])
    with open(vars_file, 'w') as file:
        json.dump(data, file)

learning_rate = tf.placeholder(tf.float32)
(prediction, loss, optimizer) = make_dcnn_sc1()

avg_1 = tf.reduce_sum(labels * prediction) / tf.cast(tf.count_nonzero(labels), tf.float32)
avg_0 = tf.reduce_sum((1 - labels) * prediction) / tf.cast(tf.count_nonzero(1 - labels), tf.float32)
accuracy = tf.reduce_mean(tf.nn.relu(tf.sign((prediction - 0.5) * (labels - 0.5))))

for x in tf.trainable_variables():
    print(x)

with tf.Session() as session:
    iterator_main = dataset_main.make_initializable_iterator()
    next_batch_main = iterator_main.get_next()    
    session.run(iterator_main.initializer)

    iterator_test = dataset_test.make_initializable_iterator()
    next_batch_test = iterator_test.get_next()    
    session.run(iterator_test.initializer)    

    tprint('Initializing global variables...')
    session.run(tf.global_variables_initializer())    

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('avg_1', avg_1)
    tf.summary.scalar('avg_0', avg_0)
    merged = tf.summary.merge_all()

    n_tb_logs = len(os.listdir(logs_path)) if os.path.isdir(logs_path) else 0
    lg_path = '%s/%d' % (logs_path, n_tb_logs + 1)

    tprint('TensorBoard logs: ' + lg_path)

    test_writer = tf.summary.FileWriter(lg_path + '/validation')
    main_writer = tf.summary.FileWriter(lg_path + '/training', session.graph)

    lr = 0.5
    lri = 0
    EPOCH_LENGTH = 100
    LR_DECAY = 1e6

    try:
        for i in range(1000):
            t0 = time.time()            
            save_vars()

            t1 = time.time()
            _labels, _images = session.run(next_batch_test)
            summary, _accuracy = session.run([merged, accuracy], feed_dict={
                phase_train: False,
                learning_rate: lr,                    
                labels: _labels,
                images: _images })
            test_writer.add_summary(summary, i * EPOCH_LENGTH * BATCH_SIZE)

            t2 = time.time()
            tprint('[%d] accuracy = %.2f; save = %.1fs; tensorboard = %.1fs' % (i, _accuracy, t1 - t0, t2 - t1))

            # adjust the DCNN weights on the main dataset
            for k in range(EPOCH_LENGTH):
                _labels, _images = session.run(next_batch_main)
                summary, _ = session.run([merged, optimizer], feed_dict={
                    phase_train: True,
                    learning_rate: lr,                    
                    labels: _labels,
                    images: _images })
                main_writer.add_summary(summary, (i * EPOCH_LENGTH + k) * BATCH_SIZE)

            # apply exp decay to the learning rate
            lri += EPOCH_LENGTH * BATCH_SIZE
            if lri > LR_DECAY:
                lr *= 0.5
                lri = 0
                tprint('learning rate = %f' % (lr))
                
    except KeyboardInterrupt:
        tprint('Terminated by Ctrl+C')
