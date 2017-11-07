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
    dataset = dataset.shuffle(1024)
    dataset = dataset.repeat()
    dataset = dataset.batch(16)
    return dataset

tprint('Initializing the main dataset...')    
dataset_main = make_dataset(ds_main)

tprint('Initializing the test dataset...')    
dataset_test = make_dataset(ds_test)

def error(count, next_batch):
    n = 0
    err_0 = 0
    err_1 = 0
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x2 = 0
    sum_y2 = 0
    t = time.time()

    for _ in range(count):
        (_labels, _images) = next_batch()

        results = prediction.eval(feed_dict={
            labels: _labels,
            images: _images })

        assert _labels.shape[0] == _images.shape[0]
        assert _labels.shape == results.shape

        for i in range(results.shape[0]):
            _label = _labels[i]
            _image = _images[i]
            result = results[i]

            # 1 = safe; 0 = unsafe
            x = result
            y = _label

            sum_x += x
            sum_y += y
            sum_x2 += x*x
            sum_y2 += y*y
            sum_xy += x*y

            if y == 0 and x > 0.5:
                err_0 += 1

            if y == 1 and x < 0.5:
                err_1 += 1

            n += 1.0
    
    correlation = (n*sum_xy - sum_x*sum_y) / ((n*sum_x2 - sum_x**2)*(n*sum_y2 - sum_y**2))**0.5

    return (1.0*err_0/n, 1.0*err_1/n, correlation, time.time() - t)

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

# perhaps the simplest NN possible: a weighthed sum of all features;
# highest observed accuracy: 0.61
def make_dcnn_0():    
    x = tf.reshape(images, [-1, N*N*F])
    b = bias([1])
    w = weights([N*N*F, 1])
    y = tf.sigmoid(tf.matmul(x, w) + b)
    y = tf.reshape(y, [-1])
    e = tf.square(y - labels)
    return (y, tf.train.GradientDescentOptimizer(0.5).minimize(e))

# applies 3x3 convolutions, then a dense layer, then readout
# highest observed accuracy: 0.79
def make_dcnn_2(n_conv = 3, n_filters = 16, n_output = 128):
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
    print(x.shape)

    # 3x3 convolutions with 16 filters
    for i in range(n_conv):
        x = conv(x, 3, n_filters)
        print(x.shape)

    # dense layer
    x = dense(x, n_output)
    print(x.shape)

    # readout
    y = readout(x)
    print(y.shape)

    y = tf.reshape(y, [-1])    
    e = tf.square(y - labels)
    return (y, tf.train.GradientDescentOptimizer(0.003).minimize(e))

# An AlpgaGo-style value network. The input is a 11x11:F image stack.
# 1. The first 5x5:K convolution maps it to a 7x7:K image stack.
# 2. A few 3x3:K convolutions with 0-padded input.
# 3. A 1x1:1 convolution yields a 7x7:1 image.
# 4. Finally a fully connected layer with 64 outputs and a readout.
# Highest observed accuracy: 0.84 (3 layers, 32 filters)
def make_dcnn_ag(n_conv = 3, n_filters = 32, n_output = 64):
    # "SAME" for 0 padding, "VALID" for no padding
    def conv(x, k, n, padding):
        f = int(x.shape[3]) # [-1, 9, 9, 5]
        w = weights([k, k, f, n])
        return tf.nn.relu(tf.nn.conv2d(x, w, [1, 1, 1, 1], padding))

    def conn(x, n):
        s = x.shape # [-1, 9, 9, 32]
        m = int(s[1]*s[2]*s[3])
        x = tf.reshape(x, [-1, m])
        b = bias([n])
        w = weights([m, n])
        return tf.nn.relu(tf.matmul(x, w) + b)

    def readout(x):
        n = int(x.shape[1]) # [-1, 128]
        w = weights([n, 1])
        return tf.sigmoid(tf.matmul(x, w))

    x = images
    print('input', x.shape)

    # a 5x5:K convolution
    x = conv(x, 5, n_filters, 'VALID')
    print('5x5', x.shape)

    # 3x3:K convolutions
    for i in range(n_conv):
        x = conv(x, 3, n_filters, 'SAME')
        print('3x3', i, x.shape)

    # 1x1:1 convolution
    x = conv(x, 1, 1, 'SAME')
    print('1x1', x.shape)

    # a fully connected layer
    x = conn(x, n_output)
    print(x.shape)

    # readout
    y = readout(x)
    print(y.shape)

    y = tf.reshape(y, [-1])    
    e = tf.square(y - labels)
    return (y, tf.train.GradientDescentOptimizer(learning_rate).minimize(e))

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
    avg_error = tf.square(output - labels)
    return (output, tf.train.AdamOptimizer(1e-4).minimize(avg_error))

learning_rate = tf.placeholder(tf.float32)
(prediction, optimizer) = make_dcnn_ag()

with tf.Session() as session:
    iterator_main = dataset_main.make_initializable_iterator()
    next_batch_main = iterator_main.get_next()    
    session.run(iterator_main.initializer)

    iterator_test = dataset_test.make_initializable_iterator()
    next_batch_test = iterator_test.get_next()    
    session.run(iterator_test.initializer)    

    tprint('Initializing global variables...')
    session.run(tf.global_variables_initializer())

    lr = 0.01

    try:
        for i in range(1000):
            # estimate the error on the test dataset
            (err_0, err_1, corr, err_t) = error(150, lambda: session.run(next_batch_test))
            tprint("error %.2f = %.2f + %.2f, correlation %.2f, iteration %d, delay %.1fs"
                % (err_0 + err_1, err_0, err_1, corr, i, err_t))

            # apply exp decay to the learning rate
            if i % 100 == 0:
                lr *= 0.5
                tprint('learning rate = %f' % (lr))

            # adjust the DCNN weights on the main dataset
            for _ in range(1500):
                (_labels, _images) = session.run(next_batch_main)
                optimizer.run(feed_dict={
                    learning_rate: lr,
                    labels: _labels,
                    images: _images })
    except KeyboardInterrupt:
        tprint('Terminated by Ctrl+C')

    tprint('Estimating accuracy on the entire test set...')
    (err_0, err_1, corr, err_t) = error(1000, lambda: session.run(next_batch_test))
    tprint("error %.2f = %.2f + %.2f, correlation %.2f" % (err_0 + err_1, err_0, err_1, corr))
