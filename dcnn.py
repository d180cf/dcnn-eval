import os
import sys
import json
import time
import datetime
import numpy as np
import random
import tensorflow as tf

ds_main = sys.argv[1] # .tfrecords file with the main dataset
ds_test = sys.argv[2] # .tfrecords file with the test dataset
N = int(sys.argv[3]) # board frame size, e.g. 11 x 11
F = int(sys.argv[4]) # the number of features features, e.g. 5
suppress_tf_warning = len(sys.argv) > 5 and sys.argv[5]

if suppress_tf_warning:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

T = time.time()
print('T = ' + datetime.datetime.now().isoformat())

def tprint(text):
    print('[T+%06.1fs] %s' % (time.time() - T, text))

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

    return (label, image)

def make_dataset(filepath):
    dataset = tf.contrib.data.TFRecordDataset(filepath)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(1024)
    dataset = dataset.repeat()
    dataset = dataset.batch(32)
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

    for _ in range(count):
        (_labels, _images) = next_batch()

        results = prediction.eval(feed_dict={
            labels: _labels,
            images: _images })

        assert _labels.shape[0] == _images.shape[0]
        assert _labels.shape[0] == results.shape[0]

        for i in range(results.shape[0]):
            _label = _labels[i]
            _image = _images[i]
            result = results[i]

            # 1 = safe; 0 = unsafe
            x = result[1]
            y = _label[1]

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

    return (1.0*err_0/n, 1.0*err_1/n, correlation)

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
labels = tf.placeholder(tf.float32, shape=[None, 2])

def make_dcnn():
    def conv(n):
        _krnl = weights([n, n, F, 1])
        _bias = bias([1])
        _conv = tf.nn.elu(conv2d(images, _krnl) + _bias)
        return tf.reshape(_conv, [-1, (N - (n - 1))**2])

    def conn(x, m, n):
        _krnl = weights([m, n])
        _bias = bias([n])
        return tf.matmul(x, _krnl) + _bias

    # shape = [(N - 0)**2 + (N - 1)**2 + (N - 2)**2 + ...]
    layer_1 = tf.concat([
        conv(1),
        conv(2),
        conv(3),
        conv(4),
        conv(5)], 1)

    layer_2 = tf.nn.elu(conn(layer_1, 415, 150))
    layer_3 = tf.nn.elu(conn(layer_2, 150, 80))
    layer_4 = tf.nn.elu(conn(layer_3, 80, 20))

    output = conn(layer_4, 20, 2)

    return (
        tf.nn.softmax(output),
        tf.train.AdamOptimizer(1e-4).minimize(
            tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=output))))

(prediction, optimizer) = make_dcnn()

with tf.Session() as session:
    iterator_main = dataset_main.make_initializable_iterator()
    next_batch_main = iterator_main.get_next()    
    session.run(iterator_main.initializer)

    iterator_test = dataset_test.make_initializable_iterator()
    next_batch_test = iterator_test.get_next()    
    session.run(iterator_test.initializer)    

    tprint('Initializing global variables...')
    session.run(tf.global_variables_initializer())

    try:
        for i in range(1000):
            # estimate the error on the test dataset
            (err_0, err_1, corr) = error(50, lambda: session.run(next_batch_test))
            tprint("error %.2f = %.2f + %.2f, correlation %.2f, iteration %d"
                % (err_0 + err_1, err_0, err_1, corr, i))

            # adjust the DCNN weights on the main dataset
            for _ in range(1000):
                (_labels, _images) = session.run(next_batch_main)
                optimizer.run(feed_dict={
                    labels: _labels,
                    images: _images })
    except KeyboardInterrupt:
        sys.exit()
