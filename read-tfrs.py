# Reads a .tfrecords with TF Dataset API

import os
import sys
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filepath = sys.argv[1]
maxcount = int(sys.argv[2])

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)

def parse(example):
    features = tf.parse_single_example(example, {
        "size": tf.FixedLenFeature((), tf.int64),
        "label": tf.FixedLenFeature((2), tf.int64),
        "shape": tf.FixedLenFeature((3), tf.int64),
        "planes": tf.VarLenFeature(tf.int64),
        "target": tf.VarLenFeature(tf.int64) })

    size = features["size"] # size of the board, can be anything
    label = features["label"] # [0, 1] or [1, 0]
    shape = features["shape"] # [size + 2, size + 2, F] where F is the number of features
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
    image = tf.pad(planes, [[5, 5], [5, 5], [0, 0]])
    image = image[tx : tx + 11, ty : ty + 11, :]
    
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

dataset = dataset.map(parse)
dataset = dataset.shuffle(1024)
dataset = dataset.repeat()
dataset = dataset.batch(16)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={filenames: [filepath]})
    index = 0

    for _ in range(maxcount):
        (label, image) = sess.run(next_element)
        index += 1
        print('[run] #%d' % (index), label, image.shape)
        #print(image)
