import os
import json
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N = 11 # board frame size
F = 5 # features
K = 3 # kernel

def inputs():
    for name in os.listdir(".bin/features"):
        config = json.load(open(".bin/features/" + name))
        image = np.array(config["features"]) # [11, 11, 5] - NHWC
        label = config["safe"]
        yield (label, image)

def testnn(_label, _image):
    def weights(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # e.g. [?, 11, 11, 5] x [11, 11, 5, 32] -> [?, 9, 9, 32]
    def conv2d(x, W):
        return tf.nn.conv2d(x, W,
            strides=[1, 1, 1, 1],
            padding='VALID')

    # keeps size of the input
    def maxpool(x):
        return tf.nn.max_pool(x,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding='SAME')    

    images = tf.placeholder(tf.float32, shape=[None, N, N, F])
    kernel = weights([K, K, F, 1])
    output = maxpool(tf.nn.relu(conv2d(images, kernel) + bias([1])))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        result = output.eval(feed_dict={
            images: [_image]
        })

        return np.reshape(result, [N - K + 1, N - K + 1])

for (label, image) in inputs():
    print("image.shape:", image.shape)
    print(testnn(label, image))
