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
labels = tf.placeholder(tf.float32, shape=[None, 2])

# [11, 11, 5] x [3, 3, 5, 32] -> [9, 9, 32]
kernel_1 = weights([K, K, F, 32])
output_1 = maxpool(tf.nn.relu(conv2d(images, kernel_1) + bias([32])))

# [9, 9, 32] x [3, 3, 32, 32] -> [7, 7, 32]
kernel_2 = weights([K, K, 32, 32])
output_2 = maxpool(tf.nn.relu(conv2d(output_1, kernel_2) + bias([32])))

# [7, 7, 32] x [3, 3, 32, 32] -> [5, 5, 32]
kernel_3 = weights([K, K, 32, 32])
output_3 = maxpool(tf.nn.relu(conv2d(output_2, kernel_3) + bias([32])))

# [5, 5, 32] -> [1024]
kernel_4 = weights([5*5*32, 1024])
output_4 = tf.matmul(tf.reshape(output_3, [-1, 5*5*32]), kernel_4) + bias([1024])

# [1024] -> [2]
kernel_5 = weights([1024, 2]);
output_5 = tf.matmul(output_4, kernel_5) + bias([2])

prediction = tf.nn.softmax(output_5)

optimizer = tf.train.AdamOptimizer(1e-4).minimize(
    tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=output_5)))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print("Training DCNN...")

    for (label, image) in inputs():
        if (image.shape[0] == 11 and image.shape[1] == 11):
            optimizer.run(feed_dict={
                labels: [[1, 0] if label == 0 else [0, 1]],
                images: [image] })

    print("Evaluating DCNN...")
    sum = 0
    count = 0

    for (label, image) in inputs():
        if (image.shape[0] == 11 and image.shape[1] == 11):
            result = prediction.eval(feed_dict={
                labels: [[1, 0] if label == 0 else [0, 1]],
                images: [image] })
            sum += (label - result[0][1])**2
            count += 1
    
    print("Error:", (sum / count)**0.5)

