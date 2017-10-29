import os
import sys
import json
import numpy as np
import random
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N = 11 # board frame size
F = 5 # features
K = 3 # kernel

# returns tensor[xmin..xmax, ymin..ymax] with zero padding
def submatrix(tensor, xmin, xmax, ymin, ymax):
    (width, height, depth) = tensor.shape
    result = np.zeros((xmax - xmin + 1, ymax - ymin + 1, depth))

    x1 = max(xmin, 0)
    x2 = min(xmax, width - 1)
    y1 = max(ymin, 0)
    y2 = min(ymax, height - 1)

    result[x1 - xmin : x2 - xmin + 1, y1 - ymin : y2 - ymin + 1] = tensor[x1 : x2 + 1, y1 : y2 + 1]
    
    return result

def get_configs(min_area_size):
    for name in os.listdir(".bin/features"):
        config = json.load(open(".bin/features/" + name))
        areasize = config["area"]

        if areasize < min_area_size: # skip too easy problems
            continue

        yield config

print("Parsing JSON files...")
configs = [x for x in get_configs(8)] # preload all the relevant JSON files
print("configs: %d" % (len(configs)))

def inputs(prob):
    for config in configs:
        if random.random() > prob: # pick only 10% of the inputs
            continue

        target = np.array(config["target"]) # [M, 2] - a list of (x, y) coords
        image = np.array(config["features"]) # [board.size + 2, board.size + 2, 5] - NHWC
        label = config["safe"]        
        [tx, ty] = random.choice(target)        
        frame = submatrix(image, tx - N//2, tx + N//2, ty - N//2, ty + N//2)

        # the result is invariant wrt transposition and rotation
        if (random.randint(0, 1) == 1): 
            frame = frame.transpose((1, 0, 2))
        for i in range(random.randint(0, 3)):
            frame = np.rot90(frame)
        
        yield ([1, 0] if label == 0 else [0, 1], frame)

def batches(size, prob):
    _labels = []
    _images = []

    for (label, image) in inputs(prob):
        _labels.append(label)
        _images.append(image)

        if (len(_labels) == size):
            yield (np.array(_labels), np.array(_images))
            _labels = []
            _images = []

def error():
    wrong = 0
    count = 0

    for (_label, _image) in inputs(0.05): # quickly estimate error on 5% of inputs
        result = prediction.eval(feed_dict={
            keep_prob: 1.0,
            labels: [_label],
            images: [_image] })
        if (result[0][1] > result[0][0]) != (_label[1] > _label[0]):
            wrong += 1
        count += 1
    
    return wrong/count

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

# dropout: [1024] -> [1024]
keep_prob = tf.placeholder(tf.float32)
output_d = tf.nn.dropout(output_4, keep_prob)

# [1024] -> [2]
kernel_5 = weights([1024, 2])
output_5 = tf.matmul(output_d, kernel_5) + bias([2])

prediction = tf.nn.softmax(output_5)

optimizer = tf.train.AdamOptimizer(1e-4).minimize(
    tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=output_5)))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print("Error: %.2f" % (error()))

    for i in range(1000):
        for (_labels, _images) in batches(50, 0.25):
            optimizer.run(feed_dict={
                keep_prob: 0.5,
                labels: _labels,
                images: _images })

        print("Error: %.2f epoch %d" % (error(), i + 1))
