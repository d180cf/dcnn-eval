import os
import sys
import json
import time
import numpy as np
import random
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N = 11 # board frame size
F = int(sys.argv[1]) # features
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

def get_configs():
    for name in os.listdir(".bin/features"):
        data = json.load(open(".bin/features/" + name))
        yield {
            'features': np.array(data["features"]),
            'target': np.array(data["target"]),
            'label': np.array([1, 0] if data['safe'] == 0 else [0, 1]),
            'name': name
        }

print("Parsing JSON files...")
all_configs = list(get_configs()) # preload all the relevant JSON files
train_configs = [x for x in all_configs if x['name'][0] != '0']
check_configs = [x for x in all_configs if x['name'][0] == '0'] # 1/16 of all inputs
print("Inputs: %dK (check = %dK, train = %dK)" % (
    len(all_configs)//1000,
    len(check_configs)//1000,
    len(train_configs)//1000))

def inputs(prob, configs = train_configs):
    for config in configs:
        if random.random() > prob: # pick only 10% of the inputs
            continue

        target = config["target"] # [M, 2] - a list of (x, y) coords
        image = config["features"] # [board.size + 2, board.size + 2, 5] - NHWC
        label = config["label"]        
        [tx, ty] = random.choice(target)        
        frame = submatrix(image, tx - N//2, tx + N//2, ty - N//2, ty + N//2)

        # the result is invariant wrt transposition
        if (random.randint(0, 1) == 1): 
            frame = frame.transpose((1, 0, 2))

        # the result is invariant wrt rotation
        for i in range(random.randint(0, 3)):
            frame = np.rot90(frame)
        
        yield (label, frame)

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
    
    if len(_labels) > 0:
        yield (np.array(_labels), np.array(_images))

def error():
    n = 0
    err_0 = 0
    err_1 = 0
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x2 = 0
    sum_y2 = 0

    for (_label, _image) in inputs(1.0, check_configs):
        result = prediction.eval(feed_dict={
            keep_prob: 1.0,
            labels: [_label],
            images: [_image] })

        # 1 = safe; 0 = unsafe
        x = result[0][1]
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

        n += 1
    
    correlation = (n*sum_xy - sum_x*sum_y) / ((n*sum_x2 - sum_x**2)*(n*sum_y2 - sum_y**2))**0.5

    return (err_0/n, err_1/n, correlation)

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


images = tf.placeholder(tf.float32, shape=[None, N, N, F])
labels = tf.placeholder(tf.float32, shape=[None, 2])

def make_dcnn():
    # [11, 11, F] x [3, 3, F, 32] -> [9, 9, 32]
    kernel_1 = weights([K, K, F, 32])
    bias_1 = bias([32])
    output_1 = tf.nn.relu(conv2d(images, kernel_1) + bias_1)

    # [9, 9, 32] x [3, 3, 32, 32] -> [7, 7, 32]
    kernel_2 = weights([K, K, 32, 32])
    bias_2 = bias([32])
    output_2 = tf.nn.relu(conv2d(output_1, kernel_2) + bias_2)

    # [7, 7, 32] x [3, 3, 32, 32] -> [5, 5, 32]
    kernel_3 = weights([K, K, 32, 32])
    bias_3 = bias([32])
    output_3 = tf.nn.relu(conv2d(output_2, kernel_3) + bias_3)

    # [5, 5, 32] -> [1024]
    kernel_4 = weights([5*5*32, 1024])
    bias_4 = bias([1024])
    output_4 = tf.matmul(tf.reshape(output_3, [-1, 5*5*32]), kernel_4) + bias_4

    # dropout: [1024] -> [1024]
    keep_prob = tf.placeholder(tf.float32)
    output_5 = tf.nn.dropout(output_4, keep_prob)

    # [1024] -> [2]
    kernel_6 = weights([1024, 2])
    bias_6 = bias([2])
    output_6 = tf.matmul(output_5, kernel_6) + bias_6

    return (
        keep_prob,
        tf.nn.softmax(output_6),
        tf.train.AdamOptimizer(1e-4).minimize(
            tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=output_6))))

(keep_prob, prediction, optimizer) = make_dcnn()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(500):
        _total = time.time()
        _train = 0

        # pick 10% of inputs and split them into 25-item batches
        for (_labels, _images) in batches(25, 0.1):
            _ts = time.time()

            optimizer.run(feed_dict={
                keep_prob: 0.5,
                labels: _labels,
                images: _images })

            _train += time.time() - _ts

        _total = time.time() - _total
        (err_0, err_1, corr) = error()
        print("error %.2f = %.2f + %.2f, correlation %.2f, iteration %d, spent on training %.2f, total %.1fs"
            % (err_0 + err_1, err_0, err_1, corr, i + 1, _train/_total, _total))
