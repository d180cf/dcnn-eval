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
MAS = 7 # min area size

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

def get_configs(count = 0):
    index = 0
    for name in os.listdir(".bin/features"):
        index += 1

        if count and index > count:
            break

        data = json.load(open(".bin/features/" + name))
        size = data["area"]

        if size < MAS:
            continue

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
    session.run(tf.global_variables_initializer())

    for i in range(1000):
        _total = time.time()
        _train = 0

        for (_labels, _images) in batches(25, 0.5):
            _ts = time.time()

            optimizer.run(feed_dict={
                labels: _labels,
                images: _images })

            _train += time.time() - _ts

        _total = time.time() - _total
        (err_0, err_1, corr) = error()
        print("error %.2f = %.2f + %.2f, correlation %.2f, iteration %d, spent on training %.2f, total %.1fs"
            % (err_0 + err_1, err_0, err_1, corr, i + 1, _train/_total, _total))
