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
print('sys.argv', sys.argv)

def sysarg(i):
    try:
        return sys.argv[i]
    except:
        return None

ds_main = sysarg(1) # .tfrecords file with the main dataset
ds_test = sysarg(2) # .tfrecords file with the test dataset
N = int(sysarg(3)) # board frame size, e.g. 11 x 11
F = int(sysarg(4)) # the number of features features, e.g. 5
models_dir = sysarg(5) # the checkpoint file for weights
logs_path = sysarg(6) # tensorboard logs
model_name = sysarg(7) or 'test' # for tensorboard
duration = float(sysarg(8) or 1.0) # hours

SHUFFLE_WINDOW = 8192
BATCH_SIZE = 256
EPOCH_DURATION = 30.0 # seconds
NUM_EPOCHS = 1000
LR_INITIAL = 0.5
LR_FACTOR = 0.5
LR_DECAY = 1e6

print('Target frame: %dx%d' % (N, N))
print('Features: %d' % (F))

model_file = models_dir + '/' + model_name + '.json'
os.makedirs(models_dir, exist_ok=True)
print('Model file: ' + model_file)

T = time.time()
print('T = ' + datetime.datetime.now().isoformat())
print('Training ends in %.1f hours' % duration)

def tprint(text):
    dt = time.time() - T
    s = '%.1fs' % dt if dt < 60 else '%.1fm' % (dt/60) if dt < 3600 else '%.1fh' % (dt/3600)
    print('%8s %s' % ('T+' + s, text))

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
    # note, that the output format of features.js is [y, x, f]
    image = tf.pad(planes, [[N//2, N//2], [N//2, N//2], [0, 0]])
    image = image[ty : ty + N, tx : tx + N, :]
    
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

images = tf.placeholder(tf.float32, shape=[None, N, N, F])
labels = tf.placeholder(tf.float32, shape=[None])
is_training = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)

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
        x = fconn(x, n, name='conn1')
        x = tf.nn.relu(x)
        x = fconn(x, n, name='conn2')
        return tf.nn.relu(x + y)

# perhaps the simplest NN possible: a weighthed sum of all features
# maximum observed accuracy:
#   0.70 when d=0 n=16
#   0.80 when d=1 n=16
#   0.84 when d=1 n=64
#   0.85 when d=1 n=128
#   0.80 when d=2 n=16
#   0.86 when d=2 n=64
def make_dcnn_fc1(d = 2, n = 64):
    x = tf.reshape(images, [-1, N*N*F])
    print(1, x.shape)

    for i in range(d):
        x = fconn(x, n, name='internal')
        x = tf.nn.relu(x)
        print(2, x.shape)

    x = fconn(x, 1, name='readout')
    x = tf.sigmoid(x)
    print(3, x.shape)

    y = tf.reshape(x, [-1])
    e = tf.losses.mean_squared_error(labels, y)
    return (y, e, tf.train.GradientDescentOptimizer(learning_rate).minimize(e))

# a few residual blocks followed by a fully connected layer
def make_dcnn_rb1(d = 3, n = 64):
    x = tf.reshape(images, [-1, N*N*F])
    print(1, x.shape)

    x = fconn(x, n, name='internal')
    x = tf.nn.relu(x)
    print(2, x.shape)

    for i in range(d):
        x = resb(x, n, name='residual')
        print(3, x.shape)

    x = fconn(x, 1, name='readout')
    x = tf.sigmoid(x)
    print(4, x.shape)

    y = tf.reshape(x, [-1])
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

        mean, var = tf.cond(is_training,
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

# TF must have a better way to save the model, but I haven't figured it out.
def save_model(file_path):
    vars = {}

    for v in tf.trainable_variables():
        data = v.eval().reshape([-1]).tolist()

        vars[v.name] = {
            'shape': v.shape.as_list(),
            'min': np.amin(data),
            'max': np.amax(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'data': data }    

    with open(file_path, 'w') as file:
        json.dump({
            'vars': vars,
            'ops': {}
        }, file)

def correlation(x, y):
    xm, xv = tf.nn.moments(x, [0])
    ym, yv = tf.nn.moments(y, [0])
    xym, xyv = tf.nn.moments((x - xm)*(y - ym), [0])
    return xym / tf.sqrt(xv * yv)

print('Constructing DCNN...')
(prediction, loss, optimizer) = make_dcnn_rb1()

avg_1 = tf.reduce_sum(labels * prediction) / tf.cast(tf.count_nonzero(labels), tf.float32)
avg_0 = tf.reduce_sum((1 - labels) * prediction) / tf.cast(tf.count_nonzero(1 - labels), tf.float32)
accuracy = tf.reduce_mean(tf.nn.relu(tf.sign((prediction - 0.5) * (labels - 0.5))))
corr = correlation(prediction, labels)

print('DCNN variables:')
for v in tf.trainable_variables():
    print(v)

with tf.Session() as session:
    iterator_main = dataset_main.make_initializable_iterator()
    next_batch_main = iterator_main.get_next()    
    session.run(iterator_main.initializer)

    iterator_test = dataset_test.make_initializable_iterator()
    next_batch_test = iterator_test.get_next()    
    session.run(iterator_test.initializer)    

    tprint('Initializing global variables...')
    session.run(tf.global_variables_initializer())    

    tf.summary.scalar('A_accuracy', accuracy)
    tf.summary.scalar('A_loss', loss)
    tf.summary.scalar('A_correlation', corr)
    tf.summary.scalar('B_avg_0', avg_0)
    tf.summary.scalar('B_avg_1', avg_1)
    tf.summary.scalar('C_learning_rate', learning_rate)
    merged = tf.summary.merge_all()

    lg_path = '%s/%s' % (logs_path, model_name)
    tprint('TensorBoard logs: ' + lg_path)

    test_writer = tf.summary.FileWriter(lg_path + '/validation')
    main_writer = tf.summary.FileWriter(lg_path + '/training', session.graph)

    lr = LR_INITIAL
    lr_next_decay = LR_DECAY
    step = 0 # the number of samples used for training

    tprint('%5s %5s %5s %5s' % ('acc', 'corr', 'save', 'tb'))

    try:
        while time.time() < T + duration * 3600:
            t0 = time.time()
            save_model(model_file)

            t1 = time.time()
            _labels, _images = session.run(next_batch_test)
            summary, _accuracy, _corr = session.run([merged, accuracy, corr], feed_dict={
                is_training: False,
                learning_rate: lr,                    
                labels: _labels,
                images: _images })
            test_writer.add_summary(summary, step)

            t2 = time.time()
            tprint('%5.2f %5.2f %5.1f %5.1f' % (_accuracy, _corr, t1 - t0, t2 - t1))

            while time.time() < t2 + EPOCH_DURATION:
                _labels, _images = session.run(next_batch_main)
                summary, _ = session.run([merged, optimizer], feed_dict={
                    is_training: True,
                    learning_rate: lr,                    
                    labels: _labels,
                    images: _images })
                step += BATCH_SIZE
                main_writer.add_summary(summary, step)

            if step >= lr_next_decay:
                lr *= LR_FACTOR
                lr_next_decay += LR_DECAY
                tprint('learning rate = %f' % lr)
                
    except KeyboardInterrupt:
        tprint('Terminated by Ctrl+C')
