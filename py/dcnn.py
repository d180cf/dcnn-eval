import os
import sys
import json
import time
import datetime
import numpy as np
import random
import tensorflow as tf
from importlib import import_module

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
duration = float(sysarg(7) or 1.5) # hours
NN_INFO = sysarg(8) or 'rb1(3,64)' # model name + params

if not NN_INFO:
    print('NN name not specified')
    sys.exit()

NN_NAME = NN_INFO.split('(')[0] # e.g. "rb1(3,64)" -> "rb1"
NN_ARGS = () if NN_NAME == NN_INFO else eval(NN_INFO[len(NN_NAME):])
SHUFFLE_WINDOW = 8192
BATCH_SIZE = 256
EPOCH_DURATION = 30.0 # seconds
NUM_EPOCHS = 1000
LR_INITIAL = 1.0
LR_FACTOR = 0.5
LR_DECAY = 20e6

model_name = NN_NAME + '-' + '-'.join([str(x) for x in NN_ARGS])

print('Target frame: %dx%d' % (N, N))
print('Features: %d' % (F))
print('NN info: ' + NN_INFO)

model_file = models_dir + '/' + model_name + '.json'
os.makedirs(models_dir, exist_ok=True)
print('Model file: ' + model_file)

tb_indx = 1
while os.path.isdir(logs_path + '/' + model_name + '/' + str(tb_indx)):
    tb_indx += 1
tb_path = logs_path + '/' + model_name + '/' + str(tb_indx)
print('TensorBoard logs: ' + tb_path)

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

    # the feature planes include the board wall, hence (x + 1, y + 1)
    tx = t[0] + 1
    ty = t[1] + 1

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

def make_dataset(filepath, batchsize):
    dataset = tf.data.TFRecordDataset(filepath, buffer_size=2**29)
    dataset = dataset.map(parse, num_parallel_calls=3)
    dataset = dataset.shuffle(SHUFFLE_WINDOW)
    dataset = dataset.repeat()
    dataset = dataset.batch(batchsize)
    return dataset

tprint('Initializing the main dataset...')    
dataset_main = make_dataset(ds_main, BATCH_SIZE)

tprint('Initializing the test dataset...')    
dataset_test = make_dataset(ds_test, SHUFFLE_WINDOW)

images = tf.placeholder(tf.float32, shape=[None, N, N, F])
labels = tf.placeholder(tf.float32, shape=[None])
is_training = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)

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
make_dcnn = import_module('graphs.' + NN_NAME).make_dcnn
(prediction, loss, optimizer) = make_dcnn(images, labels, learning_rate, is_training, *NN_ARGS)

avg_1 = tf.reduce_sum(labels * prediction) / tf.cast(tf.count_nonzero(labels), tf.float32)
avg_0 = tf.reduce_sum((1 - labels) * prediction) / tf.cast(tf.count_nonzero(1 - labels), tf.float32)
error = 1 - tf.reduce_mean(tf.nn.relu(tf.sign((prediction - 0.5) * (labels - 0.5))))
corr = correlation(prediction, labels)

print('DCNN variables:')
for v in tf.trainable_variables():
    print(v.shape, v.name)

print('Starting the session...')
with tf.Session() as session:
    iterator_main = dataset_main.make_initializable_iterator()
    next_batch_main = iterator_main.get_next()    
    session.run(iterator_main.initializer)

    iterator_test = dataset_test.make_initializable_iterator()
    next_batch_test = iterator_test.get_next()    
    session.run(iterator_test.initializer)

    session.run(tf.global_variables_initializer())    

    tf.summary.scalar('A_error', error)
    tf.summary.scalar('A_loss', loss)
    tf.summary.scalar('A_correlation', corr)
    tf.summary.scalar('B_avg_0', avg_0)
    tf.summary.scalar('B_avg_1', avg_1)
    # tf.summary.scalar('C_learning_rate', learning_rate)
    merged = tf.summary.merge_all()

    test_writer = tf.summary.FileWriter(tb_path + '/validation')
    main_writer = tf.summary.FileWriter(tb_path + '/training', session.graph)

    lr = LR_INITIAL
    lr_next_decay = LR_DECAY
    step = 0 # the number of samples used for training so far
    prev = 0

    tprint('')
    tprint('%5s %5s %5s %5s %5s' % ('error', 'corr', 'save', 'test', 'M/hr'))
    tprint('')

    try:
        while time.time() < T + duration * 3600:
            t0 = time.time()
            save_model(model_file)

            t1 = time.time()
            _labels, _images = session.run(next_batch_test)
            summary, _error, _corr = session.run([merged, error, corr], feed_dict={
                is_training: False,
                learning_rate: lr,                    
                labels: _labels,
                images: _images })
            test_writer.add_summary(summary, step)

            t2 = time.time()
            speed = (step - prev) / EPOCH_DURATION * 3600 / 1e6 # millions samples per hour
            prev = step
            tprint('%5.2f %5.2f %5.1f %5.1f %5.1f' % (_error, _corr, t1 - t0, t2 - t1, speed))

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
