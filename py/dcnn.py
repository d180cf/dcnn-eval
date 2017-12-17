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
EPOCH_DURATION = 60.0 # seconds
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
print('Weights: ' + model_file)

tb_indx = 1
while os.path.isdir(logs_path + '/' + model_name + '/' + str(tb_indx)):
    tb_indx += 1
tb_path = logs_path + '/' + model_name + '/' + str(tb_indx)
print('TensorBoard logs: ' + tb_path)

T = time.time()
print('T = ' + datetime.datetime.now().isoformat())
print('Training ends in %.1f hours' % duration)

def tprint(text):
    if text:
        dt = time.time() - T
        s = '%.1fs' % dt if dt < 60 else '%.1fm' % (dt/60) if dt < 3600 else '%.1fh' % (dt/3600)
        print('%8s %s' % ('T+' + s, text))
    else:
        print('')

def parse(example):
    features = tf.parse_single_example(example, {
        "label": tf.FixedLenFeature((1), tf.int64),
        "shape": tf.FixedLenFeature((3), tf.int64),
        "planes": tf.VarLenFeature(tf.int64),
        "target": tf.VarLenFeature(tf.int64),
        "moves": tf.VarLenFeature(tf.int64) })

    label = features["label"] # 0 or 1
    shape = features["shape"] # [N, N, F] where F is the number of features and N is the size of the board
    planes = features["planes"] # the features tensor with the shape above
    target = features["target"] # list of [target.x, target.y] pointers
    moves = features["moves"] # list of [move.x, move.y] pointers

    shape = tf.cast(shape, tf.int32) # otherwise TF crashes with weird CPU/GPU related error

    planes = tf.sparse_tensor_to_dense(planes) # when TF was writing the file, it apparently compressed it
    planes = tf.reshape(planes, shape)

    target = tf.sparse_tensor_to_dense(target)
    target = tf.reshape(target, [-1, 2])

    moves = tf.sparse_tensor_to_dense(moves)
    moves = tf.reshape(moves, [-1, 2])

    count = tf.shape(target)[0]
    index = tf.random_uniform([1], 0, count, tf.int32)[0]

    t = target[index]

    tx = t[0]
    ty = t[1]

    def make_policy_image():
        m = moves[0] # TODO: pick a random move
        mx = m[0] - (tx - N//2)
        my = m[1] - (ty - N//2)
        w = tf.one_hot(my * N + mx, N**2)
        return tf.reshape(w, [N, N])

    def make_random_image():
        rn = tf.random_uniform([2], 0, N - 1, tf.int32)
        mx = rn[0]
        my = rn[1]
        w = tf.one_hot(my * N + mx, N**2)
        return tf.reshape(w, [N, N])

    bmoves = tf.cond(tf.size(moves) > 0,
        make_policy_image,
        make_random_image)

    # `image` = 11 x 11 slice around [tx, ty] from `planes`, padded with 0s
    # note, that the output format of features.js is [y, x, f]
    image = tf.pad(planes, [[N//2, N//2], [N//2, N//2], [0, 0]])
    image = image[ty : ty + N, tx : tx + N, :]
    
    transpose = tf.random_uniform([1], 0, 2, tf.int32)[0]
    image = tf.cond(transpose > 0,
        lambda: tf.transpose(image, [1, 0, 2]),
        lambda: image)

    # rotate up to 3 times
    rotate = tf.random_uniform([1], 0, 4, tf.int32)[0]
    image = tf.image.rot90(image, rotate)

    return (label[0], bmoves, image)

def print_sample(label, moves, image):
    def get_char(x):
        return '-' if x == 0 else '#' if x == 1 else '?'

    def m2str(m, f):
        return '\n'.join([' '.join([ f(x) for x in r]) for r in m])

    print('label = %d' % label)
    print('moves = ')    
    print(m2str(moves, get_char))
    planes = np.transpose(image, [2, 0, 1]) # [N, N, F] -> [F, N, N]

    for index, plane in enumerate(planes):
        print('image[%d] = ' % index)
        print(m2str(plane, get_char))

def make_dataset(filepath, batchsize):
    dataset = tf.data.TFRecordDataset(filepath, buffer_size=2**29)
    dataset = dataset.map(parse, num_parallel_calls=3)
    dataset = dataset.shuffle(SHUFFLE_WINDOW)
    dataset = dataset.repeat()
    dataset = dataset.batch(batchsize)
    return dataset

dataset_main = make_dataset(ds_main, BATCH_SIZE)
dataset_test = make_dataset(ds_test, SHUFFLE_WINDOW)

images = tf.placeholder(tf.float32, shape=[None, N, N, F])
bmoves = tf.placeholder(tf.float32, shape=[None, N, N])
labels = tf.placeholder(tf.float32, shape=[None])
is_training = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)

def save_model(file_path):
    # TODO: TF must have a better way to save the model, but I haven't figured it out.
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
(eval_pred, move_pred, loss) = make_dcnn(images, labels, bmoves, is_training, *NN_ARGS)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# the average value predictions for cases when the true value is 1 or 0
eval_1 = tf.reduce_sum(labels * eval_pred) / tf.cast(tf.count_nonzero(labels), tf.float32)
eval_0 = tf.reduce_sum((1 - labels) * eval_pred) / tf.cast(tf.count_nonzero(1 - labels), tf.float32)

# average error and correlation for the value prediction
eval_error = 1 - tf.reduce_mean(tf.nn.relu(tf.sign((eval_pred - 0.5) * (labels - 0.5))))
eval_corr = correlation(eval_pred, labels)

# average error for move prediction
def get_move_error():
    moves_1 = tf.reshape(bmoves, [-1, N**2])
    moves_2 = tf.reshape(move_pred, [-1, N**2])
    index_1 = tf.argmax(moves_1, 1)
    index_2 = tf.argmax(moves_2, 1)
    correct = tf.equal(index_1, index_2)
    return 1 - tf.reduce_mean(tf.cast(correct, tf.float32))

move_error = get_move_error()

print('Trainable variables:')
for v in tf.trainable_variables():
    print('%15s %s' % (v.shape, v.name))

# print('UPDATE_OPS:')
# for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
#     print('%15s %s' % (v.shape, v.name))

print('Starting the session...')
with tf.Session() as session:
    iterator_main = dataset_main.make_initializable_iterator()
    next_batch_main = iterator_main.get_next()    
    session.run(iterator_main.initializer)

    iterator_test = dataset_test.make_initializable_iterator()
    next_batch_test = iterator_test.get_next()    
    session.run(iterator_test.initializer)

    session.run(tf.global_variables_initializer())    

    tf.summary.scalar('eval error', eval_error)
    tf.summary.scalar('move error', move_error)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('correlation', eval_corr)
    tf.summary.scalar('eval 0', eval_0)
    tf.summary.scalar('eval 1', eval_1)
    tf.summary.scalar('learning rate', learning_rate)

    merged = tf.summary.merge_all()

    test_writer = tf.summary.FileWriter(tb_path + '/validation')
    main_writer = tf.summary.FileWriter(tb_path + '/training', session.graph)

    lr = LR_INITIAL
    lr_next_decay = LR_DECAY
    step = 0 # the number of samples used for training so far
    prev = 0

    tprint('')
    tprint('%5s %5s %5s %5s %5s %5s %7s' % ('step', 'e_err', 'm_err', 'corr', 'save', 'test', 'perf'))
    tprint('')

    try:
        while time.time() < T + duration * 3600:
            t0 = time.time()
            save_model(model_file)

            t1 = time.time()
            _labels, _bmoves, _images = session.run(next_batch_test)

            # for i in range(5):
            #   print('sample # ', i)
            #   print_sample(_labels[i], _bmoves[i], _images[i])
            # sys.exit()

            summary, _err_eval, _err_move, _corr, _values = session.run([merged, eval_error, move_error, eval_corr, eval_pred], feed_dict={
                is_training: False,
                learning_rate: lr,
                bmoves: _bmoves,
                labels: _labels,
                images: _images })

            # print('values = ', _values)
            # print('labels = ', _labels)
            # sys.exit()

            test_writer.add_summary(summary, step)

            t2 = time.time()
            speed = (step - prev) / EPOCH_DURATION * 3600 / 1e6 # thousands samples per hour
            prev = step
            tprint('%5s %5.2f %5.2f %5.2f %5s %5s %7s' % (
                '%4.1fM' % (step / 1e6),
                _err_eval,
                _err_move,
                _corr,
                '%4.1fs' % (t1 - t0),
                '%4.1fs' % (t2 - t1),
                '%4.1fM/h' % speed))

            while time.time() < t2 + EPOCH_DURATION:
                _labels, _bmoves, _images = session.run(next_batch_main)
                summary, _ = session.run([merged, optimizer], feed_dict={
                    is_training: True,
                    learning_rate: lr,                    
                    bmoves: _bmoves,
                    labels: _labels,
                    images: _images })
                step += BATCH_SIZE
                main_writer.add_summary(summary, step)

            if step >= lr_next_decay:
                lr *= LR_FACTOR
                lr_next_decay += LR_DECAY
                tprint('learning rate = %f' % lr)
                
    except KeyboardInterrupt:
        sys.exit()
