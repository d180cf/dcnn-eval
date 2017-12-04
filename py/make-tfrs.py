# Packs all JSON files with features into one .tfrecords file.

import os
import sys
import json
import time
import numpy as np
import random
import tensorflow as tf

dirpath = sys.argv[1] # dir with *.json files
mainfilepath = sys.argv[2] # *.tfrecords file
testfilepath = sys.argv[3] # *.tfrecords file
testprob = float(sys.argv[4]) # 0.1 = 10% goes to test

def _int64s(items):
  flat = np.array(items).reshape([-1])
  ints = [int(x) for x in flat]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=ints))

def listfiles(dirpath):
  for root, subdirs, files in os.walk(dirpath):
    for name in files:
      yield os.path.join(root, name)

def main(args):
  print('Initializing...')
  wmain = tf.python_io.TFRecordWriter(mainfilepath)
  wtest = tf.python_io.TFRecordWriter(testfilepath)
  tprev = time.time()
  nprev = 0
  count = 0
  total = sum(1 for _ in listfiles(dirpath))

  print('Converting %d files...' % total)
  for path in listfiles(dirpath):
    count += 1

    if time.time() > tprev + 5:
      speed = (count - nprev) / (time.time() - tprev)
      print('%d%% files converted: %d mins remaining...' % (
        count/total*100,
        (total - count) / speed / 60))
      tprev = time.time()
      nprev = count

    data = json.load(open(path))
    planes = data['planes']
    status = data['status'] # -1, 0, +1

    props = {
      'planes': _int64s(planes),
      'shape': _int64s(data["shape"]),
      'target': _int64s(data['target']),
      'status': _int64s([status]),
    }

    record = tf.train.Example(features=tf.train.Features(feature=props))

    writer = wtest if random.random() < testprob else wmain
    writer.write(record.SerializeToString())    

  wmain.close()
  wtest.close()

tf.app.run(main=main)
