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
minareasize = int(sys.argv[5])

def _int64s(items):
  flat = np.array(items).reshape([-1])
  ints = [int(x) for x in flat]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=ints))

def main(args):
  wmain = tf.python_io.TFRecordWriter(mainfilepath)
  wtest = tf.python_io.TFRecordWriter(testfilepath)
  tprev = time.time()
  count = 0
  total = sum(1 for name in os.listdir(dirpath))

  for name in os.listdir(dirpath):
    count += 1

    if time.time() > tprev + 5:
      tprev = time.time()
      print('%d%% files converted: %d out of %d' % (count/total*100, count, total))

    data = json.load(open(os.path.join(dirpath, name)))
    asize = data["asize"] # area size

    if size < minareasize:
      continue

    planes = data['features']

    props = {
      'planes': _int64s(planes),
      'shape': _int64s(data["shape"]),
      'target': _int64s(data['target']),
      'label': _int64s([1, 0] if data['safe'] == 0 else [0, 1]),
      'size': _int64s([asize]),
    }

    record = tf.train.Example(features=tf.train.Features(feature=props))

    writer = wtest if random.random() < testprob else wmain
    writer.write(record.SerializeToString())    

  wmain.close()
  wtest.close()

tf.app.run(main=main)
