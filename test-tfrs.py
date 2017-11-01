# Reads a .tfrecords file and prints their contents

import os
import sys
import numpy as np
import tensorflow as tf

filepath = sys.argv[1]
maxcount = int(sys.argv[2])

count = 0

for string_record in tf.python_io.tf_record_iterator(path=filepath):
    count += 1

    if count > maxcount:
        break

    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    size = int(example.features.feature['size'].int64_list.value[0])
    label = example.features.feature['label'].int64_list.value
    target = np.array(example.features.feature['target'].int64_list.value).reshape([-1, 2])
    shape = np.array(example.features.feature['shape'].int64_list.value)
    planes = np.array(example.features.feature['planes'].int64_list.value).reshape(shape)
    
    print('\n### entry %d ###' % (count))
    print('size', size)
    print('label', label)
    print('target', target)
    print('shape', shape)
    print('planes', planes)
