import os
import json
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def inputs():
    for name in os.listdir(".bin/features"):
        config = json.load(open(".bin/features/" + name))
        image = np.array(config["features"]) # [11, 11, 5] - NHWC
        label = config["safe"]
        yield (label, image)

def testnn(_label, _image):
    images = tf.placeholder(tf.float32, shape=[1, 11, 11, 5])
    kernel = tf.random_normal(dtype=tf.float32, shape=[3, 3, 5, 1])

    output = tf.nn.conv2d(images, kernel,
        strides=[1, 1, 1, 1],
        padding='VALID')

    with tf.Session() as session:
        result = session.run(output, feed_dict={
            images: [_image]
        })

        return np.reshape(result, [9, 9])

for (label, image) in inputs():
    print("image.shape:", image.shape)
    print(testnn(label, image))
