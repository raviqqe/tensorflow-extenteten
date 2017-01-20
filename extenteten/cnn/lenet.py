import tensorflow as tf

from ..layer import linear
from ..util import func_scope


@func_scope()
def lenet(images, output_size):
    h = tf.contrib.slim.conv2d(images, 32, 5, scope='conv0')
    h = tf.contrib.slim.max_pool2d(h, 2, 2, scope='pool0')
    h = tf.contrib.slim.conv2d(h, 64, 5, scope='conv1')
    h = tf.contrib.slim.max_pool2d(h, 2, 2, scope='pool1')
    h = tf.contrib.slim.flatten(h)
    return linear(h, output_size)
