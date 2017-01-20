import tensorflow as tf

from .lenet import lenet


def test_lenet():
    lenet(tf.zeros([64, 24, 24, 1]), 200)
