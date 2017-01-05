import tensorflow as tf


def dropout(x, dropout_prob, mode):
    return tf.nn.dropout(
        x,
        1 - dropout_prob if mode == tf.contrib.learn.ModeKeys.TRAIN else 0)
