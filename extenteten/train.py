import tensorflow as tf


__all__ = ['global_step', 'minimize']


def global_step():
    return tf.contrib.framework.get_or_create_global_step()


def minimize(loss):
    return tf.train.AdamOptimizer().minimize(loss, global_step())
