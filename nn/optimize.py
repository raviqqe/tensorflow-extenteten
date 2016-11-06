import tensorflow as tf
import .train



def minimize(loss):
  return tf.AdamOptimizer().minimize(loss, train.global_step())
