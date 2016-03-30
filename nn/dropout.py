import tensorflow as tf



def dropout(x, dropout_ratio):
  return tf.dropout(x, 1 - dropout_ratio)
