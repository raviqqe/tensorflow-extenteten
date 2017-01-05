import tensorflow as tf



def dropout(x, dropout_prob):
  return tf.nn.dropout(x, 1 - dropout_prob)
