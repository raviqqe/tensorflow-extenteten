import tensorflow as tf



def variable(shape, name=None):
  return tf.Variable(
      (tf.contrib.layers.xavier_initializer()
       if len(shape) == 2 else
       tf.random_normal_initializer(stddev=0.1))(shape),
      name=name)
