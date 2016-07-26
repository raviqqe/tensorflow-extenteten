import tensorflow as tf



def global_step():
  return tf.Variable(0, trainable=False, name="global_step")
