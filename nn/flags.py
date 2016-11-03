import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("dropout-prob", 0, "Dropout probability")
