import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("batch-size", 32, "")
tf.app.flags.DEFINE_float("dropout-prob", 0, "")
tf.app.flags.DEFINE_string("word-file", None, "")
