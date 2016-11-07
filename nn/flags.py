import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("batch-size", 64, "")
tf.app.flags.DEFINE_float("dropout-prob", 0, "")
tf.app.flags.DEFINE_string("word-file", None, "")
tf.app.flags.DEFINE_integer("num-threads-per-queue", 2, "")
tf.app.flags.DEFINE_integer("queue-capacity", 2, "")
