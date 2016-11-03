import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("dropout-prob", 0, "Dropout probability")
tf.app.flags.DEFINE_int("n-epochs", 1, "Number of epochs")
