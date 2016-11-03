import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
DROPOUT_PROB = FLAGS.dropout_prob
DROPOUT_KEEP_PROB = 1 - DROPOUT_PROB
