import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("dropout-prob", 0, "Dropout probability")
DROPOUT_PROB = FLAGS.dropout_prob
DROPOUT_KEEP_PROB = 1 - DROPOUT_PROB

tf.app.flags.DEFINE_int("n-epochs", 1, "Number of epochs")
N_EPOCHS = FLAGS.n_epochs
