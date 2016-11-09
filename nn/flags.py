import functools
import tensorflow as tf



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("batch-size", 64, "")
tf.app.flags.DEFINE_float("dropout-prob", 0, "")
tf.app.flags.DEFINE_string("word-file", None, "")
tf.app.flags.DEFINE_integer("num-threads-per-queue", 2, "")
tf.app.flags.DEFINE_integer("queue-capacity", 2, "")
tf.app.flags.DEFINE_string("length-boundaries", "", "")
tf.app.flags.DEFINE_string("rnn-cell", "ln_lstm", "Default RNN cell")
tf.app.flags.DEFINE_string("float-type", "float32", "")
tf.app.flags.DEFINE_string("int-type", "int32", "")


@functools.lru_cache()
def _original_words():
  with open(tf.app.flags.FLAGS.word_file) as file_:
    return sorted([line.strip() for line in file_.readlines()])


@functools.lru_cache()
def word_indices():
  indices = { word: index + 2 for index, word in enumerate(_original_words()) }
  indices.update({ '<NULL>': 0, '<UNKNOWN>': 1 })
  return indices


@functools.lru_cache()
def word_space_size():
  return len(word_indices())


def rnn_cell():
  from .rnn import cell
  return getattr(cell, FLAGS.rnn_cell)


def float_type():
  return getattr(tf, FLAGS.float_type)


def int_type():
  return getattr(tf, FLAGS.int_type)
