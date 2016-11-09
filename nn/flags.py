import functools
import tensorflow as tf



FLAGS = tf.flags.FLAGS


# TODO: Use gflags and handle required option
def define_int(name, default=None, *, desc="", required=False):
  tf.flags.DEFINE_integer(name, default or 0, desc)

def define_str(name, default=None, *, desc="", required=False):
  tf.flags.DEFINE_string(name, default or "", desc)

def define_float(name, default=None, *, desc="", required=False):
  tf.flags.DEFINE_float(name, default or 0.0, desc)


# Hyperparameters

define_int("num-epochs", 64)
define_str("batch-size", 64)
define_float("dropout-prob", 0)
define_str("word-file")
define_int("num-threads-per-queue", 2)
define_int("queue-capacity", 2)
define_str("length-boundaries")
define_str("rnn-cell", "ln_lstm", desc="Default RNN cell")

# Data type

define_str("float-type", "float32")
define_str("int-type", "int32")

# Data files

define_str("file-glob", desc="File path glob to search data files")
define_str("file-format", desc="Data format of files")

# Others

define_str("log-dir", desc="Directory containing checkpoint and event files")


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
