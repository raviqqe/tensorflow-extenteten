import functools
import tensorflow as tf

from . import cnn_dailymail_rc
from ..flags import FLAGS



READERS = { "cnn_dailymail_rc": cnn_dailymail_rc.read_files }


def read_files(file_pattern, file_format):
  return READERS[file_format](_file_pattern_to_names(file_pattern))


def _file_pattern_to_names(pattern):
  return tf.train.string_input_producer(tf.train.match_filenames_once(pattern),
                                        num_epochs=FLAGS.num_epochs,
                                        capacity=NotImplemented)
