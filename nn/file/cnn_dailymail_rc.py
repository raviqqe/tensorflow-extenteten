import functools
import numpy as np
import tensorflow as tf

from .. import flags
from ..flags import FLAGS



class _RcFileReader:
  def __init__(self):
    # 0 -> null, 1 -> unknown
    self._word_indices = flags.word_indices()

  def read(self, filename_queue):
    key, value = tf.WholeFileReader().read(filename_queue)
    return (key, *self._read_record(value))

  def _read_record(self, string):
    def read_record(string):
      context, question, answer = string.decode().split("\n\n")[1:4]
      context = self._map_document_to_indices(context)
      question = self._map_document_to_indices(question)
      return (context, question) \
             + tuple(map(lambda x: np.array(x, dtype=np.int32),
                     [self._map_word_to_index(answer),
                      len(context),
                      len(question)]))

    context, question, answer, context_length, question_length = tf.py_func(
        read_record,
        [string],
        [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32],
        name="read_rc_data")

    context_length.set_shape([])
    question_length.set_shape([])

    return (tf.reshape(context, [context_length]),
            tf.reshape(question, [question_length]),
            tf.reshape(answer, []))

  def _map_word_to_index(self, word):
    return self._word_indices[word] if word in self._word_indices else 1 # unknown

  def _map_document_to_indices(self, document):
    return np.array([self._map_word_to_index(word)
                     for word in document.split()],
                    dtype=np.int32)


def read_files(filename_queue):
  tensors = _RcFileReader().read(filename_queue)
  return tf.contrib.training.bucket_by_sequence_length(
      tf.shape(tensors[1])[0],
      list(tensors),
      FLAGS.batch_size,
      [int(num) for num in FLAGS.length_boundaries.split(",")],
      num_threads=FLAGS.num_threads_per_queue,
      capacity=FLAGS.queue_capacity,
      dynamic_pad=True,
      allow_smaller_final_batch=True)[1]
