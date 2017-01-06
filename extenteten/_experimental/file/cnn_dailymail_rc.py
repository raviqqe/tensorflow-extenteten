import numpy as np
import tensorflow as tf

from .batch_by_sequence_length import batch_by_sequence_length
from .sort import sorted_batch


class _RcFileReader:

    def __init__(self, word_indices):
        self._word_indices = word_indices

    def read(self, filename_queue):
        key, value = tf.WholeFileReader().read(filename_queue)
        return (key, *self._read_record(value))

    def _read_record(self, string):
        def read_record(string):
            context, question, answer = string.decode().split("\n\n")[1:4]
            context = self._map_document_to_indices(context)
            question = self._map_document_to_indices(question)
            return (context,
                    question,
                    *map(lambda x: np.array(x, dtype=np.int32),
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
        # unknown
        return self._word_indices[word] if word in self._word_indices else 1

    def _map_document_to_indices(self, document):
        return np.array(
            [self._map_word_to_index(word) for word in document.split()],
            dtype=np.int32)


def read_files(filename_queue):
    return sorted_batch(*_RcFileReader().read(filename_queue))
