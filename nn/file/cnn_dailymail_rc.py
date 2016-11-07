import functools
import tensorflow as tf



class _RcFileReader:
  def __init__(self):
    # 0 -> null, 1 -> unknown
    self._word_indices = { word: index + 2 for index, word
                           in enumerate(self._read_words_file()) }

  def read(self, filename_queue):
    key, value = tf.WholeFileReader().read(filename_queue)
    return key, *self._read_record(value)

  def _read_record(self, string):
    def read_record(string):
      context, question, answer = string.split("\n\n")[1:4]
      return (self._map_words_to_indices(context),
              self._map_words_to_indices(question),
              self._map_word_to_index(answer))
    return tf.py_func(read_record,
                      [string],
                      [tf.int32, tf.int32, tf.int32],
                      name="read_rc_data")

  def _map_word_to_index():
    return word_indices[word] if word in self._word_indices else 1 # unknown

  def _map_document_to_indices(self, document):
    return np.array([self._map_word_to_index(word)
                     for word in document.split()],
                    dtype=np.int32)

  def _read_words_file(self):
    with open(tf.app.flags.FLAGS.words_file) as file_:
      return sorted([line.strip() for line in file_.readlines()])


def read_files(filename_queue):
  return tf.train.shuffle_batch(
      _RcFileReader().read(filename_queue),
      FLAGS.batch_size,
      capacity=4*FLAGS.batch_size,
      min_after_dequeue=2**8,
      num_threads=2**4,
      allow_smaller_final_batch=True)
