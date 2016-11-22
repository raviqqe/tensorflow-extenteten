import functools
import tensorflow as tf

from . import cnn_dailymail_rc
from .. import collections
from ..flags import FLAGS
from ..util import func_scope, dtypes
from .util import batch_queue, add_queue_runner



READERS = { "cnn_dailymail_rc": cnn_dailymail_rc.read_files }


@func_scope()
def read_files(file_pattern, file_format):
  return monitored_batch_queue(
      *READERS[file_format](_file_pattern_to_names(file_pattern)))


@func_scope()
def _file_pattern_to_names(pattern):
  return tf.train.string_input_producer(tf.train.match_filenames_once(pattern),
                                        num_epochs=FLAGS.num_epochs,
                                        capacity=FLAGS.filename_queue_capacity)


@func_scope()
def monitored_batch_queue(*tensors):
  queue = batch_queue(dtypes(*tensors))
  collections.add_metric(queue.size(), "batches_in_queue")

  add_queue_runner(queue, [queue.enqueue(tensors)])

  results = queue.dequeue()
  for tensor, result in zip(tensors, results):
    result.set_shape(tensor.get_shape())

  return results
