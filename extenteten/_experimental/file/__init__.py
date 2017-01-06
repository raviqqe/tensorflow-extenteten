import functools
import tensorflow as tf

from . import cnn_dailymail_rc
from .. import collections
from ..flags import FLAGS
from ..util import func_scope, dtypes
from .util import add_queue_runner


READERS = {"cnn_dailymail_rc": cnn_dailymail_rc.read_files}


@func_scope()
def read_files(file_pattern, file_format):
    return monitored_queue(
        *READERS[file_format](_file_pattern_to_names(file_pattern)),
        metric_name="batches_in_queue",
        capacity=FLAGS.batch_queue_capacity)


@func_scope()
def _file_pattern_to_names(pattern):
    return monitored_queue(
        tf.train.string_input_producer(tf.train.match_filenames_once(pattern),
                                       num_epochs=FLAGS.num_epochs,
                                       capacity=FLAGS.filename_queue_capacity)
        .dequeue(),
        metric_name="filenames_in_queue",
        capacity=FLAGS.filename_queue_capacity,
        return_queue=True)


@func_scope()
def monitored_queue(*tensors,
                    capacity,
                    metric_name="items_in_queue",
                    return_queue=False):
    queue = tf.FIFOQueue(capacity, dtypes(*tensors))
    collections.add_metric(queue.size(), metric_name)

    add_queue_runner(queue, [queue.enqueue(tensors)])

    if return_queue:
        return queue

    results = queue.dequeue()

    for tensor, result \
            in zip(tensors, results if isinstance(results, list) else [results]):
        result.set_shape(tensor.get_shape())

    return results
