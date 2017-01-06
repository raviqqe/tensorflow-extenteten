import tensorflow as tf

from ..flags import FLAGS
from ..util import func_scope, dtypes, static_shapes


@func_scope()
@func_scope()
def requeue(*tensors,
            capacity=FLAGS.num_threads_per_queue,
            num_threads=FLAGS.num_threads_per_queue):
    queue = tf.PaddingFIFOQueue(capacity,
                                dtypes(*tensors),
                                static_shapes(*tensors))
    add_queue_runner(queue, [queue.enqueue(tensors)] * num_threads)
    return queue


@func_scope()
def add_queue_runner(queue, enqueues):
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueues))
