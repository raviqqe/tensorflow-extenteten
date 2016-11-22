import tensorflow as tf

from ..flags import FLAGS
from ..util import func_scope, dtypes, static_shapes



@func_scope()
def batch_queue(dtypes):
  return tf.FIFOQueue(FLAGS.batch_queue_capacity, dtypes)


@func_scope()
def requeue(*tensors, capacity=2):
  queue = tf.PaddingFIFOQueue(capacity,
                              dtypes(*tensors),
                              static_shapes(*tensors))
  add_queue_runner(queue, [queue.enqueue(tensors)])
  return queue


@func_scope()
def add_queue_runner(queue, enqueues):
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueues))
