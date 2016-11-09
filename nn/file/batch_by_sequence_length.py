import tensorflow as tf

from ..flags import FLAGS
from ..util import funcname_scope



@funcname_scope
def batch_by_sequence_length(tensors, length_fn=(lambda x: tf.shape(x[1])[0])):
  return tf.contrib.training.bucket_by_sequence_length(
      length_fn(tensors), # first tensor is key
      list(tensors),
      FLAGS.batch_size,
      [int(num) for num in FLAGS.length_boundaries],
      num_threads=FLAGS.num_threads_per_queue,
      capacity=FLAGS.batch_queue_capacity,
      dynamic_pad=True)[1]
