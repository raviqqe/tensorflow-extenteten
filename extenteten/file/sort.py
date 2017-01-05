import functools
import operator
import tensorflow as tf

from .. import control, collections, transform
from ..flags import FLAGS
from ..util import func_scope, dtypes, static_shapes, static_shape
from .util import requeue, add_queue_runner



def _num_prefetched_samples():
  return FLAGS.batch_queue_capacity * FLAGS.batch_size


@func_scope()
def sorted_batch(*tensors):
  queue = requeue(*tensors, capacity=2*_num_prefetched_samples())
  collections.add_metric(queue.size(), "unsorted_samples_in_queue")

  return _gather_into_queue(*_sort_by_length(
      *[queue.dequeue() for _ in range(_num_prefetched_samples())]))


@func_scope()
def _gather_into_queue(*tensor_lists):
  assert len(tensor_lists) % FLAGS.batch_size == 0

  queue = tf.RandomShuffleQueue(FLAGS.batch_queue_capacity,
                                FLAGS.batch_queue_capacity // 2,
                                dtypes(*tensor_lists[0]))
  collections.add_metric(queue.size(), "sorted_batches_in_queue")

  add_queue_runner(
      queue,
      [tf.group(*[
          queue.enqueue(transform.batch(*tensor_lists[i:i+FLAGS.batch_size]))
          for i in range(0, len(tensor_lists), FLAGS.batch_size)])])

  results = queue.dequeue()

  for result, tensor in zip(results, tensor_lists[0]):
    result.set_shape([None, *static_shape(tensor)])

  return results


@func_scope()
def _sort_by_length(*tensor_lists,
                    length_fn=(lambda arrays: arrays[1].shape[0])):
  tensor_list = tensor_lists[0]
  list_length = len(tensor_list)
  list_dtypes = dtypes(*tensor_list)

  def pack(*tensors):
    assert len(tensors) % list_length == 0
    return [tensors[i:i+list_length]
            for i in range(0, len(tensors), list_length)]

  def op(*arrays):
    array_lists = pack(*arrays)

    indices = [*zip(*sorted([(length, index) for index, length
                             in enumerate(map(length_fn, array_lists))]))][1]

    return _merge_lists(*[array_lists[index] for index in indices])

  flatten_tensors = _merge_lists(*tensor_lists)
  results = tf.py_func(op,
                       flatten_tensors,
                       dtypes(*flatten_tensors),
                       name="sort_by_length")

  for result, tensor in zip(results, flatten_tensors):
    result.set_shape(tensor.get_shape())

  return pack(*results)


def _merge_lists(*lists):
  return functools.reduce(operator.add, lists)
