import functools
import operator
import tensorflow as tf

from .. import control, collections
from ..flags import FLAGS
from ..util import func_scope, dtypes, static_shapes
from .util import requeue, add_queue_runner



def _num_prefetched_samples():
  return FLAGS.batch_queue_capacity * FLAGS.batch_size


@func_scope()
def sorted_batch(*tensors):
  return prefetch_and_sort(*tensors).dequeue_many(FLAGS.batch_size)


@func_scope()
def prefetch_and_sort(*tensors):
  queue = requeue(*tensors)
  collections.add_metric(queue.size(), "unsorted_samples_in_queue")

  return _gather_into_queue(*_sort_by_length(
      *[queue.dequeue() for _ in range(_num_prefetched_samples())]))


@func_scope()
def _gather_into_queue(*tensor_lists):
  tensor_list = tensor_lists[0]

  queue = tf.PaddingFIFOQueue(_num_prefetched_samples(),
                              dtypes(*tensor_list),
                              static_shapes(*tensor_list))
  collections.add_metric(queue.size(), "sorted_samples_in_queue")

  add_queue_runner(queue,
                   [control.sequential(*[queue.enqueue(tensor_list)
                                         for tensor_list in tensor_lists])])

  return queue


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
