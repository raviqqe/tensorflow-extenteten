import collections
import numpy
import tensorflow as tf

from .util import func_scope


def is_int(num):
    return (isinstance(num, int)
            or isinstance(num, numpy.integer)
            or (isinstance(num, numpy.ndarray)
                and num.ndim == 0
                and issubclass(num.dtype.type, numpy.integer)))


def is_natural_num(num):
    return is_int(num) and num > 0


def is_natural_num_sequence(num_list, length=None):
    return (is_sequence(num_list)
            and all(is_natural_num(num) for num in num_list)
            and (length == None or len(num_list) == length))


def is_sequence(obj):
    return isinstance(obj, collections.Sequence)


@func_scope()
def assert_no_nan(tensor):
    return tf.assert_equal(tf.reduce_any(tf.is_nan(tensor)), False)
