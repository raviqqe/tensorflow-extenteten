import functools
import numpy
import tensorflow as tf


def static_shape(tensor):
    return tf.convert_to_tensor(tensor).get_shape().as_list()


def static_shapes(*tensors):
    return _map_to_list(static_shape, tensors)


def static_rank(tensor):
    return len(static_shape(tf.convert_to_tensor(tensor)))


def static_ranks(*tensors):
    return _map_to_list(static_rank, tensors)


def _map_to_list(func, xs):
    return list(map(func, xs))


def dtypes(*tensors):
    return [tensor.dtype for tensor in tensors]


def func_scope(name=None, initializer=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tf.variable_scope(name or func.__name__,
                                   initializer=initializer):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def on_device(device_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tf.device(device_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def dimension_indices(tensor, start=0):
    return [*range(static_rank(tensor))][start:]


@func_scope()
def dtype_min(dtype):
    return tf.constant(_numpy_min(dtype.as_numpy_dtype))


def _numpy_min(dtype):
    return numpy.finfo(dtype).min


@func_scope()
def dtype_epsilon(dtype):
    return tf.constant(_numpy_epsilon(dtype.as_numpy_dtype))


def _numpy_epsilon(dtype):
    return numpy.finfo(dtype).eps


def flatten(x):
    return tf.reshape(x, [-1])


def rename(x, name):
    return tf.identity(x, name)
