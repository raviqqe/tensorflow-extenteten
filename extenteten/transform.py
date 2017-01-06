import tensorflow as tf

from .util import func_scope, static_rank, static_shape


__all__ = ['batch', 'dynamic_pack']


@func_scope()
def batch(*tensor_lists_or_dicts):
    type_ = type(tensor_lists_or_dicts[0])

    assert type_ in [list, tuple, dict]
    assert all(type_ == type(obj) for obj in tensor_lists_or_dicts)

    if type_ == tuple:
        type_ = list
        tensor_lists_or_dicts = [list(tuple_)
                                 for tuple_ in tensor_lists_or_dicts]

    assert type_ != dict or all(tensor_lists_or_dicts[0].keys() == dict_.keys()
                                for dict_ in tensor_lists_or_dicts)

    tensor_lists = (tensor_lists_or_dicts if type_ == list else
                    [dict_.values() for dict_ in tensor_lists_or_dicts])

    tensor_list = [dynamic_pack(*tensor_list)
                   for tensor_list in zip(*tensor_lists)]

    if type_ == list:
        return tensor_list

    return {key: tensor for key, tensor
            in zip(tensor_lists_or_dicts[0].keys(), tensor_list)}


@func_scope()
def dynamic_pack(*tensors):
    if tensors[0].dtype == tf.string:
        return tf.pack(tensors)

    shape = _max_shape(tensors)
    return tf.pack([_pad_to_shape(tensor, shape) for tensor in tensors])


@func_scope()
def _pad_to_shape(x, shape):
    shape = tf.to_int32(shape)  # for scalars
    assert static_rank(shape) == 1
    assert static_rank(x) == static_shape(shape)[0]

    paddings = tf.concat(
        1,
        [tf.expand_dims(paddings, 1) for paddings
         in [tf.zeros_like(shape),  shape - tf.shape(x)]])

    return tf.pad(x, paddings)


@func_scope()
def _max_shape(tensors):
    return [_max_dim(i, tensors) for i in range(_rank_of_tensors(tensors))]


@func_scope()
def _rank_of_tensors(tensors):
    rank = static_rank(tensors[0])
    assert all(rank == static_rank(tensor) for tensor in tensors)
    return rank


@func_scope()
def _max_dim(i, tensors):
    return tf.reduce_max([tf.shape(tensor)[i] for tensor in tensors])
