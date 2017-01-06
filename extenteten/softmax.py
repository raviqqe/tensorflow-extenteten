import tensorflow as tf

from . import batch
from .util import static_rank, func_scope, dtype_min, dtype_epsilon


__all__ = ['softmax']


@func_scope()
def softmax(vector, sequence_length=None):
    assert static_rank(vector) == 2

    return (tf.nn.softmax(vector)
            if sequence_length is None else
            _dynamic_softmax(vector, sequence_length))


@func_scope()
def _dynamic_softmax(vector, sequence_length):
    mask_ = tf.cast(tf.sequence_mask(sequence_length, tf.shape(vector)[1]),
                    vector.dtype)
    vector_with_min = mask_ * vector + (1 - mask_) * dtype_min(vector.dtype)

    unnormal_dist = tf.exp(vector_with_min
                           - batch.max(vector_with_min, keep_dims=True)) * mask_

    return unnormal_dist / (batch.sum(unnormal_dist, keep_dims=True)
                            + dtype_epsilon(unnormal_dist.dtype))
