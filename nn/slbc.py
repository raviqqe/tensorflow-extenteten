import functools
import tensorflow as tf

from .util import static_shape, static_rank, funcname_scope



@funcname_scope
@_squeeze_output_layer
def classify(output_layer, true_label):
  assert static_rank(output_layer) == static_rank(true_label) == 1
  assert static_shape(output_layer) == static_shape(true_label)
  assert true_label.dtype == tf.bool

  return loss(output_layer, true_label), \
         accuracy(output_layer, true_label), \
         predicted_label(output_layer)


@funcname_scope
@_squeeze_output_layer
def loss(output_layer, true_label):
  assert static_rank(output_layer) == static_rank(true_label) == 1
  assert static_shape(output_layer) == static_shape(true_label)
  assert true_label.dtype == tf.bool

  return tf.nn.sigmoid_cross_entropy_with_logits(output_layer,
                                                 tf.to_float(true_label))


@funcname_scope
@_squeeze_output_layer
def accuracy(output_layer, true_label):
  assert static_rank(output_layer) == static_rank(true_label) == 1
  assert static_shape(output_layer) == static_shape(true_label)
  assert true_label.dtype == tf.bool

  return tf.reduce_mean(tf.to_float(tf.equal(predicted_label(output_layer),
                                             true_label)))


@funcname_scope
@_squeeze_output_layer
def predicted_label(output_layer):
  assert static_rank(output_layer) == 1
  return tf.sigmoid(output_layer) > 0.5


def _squeeze_output_layer(func):
  @functools.wraps(func)
  def wrapper(output_layer, *args, **kwargs):
    assert static_rank(output_layer) == 1 or static_rank(output_layer) == 2
    return func(
        tf.squeeze(output_layer, [1]) if static_rank(output_layer) == 2 else
          output_layer,
        *args,
        **kwargs)
  return wrapper
