import functools
import tensorflow as tf

from .util import static_shape, static_rank, funcname_scope



# constants

TRUE_LABEL_TYPE = tf.int64



# functions

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


@funcname_scope
@_squeeze_output_layer
def classify(output_layer, true_label):
  assert static_rank(output_layer) == static_rank(true_label) == 1
  assert static_shape(output_layer) == static_shape(true_label)
  assert true_label.dtype == TRUE_LABEL_TYPE

  return loss(output_layer, true_label), \
         accuracy(output_layer, true_label), \
         predicted_label(output_layer)


@funcname_scope
@_squeeze_output_layer
def loss(output_layer, true_label):
  assert static_rank(output_layer) == static_rank(true_label) == 1
  assert static_shape(output_layer) == static_shape(true_label)
  assert true_label.dtype == TRUE_LABEL_TYPE

  return tf.nn.sigmoid_cross_entropy_with_logits(
      output_layer,
      tf.cast(true_label, output_layer.dtype))


@funcname_scope
@_squeeze_output_layer
def accuracy(output_layer, true_label):
  assert static_rank(output_layer) == static_rank(true_label) == 1
  assert static_shape(output_layer) == static_shape(true_label)
  assert true_label.dtype == TRUE_LABEL_TYPE

  return tf.reduce_mean(tf.to_float(tf.equal(predicted_label(output_layer),
                                             true_label)))


@funcname_scope
@_squeeze_output_layer
def predicted_label(output_layer):
  assert static_rank(output_layer) == 1
  return tf.cast(tf.sigmoid(output_layer) > 0.5, TRUE_LABEL_TYPE)
