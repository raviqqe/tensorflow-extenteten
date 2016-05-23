import tensorflow as tf

from .conv import lenet
from .layer import fully_connected
from .util import static_rank, static_shape, funcname_scope



@funcname_scope
def font2char(fonts, *, dropout_prob, char_embedding_size):
  assert static_rank(fonts) == 3

  return fully_connected(tf.reshape(lenet(tf.expand_dims(fonts, -1)),
                                    [static_shape(fonts)[0], -1]),
                         dropout_prob=dropout_prob,
                         output_layer_size=char_embedding_size,
                         activate=None)
