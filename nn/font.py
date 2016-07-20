import tensorflow as tf

from .conv import multi_conv
from .layer import fully_connected
from .util import static_rank, static_shape, funcname_scope



@funcname_scope
def font2char(fonts, *, dropout_prob, char_embedding_size, **conv_hyperparams):
  assert static_rank(fonts) == 3

  images = tf.expand_dims(fonts, -1)
  tf.image_summary("fonts", images, max_images=4096)

  return fully_connected(
      tf.reshape(multi_conv(images, **conv_hyperparams),
                 [static_shape(fonts)[0], -1]),
      dropout_prob=dropout_prob,
      output_layer_size=char_embedding_size,
      activate=tf.nn.elu)
