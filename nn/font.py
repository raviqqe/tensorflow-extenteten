import tensorflow as tf

from .conv import invertible_multi_conv
from .layer import fully_connected
from .random import sample_crop
from .summary import image_summary, num_of_summary_images
from .util import static_rank, static_shape, func_scope



@func_scope()
def font2char(fonts, *, dropout_prob, char_embedding_size, **conv_hyperparams):
  assert static_rank(fonts) == 3

  images = tf.expand_dims(fonts, -1)
  image_summary(sample_crop(images, num_of_summary_images))

  return fully_connected(
      tf.reshape(invertible_multi_conv(images, **conv_hyperparams),
                 [static_shape(fonts)[0], -1]),
      dropout_prob=dropout_prob,
      output_layer_size=char_embedding_size,
      activate=tf.nn.elu)
