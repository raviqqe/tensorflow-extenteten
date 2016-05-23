import tensorflow as tf

from .conv import lenet
from .util import static_rank, static_shape



def font2char(fonts):
  assert static_rank(fonts) == 3

  return tf.reshape(lenet(tf.expand_dims(fonts, -1)),
                    static_shape(fonts)[0] + [-1])
