import tensorflow as tf



def linear(x, output_layer_size, regularizer_scale=1e-8):
  return tf.contrib.layers.fully_connected(
    x,
    output_layer_size,
    activation_fn=tf.nn.elu,
    weight_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale),
  )
