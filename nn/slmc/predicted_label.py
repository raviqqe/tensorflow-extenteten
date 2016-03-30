import tensorflow



def predicted_label(output_layer):
  assert len(output_layer.get_shape()) == 2
  return tf.argmax(output_layer, 1)
