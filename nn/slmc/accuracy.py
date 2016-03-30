import tensorflow as tf



def accuracy(output_layer, true_label):
  assert len(output_layer.get_shape()) == 2
  #assert output_layer.get_shape()[0] == (batch size)
  #assert output_layer.get_shape()[1] == (number of classes)
  assert len(true_label.get_shape()) == 1
  #assert true_label.get_shape()[0] == (batch size)
  assert output_layer.get_shape()[0] == true_label.get_shape()[0]

  correct_prediction = tf.equal(tf.argmax(output_layer, 1), true_label)
  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
