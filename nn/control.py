import tensorflow as tf



def unpack_to_array(tensor):
  return tf.TensorArray(tensor.dtype, tf.shape(tensor)[0]).unpack(tensor)
