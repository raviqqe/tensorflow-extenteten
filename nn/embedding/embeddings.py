import tensorflow as tf

from .. import var_init



def embeddings(*, id_space_size, embedding_size):
  return tf.Variable(var_init.normal([id_space_size, embedding_size]))
