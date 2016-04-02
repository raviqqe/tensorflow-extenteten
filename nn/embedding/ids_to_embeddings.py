import tensorflow as tf

from .. import var_init



def ids_to_embeddings(ids, id_space_size, embedding_size):
  return tf.nn.embedding_lookup(
      tf.Variable(var_init.normal([id_space_size, embedding_size])),
      ids)
