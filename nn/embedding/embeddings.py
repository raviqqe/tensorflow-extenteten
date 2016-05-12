import tensorflow as tf

from ..variable import variable



def embeddings(*, id_space_size, embedding_size, name=None):
  return variable([id_space_size, embedding_size], name=name)
