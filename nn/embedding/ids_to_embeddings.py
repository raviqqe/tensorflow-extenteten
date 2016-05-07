import tensorflow as tf

from ..util import funcname_scope



@funcname_scope
def ids_to_embeddings(ids, embeddings):
  return tf.nn.embedding_lookup(embeddings, ids)
