import tensorflow as tf

from ..util import func_scope



@func_scope()
def ids_to_embeddings(ids, embeddings):
  return tf.nn.embedding_lookup(embeddings, ids)
