import tensorflow as tf



def ids_to_embeddings(ids, embeddings):
  return tf.nn.embedding_lookup(embeddings, ids)
