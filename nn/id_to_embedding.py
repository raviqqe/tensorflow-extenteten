import tensorflow as tf



def id_to_embedding(ids, max_id, embedding_size):
  embeddings = tf.Variable(tf.truncated_normal((max_id + 1, embedding_size),
                                               stddev=0.1))
  return tf.nn.embedding_lookup(embeddings, ids)
