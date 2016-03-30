import tensorflow as tf



def id_to_embedding(ids, id_space_size, embedding_size):
  embeddings = tf.Variable(tf.truncated_normal([id_space_size, embedding_size],
                                               stddev=0.1))
  return tf.nn.embedding_lookup(embeddings, ids)
