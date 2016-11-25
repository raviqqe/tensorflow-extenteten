import tensorflow as tf

from ..flags import FLAGS
from ..variable import variable



def embeddings(*, id_space_size, embedding_size, name=None):
  return variable([id_space_size, embedding_size], name=name)


def word_embeddings(name="word_embeddings"):
  if FLAGS.word_embeddings is None:
    return embeddings(id_space_size=FLAGS.word_space_size,
                      embedding_size=FLAGS.word_embedding_size,
                      name=name)
  return tf.Variable(tf.cast(FLAGS.word_embeddings, FLAGS.float_type),
                     name=name)
