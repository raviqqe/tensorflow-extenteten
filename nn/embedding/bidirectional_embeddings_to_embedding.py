import functools
import tensorflow as tf

from ..util import static_shape
from .embeddings_to_embedding import embeddings_to_embedding



def bidirectional_embeddings_to_embedding(child_embeddings,
                                          **kwargs):
  child_embeddings_to_embedding = functools.partial(
      embeddings_to_embedding,
      **kwargs)

  with tf.variable_scope("forward"):
    forward_embedding = child_embeddings_to_embedding(child_embeddings)

  with tf.variable_scope("backward"):
    backward_embedding = child_embeddings_to_embedding(
        _reverse_embedding_sequence(child_embeddings))

  return _concat_embeddings(forward_embedding, backward_embedding)


def _concat_embeddings(*embeddings):
  return tf.concat(1, list(embeddings))


def _reverse_embedding_sequence(embedding_sequence):
  embedding_dim = 1
  return tf.concat(embedding_dim,
                   list(reversed(tf.split(
                     embedding_dim,
                     static_shape(embedding_sequence)[embedding_dim],
                     embedding_sequence))))
