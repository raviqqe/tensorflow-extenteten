import functools
import tensorflow as tf

from ..util import static_shape, static_rank, funcname_scope
from .embeddings_to_embedding import embeddings_to_embedding



@funcname_scope
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
  assert static_rank(embedding_sequence) == 3
  return tf.reverse(embedding_sequence, [False, True, False])
