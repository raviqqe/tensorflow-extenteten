import functools
import tensorflow as tf

from ..util import static_shape, static_rank, funcname_scope
from ..rnn import rnn
from .embeddings_to_embedding import embeddings_to_embedding
from ..attention import attention_please



@funcname_scope
def bidirectional_embeddings_to_embedding(child_embeddings,
                                          *,
                                          context_vector_size,
                                          output_embedding_size,
                                          **rnn_hyperparams):
  assert output_embedding_size % 2 == 0

  rnn_ = functools.partial(rnn,
                           output_embedding_size=output_embedding_size // 2,
                           **rnn_hyperparams)

  with tf.variable_scope("forward"):
    forward_outputs = rnn_(child_embeddings)

  with tf.variable_scope("backward"):
    backward_outputs = rnn_(_reverse_embedding_sequence(child_embeddings))

  return attention_please(
      _concat_rnn_outputs(forward_outputs, backward_outputs),
      context_vector_size=context_vector_size)


def _concat_rnn_outputs(*rnn_outputs):
  assert all(static_rank(rnn_output) == 3 for rnn_output in rnn_outputs)
  return tf.concat(2, list(rnn_outputs))


def _reverse_embedding_sequence(embedding_sequence):
  assert static_rank(embedding_sequence) == 3
  return tf.reverse(embedding_sequence, [False, True, False])
