import tensorflow as tf

from ..util import static_rank, funcname_scope
from ..attention import attention_please
from ..rnn import rnn



@funcname_scope
def embeddings_to_embedding(embeddings,
                            *,
                            context_vector_size,
                            sequence_length=None,
                            **rnn_hyperparams):
  assert static_rank(embeddings) == 3

  return attention_please(
      rnn(embeddings, sequence_length=sequence_length, **rnn_hyperparams),
      context_vector_size=context_vector_size,
      sequence_length=sequence_length)
