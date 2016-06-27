import tensorflow as tf

from ..util import funcname_scope
from ..rnn import bidirectional_rnn
from ..attention import attention_please



@funcname_scope
def bidirectional_embeddings_to_embedding(embeddings,
                                          *,
                                          context_vector_size,
                                          sequence_length=None,
                                          **rnn_hyperparams):
  return attention_please(
      bidirectional_rnn(embeddings,
                        sequence_length=sequence_length,
                        **rnn_hyperparams),
      context_vector_size,
      sequence_length=sequence_length)
