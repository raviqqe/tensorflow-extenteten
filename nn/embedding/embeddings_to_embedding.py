import tensorflow as tf

from ..util import static_rank, funcname_scope
from ..attention import attention_please
from ..rnn import rnn



@funcname_scope
def embeddings_to_embedding(child_embeddings,
                            *,
                            context_vector_size,
                            **rnn_hyperparams):
  assert static_rank(child_embeddings) == 3

  return attention_please(rnn(child_embeddings, **rnn_hyperparams),
                          context_vector_size)
