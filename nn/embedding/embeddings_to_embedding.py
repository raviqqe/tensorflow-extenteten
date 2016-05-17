import tensorflow as tf

from ..util import static_rank, funcname_scope, dimension_indices
from ..linear import linear
from ..variable import variable
from ..attention import attention_please
from ..rnn import rnn



@funcname_scope
def embeddings_to_embedding(child_embeddings,
                            *,
                            output_embedding_size,
                            context_vector_size,
                            dropout_prob):
  assert static_rank(child_embeddings) == 3

  cell_outputs = rnn(child_embeddings,
                     output_embedding_size=output_embedding_size,
                     dropout_prob=dropout_prob)

  return attention_please(cell_outputs, context_vector_size)
