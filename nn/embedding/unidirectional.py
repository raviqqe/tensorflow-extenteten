import tensorflow as tf

from ..attention import attention_please
from ..dynamic_length import id_sequence_to_length
from ..rnn import rnn
from ..util import static_rank, funcname_scope
from .ids_to_embeddings import ids_to_embeddings



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


@funcname_scope
def id_sequence_to_embedding(id_sequence,
                             embeddings,
                             *,
                             dynamic_length=False,
                             **kwargs):
  assert static_rank(id_sequence) == 2

  return embeddings_to_embedding(
      ids_to_embeddings(id_sequence, embeddings),
      sequence_length=id_sequence_to_length(id_sequence)
                      if dynamic_length else None,
      **kwargs)
