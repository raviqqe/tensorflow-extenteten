from .ids_to_embeddings import ids_to_embeddings
from .embeddings_to_embedding import embeddings_to_embedding
from ..util import static_rank, funcname_scope
from ..dynamic_length import id_sequence_to_length



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
