from .ids_to_embeddings import ids_to_embeddings
from .embeddings_to_embedding import embeddings_to_embedding
from ..util import static_rank, funcname_scope
from ..dynamic_length import id_sequence_to_length



@funcname_scope
def id_sequence_to_embedding(child_id_sequence,
                             child_embeddings,
                             *,
                             dynamic_length=False,
                             **kwargs):
  assert static_rank(child_id_sequence) == 2

  return embeddings_to_embedding(
      ids_to_embeddings(child_id_sequence, child_embeddings),
      sequence_length=id_sequence_to_length(child_id_sequence)
                      if dynamic_length else None,
      **kwargs)
