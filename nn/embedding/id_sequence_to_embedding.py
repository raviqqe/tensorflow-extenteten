from .ids_to_embeddings import ids_to_embeddings
from .embeddings_to_embedding import embeddings_to_embedding
from ..util import static_rank


def id_sequence_to_embedding(child_id_sequence,
                             child_embeddings,
                             *,
                             output_embedding_size,
                             context_vector_size):
  assert static_rank(child_id_sequence) == 2

  return embeddings_to_embedding(
      ids_to_embeddings(child_id_sequence, child_embeddings),
      output_embedding_size=output_embedding_size,
      context_vector_size=context_vector_size)
