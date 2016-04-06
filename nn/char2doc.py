import tensorflow as tf

from .embedding import embeddings_to_embedding, ids_to_embeddings, embeddings
from .linear import linear
from .dropout import dropout



def char2doc(forward_document,
             backward_document,
             char_space_size,
             char_embedding_size,
             document_embedding_size,
             dropout_prob,
             hidden_layer_size,
             output_layer_size,
             context_vector_size):
  with tf.name_scope("char2doc"):
    char_embeddings = embeddings(id_space_size=char_space_size,
                                 embedding_size=char_embedding_size)

    def char_ids_to_doc_embedding(document):
      return embeddings_to_embedding(
          ids_to_embeddings(document, char_embeddings),
          output_embedding_size=document_embedding_size,
          context_vector_size=context_vector_size)

    document_embedding = tf.concat(
        1,
        list(map(char_ids_to_doc_embedding,
                 [forward_document, backward_document])))

    hidden_layer = dropout(_activate(linear(_activate(document_embedding),
                                            hidden_layer_size)),
                           dropout_prob)
    return linear(hidden_layer, output_layer_size)


def _activate(tensor):
  return tf.nn.elu(tensor)
