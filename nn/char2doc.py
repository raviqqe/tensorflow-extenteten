import tensorflow as tf

from .embedding import embeddings_to_embedding, ids_to_embeddings
from .linear import linear
from .dropout import dropout



def char2doc(document_forward,
             document_backward,
             char_space_size,
             char_embedding_size,
             document_embedding_size,
             dropout_ratio,
             hidden_layer_size,
             output_layer_size):
  char_embeddings = ids_to_embeddings(document_forward,
                                      id_space_size=char_space_size,
                                      embedding_size=char_embedding_size)
  document_embedding = embeddings_to_embedding(char_embeddings,
                                               document_embedding_size)

  hidden_layer = dropout(_activate(linear(_activate(document_embedding),
                                          hidden_layer_size)),
                         dropout_ratio)
  return linear(hidden_layer, output_layer_size)


def _activate(tensor):
  return tf.nn.elu(tensor)
