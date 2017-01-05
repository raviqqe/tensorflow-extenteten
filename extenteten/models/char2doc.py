import tensorflow as tf

from ..embedding import id_sequence_to_embedding, embeddings
from ..layer import linear
from ..dropout import dropout
from ..util import func_scope


@func_scope()
def char2doc(document,
             *,
             char_space_size,
             char_embedding_size,
             document_embedding_size,
             dropout_keep_prob,
             hidden_layer_size,
             output_layer_size,
             context_vector_size):
    char_embeddings = embeddings(id_space_size=char_space_size,
                                 embedding_size=char_embedding_size,
                                 name="char_embeddings")

    document_embedding = id_sequence_to_embedding(
        document,
        char_embeddings,
        output_embedding_size=document_embedding_size,
        context_vector_size=context_vector_size)

    hidden_layer = dropout(_activate(linear(_activate(document_embedding),
                                            hidden_layer_size)),
                           dropout_keep_prob)
    return linear(hidden_layer, output_layer_size)


def _activate(tensor):
    return tf.nn.elu(tensor)
