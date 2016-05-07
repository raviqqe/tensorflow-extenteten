import tensorflow as tf

from ..embedding import bidirectional_embeddings_to_embedding, \
                        bidirectional_id_sequence_to_embedding, \
                        embeddings
from ..linear import linear
from ..dropout import dropout
from ..util import static_shape, static_rank



def char2word2sent2doc(document,
                       *,
                       words,
                       char_space_size,
                       char_embedding_size,
                       word_embedding_size,
                       sentence_embedding_size,
                       document_embedding_size,
                       dropout_prob,
                       hidden_layer_size,
                       output_layer_size,
                       context_vector_size):
  with tf.name_scope("char2word2sent2doc"):
    with tf.variable_scope("char_embedding"):
      char_embeddings = embeddings(id_space_size=char_space_size,
                                   embedding_size=char_embedding_size)

    with tf.variable_scope("word_embedding"):
      word_embeddings = bidirectional_id_sequence_to_embedding(
          words,
          char_embeddings,
          output_embedding_size=word_embedding_size,
          context_vector_size=context_vector_size,
          dropout_prob=dropout_prob)

    with tf.variable_scope("sentence_embedding"):
      sentence_embeddings = bidirectional_id_sequence_to_embedding(
          _flatten_document_to_sentences(document),
          word_embeddings,
          output_embedding_size=sentence_embedding_size,
          context_vector_size=context_vector_size,
          dropout_prob=dropout_prob)

    with tf.variable_scope("document_embedding"):
      document_embedding = bidirectional_embeddings_to_embedding(
          _restore_document_shape(sentence_embeddings, document),
          output_embedding_size=document_embedding_size,
          context_vector_size=context_vector_size,
          dropout_prob=dropout_prob)

    hidden_layer = dropout(_activate(linear(_activate(document_embedding),
                                            hidden_layer_size)),
                           dropout_prob)
    return linear(hidden_layer, output_layer_size)


def _concat(tensors):
  return tf.concat(1, list(tensors))


def _restore_document_shape(sentences, document):
  return tf.reshape(sentences, [-1] + [static_shape(document)[1]]
                                    + static_shape(sentences)[1:])


def _flatten_document_to_sentences(document):
  return tf.reshape(document, [-1] + static_shape(document)[2:])


def _activate(tensor):
  return tf.nn.elu(tensor)
