import tensorflow as tf

from ..embedding import bidirectional_embeddings_to_embedding, \
                        bidirectional_id_sequence_to_embedding, \
                        embeddings
from ..layer import linear
from ..dropout import dropout
from ..util import static_shape, static_rank, funcname_scope
from ..mlp import mlp



@funcname_scope
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
  """
  The argument `document` is in the shape of
  (batch size, #sentences / document, #words / sentence).
  """

  assert static_rank(document) == 3
  assert static_rank(words) == 2

  with tf.variable_scope("char_embedding"):
    char_embeddings = embeddings(id_space_size=char_space_size,
                                 embedding_size=char_embedding_size,
                                 name="char_embeddings")

  with tf.variable_scope("word_embedding"):
    word_embeddings = bidirectional_id_sequence_to_embedding(
        tf.gather(words, _flatten_document_to_words(document)),
        char_embeddings,
        output_embedding_size=word_embedding_size,
        context_vector_size=context_vector_size,
        dropout_prob=dropout_prob)

  with tf.variable_scope("sentence_embedding"):
    sentence_embeddings = bidirectional_embeddings_to_embedding(
        _restore_sentence_shape(word_embeddings, document),
        output_embedding_size=sentence_embedding_size,
        context_vector_size=context_vector_size,
        dropout_prob=dropout_prob)

  with tf.variable_scope("document_embedding"):
    document_embedding = bidirectional_embeddings_to_embedding(
        _restore_document_shape(sentence_embeddings, document),
        output_embedding_size=document_embedding_size,
        context_vector_size=context_vector_size,
        dropout_prob=dropout_prob)

  return mlp(document_embedding,
             layer_sizes=[hidden_layer_size, output_layer_size],
             dropout_prob=dropout_prob)


def _restore_document_shape(sentences, document):
  return tf.reshape(
      sentences,
      [-1, static_shape(document)[1]] + static_shape(sentences)[1:])


def _restore_sentence_shape(words, document):
  return tf.reshape(
      words,
      [-1, static_shape(document)[2]] + static_shape(words)[1:])


def _flatten_document_to_words(document):
  return tf.reshape(document, [-1])


def _activate(tensor):
  return tf.nn.elu(tensor)
