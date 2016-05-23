import tensorflow as tf

from ..embedding import bidirectional_embeddings_to_embedding, \
                        bidirectional_id_sequence_to_embedding
from ..util import static_shape, static_rank, funcname_scope
from ..mlp import mlp



def ar2word2sent2doc(document,
                     *,
                     words,
                     char_embeddings,
                     word_embedding_size,
                     sentence_embedding_size,
                     document_embedding_size,
                     dropout_prob,
                     hidden_layer_sizes,
                     output_layer_size,
                     context_vector_size):
  """
  char2word2sent2doc model lacking character embeddings

  The argument `document` is in the shape of
  (batch size, #sentences / document, #words / sentence).
  """

  assert static_rank(document) == 3
  assert static_rank(words) == 2
  assert static_rank(char_embeddings) == 2

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
             layer_sizes=list(hidden_layer_sizes)+[output_layer_size],
             dropout_prob=dropout_prob)


@funcname_scope
def _restore_document_shape(sentences, document):
  return tf.reshape(
      sentences,
      [-1, static_shape(document)[1]] + static_shape(sentences)[1:])


@funcname_scope
def _restore_sentence_shape(words, document):
  return tf.reshape(
      words,
      [-1, static_shape(document)[2]] + static_shape(words)[1:])


@funcname_scope
def _flatten_document_to_words(document):
  return tf.reshape(document, [-1])