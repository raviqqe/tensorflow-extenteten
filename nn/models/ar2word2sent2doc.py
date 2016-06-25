import tensorflow as tf

from ..embedding import bidirectional_id_sequence_to_embedding
from ..util import static_shape, static_rank, funcname_scope
from .rd2sent2doc import rd2sent2doc



def ar2word2sent2doc(document,
                     *,
                     words,
                     char_embeddings,
                     word_embedding_size,
                     dropout_prob,
                     context_vector_size,
                     **rd2sent2doc_hyperparams):
  """
  char2word2sent2doc model lacking character embeddings as parameters
  """

  assert static_rank(document) == 3
  assert static_rank(words) == 2
  assert static_rank(char_embeddings) == 2

  with tf.variable_scope("char2word"):
    word_embeddings = _restore_document_shape(
        bidirectional_id_sequence_to_embedding(
            tf.gather(words, _flatten_document_into_word_ids(document)),
            char_embeddings,
            output_embedding_size=word_embedding_size,
            context_vector_size=context_vector_size,
            dropout_prob=dropout_prob,
            dynamic_length=True),
        document)

  return rd2sent2doc(word_embeddings,
                     dropout_prob=dropout_prob,
                     context_vector_size=context_vector_size,
                     **rd2sent2doc_hyperparams)


@funcname_scope
def _flatten_document_into_word_ids(document):
  return tf.reshape(document, [-1])


@funcname_scope
def _restore_document_shape(words, document):
  return tf.reshape(
      words,
      [-1] + static_shape(document)[1:3] + static_shape(words)[1:])
