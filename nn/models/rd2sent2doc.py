import functools
import tensorflow as tf

from ..embedding import bidirectional_id_sequence_to_embedding, \
                        bidirectional_embeddings_to_embedding
from ..util import static_shape, static_rank, funcname_scope
from ..mlp import mlp
from ..dynamic_length import id_tree_to_root_width, id_sequence_to_length



def rd2sent2doc(document,
                word_embeddings,
                *,
                sentence_embedding_size,
                document_embedding_size,
                dropout_prob,
                hidden_layer_sizes,
                output_layer_size,
                context_vector_size,
                save_memory=False):
  """
  word2sent2doc model lacking word embeddings as parameters
  """

  assert static_rank(document) == 3
  assert static_rank(word_embeddings) == 2

  embeddings_to_embedding = functools.partial(
      bidirectional_embeddings_to_embedding,
      context_vector_size=context_vector_size,
      dropout_prob=dropout_prob)

  with tf.variable_scope("word2sent"):
    # word_embeddings.shape == (#batch * #sent * #word, emb_size)
    #                          if save_memory else
    #                          (vocab_size, emb_size)

    sentences = _flatten_document_into_sentences(document)

    sentence_embeddings = _restore_document_shape(
        embeddings_to_embedding(
            _restore_sentence_shape(word_embeddings, sentences)
                if save_memory else tf.gather(word_embeddings, sentences),
            sequence_length=id_sequence_to_length(sentences),
            output_embedding_size=sentence_embedding_size),
        document)

  with tf.variable_scope("sent2doc"):
    document_embedding = embeddings_to_embedding(
        sentence_embeddings,
        sequence_length=id_tree_to_root_width(document),
        output_embedding_size=document_embedding_size)

  return mlp(document_embedding,
             layer_sizes=[*hidden_layer_sizes, output_layer_size],
             dropout_prob=dropout_prob)


@funcname_scope
def _flatten_document_into_sentences(document):
  return tf.reshape(document, [-1] + static_shape(document)[2:])


@funcname_scope
def _restore_document_shape(sentences, document):
  return tf.reshape(
      sentences,
      [-1, static_shape(document)[1]] + static_shape(sentences)[1:])


@funcname_scope
def _restore_sentence_shape(words, sentences):
  return tf.reshape(
      words,
      [-1, static_shape(sentences)[1]] + static_shape(words)[1:])
