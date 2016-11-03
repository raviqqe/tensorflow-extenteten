from functools import partial
import tensorflow as tf

from ..embedding import embeddings, bidirectional_id_sequence_to_embeddings
from ..util import static_rank, funcname_scope



@funcname_scope
def attention_sum_reader(document,
                         query,
                         *,
                         word_space_size,
                         word_embedding_size,
                         **bi_rnn_hp):
  assert static_rank(document) == 2
  assert static_rank(words) == 2

  with tf.variable_scope("word_embeddings"):
    word_embeddings = embeddings(id_space_size=word_space_size,
                                 embedding_size=word_embedding_size,
                                 name="word_embeddings")

  bi_rnn = partial(bidirectional_id_sequence_to_embeddings,
                   embeddings=word_embeddings,
                   dynamic_length=True,
                   **bi_rnn_hp)

  with tf.variable_scope("query"):
    query_word_embs = bi_rnn(query)
    query_embedding = tf.reshape(
        tf.concat(1, [query_word_embs[:, 0:1], query_word_embs[:, -2:-1]]),
        [static_shape(query)[0], word_embedding_size * 2])

  with tf.variable_scope("word2attention"):
    return tf.unsorted_segment_sum(
        _calculate_attention(bi_rnn(document),
                             query_embedding,
                             id_sequence_to_length(document)),
        document,
        tf.reduce_max(document) + 1)


@funcname_scope
def _calculate_attention(es: ("batch", "sequence", "embedding"),
                         q: ("batch", "embedding"),
                         sequence_length=None):
  assert static_rank(es) == 3
  assert static_rank(q) == 2

  return softmax(tf.squeeze(tf.batch_matmul(es, tf.expand_dims(q, 2)), [2]),
                 sequence_length)
