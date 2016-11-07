from functools import partial
import tensorflow as tf

from .. import flags, slmc, train
from ..embedding import embeddings, bidirectional_id_sequence_to_embeddings
from ..flags import FLAGS
from ..optimize import minimize
from ..util import static_rank, funcname_scope



tf.app.flags.DEFINE_integer("word-embedding-size", 200, "")
tf.app.flags.DEFINE_integer("first-entity-index", None, "")
tf.app.flags.DEFINE_integer("last-entity-index", None, "")



@funcname_scope
def attention_sum_reader(document: ("batch", "words"),
                         query: ("batch", "words"),
                         answer: ("batch",)):
  assert static_rank(document) == 2
  assert static_rank(query) == 2
  assert static_rank(answer) == 1

  with tf.variable_scope("word_embeddings"):
    word_embeddings = embeddings(id_space_size=flags.word_space_size(),
                                 embedding_size=FLAGS.word_embedding_size,
                                 name="word_embeddings")

  bi_rnn = partial(bidirectional_id_sequence_to_embeddings,
                   embeddings=word_embeddings,
                   dynamic_length=True,
                   output_embedding_size=FLAGS.word_embedding_size)

  with tf.variable_scope("query"):
    query_word_embs = bi_rnn(query)
    query_embedding = tf.reshape(
        tf.concat(1, [query_word_embs[:, 0:1], query_word_embs[:, -2:-1]]),
        [static_shape(query)[0], word_embedding_size * 2])

  with tf.variable_scope("document_to_attention"):
    # entity_mask = tf.cast(tf.logical_and(document > FLAGS.first_entity_index,
    #                                      document < FLAGS.last_entity_index),
    #                       tf.int32)

    # entity_embeddings = tf.dynamic_partition(bi_rnn(document), entity_mask, 2)[0]

    prob = tf.unsorted_segment_sum(
        _calculate_attention(bi_rnn(document),
                             query_embedding,
                             id_sequence_to_length(document)),
        document,
        tf.reduce_max(document) + 1)

  return minimize(slmc.loss(tf.log(prob), answer)), slmc.label(prob)


@funcname_scope
def _calculate_attention(es: ("batch", "sequence", "embedding"),
                         q: ("batch", "embedding"),
                         sequence_length=None):
  assert static_rank(es) == 3
  assert static_rank(q) == 2

  return softmax(tf.squeeze(tf.batch_matmul(es, tf.expand_dims(q, 2)), [2]),
                 sequence_length)
