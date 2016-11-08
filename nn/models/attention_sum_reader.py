from functools import partial
import tensorflow as tf

from .. import flags, slmc, train, flags, batch
from ..embedding import embeddings, bidirectional_id_sequence_to_embeddings
from ..dynamic_length import id_sequence_to_length
from ..softmax import softmax
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
        [tf.shape(query)[0], FLAGS.word_embedding_size * 2])

  with tf.variable_scope("document_to_attention"):
    attentions = _calculate_attention(
        bi_rnn(document), query_embedding, id_sequence_to_length(document))
    prob = _sum_attentions(attentions, document)

  return minimize(slmc.loss(tf.log(prob), answer)), slmc.label(prob)


@funcname_scope
def _sum_attentions(attentions, document):
  assert static_rank(attentions) == 2 and static_rank(document) == 2

  num_entities = tf.reduce_max(document) + 1

  def _sum_attention(args):
    attentions, document = args
    assert static_rank(attentions) == 1 and static_rank(document) == 1
    return tf.unsorted_segment_sum(attentions, document, num_entities)

  return tf.map_fn(_sum_attention,
                   [attentions, document],
                   dtype=FLAGS.float_type)


@funcname_scope
def _calculate_attention(document: ("batch", "sequence", "embedding"),
                         query: ("batch", "embedding"),
                         sequence_length):
  assert static_rank(document) == 3
  assert static_rank(query) == 2

  return softmax(batch.mat_vec_mul(document, query), sequence_length)
