from functools import partial
import tensorflow as tf

from .. import slmc, batch
from ..embedding import embeddings, bidirectional_id_sequence_to_embeddings
from ..dynamic_length import id_sequence_to_length
from ..softmax import softmax
from ..flags import FLAGS
from ..optimize import minimize
from ..util import static_rank, func_scope
from ..model import Model
from ..math import softmax_inverse



class AttentionSumReader(Model):
  @func_scope("attention_sum_reader")
  def __init__(self,
               document: ("batch", "words"),
               query: ("batch", "words"),
               answer: ("batch",)):
    assert static_rank(document) == 2
    assert static_rank(query) == 2
    assert static_rank(answer) == 1

    with tf.variable_scope("word_embeddings"):
      word_embeddings = embeddings(id_space_size=FLAGS.word_space_size,
                                   embedding_size=FLAGS.word_embedding_size,
                                   name="word_embeddings")

    bi_rnn = partial(bidirectional_id_sequence_to_embeddings,
                     embeddings=word_embeddings,
                     dynamic_length=True,
                     output_embedding_size=FLAGS.word_embedding_size)

    with tf.variable_scope("query"):
      query_embedding = bi_rnn(query, output_state=True)
      assert static_rank(query_embedding) == 2

    with tf.variable_scope("document_to_attention"):
      attentions = _calculate_attention(
          bi_rnn(document), query_embedding, id_sequence_to_length(document))
      logits = softmax_inverse(_sum_attentions(attentions, document))

    answer -= FLAGS.first_entity_index
    loss = slmc.loss(logits, answer)

    self._train_op = minimize(loss)
    self._labels = slmc.label(logits)
    self._metrics = {
      "loss": loss,
      "accuracy": slmc.accuracy(logits, answer),
    }

    with tf.variable_scope("debug_values"):
      self._debug_values = {
        "document": tf.reduce_any(tf.is_nan(bi_rnn(document))),
        "query": tf.reduce_any(tf.is_nan(query_embedding)),
        "true_label": answer,
        "predicted_label": self._labels,
      }

  @property
  def train_op(self):
    return self._train_op

  @property
  def labels(self):
    return self._labels

  @property
  def metrics(self):
    return self._metrics

  @property
  def debug_values(self):
    return self._debug_values


@func_scope
def _sum_attentions(attentions, document):
  assert static_rank(attentions) == 2 and static_rank(document) == 2

  num_entities = tf.reduce_max(document) + 1

  @func_scope
  def _sum_attention(args):
    attentions, document = args
    assert static_rank(attentions) == 1 and static_rank(document) == 1
    return tf.unsorted_segment_sum(attentions, document, num_entities)

  attentions = tf.map_fn(_sum_attention,
                         [attentions, document],
                         dtype=FLAGS.float_type)

  return attentions[:, FLAGS.first_entity_index:FLAGS.last_entity_index+1]


@func_scope
def _calculate_attention(document: ("batch", "sequence", "embedding"),
                         query: ("batch", "embedding"),
                         sequence_length):
  assert static_rank(document) == 3
  assert static_rank(query) == 2

  return softmax(batch.mat_vec_mul(document, query), sequence_length)
