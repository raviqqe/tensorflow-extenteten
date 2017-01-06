from functools import partial
import tensorflow as tf

from .. import slmc, batch, collections
from ..embedding import bidirectional_id_sequence_to_embeddings
from ..embedding.embeddings import word_embeddings as _word_embeddings
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

        collections.add_metric(tf.shape(document)[1], "document_2nd_dim")
        collections.add_metric(tf.reduce_max(id_sequence_to_length(document)),
                               "max_document_length")
        collections.add_metric(tf.reduce_min(id_sequence_to_length(document)),
                               "min_document_length")

        word_embeddings = _word_embeddings()

        bi_rnn = partial(bidirectional_id_sequence_to_embeddings,
                         embeddings=word_embeddings,
                         dynamic_length=True,
                         output_size=FLAGS.word_embedding_size)

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
        collections.add_metric(loss, "loss")
        collections.add_metric(slmc.accuracy(logits, answer), "accuracy")

    @property
    def train_op(self):
        return self._train_op

    @property
    def labels(self):
        return self._labels


@func_scope()
def _sum_attentions(attentions, document):
    assert static_rank(attentions) == 2 and static_rank(document) == 2

    num_entities = tf.reduce_max(document) + 1

    @func_scope()
    def _sum_attention(args):
        attentions, document = args
        assert static_rank(attentions) == 1 and static_rank(document) == 1
        return tf.unsorted_segment_sum(attentions, document, num_entities)

    attentions = tf.map_fn(_sum_attention,
                           [attentions, document],
                           dtype=FLAGS.float_type)

    return attentions[:, FLAGS.first_entity_index:FLAGS.last_entity_index + 1]


@func_scope()
def _calculate_attention(document: ("batch", "sequence", "embedding"),
                         query: ("batch", "embedding"),
                         sequence_length):
    assert static_rank(document) == 3
    assert static_rank(query) == 2

    return softmax(batch.mat_vec_mul(document, query), sequence_length)
