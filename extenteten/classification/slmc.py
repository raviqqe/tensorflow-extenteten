import tensorflow as tf

from .util import static_shape, static_rank, func_scope


@func_scope()
def loss(output_layer, true_label):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        output_layer,
        true_label))


@func_scope()
def accuracy(output_layer: ("batch", "class"), true_label: ("batch",)):
    assert static_rank(output_layer) == 2
    assert static_rank(true_label) == 1
    assert static_shape(output_layer)[0] == static_shape(true_label)[0]

    return tf.reduce_mean(tf.to_float(tf.equal(
        label(output_layer, dtype=true_label.dtype),
        true_label)))


@func_scope()
def label(output_layer, dtype=None):
    assert static_rank(output_layer) == 2
    label = tf.argmax(output_layer, 1)
    return label if dtype == None else tf.cast(label, dtype)


@func_scope()
def loss_with_summaries(output_layer, true_label):
    tf.scalar_summary("accuracy", accuracy(output_layer, true_label))
    los = loss(output_layer, true_label)
    tf.scalar_summary("loss", los)
    return los
