import tensorflow as tf

from . import train
from .util import static_shape, static_rank, func_scope


def num_labels(labels):
    assert static_rank(labels) in {1, 2}

    return 1 if static_rank(labels) == 1 else static_shape(labels)[1]


@func_scope()
def classify(logits, label, binary=True):
    assert static_rank(logits) == 2
    assert static_rank(label) in {1, 2}

    predictions, loss = (
        (_classify_label if num_labels(label) == 1 else _classify_labels)
        (logits, label, binary=binary))

    return (predictions,
            loss,
            train.minimize(loss),
            _evaluate(predictions, label))


@func_scope()
def _evaluate(predictions, label):
    recall = tf.contrib.metrics.streaming_recall(predictions, label)
    precision = tf.contrib.metrics.streaming_precision(predictions, label)

    return {
        "accuracy": tf.contrib.metrics.streaming_accuracy(predictions, label),
        "recall": recall,
        "precision": precision,
        "F1": 2 * recall * precision / (recall + precision),
    }


@func_scope()
def _classify_label(logits, label, binary):
    assert static_rank(labels) == 1

    return (_classify_binary_label(logits, label)
            if binary else
            _classify_multi_class_label(logits, label))


@func_scope()
def _classify_binary_label(logits, label):
    return (tf.cast(tf.sigmoid(logits) > 0.5, label.dtype),
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits,
                tf.cast(label, logits.dtype))))


@func_scope()
def _classify_multi_class_label(logits, label):
    return (tf.argmax(logits, 1),
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits,
                label)))


@func_scope()
def _classify_labels(logits, labels, binary):
    assert static_rank(labels) == 2

    predictions, losses = map(list, zip(*[
        _classify_label(logits_per_label, label, binary)
        for logits_per_label, label
        in zip(tf.split(1, num_labels(labels), logits),
               tf.unstack(tf.transpose(labels)))
    ]))

    return (tf.transpose(tf.stack(predictions)),
            tf.reduce_mean(tf.stack(losses)))
