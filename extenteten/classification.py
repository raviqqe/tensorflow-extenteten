import functools
import tensorflow as tf

from . import train
from .regularization import l2_regularization_loss
from .util import static_shape, static_rank, func_scope


__all__ = ['num_logits', 'num_labels', 'classify']


def num_logits(num_labels, num_classes):
    return num_labels if num_classes == 2 else num_labels * num_classes


def num_labels(labels):
    assert static_rank(labels) in {1, 2}

    return 1 if static_rank(labels) == 1 else static_shape(labels)[1]


_calc_num_labels = num_labels


@func_scope()
def classify(logits,
             true_label=None,
             *,
             num_classes,
             num_labels=None,
             key=None,
             predictions={},
             regularization_scale=1e-8):
    assert isinstance(predictions, dict)

    if num_labels is None:
        assert true_label is not None
        num_labels = _calc_num_labels(true_label)

    assert static_rank(logits) in {1, 2}
    if true_label is not None:
        assert static_rank(true_label) in {1, 2}
        assert num_labels == _calc_num_labels(true_label)
    assert num_classes >= 2
    assert num_labels >= 1

    predicted_labels, loss = (
        (_classify_label
         if num_labels == 1 else
         functools.partial(_classify_labels, num_labels=num_labels))
        (logits, true_label, num_classes=num_classes))

    predictions['label'] = predicted_labels
    if key is not None:
        predictions['key'] = key

    if true_label is None:
        return predictions

    return (predictions,
            loss + l2_regularization_loss(regularization_scale),
            train.minimize(loss),
            _evaluate(predicted_labels, true_label))


@func_scope()
def _evaluate(predictions, label):
    recall = tf.contrib.metrics.streaming_recall(predictions, label)[1]
    precision = tf.contrib.metrics.streaming_precision(predictions, label)[1]

    return {
        "accuracy": tf.contrib.metrics.streaming_accuracy(predictions,
                                                          label)[1],
        "recall": recall,
        "precision": precision,
        "F1": 2 * recall * precision / (recall + precision),
    }


@func_scope()
def _classify_labels(logits, labels=None, *, num_classes, num_labels):
    if labels is not None:
        assert static_rank(labels) == 2

    predictions, losses = map(list, zip(*[
        _classify_label(logits_per_label, label, num_classes=num_classes)
        for logits_per_label, label
        in zip(tf.split(logits, num_labels, axis=1),
               ([None] * num_labels
                if labels is None else
                tf.unstack(tf.transpose(labels))))
    ]))

    return (tf.transpose(tf.stack(predictions)),
            None if labels is None else tf.reduce_mean(tf.stack(losses)))


@func_scope()
def _classify_label(logits, label=None, *, num_classes):
    if label is not None:
        assert static_rank(label) == 1

    return (_classify_binary_label(logits, label)
            if num_classes == 2 else
            _classify_multi_class_label(logits, label))


@func_scope()
def _classify_binary_label(logits, label=None):
    logits = tf.squeeze(logits)
    return (tf.cast(tf.sigmoid(logits) > 0.5,
                    (tf.int64 if label is None else label.dtype)),
            (None
             if label is None else
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                 logits=logits,
                 labels=tf.cast(label, logits.dtype)))))


@func_scope()
def _classify_multi_class_label(logits, label=None):
    return (tf.argmax(logits, 1),
            (None
             if label is None else
             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                 logits=logits,
                 labels=label))))
