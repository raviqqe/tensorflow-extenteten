import numpy as np
import tensorflow as tf

from .classification import classify


def test_classify():
    for logits_shape, labels_shape, num_classes, num_labels in [
            # 1 label, 2 classes
            [64, 64, 2, None],
            [64, 64, 2, 1],
            [64, None, 2, 1],
            # 3 labels, 2 classes
            [[64, 3], [64, 3], 2, None],
            [[64, 3], [64, 3], 2, 3],
            [[64, 3], None, 2, 3],
            # 1 label, 5 classes
            [[64, 5], 64, 5, None],
            [[64, 5], 64, 5, 1],
            [[64, 5], None, 5, 1],
            # 3 labels, 5 classes
            [[64, 15], [64, 3], 5, None],
            [[64, 15], [64, 3], 5, 3],
            [[64, 15], None, 5, 3],
    ]:
        print(logits_shape, labels_shape, num_classes, num_labels)
        print(classify(tf.Variable(np.zeros(logits_shape)),
                       (None
                        if labels_shape is None else
                        tf.constant(np.zeros(labels_shape, np.int32))),
                       num_classes=num_classes,
                       num_labels=num_labels))
