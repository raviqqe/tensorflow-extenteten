import numpy as np
import tensorflow as tf

from . import dropout


def test_dropout():
    with tf.Session() as session:
        nums = [1.0, 2, 3]
        x = session.run(dropout.dropout(nums,
                                        0,
                                        tf.contrib.learn.ModeKeys.TRAIN))

        assert (x == np.array(nums, dtype=x.dtype)).all()
