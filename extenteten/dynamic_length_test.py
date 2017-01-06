import numpy as np
import tensorflow as tf

from .dynamic_length import *


def test_id_tree_to_root_width():
    with tf.Session() as session, session.as_default():
        id_tree = tf.constant([[[1], [2], [3], [0], [0]]])
        assert id_tree_to_root_width(id_tree).eval() == np.array([3])


def test_id_vector_to_length():
    with tf.Session() as session, session.as_default():
        id_vector = tf.constant([[1, 2, 3, 0, 0]])
        assert id_vector_to_length(id_vector).eval() == np.array([3])
