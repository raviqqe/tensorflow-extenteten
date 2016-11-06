import tensorflow as tf



class Model:
  def __init__(self, train_op, labels):
    assert isinstance(train_op, tf.Operation)
    assert isinstance(labels, tf.Tensor)

    self._train_op = train_op
    self._labels = labels

  @property
  def train_op(self):
    return self._train_op

  @property
  def labels(self):
    return self._labels
