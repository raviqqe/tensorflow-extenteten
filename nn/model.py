import abc



class Model(metaclass=abc.ABCMeta):
  @abc.abstractproperty
  def train_op(self):
    return NotImplemented

  @abc.abstractproperty
  def labels(self):
    return NotImplemented

  @property
  def metrics(self):
    return {}

  @property
  def debug_values(self):
    return {}
