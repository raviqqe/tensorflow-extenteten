import abc



class Model(metaclass=abc.ABCMeta):
  @abc.abstractproperty
  def train_op(self):
    return NotImplemented

  @abc.abstractproperty
  def labels(self):
    return NotImplemented