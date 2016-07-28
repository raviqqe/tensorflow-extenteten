import abc



class InvertibleLayer(abc.ABC):
  def forward(self, *args, **kwargs):
    return NotImplemented

  def backward(self, *args, **kwargs):
    return NotImplemented
