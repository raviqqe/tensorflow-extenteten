import abc
import functools



_FORWARD = "forward"
_BACKWARD = "backward"


class InvertibleLayer(abc.ABC):
  def forward(self, *args):
    return NotImplemented

  def backward(self, *args):
    return NotImplemented


class InvertibleNetwork(InvertibleLayer):
  def __init__(self, *layers):
    assert all(isinstance(layer, InvertibleLayer) for layer in layers)
    self._layers = layers

  def forward(self, *args):
    return _reduce_layers(_FORWARD, *args, **kwargs)

  def backward(self, *args):
    return _reduce_layers(_BACKWARD, *args, **kwargs)

  def _reduce_layers(self, method, *args):
    assert method in {_FORWARD, _BACKWARD}
    return functools.reduce(lambda args, layer: getattr(layer, method)(*args),
                            self._layers,
                            args)
