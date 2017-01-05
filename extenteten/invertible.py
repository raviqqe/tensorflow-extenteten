import abc
import functools


_FORWARD = "forward"
_BACKWARD = "backward"


class InvertibleLayer(abc.ABC):

    @abc.abstractmethod
    def forward(self, x):
        return NotImplemented

    @abc.abstractmethod
    def backward(self, x):
        return NotImplemented


class InvertibleNetwork(InvertibleLayer):

    def __init__(self, *layers):
        assert all(isinstance(layer, InvertibleLayer) for layer in layers)
        self._layers = layers

    def forward(self, x):
        return self._reduce_layers(_FORWARD, x)

    def backward(self, x):
        return self._reduce_layers(_BACKWARD, x)

    def _reduce_layers(self, method, x):
        assert method in {_FORWARD, _BACKWARD}
        return functools.reduce(
            lambda x, layer: getattr(layer, method)(x),
            self._layers if method == _FORWARD else reversed(self._layers),
            x)
