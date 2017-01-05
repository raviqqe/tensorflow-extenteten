from .util import static_shape, static_rank
from .variable import variable


def test_variable():
    shape = [123, 456]
    assert static_shape(variable(shape)) == shape

    initial = [float(n) for n in shape]
    assert static_rank(variable(initial)) == 1
