from ..util import static_shape, static_rank


def num_labels(labels):
    assert static_rank(labels) in {1, 2}

    return 1 if static_rank(labels) == 1 else static_shape(labels)[1]
