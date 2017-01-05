import tensorflow as tf

from ..variable import variable
from ..util import func_scope


@func_scope()
def embeddings(initial=None,
               id_space_size=None,
               embedding_size=None,
               name=None):
    return variable(initial or [id_space_size, embedding_size], name=name)
