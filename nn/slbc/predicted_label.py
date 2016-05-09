import tensorflow as tf

from ..util import static_rank, funcname_scope



@funcname_scope
def predicted_label(output_layer):
  assert static_rank(output_layer) == 1
  return output_layer > 0.5
