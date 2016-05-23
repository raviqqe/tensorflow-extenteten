import tensorflow as tf

from ..embedding import embeddings
from ..util import static_rank, funcname_scope
from .ar2word2sent2doc import ar2word2sent2doc



@funcname_scope
def char2word2sent2doc(document,
                       *,
                       words,
                       char_space_size,
                       char_embedding_size,
                       **ar2word2sent2doc_hyper_params):
  """
  The argument `document` is in the shape of
  (batch size, #sentences / document, #words / sentence).
  """

  assert static_rank(document) == 3
  assert static_rank(words) == 2

  with tf.variable_scope("char_embedding"):
    char_embeddings = embeddings(id_space_size=char_space_size,
                                 embedding_size=char_embedding_size,
                                 name="char_embeddings")

  return ar2word2sent2doc(document,
                          words=words,
                          char_embeddings=char_embeddings,
                          **ar2word2sent2doc_hyper_params)
