import tensorflow as tf

from ..font import font2char
from ..util import static_rank, funcname_scope
from .ar2word2sent2doc import ar2word2sent2doc



@funcname_scope
def font2char2word2sent2doc(document,
                            *,
                            words,
                            fonts,
                            **ar2word2sent2doc_hyper_params):
  assert static_rank(document) == 3
  assert static_rank(words) == 2
  assert static_rank(fonts) == 3

  with tf.variable_scope("char_embedding"):
    char_embeddings = font2char(fonts)

  return ar2word2sent2doc(document,
                          words=words,
                          char_embeddings=char_embeddings,
                          **ar2word2sent2doc_hyper_params)