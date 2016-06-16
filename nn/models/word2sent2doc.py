import tensorflow as tf

from ..embedding import embeddings
from ..util import static_rank, funcname_scope
from .rd2sent2doc import rd2sent2doc



@funcname_scope
def word2sent2doc(document,
                  *,
                  word_space_size,
                  word_embedding_size,
                  **rd2sent2doc_hyperparams):
  assert static_rank(document) == 3

  with tf.variable_scope("word_embeddings"):
    word_embeddings = tf.gather(
        embeddings(id_space_size=word_space_size,
                   embedding_size=word_embedding_size,
                   name="word_embeddings"),
        document)

  return rd2sent2doc(word_embeddings, **rd2sent2doc_hyperparams)
