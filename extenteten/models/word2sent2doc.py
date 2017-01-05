import tensorflow as tf

from ..embedding import embeddings
from ..util import static_rank, func_scope, flatten
from .rd2sent2doc import rd2sent2doc



@func_scope()
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
        flatten(document))

  return rd2sent2doc(document,
                     word_embeddings,
                     save_memory=True,
                     **rd2sent2doc_hyperparams)
