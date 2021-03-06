import tensorflow as tf

from ..font import font2char
from ..util import static_rank, func_scope
from .ar2word2sent2doc import ar2word2sent2doc


@func_scope()
def font2char2word2sent2doc(document,
                            *,
                            words,
                            fonts,
                            char_embedding_size,
                            dropout_keep_prob,
                            nums_of_channels,
                            kernel_shape,
                            **ar2word2sent2doc_hyperparams):
    assert static_rank(document) == 3
    assert static_rank(words) == 2
    assert static_rank(fonts) == 3

    return ar2word2sent2doc(
        document,
        words=words,
        char_embeddings=font2char(fonts,
                                  dropout_keep_prob=dropout_keep_prob,
                                  char_embedding_size=char_embedding_size,
                                  nums_of_channels=nums_of_channels,
                                  kernel_shape=kernel_shape),
        dropout_keep_prob=dropout_keep_prob,
        **ar2word2sent2doc_hyperparams)
