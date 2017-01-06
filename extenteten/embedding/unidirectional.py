import tensorflow as tf

from ..attention import attention_please
from ..dynamic_length import id_vector_to_length
from ..rnn import rnn
from ..util import static_rank, func_scope


@func_scope()
def embeddings_to_embedding(embeddings,
                            *,
                            context_vector_size,
                            sequence_length=None,
                            **rnn_hyperparams):
    assert static_rank(embeddings) == 3

    return attention_please(
        rnn(embeddings, sequence_length=sequence_length, **rnn_hyperparams),
        context_vector_size=context_vector_size,
        sequence_length=sequence_length)


@func_scope()
def id_vector_to_embedding(id_vector,
                           embeddings,
                           *,
                           dynamic_length=False,
                           **kwargs):
    assert static_rank(id_vector) == 2

    return embeddings_to_embedding(
        tf.nn.embedding_lookup(embeddings, id_vector),
        sequence_length=(id_vector_to_length(id_vector)
                         if dynamic_length else
                         None),
        **kwargs)
