import tensorflow as tf

from ..util import static_rank, func_scope
from ..rnn import bidirectional_rnn
from ..attention import attention_please
from ..dynamic_length import id_sequence_to_length
from .ids_to_embeddings import ids_to_embeddings


@func_scope()
def bidirectional_embeddings_to_embedding(embeddings,
                                          *,
                                          context_vector_size,
                                          sequence_length=None,
                                          **rnn_hyperparams):
    return attention_please(
        bidirectional_rnn(embeddings,
                          sequence_length=sequence_length,
                          **rnn_hyperparams),
        context_vector_size,
        sequence_length=sequence_length)


@func_scope()
def bidirectional_id_sequence_to_embedding(id_sequence,
                                           embeddings,
                                           *,
                                           dynamic_length=False,
                                           **kwargs):
    assert static_rank(id_sequence) == 2

    return bidirectional_embeddings_to_embedding(
        ids_to_embeddings(id_sequence, embeddings),
        sequence_length=id_sequence_to_length(id_sequence)
        if dynamic_length else None,
        **kwargs)


@func_scope()
def bidirectional_id_sequence_to_embeddings(id_sequence,
                                            embeddings,
                                            *,
                                            dynamic_length=False,
                                            **kwargs):
    assert static_rank(id_sequence) == 2

    return bidirectional_rnn(
        ids_to_embeddings(id_sequence, embeddings),
        sequence_length=id_sequence_to_length(id_sequence)
        if dynamic_length else None,
        **kwargs)
