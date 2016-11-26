import functools
import logging
import os

import numpy
import tensorflow as tf
import gargparse
from gargparse import ARGS

from . import log



def add_flag(name, *args, **kwargs):
  gargparse.add_argument("--" + name, *args, **kwargs)


def str_list(string):
  return string.split(',')


def int_list(string):
  return [int(num) for num in string.split(',')]


# Cluster (Distributed TensorFlow)

_list_of_hosts = "Comma-separated list of hostname:port pairs"
add_flag("ps-hosts", type=str_list, required=True, help=_list_of_hosts)
add_flag("worker-hosts", type=str_list, required=True, help=_list_of_hosts)
add_flag("job-name", required=True, help="'ps' or 'worker'")
add_flag("task-index", type=int, required=True, help="Task index within a job")

# Hyperparameters

add_flag("num-epochs", type=int, default=64)
add_flag("batch-size", type=int, default=64)
add_flag("dropout-prob", type=float, default=0)

add_flag("word-file")
add_flag("word-embedding-file")
add_flag("rnn-cell", dest="_rnn_cell", default="gru")
add_flag("word-embedding-size", dest="_word_embedding_size", type=int)

# NLP

add_flag("null-word", default="<NULL>")
add_flag("unknown-word", default="<UNKNOWN>")

## QA

add_flag("first-entity")
add_flag("last-entity")

# Data types

add_flag("float-type", type=(lambda name: getattr(tf, name)), default="float32")
add_flag("int-type", type=(lambda name: getattr(tf, name)), default="int32")

# Data files

add_flag("file-glob", required=True, help="File path glob to search data files")
add_flag("file-format", required=True, help="Data format of files")

# Others

add_flag("log-level", default="info", type=log.init_logger)
add_flag("log-dir",
         default="log",
         help="Directory containing checkpoint and event files")
add_flag("debug", action="store_true")
add_flag("num-cpus", type=int)
add_flag("num-threads-per-queue", type=int, default=2)
add_flag("batch-queue-capacity", type=int, default=2)
add_flag("length-boundaries", type=int_list)


def _read_lines(filename):
  with open(filename) as file_:
    return [line.strip() for line in file_.readlines()]


def _cached_property(func):
  @property
  @functools.wraps(func)
  def wrapper(self):
    attr = "_cached_" + func.__name__

    if not hasattr(self, attr):
      setattr(self, attr, func(self))
    return getattr(self, attr)

  return wrapper


class _Flags:
  def __getattr__(self, name):
    try:
      return getattr(ARGS, name)
    except AttributeError:
      return object.__getattribute__(self, name)

  @_cached_property
  def _raw_words(self):
    return _read_lines(ARGS.word_file)

  @_cached_property
  def words(self):
    words = self._raw_words

    for word in [ARGS.unknown_word, ARGS.null_word]:
      if word in words:
        words.remove(word)
      else:
        logging.info('"{}" is added to a word vocabulary.'.format(word))
      words.insert(0, word)

    return words

  @_cached_property
  def word_embeddings(self):
    if ARGS.word_embedding_file is None:
      return None

    embeddings = numpy.array([
        [float(num) for num in line.split(",")]
        for line in _read_lines(ARGS.word_embedding_file)])
    assert len(embeddings) == len(self._raw_words)

    word_to_vector = dict(zip(self._raw_words, embeddings))

    return numpy.array([
        (word_to_vector[word] if word in word_to_vector else
         numpy.random.uniform(high=0.1, size=embeddings.shape[1]))
        for word in self.words])

  @_cached_property
  def word_indices(self):
    return { word: index for index, word in enumerate(self.words) }

  @_cached_property
  def word_space_size(self):
    return len(self.words)

  @property
  def rnn_cell(self):
    from .rnn import cell
    return getattr(cell, ARGS._rnn_cell)

  @property
  def filename_queue_capacity(self):
    return ARGS.batch_queue_capacity * ARGS.batch_size

  @property
  def first_entity_index(self):
    return self.words.index(ARGS.first_entity)

  @property
  def last_entity_index(self):
    return self.words.index(ARGS.last_entity)

  @property
  def word_embedding_size(self):
    assert (self.word_embedding_file is None or
            self._word_embedding_size is None)

    return self._word_embedding_size or self.word_embeddings.shape[1]


FLAGS = _Flags()
