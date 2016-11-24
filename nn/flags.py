import functools
import os

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
add_flag("rnn-cell", dest="_rnn_cell", default="gru")
add_flag("word-embedding-size", type=int, default=200)

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

  @property
  def default_words(self):
    return [ARGS.null_word, ARGS.unknown_word]

  @_cached_property
  def words(self):
    with open(ARGS.word_file) as file_:
      return [*self.default_words,
              *[line.strip() for line in file_.readlines()]]

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


FLAGS = _Flags()
