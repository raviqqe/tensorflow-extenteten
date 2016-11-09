import functools
import tensorflow as tf
import gargparse
from gargparse import ARGS

from .rnn import cell



def add_flag(name, *args, **kwargs):
  gargparse.add_argument("--" + name, *args, **kwargs)


def str_list(string):
  return string.split(',')


# Hyperparameters

add_flag("num-epochs", type=int, default=64)
add_flag("batch-size", type=int, default=64)
add_flag("dropout-prob", type=float, default=0)

def _read_words(filename):
  with open(filename) as file_:
    return sorted([line.strip() for line in file_.readlines()])

add_flag("word-file", metavar="words", type=_read_words)
add_flag("num-threads-per-queue", type=int, default=2)
add_flag("queue-capacity", type=int, default=2)
add_flag("length-boundaries", type=str_list)
add_flag("rnn-cell", type=(lambda name: getattr(cell, name)), default="ln_lstm")

add_flag("word-embedding-size", type=int, default=200)
add_flag("first-entity-index", type=int)
add_flag("last-entity-index", type=int)

# Data types

add_flag("float-type", type=(lambda name: getattr(tf, name)), default="float32")
add_flag("int-type", type=(lambda name: getattr(tf, name)), default="int32")

# Data files

add_flag("file-glob", required=True, help="File path glob to search data files")
add_flag("file-format", required=True, help="Data format of files")

# Others

add_flag("log-dir",
         default="log",
         help="Directory containing checkpoint and event files")



def _cached_property(func):
  @property
  @functools.wraps()
  def wrapper(self):
    attr = "_cached_" + func.__name__

    if not hasattr(self, attr):
      setattr(func(self), attr)
    return getattr(self, attr)

  return wrapper


class _Flags:
  def __getattr__(self, name):
    return getattr(ARGS, name)

  @_cached_property
  def word_indices(self):
    indices = { word: index + 2 for index, word in enumerate(ARGS.words) }
    indices.update({ '<NULL>': 0, '<UNKNOWN>': 1 })
    return indices

  @_cached_property
  def word_space_size(self):
    return len(self.word_indices)


FLAGS = _Flags()
