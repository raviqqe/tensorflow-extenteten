import functools
import logging
import os

import tensorflow as tf
import gargparse
from gargparse import ARGS



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

def _read_words(filename):
  with open(filename) as file_:
    return sorted([line.strip() for line in file_.readlines()])

add_flag("word-file", dest="words", type=_read_words)
add_flag("rnn-cell", dest="_rnn_cell", default="gru")
add_flag("word-embedding-size", type=int, default=200)

# QA

add_flag("first-entity-index", type=int)
add_flag("last-entity-index", type=int)

# Data types

add_flag("float-type", type=(lambda name: getattr(tf, name)), default="float32")
add_flag("int-type", type=(lambda name: getattr(tf, name)), default="int32")

# Data files

add_flag("file-glob", required=True, help="File path glob to search data files")
add_flag("file-format", required=True, help="Data format of files")

# Others

def _set_up_logger(level):
  logging.getLogger().setLevel(getattr(logging, level.upper()))
  logging.basicConfig(format="%(asctime)s::PID-%(process)d:%(module)s:"
                             "%(funcName)s: %(message)s")

add_flag("log-level", default="info", type=_set_up_logger)
add_flag("log-dir",
         default="log",
         help="Directory containing checkpoint and event files")
add_flag("debug", action="store_true")
add_flag("num-cpus", type=int, default=os.cpu_count())
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

  @_cached_property
  def word_indices(self):
    indices = { word: index + 2 for index, word in enumerate(ARGS.words) }
    indices.update({ '<NULL>': 0, '<UNKNOWN>': 1 })
    return indices

  @_cached_property
  def word_space_size(self):
    return len(self.word_indices)

  @property
  def rnn_cell(self):
    from .rnn import cell
    return getattr(cell, ARGS._rnn_cell)

  @property
  def filename_queue_capacity(self):
    return ARGS.batch_queue_capacity * ARGS.batch_size


FLAGS = _Flags()
