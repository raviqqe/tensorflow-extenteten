import collections
import numpy



class Data:
  def __init__(self, documents, labels):
    assert isinstance(documents, numpy.ndarray)
    assert isinstance(labels, numpy.ndarray)
    assert documents.shape[0] == labels.shape[0]
    self._documents = documents
    self._labels = labels

  @property
  def documents(self):
    return self._documents

  @property
  def labels(self):
    return self._labels
