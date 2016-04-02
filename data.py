import collections
import numpy



# classes

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
  def forward_documents(self):
    if not hasattr(self, "_forward_documents"):
      self._forward_documents \
          = numpy.array([self._reverse_document(array)
                         for array in self.backward_documents])
    return self._forward_documents

  @property
  def backward_documents(self):
    if not hasattr(self, "_backward_documents"):
      self._backward_documents = numpy.fliplr(self._documents)
    return self._backward_documents

  @property
  def labels(self):
    return self._labels

  @property
  def size(self):
    return self._documents.shape[0]

  @staticmethod
  def _reverse_document(array):
    if array[0] != 0:
      return array[::-1]
    padding_length = (array != 0).argmax(axis=0)
    return numpy.concatenate((numpy.zeros((padding_length,),
                                          dtype=numpy.int32),
                             array[:(padding_length - 1):-1]))



# functions

def batches(data, batch_size):
  for index in range(0, data.size, batch_size):
    index_range = slice(index, index + batch_size)
    yield Data(data.documents[index_range],
               data.labels[index_range])


def sample(data, sample_data_size):
  shuffled_data = shuffle(data)
  return Data(shuffled_data.documents[:sample_data_size],
              shuffled_data.labels[:sample_data_size])


def shuffle(data):
  indices = numpy.random.permutation(data.size)
  return Data(data.documents[indices], data.labels[indices])
