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
  def labels(self):
    return self._labels

  @property
  def size(self):
    return self._documents.shape[0]



# functions

def batches(data, batch_size):
  for index in range(0, data.size, batch_size):
    index_range = slice(index, index + batch_size)
    yield data.Data(data.documents[index_range],
                    data.labels[index_range])


def sample(data, sample_data_size):
  shuffled_data = shuffle(data)
  return data.Data(shuffled_data.documents[:sample_data_size],
                   shuffled_data.labels[:sample_data_size])


def shuffle(data):
  indices = numpy.random.permutation(data.size)
  return data.Data(data.documents[indices], data.labels[indices])
