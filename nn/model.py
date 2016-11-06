import abc
import tensorflow as tf



class Model(metaclass=abc.ABCMeta):
  @abc.astractmethod
  def __init__(self, **hyperparameters_and_initial_parameters):
    return NotImplemented

  @abc.astractmethod
  def train(self, *input_tensors) -> tf.Operation: # training operation
    return NotImplemented

  @abc.astractmethod
  def test(self, *input_tensors) -> tf.Tensor: # labels
    return NotImplemented
