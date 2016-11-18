import tensorflow as tf

from . import util



ATTENTIONS = "attentions"
METRICS = "metrics"



def add_attention(tensor):
  return tf.add_to_collection(ATTENTIONS, tensor)

def get_attentions():
  return tf.get_collection(ATTENTIONS)


def add_metric(tensor, name=None):
  return tf.add_to_collection(
      METRICS,
      tensor if name is None else util.rename(tensor, name))

def get_metrics():
  return tf.get_collection(METRICS)
