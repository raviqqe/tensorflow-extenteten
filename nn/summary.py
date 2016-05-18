import tensorflow as tf

from .util import funcname_scope



@funcname_scope
def summarize(variable):
  name = variable.name

  mean = tf.reduce_mean(variable)
  tf.scalar_summary("mean/" + name, mean)
  tf.scalar_summary("stddev/" + name,
                    tf.sqrt(tf.reduce_sum(tf.square(variable - mean))))
  tf.scalar_summary("max/" + name, tf.reduce_max(variable))
  tf.scalar_summary("min/" + name, tf.reduce_min(variable))
  tf.histogram_summary(name, variable)
