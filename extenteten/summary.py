import tensorflow as tf

from .util import func_scope, static_rank


num_of_summary_images = 8


@func_scope()
def summarize(variable, name=None):
    summary_name = name or variable.name

    mean = tf.reduce_mean(variable)

    return tf.merge_summary([
        tf.scalar_summary("mean/" + summary_name, mean),
        tf.scalar_summary("stddev/" + summary_name,
                          tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))),
        tf.scalar_summary("max/" + summary_name, tf.reduce_max(variable)),
        tf.scalar_summary("min/" + summary_name, tf.reduce_min(variable)),
        tf.histogram_summary(summary_name, variable),
    ])


@func_scope()
def image_summary(variable, name=None):
    summary_name = name or variable.name
    rank = static_rank(variable)

    return tf.image_summary(
        summary_name,
        variable if rank == 4 else
        tf.expand_dims(variable, 3) if rank == 3 else
        tf.expand_dims(tf.expand_dims(variable, 0), 3),
        max_images=256)
