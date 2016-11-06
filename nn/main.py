import tensorflow as tf

from . import train
from .flags import FLAGS
from .file import read_files



# Flags

## Cluster

list_of_hosts = "Comma-separated list of hostname:port pairs"
tf.app.flags.DEFINE_string("ps-hosts", None, list_of_hosts)
tf.app.flags.DEFINE_string("worker-hosts", None, list_of_hosts)

tf.app.flags.DEFINE_string("job-name", None, "'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task-index", None, "Task index within the job")

## Data files

tf.app.flags.DEFINE_string(
    "file-glob", None, "File path glob to search data files")
tf.app.flags.DEFINE_string("file-format", None, "Data format of files")

## Others

tf.app.flags.DEFINE_integer("num-epochs", None, "Number of epochs")
tf.app.flags.DEFINE_string(
    "log-dir", None, "Log directory containing checkpoint and event files")



def main(model):
  def run(_):
    cluster = tf.train.ClusterSpec({
      "ps": FLAGS.ps_hosts.split(","),
      "worker": FLAGS.worker_hosts.split(","),
    })

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
      server.join()
    elif FLAGS.job_name == "worker":
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:{}".format(FLAGS.task_index),
          cluster=cluster)):
        global_step = train.global_step()

        inputs = read_files(FLAGS.file_glob, FLAGS.file_format)
        train_op = model(*inputs).train_op

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()

      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                               logdir=FLAGS.log_dir,
                               init_op=init_op,
                               summary_op=summary_op,
                               saver=saver,
                               global_step=global_step)

      with sv.managed_session(server.target) as sess, sess.as_default():
        step = global_step.eval()
        while not sv.should_stop() and step < FLAGS.num_epochs: # TODO: num_epochs != num_steps
          _, step = sess.run([train_op, global_step])

      sv.stop()
    else:
      raise ValueError("Invalid job_name: {}".format(FLAGS.job_name))

  tf.app.run(run)
