import logging
import time
import tensorflow as tf

from . import train
from .flags import FLAGS
from .file import read_files



def main(model):
  cluster = tf.train.ClusterSpec({
    "ps": FLAGS.ps_hosts,
    "worker": FLAGS.worker_hosts,
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
      inputs = read_files(FLAGS.file_glob, FLAGS.file_format)
      batch_size = tf.shape(inputs[0])[0]
      train_op, _ = model(*inputs[1:])

      saver = tf.train.Saver(max_to_keep=2**16)
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir=FLAGS.log_dir,
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=train.global_step())

    config = tf.ConfigProto(
        inter_op_parallelism_threads=FLAGS.num_cpus,
        intra_op_parallelism_threads=FLAGS.num_cpus,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True))

    with sv.managed_session(server.target, config) as sess, \
          sess.as_default():
      step = train.global_step().eval()
      logging.info("Initial global step: %d", step)
      while not sv.should_stop():
        start_time = time.time()
        _, step, bsize \
            = sess.run([train_op, train.global_step(), batch_size])
        logging.info("#steps = %d, speed = %f examples/sec, batch size = %d",
                     step,
                     bsize / (time.time() - start_time),
                     bsize)
      sv.saver.save(sess, sv.save_path, step)
    sv.stop()
  else:
    raise ValueError("Invalid job_name: {}".format(FLAGS.job_name))
