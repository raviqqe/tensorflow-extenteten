import os
import os.path
import re
import tensorflow as tf
import time



_CHECKPOINT_BASENAME = "model"
_SEPARATOR = "-"
_WAIT_TIME_FOR_MODEL_FILE = 60


class Supervisor:
  def __init__(self, logdir, global_step):
    self._sv = tf.train.Supervisor(
        saver=tf.train.Saver(max_to_keep=2**16),
        save_summaries_secs=0,
        save_model_secs=900,
        checkpoint_basename=_CHECKPOINT_BASENAME,
        logdir=logdir,
        global_step=global_step)

  def session(self, config):
    with self._sv.stop_on_exception():
      return self._sv.managed_session(config=config)

  def should_stop(self):
    return self._sv.should_stop()

  def save_model(self):
    self._sv.saver.save(tf.get_default_session(),
                        self._sv.save_path,
                        self._sv.global_step.eval())

  def add_summary(self, summary):
    self._sv.summary_computed(tf.get_default_session(), summary)


class ModelWaiter:
  def __init__(self, checkpoint_dir, max_global_step):
    self._checkpoint_dir = checkpoint_dir
    self._max_global_step = max_global_step
    self._used_model_filenames = set()
    self._waiting = True

  def youngest_model(self):
    while len(self._unused_model_filenames) == 0:
      _sleep()

    model_filename = min(self._unused_model_filenames,
                         key=_model_filename_to_global_step)
    self._used_model_filenames.add(model_filename)

    if _model_filename_to_global_step(model_filename) >= self._max_global_step:
      self._waiting = False

    return model_filename

  @property
  def _unused_model_filenames(self):
    return (set(_list_model_filenames(self._checkpoint_dir))
            - self._used_model_filenames)

  @property
  def is_waiting(self):
    return self._waiting


def global_step():
  return tf.contrib.train.get_or_create_global_step()


def _global_step_to_model_filename(global_step, checkpoint_dir):
  return os.path.join(
      checkpoint_dir,
      _CHECKPOINT_BASENAME + _SEPARATOR + str(global_step))


def _model_filename_to_global_step(model_filename):
  return int(os.path.basename(model_filename).split(_SEPARATOR)[-1])


def _list_model_filenames(checkpoint_dir):
  return sorted([os.path.join(checkpoint_dir, filename)
                 for filename in _list_dir(checkpoint_dir)
                 if _is_model_filename(filename)],
                key=_model_filename_to_global_step)


def _is_model_filename(filename):
  return re.match(_CHECKPOINT_BASENAME + _SEPARATOR + r"[0-9]+$",
                  os.path.basename(filename))


def wait_nearest_model_filename(global_step, checkpoint_dir):
  while _current_global_step(checkpoint_dir) < global_step:
    _sleep()

  return _global_step_to_model_filename(
      min(_global_steps_in_dir(checkpoint_dir),
          key=lambda n: abs(n - global_step)),
      checkpoint_dir)


def _current_global_step(dirname):
  nums = _global_steps_in_dir(dirname)
  return -1 if nums == [] else max(nums)


def _global_steps_in_dir(dirname):
  return list(map(_model_filename_to_global_step,
                  _list_model_filenames(dirname)))


def _list_dir(dirname):
  return os.listdir(dirname) if os.path.isdir(dirname) else []


def _sleep():
  time.sleep(_WAIT_TIME_FOR_MODEL_FILE)
