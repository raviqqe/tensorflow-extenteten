import logging



class Metrics(dict):
  def __setitem__(self, name, item):
    if name in self:
      raise ValueError("A value for a key {!r} is already set to {!r}."
                       .format(name, self[name]))

    return super().__setitem__(name, item)


METRICS = Metrics()


def init_logger(level):
  logging.getLogger().setLevel(getattr(logging, level.upper()))
  logging.basicConfig(format="[%(asctime)s]PID-%(process)d:"
                             "%(module)s.%(funcName)s(): %(message)s")
