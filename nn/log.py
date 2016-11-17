import logging



def init_logger(level):
  logging.getLogger().setLevel(getattr(logging, level.upper()))
  logging.basicConfig(format="[%(asctime)s]PID-%(process)d:"
                             "%(module)s.%(funcName)s(): %(message)s")
