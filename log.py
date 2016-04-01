import os.path
import sys



# constants

COMMAND_NAME = os.path.basename(sys.argv[0])



# functions

def message(*messages):
  print("{}:".format(COMMAND_NAME), *messages, file=sys.stderr)


def warn(*messages):
  message("WARNING:", *messages)


def error(*messages):
  message("ERROR:", *messages)
  exit(1)
