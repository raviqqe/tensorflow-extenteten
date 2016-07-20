from .util import is_int


def is_natural_num(num):
  return is_int(num) and num > 0


def is_natural_num_list(num_list, length=None):
  return isinstance(num_list, list) and \
         all(is_natural_num(num) for num in num_list) and \
         (length == None or len(num_list) == length)
