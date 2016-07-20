import collections



def is_int(num):
  return isinstance(num, int) \
         or isinstance(num, numpy.integer) \
         or (isinstance(num, numpy.ndarray)
             and num.ndim == 0
             and issubclass(num.dtype.type, numpy.integer))


def is_natural_num(num):
  return is_int(num) and num > 0


def is_natural_num_sequence(num_list, length=None):
  return isinstance(num_list, collections.Sequence) and \
         all(is_natural_num(num) for num in num_list) and \
         (length == None or len(num_list) == length)
