import functools

from .multi_label import classify_with_single_label_module
from . import slbc



classify = functools.partial(classify_with_single_label_module,
                             single_label_module=slbc)
