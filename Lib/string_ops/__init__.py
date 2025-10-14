#
# NO LICENSE IS AVAILABLE YET
#

import enum
import _string_ops

assert _string_ops.MAGIC == 777, "_string_ops module mismatch"


@enum.global_enum
@enum._simple_enum(enum.IntFlag, boundary=enum.KEEP)
class SearchDirection:
    LEFT = 0
    RIGHT = 1
    __str__ = object.__str__
    _numeric_repr_ = hex
